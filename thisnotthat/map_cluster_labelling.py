from warnings import warn

import pandas as pd
from sklearn.utils import check_random_state
from umap import UMAP
from hdbscan import HDBSCAN
from hdbscan._hdbscan_tree import recurse_leaf_dfs, get_cluster_tree_leaves
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import pairwise_distances
from pynndescent import NNDescent

import numpy as np
import numpy.typing as npt

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import apricot

    HAS_APRICOT = True
except ImportError:
    HAS_APRICOT = False

from typing import *


def string_label_formatter(label_list: List[str]):
    return "\n".join(label_list)


def build_fine_grained_cluster_centers(
    source_vectors,
    map_representation,
    *,
    umap_n_components=5,
    umap_metric="cosine",
    umap_n_neighbors=15,
    hdbscan_min_samples=10,
    hdbscan_min_cluster_size=20,
    random_state=None,
):
    clusterable_representation = UMAP(
        n_components=umap_n_components,
        metric=umap_metric,
        n_neighbors=umap_n_neighbors,
        min_dist=1e-8,
        random_state=random_state,
    ).fit_transform(source_vectors)

    clusterer = HDBSCAN(
        min_samples=hdbscan_min_samples,
        min_cluster_size=hdbscan_min_cluster_size,
        cluster_selection_method="leaf",
    ).fit(clusterable_representation)

    cluster_vectors = [
        np.average(
            source_vectors[clusterer.labels_ == cluster_id],
            weights=clusterer.probabilities_[clusterer.labels_ == cluster_id],
            axis=0,
        )
        for cluster_id in np.unique(clusterer.labels_)
        if cluster_id != -1
    ]

    map_cluster_locations = [
        np.average(
            map_representation[clusterer.labels_ == cluster_id],
            weights=clusterer.probabilities_[clusterer.labels_ == cluster_id],
            axis=0,
        )
        for cluster_id in np.unique(clusterer.labels_)
        if cluster_id != -1
    ]

    return (
        np.array(cluster_vectors),
        np.array(map_cluster_locations),
        clusterer.condensed_tree_,
    )


def hdbscan_tree_based_cluster_merger(tree, clusters_to_merge):
    to_be_merged = list(clusters_to_merge[:])
    leaves_to_merge = set(clusters_to_merge)
    result = []
    while len(to_be_merged) > 0:
        merge_candidate = to_be_merged[0]
        merged_leaves = {merge_candidate}
        while True:
            parent = tree["parent"][tree["child"] == merge_candidate][0]
            leaves_under_parent = set(recurse_leaf_dfs(tree, parent))
            if leaves_under_parent <= leaves_to_merge:
                merge_candidate = parent
                merged_leaves = leaves_under_parent
            else:
                break

        result.append(merge_candidate)
        for leaf in merged_leaves:
            to_be_merged.remove(leaf)

    return result


def point_set_from_cluster(tree, cluster_indices, topic_mask, leaf_mapping):
    cluster_tree = tree[tree["child_size"] > 1]
    cluster = np.arange(topic_mask.shape[0])[topic_mask][cluster_indices]
    leaves_to_merge = [leaf_mapping[x] for x in cluster]
    merged_cluster_set = hdbscan_tree_based_cluster_merger(
        cluster_tree, leaves_to_merge
    )
    result = sorted(sum([recurse_leaf_dfs(tree, x) for x in merged_cluster_set], []))

    return result


def build_cluster_layers(
    cluster_vectors,
    cluster_locations,
    *,
    min_clusters=4,
    contamination=0.05,
    vector_metric="cosine",
    cluster_distance_threshold=0.025,
    contamination_multiplier=1.5,
    max_contamination=0.25,
    return_pointsets=False,
    hdbscan_tree=None,
):
    vector_layers = [cluster_vectors]
    location_layers = [cluster_locations]

    if return_pointsets:
        if hdbscan_tree is None:
            raise ValueError("Must supply a hdbscan_tree if returning pointsets")
        full_tree = hdbscan_tree.to_numpy()
        cluster_tree = full_tree[full_tree["child_size"] > 1]
        leaf_mapping = {
            n: c for n, c in enumerate(sorted(get_cluster_tree_leaves(cluster_tree)))
        }
        pointset_layers = [
            [
                recurse_leaf_dfs(full_tree, leaf_mapping[x])
                for x in range(cluster_vectors.shape[0])
            ]
        ]
    else:
        pointset_layers = []
        full_tree = None
        leaf_mapping = {}

    n_clusters = cluster_vectors.shape[0] // 2
    min_cluster_size = 2

    while n_clusters >= min_clusters:
        robust_cluster_indicator = (
            LocalOutlierFactor(contamination=contamination).fit_predict(cluster_vectors)
            > 0
        )

        vectors_for_clustering = cluster_vectors[robust_cluster_indicator]
        locations_for_clusters = cluster_locations[robust_cluster_indicator]

        layer_metaclusters = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="euclidean", linkage="complete",
        ).fit_predict(locations_for_clusters)

        layer_vectors = []
        layer_locations = []
        layer_pointsets = []

        for label in np.unique(layer_metaclusters):
            if np.sum(layer_metaclusters == label) <= min_cluster_size:
                continue

            vector = vectors_for_clustering[layer_metaclusters == label].mean(axis=0)
            location = locations_for_clusters[layer_metaclusters == label].mean(axis=0)
            if return_pointsets:
                pointset = point_set_from_cluster(
                    full_tree,
                    np.where(layer_metaclusters == label),
                    robust_cluster_indicator,
                    leaf_mapping,
                )
            else:
                pointset = None

            for i, l in enumerate(vector_layers):
                if np.any(
                    pairwise_distances([vector], l, metric=vector_metric)
                    <= cluster_distance_threshold
                ):
                    break
            else:
                layer_vectors.append(vector)
                layer_locations.append(location)
                layer_pointsets.append(pointset)

        if len(layer_vectors) >= min_clusters:
            vector_layers.append(layer_vectors)
            location_layers.append(np.array(layer_locations))
            pointset_layers.append(layer_pointsets)

        n_clusters = n_clusters // 2
        contamination *= contamination_multiplier
        if contamination >= max_contamination:
            contamination = max_contamination

    if return_pointsets:
        return vector_layers, location_layers, pointset_layers
    else:
        return vector_layers, location_layers


def adjust_layer_locations(
    fixed_layer, layer_to_adjust, *, spring_constant=0.1, edge_weight=1.0
):
    fixed_node_pos = fixed_layer
    anchor_node_pos = layer_to_adjust

    position_dict = {}
    fixed_nodes = []
    g = nx.Graph()

    for i in range(fixed_node_pos.shape[0]):
        g.add_node(f"fixed_node_{i}")
        fixed_nodes.append(f"fixed_node_{i}")
        position_dict[f"fixed_node_{i}"] = fixed_node_pos[i]

    for i in range(anchor_node_pos.shape[0]):
        g.add_node(f"anchor_node_{i}")
        fixed_nodes.append(f"anchor_node_{i}")
        position_dict[f"anchor_node_{i}"] = anchor_node_pos[i]

    for i in range(layer_to_adjust.shape[0]):
        g.add_node(f"movable_node_{i}")
        g.add_edge(f"movable_node_{i}", f"anchor_node_{i}", weight=edge_weight)
        position_dict[f"movable_node_{i}"] = layer_to_adjust[i]

    layout = nx.spring_layout(
        g,
        k=spring_constant,
        pos=position_dict,
        fixed=fixed_nodes,
        scale=None,
        center=None,
        dim=fixed_node_pos.shape[1],
        seed=42,
    )
    adjusted_positions = np.array(
        [v for k, v in layout.items() if k.startswith("movable_node_")]
    )

    return adjusted_positions


def text_locations(
    location_layers,
    *,
    spring_constant=0.1,
    spring_constant_multiplier=1.5,
    edge_weight=1.0,
):
    if len(location_layers) <= 1:
        return location_layers

    fixed_layer = np.array(location_layers[0])
    adjustable_layer = np.array(location_layers[1])
    adjusted_locations = adjust_layer_locations(
        fixed_layer,
        adjustable_layer,
        spring_constant=spring_constant,
        edge_weight=edge_weight,
    )
    text_locations = [fixed_layer, adjusted_locations]

    for i in range(2, len(location_layers)):
        fixed_layer = np.vstack(text_locations)
        adjustable_layer = np.array(location_layers[i])

        adjusted_locations = adjust_layer_locations(
            fixed_layer,
            adjustable_layer,
            spring_constant=spring_constant,
            edge_weight=edge_weight,
        )
        text_locations.append(adjusted_locations)

        spring_constant *= spring_constant_multiplier

    return text_locations


def text_labels_from_joint_vector_space(
    vector_layers,
    text_representations,
    text_label_dictionary,
    *,
    items_per_label=3,
    vector_metric="cosine",
    pynnd_n_neighbors=40,
    query_size=10,
    random_state=None,
):
    text_label_nn_index = NNDescent(
        text_representations,
        metric=vector_metric,
        n_neighbors=pynnd_n_neighbors,
        random_state=random_state,
    )
    text_label_nn_index.prepare()

    keyword_set = set([])
    labels = []
    for layer in reversed(vector_layers):
        layer_keyword_set = set([])
        layer_labels = []
        for cluster in layer:
            text_rep_indices, _ = text_label_nn_index.query([cluster], k=query_size)
            nearest_text = [
                x
                for x in text_rep_indices[0]
                if x not in keyword_set and x not in layer_keyword_set
            ]
            layer_keyword_set.update(set(nearest_text[:items_per_label]))
            layer_labels.append(
                [text_label_dictionary[x] for x in nearest_text[:items_per_label]]
            )
        keyword_set.update(layer_keyword_set)
        labels.append(layer_labels)

    return labels[::-1]


def text_labels_from_source_metadata(
    pointset_layers, source_metadataframe, *, items_per_label=3,
):
    n_numeric_cols = source_metadataframe.select_dtypes(
        exclude=["object", "category"]
    ).shape[1]
    logistic_regression_dataframe = pd.get_dummies(
        source_metadataframe, prefix_sep=": "
    )
    logistic_regression_data = RobustScaler().fit_transform(
        logistic_regression_dataframe
    )
    labels = []
    for layer in pointset_layers:
        layer_labels = []
        for cluster in layer:
            target_vector = np.zeros(logistic_regression_data.shape[0], dtype=np.int32)
            target_vector[cluster] = 1

            model = LogisticRegression().fit(logistic_regression_data, target_vector)

            coeff_sign_mask = np.ones_like(model.coef_[0])
            coeff_sign_mask[:n_numeric_cols] = np.sign(model.coef_[0][:n_numeric_cols])
            coeff_order = np.argsort(model.coef_[0] * coeff_sign_mask)[::-1]
            coeff_signs = np.sign(model.coef_[0])[coeff_order]
            cluster_label = [
                f"{('low '  if coeff_signs[i] < 0 else 'high ') if coeff_order[i] < n_numeric_cols else ''}"
                f"{logistic_regression_dataframe.columns[coeff_order[i]]}"
                for i in range(items_per_label)
            ]
            layer_labels.append(cluster_label)

        labels.append(layer_labels)

    return labels


class RandomSampleSelection(object):
    def __init__(self, n_samples, random_state=None):
        self.n_samples = n_samples
        self.random_state = random_state

    def fit_transform(self, X, y=None, **fit_params):
        state = check_random_state(self.random_state)
        indices = state.random_choice(
            np.arange(X.shape[0]), size=self.n_samples, replace=False
        )
        X_result = X[indices]
        if y is not None:
            y_result = y[indices]
            return X_result, y_result
        else:
            return X_result

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y=y, **fit_params)
        return self


def text_labels_from_per_sample_labels(
    pointset_layers,
    source_vectors,
    labels_per_sample,
    *,
    items_per_label=3,
    vector_metric="cosine",
    random_state=None,
):
    if HAS_APRICOT:
        selector = apricot.FacilityLocationSelection(
            items_per_label, metric=vector_metric
        )
    else:
        selector = RandomSampleSelection(items_per_label, random_state=random_state)

    labels = []
    labels_per_sample = np.asarray(labels_per_sample)
    excluded_indices: Set[int] = set([])
    for layer in pointset_layers:
        layer_labels = []
        for cluster in layer:
            cluster_with_exclusion = list(set(cluster) - excluded_indices)
            vector_selection, label_selection = selector.fit_transform(
                source_vectors[cluster_with_exclusion],
                labels_per_sample[cluster_with_exclusion],
            )
            indices = np.where(np.isin(labels_per_sample, label_selection))
            excluded_indices.update(indices)
            layer_labels.append(list(label_selection))

        labels.append(layer_labels)

    return labels


class JointVectorLabelLayers(object):
    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        labelling_vectors: npt.ArrayLike,
        labels: Dict[int, Any],
        *,
        vector_metric="cosine",
        umap_n_components=5,
        umap_n_neighbors=15,
        hdbscan_min_samples=10,
        hdbscan_min_cluster_size=20,
        min_clusters_in_layer=4,
        contamination=0.05,
        cluster_distance_threshold=0.025,
        contamination_multiplier=1.5,
        max_contamination=0.25,
        adjust_label_locations=True,
        label_adjust_spring_constant=0.1,
        label_adjust_spring_constant_multiplier=1.5,
        label_adjust_edge_weight=1.0,
        items_per_label=3,
        pynnd_n_neighbors=40,
        pynnd_query_size=10,
        label_formatter=string_label_formatter,
        random_state=None,
    ):
        if adjust_label_locations and not HAS_NETWORKX:
            warn("NetworkX is required for label adjustments; try pip install networkx")
            adjust_label_locations = False

        cluster_vectors, cluster_locations, _ = build_fine_grained_cluster_centers(
            source_vectors,
            map_representation,
            umap_metric=vector_metric,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            random_state=random_state,
        )
        vector_layers, self.location_layers = build_cluster_layers(
            cluster_vectors,
            cluster_locations,
            min_clusters=min_clusters_in_layer,
            contamination=contamination,
            vector_metric=vector_metric,
            cluster_distance_threshold=cluster_distance_threshold,
            contamination_multiplier=contamination_multiplier,
            max_contamination=max_contamination,
        )
        if adjust_label_locations:
            self.location_layers = text_locations(
                self.location_layers,
                spring_constant=label_adjust_spring_constant,
                spring_constant_multiplier=label_adjust_spring_constant_multiplier,
                edge_weight=label_adjust_edge_weight,
            )

        self.labels = text_labels_from_joint_vector_space(
            vector_layers,
            labelling_vectors,
            labels,
            items_per_label=items_per_label,
            vector_metric=vector_metric,
            pynnd_n_neighbors=pynnd_n_neighbors,
            query_size=pynnd_query_size,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]


class MetadataLabelLayers(object):
    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        metadata_dataframe: pd.DataFrame,
        *,
        vector_metric="cosine",
        umap_n_components=5,
        umap_n_neighbors=15,
        hdbscan_min_samples=10,
        hdbscan_min_cluster_size=20,
        min_clusters_in_layer=4,
        contamination=0.05,
        cluster_distance_threshold=0.025,
        contamination_multiplier=1.5,
        max_contamination=0.25,
        adjust_label_locations=True,
        label_adjust_spring_constant=0.1,
        label_adjust_spring_constant_multiplier=1.5,
        label_adjust_edge_weight=1.0,
        items_per_label=3,
        label_formatter=string_label_formatter,
        random_state=None,
    ):
        if adjust_label_locations and not HAS_NETWORKX:
            warn("NetworkX is required for label adjustments; try pip install networkx")
            adjust_label_locations = False

        (
            cluster_vectors,
            cluster_locations,
            hdbscan_tree,
        ) = build_fine_grained_cluster_centers(
            source_vectors,
            map_representation,
            umap_metric=vector_metric,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            random_state=random_state,
        )
        (
            vector_layers,
            self.location_layers,
            self.pointset_layers,
        ) = build_cluster_layers(
            cluster_vectors,
            cluster_locations,
            min_clusters=min_clusters_in_layer,
            contamination=contamination,
            vector_metric=vector_metric,
            cluster_distance_threshold=cluster_distance_threshold,
            contamination_multiplier=contamination_multiplier,
            max_contamination=max_contamination,
            return_pointsets=True,
            hdbscan_tree=hdbscan_tree,
        )
        if adjust_label_locations:
            self.location_layers = text_locations(
                self.location_layers,
                spring_constant=label_adjust_spring_constant,
                spring_constant_multiplier=label_adjust_spring_constant_multiplier,
                edge_weight=label_adjust_edge_weight,
            )

        self.labels = text_labels_from_source_metadata(
            self.pointset_layers, metadata_dataframe, items_per_label=items_per_label,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]


class SampleLabelLayers(object):
    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        per_sample_labels: npt.ArrayLike,
        *,
        vector_metric="cosine",
        umap_n_components=5,
        umap_n_neighbors=15,
        hdbscan_min_samples=10,
        hdbscan_min_cluster_size=20,
        min_clusters_in_layer=4,
        contamination=0.05,
        cluster_distance_threshold=0.025,
        contamination_multiplier=1.5,
        max_contamination=0.25,
        adjust_label_locations=True,
        label_adjust_spring_constant=0.1,
        label_adjust_spring_constant_multiplier=1.5,
        label_adjust_edge_weight=1.0,
        items_per_label=3,
        label_formatter=string_label_formatter,
        random_state=None,
    ):
        self.per_sample_labels = per_sample_labels

        if not HAS_APRICOT:
            warn(
                "Apricot selection library not found; using random selection. Try pip install apricot-select"
            )

        if adjust_label_locations and not HAS_NETWORKX:
            warn("NetworkX is required for label adjustments; try pip install networkx")
            adjust_label_locations = False

        (
            cluster_vectors,
            cluster_locations,
            hdbscan_tree,
        ) = build_fine_grained_cluster_centers(
            source_vectors,
            map_representation,
            umap_metric=vector_metric,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            random_state=random_state,
        )
        (
            vector_layers,
            self.location_layers,
            self.pointset_layers,
        ) = build_cluster_layers(
            cluster_vectors,
            cluster_locations,
            min_clusters=min_clusters_in_layer,
            contamination=contamination,
            vector_metric=vector_metric,
            cluster_distance_threshold=cluster_distance_threshold,
            contamination_multiplier=contamination_multiplier,
            max_contamination=max_contamination,
            return_pointsets=True,
            hdbscan_tree=hdbscan_tree,
        )
        if adjust_label_locations:
            self.location_layers = text_locations(
                self.location_layers,
                spring_constant=label_adjust_spring_constant,
                spring_constant_multiplier=label_adjust_spring_constant_multiplier,
                edge_weight=label_adjust_edge_weight,
            )

        self.labels = text_labels_from_per_sample_labels(
            self.pointset_layers,
            source_vectors,
            self.per_sample_labels,
            items_per_label=items_per_label,
            vector_metric=vector_metric,
            random_state=random_state,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]
