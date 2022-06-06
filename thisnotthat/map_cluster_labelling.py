from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from pynndescent import NNDescent

import numpy as np
import numpy.typing as npt
import networkx as nx

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

    return np.array(cluster_vectors), np.array(map_cluster_locations)


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
):
    vector_layers = [cluster_vectors]
    location_layers = [cluster_locations]

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

        for label in np.unique(layer_metaclusters):
            if np.sum(layer_metaclusters == label) <= min_cluster_size:
                continue

            vector = vectors_for_clustering[layer_metaclusters == label].mean(axis=0)
            location = locations_for_clusters[layer_metaclusters == label].mean(axis=0)

            for i, l in enumerate(vector_layers):
                if np.any(
                    pairwise_distances([vector], l, metric=vector_metric)
                    <= cluster_distance_threshold
                ):
                    break
            else:
                layer_vectors.append(vector)
                layer_locations.append(location)

        if len(layer_vectors) >= min_clusters:
            vector_layers.append(layer_vectors)
            location_layers.append(layer_locations)

        n_clusters = n_clusters // 2
        contamination *= contamination_multiplier
        if contamination >= max_contamination:
            contamination = max_contamination

    return vector_layers, location_layers


def adjust_layer_locations(
    fixed_layer, layer_to_adjust, *, spring_constant=0.1, edge_weight=1.0
):
    fixed_node_pos = fixed_layer
    anchor_node_pos = layer_to_adjust
    distances = pairwise_distances(anchor_node_pos, fixed_node_pos, metric="euclidean")

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


def text_labels(
    vector_layers,
    text_representations,
    text_label_dictionary,
    *,
    items_per_label=3,
    vector_metric="cosine",
    pynnd_n_neighbors=40,
    query_size=10,
):
    text_label_nn_index = NNDescent(
        text_representations, metric=vector_metric, n_neighbors=pynnd_n_neighbors,
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


class TextVectorLabelLayers(object):
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
        cluster_vectors, cluster_locations = build_fine_grained_cluster_centers(
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

        self.labels = text_labels(
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
