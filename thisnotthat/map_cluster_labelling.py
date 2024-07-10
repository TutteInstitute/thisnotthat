from warnings import warn

import hdbscan.plots
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
from scipy.sparse import spmatrix
from vectorizers.transformers import InformationWeightTransformer

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

VALID_SELECTION_METHODS = {
    "facility_location",
    "graph_cut",
    "sum_redundancy",
    "saturated_coverage",
}


def string_label_formatter(label_list: List[str]) -> str:
    return "\n".join(label_list)


def build_fine_grained_cluster_centers(
    source_vectors: npt.ArrayLike,
    map_representation: npt.ArrayLike,
    *,
    cluster_map_representation: bool = False,
    umap_n_components: int = 5,
    umap_metric: str = "cosine",
    umap_n_neighbors: int = 15,
    hdbscan_min_samples: int = 10,
    hdbscan_min_cluster_size: int = 20,
    random_state: Optional[int] = None,
):
    """Generate a fine grained clustering either from a UMAP projection of the data, or the map representation
    itself. Return the resulting cluster centroids in the high space, their equivalent centroids in the map
    representation, and the condensed tree of the clusterig itself (useful for cases where centroids are not suitable
    cluster representations).

    Parameters
    ----------
    source_vectors: ArrayLike of shape (n_samples, n_features)
        The original high dimensional vector representation of the data

    map_representation: ArrayLike of shape (n_samples, n_map_features)
        The map representation of the data

    cluster_map_representation: bool (optional, default = False)
        Whether to directly cluster the map representation, or use UMAP to generate a representation for clustering
        using ``umap_n_components`` many dimensions.

    umap_n_components: int (optional, default = 5)
        The number of dimensions to use UMAP to reduce to if ``cluster_map_representation`` is ``False``.

    umap_metric: str (optional, default = "cosine")
        The metric to pass to UMAP for dimension reduction if ``cluster_map_representation`` is ``False``.

    umap_n_neighbors: int (optional, default = 15)
        The number of neighbors to use for UMAP  if ``cluster_map_representation`` is ``False``.

    hdbscan_min_samples: int (optional, default = 10)
        The ``min_samples`` value to use with HDBSCAN for clustering.

    hdbscan_min_cluster_size: int (optional, default = 20)
        The ``min_cluster_size`` value to use with HDBSCAN for clustering.

    random_state: int or None (optional, default = None)
        A random state seed that can be fixed to ensure reproducibility.

    Returns
    -------

    cluster_vectors: ArrayLike of shape (n_clusters, n_features)
        Centroid representations of each of the fine grained clusters found

    map_cluster_locations: ArrayLike of shape (n_clusters, n_map_features)
        Centroid bsed map locations of each of the fine grained clusters found

    condensed_tree: CondensedTree object
        The condensed tree representation of the clustering.
    """
    if cluster_map_representation:
        clusterable_representation = map_representation
    else:
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


def hdbscan_tree_based_cluster_merger(
    tree: hdbscan.plots.CondensedTree, clusters_to_merge: List[int]
) -> List[int]:
    """Given a lost of leaf nodes, find the clusters in the tree that cover all the leaf nodes in the list, and no leaf
    nodes outside of the list, using higher nodes in the tree to merge clusters whenever possible. This provides
    point based representations of sets of fine grained clusters.

    Parameters
    ----------
    tree: CondensedTree
        The condensed tree containing the relevant cluster information for merging

    clusters_to_merge: List of leaf node ids
        The leaf nodes to attempt to cover via tree based merging

    Returns
    -------
    result: List of cluster node ids
        The cluster nodes that cover the input leaf nodes optimally.
    """
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


def point_set_from_cluster(
    tree: hdbscan.plots.CondensedTree,
    cluster_indices: List[int],
    topic_mask: npt.NDArray,
    leaf_mapping: Dict[int, int],
) -> List[int]:
    """Given a list of cluster node ids return the source points falling under those cluster ids. We also need to
    keep track of any masked out clusters, and a mapping to leaf nodes ids.

    Parameters
    ----------
    tree: CondensedTree
         The condensed tree containing the relevant cluster information for merging

    cluster_indices: List of cluster node ids
        The cluster node ids of which to find the underlying points

    topic_mask: ArrayLike of bool of shape (n_clusters,)
        A mask vector determining which leaf nodes from the fine grained clustering to ignore at this time

    leaf_mapping: Dict mapping int to int
        A mapping from cluster label ids of the fine grained clustering to leaf node ids in the condensed tree

    Returns
    -------
    points: List of point indices
        The indices of the points in the input clusters
    """
    cluster_tree = tree[tree["child_size"] > 1]
    cluster = np.arange(topic_mask.shape[0])[topic_mask][cluster_indices]
    leaves_to_merge = [leaf_mapping[x] for x in cluster]
    merged_cluster_set = hdbscan_tree_based_cluster_merger(
        cluster_tree, leaves_to_merge
    )
    result = sorted(sum([recurse_leaf_dfs(tree, x) for x in merged_cluster_set], []))

    return result


def build_cluster_layers(
    cluster_vectors: npt.ArrayLike,
    cluster_locations: npt.ArrayLike,
    *,
    min_clusters: int = 4,
    contamination: float = 0.05,
    contamination_multiplier: float = 1.5,
    max_contamination: float = 0.25,
    vector_metric: str = "cosine",
    cluster_distance_threshold: float = 0.025,
    return_pointsets: bool = False,
    hdbscan_tree: Optional[hdbscan.plots.CondensedTree] = None,
):
    """Given a fine grained clustering generate hierarchical layers of clusters such that each layer is a clustering of
    fine-grained clusters. For this we want compact clusters in the map representation, so we use complete linkage
    on the map representation as the clustering approach. We also want to be wary of duplicating clusters, or
    creating higher level clusters that include otherwise distinct outlying points. We resolve the first issue by
    checking the distances to existing clusters and not using higher level clusters that are too close to existing lower
    level clusters. We resolve the second issue by using outlier detection on the fine grained clusters, progressivly
    removing more outliers for higher level clusterings.

    Depending on whether the desired result is cluster centroids or sets of points a condensed tree may be required.

    Parameters
    ----------
    cluster_vectors: ArrayLike of shape (n_clusters, n_features)
        The centroid vector representations in terms of the source vector data

    cluster_locations: ArrayLike of shape (n_clusters, n_map_features)
        The centroid map locations of the clusters

    min_clusters: int (optional, default = 4)
        The number of clusters to have at the highest layer; layers with fewer than this number of clusters will
        be discarded

    contamination: float (optional, default = 0.05)
        The base contamination score used for outlier detection of fine grained clusters. Larger values will
        prune out more outliers

    contamination_multiplier: float (optional, default = 1.5)
        The value to multiply the contamination score by as we increase the layers -- thus applying
        higher contamination and removing more outliers from higher layers. Larger values will prune
        more aggressively

    max_contamination: float (optional, default = 0.25)
        The maximum contamination value to use in outlier pruning -- once the multiplier increases
        contamination beyond this value the contamination used will simply be capped at this value.

    vector_metric: str (optional, default = "cosine")
        The metric to use on the source vector space. This is used to determine if cluster centroid representatives
        are too close and should be ignored.

    cluster_distance_threshold: float (optional, default = 0.025)
        Cluster centroid representatives from a higher layer that are within this distance of an already selected
        cluster centroid in a lower layer will be ignored (so we don't repeat clusters)

    return_pointsets: bool (optional, default = False)
        Whether to return point set data for clusters. This may be required for various approaches to cluster labelling.

    hdbscan_tree: CondensedTree or None
        If ``return_pointsets`` is ``True`` then a condensed tree must be provided to generate the relevant pointsets.
        If ``return_pointsets`` is ``False`` then this can be ``None`` as it will not be used.

    Returns
    -------
    vector_layers: List of list Arrays
        A list of layers; each layer is a list of arrays of the cluster centroids for that layer

    location_layers: List of list of Arrays
        A list of layers; each layer is a list of arrays of map locations for clusters in that layer

    pointset_layers: List of list of lists (optional; only if ``return_pointsets`` was ``True``)
        A list of layers, each layer is a list of point sets (a list of indices) for the clusters in that layer
    """
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
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="complete",
        ).fit_predict(locations_for_clusters)

        layer_vectors = []
        layer_locations = []
        layer_pointsets = []

        for label in np.unique(layer_metaclusters):
            if np.sum(layer_metaclusters == label) <= min_cluster_size:
                continue

            vector = vectors_for_clustering[layer_metaclusters == label].mean(axis=0)
            location = locations_for_clusters[layer_metaclusters == label].mean(axis=0)
            pointset: Optional[List[int]]
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
    fixed_layer: npt.NDArray,
    layer_to_adjust: npt.NDArray,
    *,
    spring_constant: float = 0.1,
    edge_weight: float = 1.0,
) -> npt.NDArray:
    """Use a spring layout style approach to adjust the locations of a layer to conflict/overlap less with a fixed layer
    (generally a lower layer, or combination of lower layers). Essentially each cluster in the layer to be adjusted is
    attached to its location by a spring with weight ``edge_weight`` and spring constant ``spring_constant`` and then
    repelled by all the points in the fixed layer.

    Parameters
    ----------
    fixed_layer: Array of shape (n_clusters, n_map_features)
        cluster positions to remain fixed and which will provide a repulsive force pushing away clusters to be adjusted

    layer_to_adjust: Array of shape (n_clusters, n_map_features)
        cluster positions to be adjusted

    spring_constant: float (optional, default = 0.1)
        The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    Returns
    -------
    adjusted_positions: Array of shape (n_clusters, n_map_features)
    """
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
    location_layers: List[npt.NDArray],
    *,
    spring_constant: float = 0.1,
    spring_constant_multiplier: float = 1.5,
    edge_weight: float = 1.0,
) -> List[npt.NDArray]:
    """Adjust locations of clusters in layers to attempt to avoid too much overlap of cluster -- move higher level layer
    clusters to avoid overlapping with lower level layer clusters, under the assumptions that higher level layers
    represent more area and thus have some freedom to be moved.

    Parameters
    ----------
    location_layers: List of Arrays
        The list of layers, where each layer is a array of positions on the map representation.

    spring_constant: float (optional, default = 0.1)
        The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    spring_constant_multiplier: float (optional, default = 1.5)
        We can increase the spring constant for higher level layers; to do this we multiply by the
        ``spring_constant_multiplier`` as we go up a layer. Smaller values (closer to 1.0) will ensure locations
        do no stray too far; this is particularly desireable in the case where there are many layers.

    edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    Returns
    -------
    text_locations: List of list of Arrays
        The resulting list of layers, where each layer is a list of positions on the map representation.
    """
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
    vector_layers: List[List[npt.NDArray]],
    text_representations: npt.ArrayLike,
    text_label_dictionary: Dict[int, Any],
    *,
    items_per_label: int = 3,
    vector_metric: str = "cosine",
    pynnd_n_neighbors: int = 40,
    query_size: int = 10,
    exclude_keyword_reuse: bool = True,
    random_state: Optional[int] = None,
) -> List[List[List[Any]]]:
    """Generate labels (usually text) for each cluster in each layer using a joint vector space representation model. To
    do this we assume we have a ``text_representation`` providing a vector to each "word" such that the vectors exist in
    the *same* vector space as the ``vector_layers`` vector representations. A cluster is then labelled by the "words"
    closest to the cluster representation in the vector space. By default we avoid keyword re-use in layers by keeping
    track of which words have already been used in a layer (starting from the top layers and working downward to the
    finest grained layers), and exclude "words" that have already been used. This behaviour can be turned off if desired.

    Parameters
    ----------
    vector_layers: List of list of Arrays
        A list of layers; each layer is a list of cluster centroids existing in the source vector space

    text_representations: Array of shape (n_possible_labels, n_features)
        An array giving a vector (in the source vector space) for each potential label item

    text_label_dictionary: Dict mapping indices to labels
        A dictionary mapping from indices in the ``text_representation`` array to labels (usually words)

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    vector_metric: str (optional, default = "cosine")
        The metric to use to measure closeness in the source vector space

    pynnd_n_neighbors: int (optional, default = 40)
        The ``n_neighbors`` parameter to use for PyNNDescent for nearest neighbour lookups

    query_size: int (optional, default = 10)
        The number of nearest neighbors to return via PyNNDescent queries; this should be at *least* ``items_per_label``
        and often larger if ``exclude_keyword_reuse`` is ``True``.

    exclude_keyword_reuse: bool (optional, default = True)
        Whether to ensure keyword/labels don't get reused for lower level clusters.

    random_state: int or None (optional, default = None)
        A random state parameter, passed to PyNNDescent which can be used to ensure fixed results for reproducibility.

    Returns
    -------
    labels: List of list of lists of label items
        The resulting layers; each layer is a list of cluster labels; each cluster label is a list of label items
    """
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

            if exclude_keyword_reuse:
                nearest_text = [
                    x
                    for x in text_rep_indices[0]
                    if x not in keyword_set and x not in layer_keyword_set
                ]
                layer_keyword_set.update(set(nearest_text[:items_per_label]))
            else:
                nearest_text = text_rep_indices[0].tolist()

            layer_labels.append(
                [text_label_dictionary[x] for x in nearest_text[:items_per_label]]
            )

        if exclude_keyword_reuse:
            keyword_set.update(layer_keyword_set)

        labels.append(layer_labels)

    return labels[::-1]


def text_labels_from_source_metadata(
    pointset_layers: List[List[npt.NDArray]],
    source_metadataframe: pd.DataFrame,
    *,
    items_per_label: int = 3,
) -> List[List[List[Any]]]:
    """Generate text labels for layers of clusters using a dataframe of metadata associated to points. To label a
    cluster in a layer we train a one versus the rest classifier to discern the cluster and use feature importance
    to label a cluster with the most discerning features.

    Parameters
    ----------
    pointset_layers: List of list of Arrays
        A list of layers; each layer is a list of clusters; each cluster is an array of point indices.

    source_metadataframe: DataFrame
        A dataframe of metadata associated to the points of data / map representation. Each row of the dataframe should
        correspond to a point in the dataset (assumed to be in the same order as the points). We will attempt to handle
        relatively diverse datatypes within the dataframe as well as possible.

    items_per_label: int (optional, default = 3)
        The number of items (features) to label a given cluster with

    Returns
    -------
    labels: List of list of lists of label items
        The resulting layers; each layer is a list of cluster labels; each cluster label is a list of label items
    """
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

    def fit_transform(self, X, y=None, sample_weights=None, **fit_params):
        state = check_random_state(self.random_state)
        if sample_weights is not None:
            weights = np.asarray(sample_weights) / np.sum(sample_weights)
        else:
            weights = None

        indices = state.choice(
            np.arange(X.shape[0]), size=self.n_samples, replace=False, p=weights
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
    pointset_layers: List[List[npt.NDArray]],
    source_vectors: npt.ArrayLike,
    labels_per_sample: npt.ArrayLike,
    *,
    sample_selection_method: str = "facility_location",
    items_per_label: int = 3,
    vector_metric: str = "cosine",
    sample_weights: npt.ArrayLike,
    random_state: Optional[int] = None,
) -> List[List[List[Any]]]:
    """Generate text labels for layers of clusters where each source vector has an associated label representation
    (usually text, usually a word). The labels are generated by sampling labels from the points in the cluster. Various
    sampling strategies are available. The cheapest approach is ``"random"``. More advanced approaches are available via
    the apricot-select library which provides submodular-selection. Here we support ``"saturated_coverage"`` which is
    fast; ``"sum_redundancy"`` and ``"graph_cut"`` which are more expensive, but do a better coverage job; and
    ``"facility_location"`` which does the best job of ensuring diversity and coverage in the selection, but can be
    quite expensive computationally. ``"facility_selection"`` is definitely the best option if ``items_per_label`` is
    very large however.

    Parameters
    ----------
    pointset_layers: List of list of Arrays
        A list of layers; each layer is a list of clusters; each cluster is an array of point indices.

    source_vectors: Array of shape (n_samples, n_features)
        The source vector data from which the map representation was generated.

    labels_per_sample: Array of shape (n_samples,)
        An array of label items for each source vector.

    sample_selection_method: str (optional, default = "facility_selection")
        The selection method to use for sampling from a cluster. Should be one of
            * ``"facility_selection"``
            * ``"graph_cut"``
            * ``"sum_redundancy"``
            * ``"saturated_coverage"``
            * ``"random"``

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    vector_metric: str (optional, default = "cosine")
        The distance metric used in the ``source_vectors`` vector space.

    sample_weights: Array of shape (n_samples,)
        An array of weights to apply to each sample. Higher weight samples may be more likely to be selected. This is
        only supported for some selection methods (random selection does support it). Check the apricot-select
        documentation for more details.

    random_state: int or None (optional, default = None)
        A random state seed to use in random selection.

    Returns
    -------
    labels: List of list of lists of label items
        The resulting layers; each layer is a list of cluster labels; each cluster label is a list of label items
    """
    if HAS_APRICOT and sample_selection_method != "random":
        if sample_selection_method == "facility_location":
            selector = apricot.FacilityLocationSelection(
                items_per_label,
                metric=vector_metric,
                n_neighbors=100,
            )
        elif sample_selection_method == "graph_cut":
            selector = apricot.GraphCutSelection(items_per_label, metric=vector_metric)
        elif sample_selection_method == "sum_redundancy":
            selector = apricot.SumRedundancySelection(
                items_per_label, metric=vector_metric
            )
        elif sample_selection_method == "saturated_coverage":
            selector = apricot.SaturatedCoverageSelection(
                items_per_label, metric=vector_metric
            )
        else:
            raise ValueError(
                f'Unrecognized sample_selection_method {sample_selection_method}! Should be one of {VALID_SELECTION_METHODS} or "random"'
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
                sample_weights=sample_weights[cluster_with_exclusion],
            )
            indices = np.where(np.isin(labels_per_sample, label_selection))[0]
            excluded_indices.update(indices)
            layer_labels.append(list(label_selection))

        labels.append(layer_labels)

    return labels


def text_labels_from_sparse_metadata(
    pointset_layers: List[List[npt.NDArray]],
    sparse_metadata: spmatrix,
    feature_name_dictionary: Dict[int, str],
    *,
    items_per_label: int = 3,
):
    labels = []
    positive_matrix = np.abs(sparse_metadata)
    for layer in pointset_layers:
        layer_labels = []
        cluster_class_labels = np.full(positive_matrix.shape[0], -1, dtype=np.int64)
        for i, pointset in enumerate(layer):
            cluster_class_labels[pointset] = i

        weighted_matrix = InformationWeightTransformer().fit_transform(
            positive_matrix, cluster_class_labels
        )
        for pointset in layer:
            cluster_scores = np.squeeze(np.array(weighted_matrix[pointset].sum(axis=0)))
            top_indices = np.argsort(cluster_scores)[-items_per_label:]
            layer_labels.append(
                [feature_name_dictionary[idx] for idx in reversed(top_indices)]
            )

        labels.append(layer_labels)

    return labels


class JointVectorLabelLayers(object):
    """Generate multiple layers of labelling for a map based on the existence of a joint vector space representation
     of the source vector data for the map, and a separate set of label vectors that exist in the same vector space. To
     do this we assume we have a ``text_representation`` providing a vector to each "word" such that the vectors exist in
     the *same* vector space as the ``vector_layers`` vector representations.

     Multiple layers of clusters are generated, with higher level layers having larger more general clusters. Each
     cluster is then labelled by the "words" closest to the cluster representation in the vector space. By default we
     avoid keyword re-use in layers by keeping track of which words have already been used in a layer (starting from
     the top layers and working downward to the finest grained layers), and exclude "words" that have already been
     used. This behaviour can be turned off if desired.

     Parameters
     ----------
     source_vectors: Array of shape (n_samples, n_features)
         The original high dimensional vector representation of the data

     map_representation: Array of shape (n_samples, n_map_features)
         The map representation of the data

     labelling_vectors: Array of shape (n_possible_labels, n_features)
         An array giving a vector (in the source vector space) for each potential label item

     labels: Dictionary mapping indices to label items
         A dictionary mapping from indices in the ``labelling_vectors`` array to labels (usually words)

     vector_metric: str (optional, default = "cosine")
         The metric to use on the source vector space.

     cluster_map_representation: bool (optional, default = False)
         Whether to directly cluster the map representation, or use UMAP to generate a representation for clustering
         using ``umap_n_components`` many dimensions.

     umap_n_components:
         The number of dimensions to use UMAP to reduce to if ``cluster_map_representation`` is ``False``.

     umap_n_neighbors: int (optional, default = 15)
         The number of neighbors to use for UMAP  if ``cluster_map_representation`` is ``False``.

     hdbscan_min_samples: int (optional, default = 10)
         The ``min_samples`` value to use with HDBSCAN for clustering.

     hdbscan_min_cluster_size: int (optional, default = 20)
         The ``min_cluster_size`` value to use with HDBSCAN for clustering.

     min_clusters: int (optional, default = 4)
         The number of clusters to have at the highest layer; layers with fewer than this number of clusters will
         be discarded

     contamination: float (optional, default = 0.05)
         The base contamination score used for outlier detection of fine grained clusters. Larger values will
         prune out more outliers

     contamination_multiplier: float (optional, default = 1.5)
         The value to multiply the contamination score by as we increase the layers -- thus applying
         higher contamination and removing more outliers from higher layers. Larger values will prune
         more aggressively

     max_contamination: float (optional, default = 0.25)
         The maximum contamination value to use in outlier pruning -- once the multiplier increases
         contamination beyond this value the contamination used will simply be capped at this value.

     cluster_distance_threshold: float (optional, default = 0.025)
         Cluster centroid representatives from a higher layer that are within this distance of an already selected
         cluster centroid in a lower layer will be ignored (so we don't repeat clusters)

     adjust_label_locations: bool (optional, default = True)
         Whether to attempt to adjust label locations to avoid overlaps with lower layers.

     label_adjust_spring_constant: float (optional, default = 0.1)
          The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    label_adjust_spring_constant_multiplier: float (optional, default = 1.5)
         We can increase the spring constant for higher level layers; to do this we multiply by the
         ``spring_constant_multiplier`` as we go up a layer. Smaller values (closer to 1.0) will ensure locations
         do no stray too far; this is particularly desireable in the case where there are many layers.

    label_adjust_edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    pynnd_n_neighbors: int (optional, default = 40)
        The ``n_neighbors`` parameter to use for PyNNDescent for nearest neighbour lookups

    query_size: int (optional, default = 10)
        The number of nearest neighbors to return via PyNNDescent queries; this should be at *least* ``items_per_label``
        and often larger if ``exclude_keyword_reuse`` is ``True``.

    exclude_keyword_reuse: bool (optional, default = True)
        Whether to ensure keyword/labels don't get reused for lower level clusters.

    label_formatter: Function (optional, default = string_label_formatter)
        A function used for format a list of label items into a usable label (usually a single string).

    random_state: int or None (optional, default = None)
        A random state parameter which can be used to ensure fixed results for reproducibility.

     Attributes
     ----------
    labels: List of list of lists of label items
        A list of layers; each layer is a list of labels; each label is a list of label ``items_per_label`` many items

    location_layers: List of Arrays of shape (n_cluster_in_layer, n_map_features)
        A list of layers; each layer is an array of locations in the map representation to place the labels of that layer

    labels_for_display: List of list of labels
        A list of layers; each layer is a list of labels; each label is formatted for display use by ``label_formatter``
    """

    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        labelling_vectors: npt.ArrayLike,
        labels: Dict[int, Any],
        *,
        vector_metric: str = "cosine",
        cluster_map_representation: bool = False,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        hdbscan_min_samples: int = 10,
        hdbscan_min_cluster_size: int = 20,
        min_clusters_in_layer: int = 4,
        contamination: float = 0.05,
        contamination_multiplier: float = 1.5,
        max_contamination: float = 0.25,
        cluster_distance_threshold: float = 0.025,
        adjust_label_locations: bool = True,
        label_adjust_spring_constant: float = 0.1,
        label_adjust_spring_constant_multiplier: float = 1.5,
        label_adjust_edge_weight: float = 1.0,
        items_per_label: int = 3,
        pynnd_n_neighbors: int = 40,
        pynnd_query_size: int = 10,
        exclude_keyword_reuse: bool = True,
        label_formatter: Callable[[List[Any]], Any] = string_label_formatter,
        random_state: Optional[int] = None,
    ):
        if adjust_label_locations and not HAS_NETWORKX:
            warn("NetworkX is required for label adjustments; try pip install networkx")
            adjust_label_locations = False

        cluster_vectors, cluster_locations, _ = build_fine_grained_cluster_centers(
            source_vectors,
            map_representation,
            cluster_map_representation=cluster_map_representation,
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
            exclude_keyword_reuse=exclude_keyword_reuse,
            random_state=random_state,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]


class MetadataLabelLayers(object):
    """Generate multiple layers of labelling for a map based on a dataframe of metadata associated to points. Multiple
    layers of clusters are generated, with higher level layers having larger more general clusters. Each cluster is
    then labelled by training a one versus the rest classifier to discern the cluster in terms of the associated
    metadata. The feature importances can then be used to label a cluster with the most discerning features.

    Parameters
    ----------
     source_vectors: Array of shape (n_samples, n_features)
         The original high dimensional vector representation of the data

     map_representation: Array of shape (n_samples, n_map_features)
         The map representation of the data

    metadata_dataframe: DataFrame
        A dataframe of metadata associated to the points of data / map representation. Each row of the dataframe should
        correspond to a point in the dataset (assumed to be in the same order as the points). We will attempt to handle
        relatively diverse datatypes within the dataframe as well as possible.

    vector_metric: str (optional, default = "cosine")
        The metric to use on the source vector space.

    cluster_map_representation: bool (optional, default = False)
        Whether to directly cluster the map representation, or use UMAP to generate a representation for clustering
        using ``umap_n_components`` many dimensions.

    umap_n_components:
        The number of dimensions to use UMAP to reduce to if ``cluster_map_representation`` is ``False``.

    umap_n_neighbors: int (optional, default = 15)
        The number of neighbors to use for UMAP  if ``cluster_map_representation`` is ``False``.

    hdbscan_min_samples: int (optional, default = 10)
        The ``min_samples`` value to use with HDBSCAN for clustering.

    hdbscan_min_cluster_size: int (optional, default = 20)
        The ``min_cluster_size`` value to use with HDBSCAN for clustering.

    min_clusters: int (optional, default = 4)
        The number of clusters to have at the highest layer; layers with fewer than this number of clusters will
        be discarded

    contamination: float (optional, default = 0.05)
        The base contamination score used for outlier detection of fine grained clusters. Larger values will
        prune out more outliers

    contamination_multiplier: float (optional, default = 1.5)
        The value to multiply the contamination score by as we increase the layers -- thus applying
        higher contamination and removing more outliers from higher layers. Larger values will prune
        more aggressively

    max_contamination: float (optional, default = 0.25)
        The maximum contamination value to use in outlier pruning -- once the multiplier increases
        contamination beyond this value the contamination used will simply be capped at this value.

    cluster_distance_threshold: float (optional, default = 0.025)
        Cluster centroid representatives from a higher layer that are within this distance of an already selected
        cluster centroid in a lower layer will be ignored (so we don't repeat clusters)

    adjust_label_locations: bool (optional, default = True)
        Whether to attempt to adjust label locations to avoid overlaps with lower layers.

    label_adjust_spring_constant: float (optional, default = 0.1)
        The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    label_adjust_spring_constant_multiplier: float (optional, default = 1.5)
         We can increase the spring constant for higher level layers; to do this we multiply by the
         ``spring_constant_multiplier`` as we go up a layer. Smaller values (closer to 1.0) will ensure locations
         do no stray too far; this is particularly desireable in the case where there are many layers.

    label_adjust_edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    label_formatter: Function (optional, default = string_label_formatter)
        A function used for format a list of label items into a usable label (usually a single string).

    random_state: int or None (optional, default = None)
        A random state parameter which can be used to ensure fixed results for reproducibility.

     Attributes
     ----------
    labels: List of list of lists of label items
        A list of layers; each layer is a list of labels; each label is a list of label ``items_per_label`` many items

    location_layers: List of Arrays of shape (n_cluster_in_layer, n_map_features)
        A list of layers; each layer is an array of locations in the map representation to place the labels of that layer

    labels_for_display: List of list of labels
        A list of layers; each layer is a list of labels; each label is formatted for display use by ``label_formatter``
    """

    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        metadata_dataframe: pd.DataFrame,
        *,
        vector_metric: str = "cosine",
        cluster_map_representation: bool = False,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        hdbscan_min_samples: int = 10,
        hdbscan_min_cluster_size: int = 20,
        min_clusters_in_layer: int = 4,
        contamination: float = 0.05,
        contamination_multiplier: float = 1.5,
        max_contamination: float = 0.25,
        cluster_distance_threshold: float = 0.025,
        adjust_label_locations: bool = True,
        label_adjust_spring_constant: float = 0.1,
        label_adjust_spring_constant_multiplier: float = 1.5,
        label_adjust_edge_weight: float = 1.0,
        items_per_label: int = 3,
        label_formatter: Callable[[List[Any]], Any] = string_label_formatter,
        random_state: Optional[int] = None,
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
            cluster_map_representation=cluster_map_representation,
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
            self.pointset_layers,
            metadata_dataframe,
            items_per_label=items_per_label,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]


class SampleLabelLayers(object):
    """Generate text labels for layers of clusters from data where each source vector has an associated label
    representation (usually text, usually a word). Multiple layers of clusters are generated, with higher level
    layers having larger more general clusters. Each cluster is then labelled by sampling labels from the points in
    the cluster. Various sampling strategies are available. The cheapest approach is ``"random"``. More advanced
    approaches are available via the apricot-select library which provides submodular-selection. Here we support
    ``"saturated_coverage"`` which is fast; ``"sum_redundancy"`` and ``"graph_cut"`` which are more expensive,
    but do a better coverage job; and ``"facility_location"`` which does the best job of ensuring diversity and
    coverage in the selection, but can be quite expensive computationally. ``"facility_selection"`` is definitely the
    best option if ``items_per_label`` is very large however.

    Parameters
    ----------
     source_vectors: Array of shape (n_samples, n_features)
         The original high dimensional vector representation of the data

     map_representation: Array of shape (n_samples, n_map_features)
         The map representation of the data

    per_sample_labels: Array of shape (n_samples,)
        An array of label items for each source vector.

    vector_metric: str (optional, default = "cosine")
        The metric to use on the source vector space.

    cluster_map_representation: bool (optional, default = False)
        Whether to directly cluster the map representation, or use UMAP to generate a representation for clustering
        using ``umap_n_components`` many dimensions.

    sample_selection_method: str (optional, default = "facility_selection")
        The selection method to use for sampling from a cluster. Should be one of
            * ``"facility_selection"``
            * ``"graph_cut"``
            * ``"sum_redundancy"``
            * ``"saturated_coverage"``
            * ``"random"``

    sample_weights: Array of shape (n_samples,)
        An array of weights to apply to each sample. Higher weight samples may be more likely to be selected. This is
        only supported for some selection methods (random selection does support it). Check the apricot-select
        documentation for more details.

    umap_n_components:
        The number of dimensions to use UMAP to reduce to if ``cluster_map_representation`` is ``False``.

    umap_n_neighbors: int (optional, default = 15)
        The number of neighbors to use for UMAP  if ``cluster_map_representation`` is ``False``.

    hdbscan_min_samples: int (optional, default = 10)
        The ``min_samples`` value to use with HDBSCAN for clustering.

    hdbscan_min_cluster_size: int (optional, default = 20)
        The ``min_cluster_size`` value to use with HDBSCAN for clustering.

    min_clusters: int (optional, default = 4)
        The number of clusters to have at the highest layer; layers with fewer than this number of clusters will
        be discarded

    contamination: float (optional, default = 0.05)
        The base contamination score used for outlier detection of fine grained clusters. Larger values will
        prune out more outliers

    contamination_multiplier: float (optional, default = 1.5)
        The value to multiply the contamination score by as we increase the layers -- thus applying
        higher contamination and removing more outliers from higher layers. Larger values will prune
        more aggressively

    max_contamination: float (optional, default = 0.25)
        The maximum contamination value to use in outlier pruning -- once the multiplier increases
        contamination beyond this value the contamination used will simply be capped at this value.

    cluster_distance_threshold: float (optional, default = 0.025)
        Cluster centroid representatives from a higher layer that are within this distance of an already selected
        cluster centroid in a lower layer will be ignored (so we don't repeat clusters)

    adjust_label_locations: bool (optional, default = True)
        Whether to attempt to adjust label locations to avoid overlaps with lower layers.

    label_adjust_spring_constant: float (optional, default = 0.1)
        The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    label_adjust_spring_constant_multiplier: float (optional, default = 1.5)
        We can increase the spring constant for higher level layers; to do this we multiply by the
        ``spring_constant_multiplier`` as we go up a layer. Smaller values (closer to 1.0) will ensure locations
        do no stray too far; this is particularly desireable in the case where there are many layers.

    label_adjust_edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    label_formatter: Function (optional, default = string_label_formatter)
        A function used for format a list of label items into a usable label (usually a single string).

    random_state: int or None (optional, default = None)
        A random state parameter which can be used to ensure fixed results for reproducibility.

     Attributes
     ----------
    labels: List of list of lists of label items
        A list of layers; each layer is a list of labels; each label is a list of label ``items_per_label`` many items

    location_layers: List of Arrays of shape (n_cluster_in_layer, n_map_features)
        A list of layers; each layer is an array of locations in the map representation to place the labels of that layer

    labels_for_display: List of list of labels
        A list of layers; each layer is a list of labels; each label is formatted for display use by ``label_formatter``

    """

    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        per_sample_labels: npt.ArrayLike,
        *,
        vector_metric: str = "cosine",
        cluster_map_representation: bool = False,
        sample_selection_method: str = "facility_location",
        sample_weights: Optional[npt.ArrayLike] = None,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        hdbscan_min_samples: int = 10,
        hdbscan_min_cluster_size: int = 20,
        min_clusters_in_layer: int = 4,
        contamination: float = 0.05,
        contamination_multiplier: float = 1.5,
        max_contamination: float = 0.25,
        cluster_distance_threshold: float = 0.025,
        adjust_label_locations: bool = True,
        label_adjust_spring_constant: float = 0.1,
        label_adjust_spring_constant_multiplier: float = 1.5,
        label_adjust_edge_weight: float = 1.0,
        items_per_label: int = 3,
        label_formatter: Callable[[List[Any]], Any] = string_label_formatter,
        random_state: Optional[int] = None,
    ):
        self.per_sample_labels = per_sample_labels

        if not HAS_APRICOT and sample_selection_method != "random":
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
            cluster_map_representation=cluster_map_representation,
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
            sample_selection_method=sample_selection_method,
            items_per_label=items_per_label,
            vector_metric=vector_metric,
            sample_weights=np.asarray(sample_weights),
            random_state=random_state,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]


class SparseMetadataLabelLayers(object):
    """Generate multiple layers of labelling for a map based on a dataframe of metadata associated to points. Multiple
    layers of clusters are generated, with higher level layers having larger more general clusters. Each cluster is
    then labelled by training a one versus the rest classifier to discern the cluster in terms of the associated
    metadata. The feature importances can then be used to label a cluster with the most discerning features.

    Parameters
    ----------
     source_vectors: Array of shape (n_samples, n_features)
         The original high dimensional vector representation of the data

     map_representation: Array of shape (n_samples, n_map_features)
         The map representation of the data

    sparse_metadata: spmatrix
        A sparse matrix of metadata associated to the points of data / map representation. Usually this is associated
        with metadata that has a high number of features, and any given sample only has non-zero values for a small
        number of features. A prime example is a bag-of-words representation of a corpus of documents.

    feature_name_dictionary: dict
        A dictionary mapping column indices of the sparse matrix to feature names. For example, if the sparse matrix
        were the output of sklearn's ``CountVectorizer`` the dict would be
        ``{idx: word for word, idx in model.vocabulary_.items()}``.

    vector_metric: str (optional, default = "cosine")
        The metric to use on the source vector space.

    cluster_map_representation: bool (optional, default = False)
        Whether to directly cluster the map representation, or use UMAP to generate a representation for clustering
        using ``umap_n_components`` many dimensions.

    umap_n_components:
        The number of dimensions to use UMAP to reduce to if ``cluster_map_representation`` is ``False``.

    umap_n_neighbors: int (optional, default = 15)
        The number of neighbors to use for UMAP  if ``cluster_map_representation`` is ``False``.

    hdbscan_min_samples: int (optional, default = 10)
        The ``min_samples`` value to use with HDBSCAN for clustering.

    hdbscan_min_cluster_size: int (optional, default = 20)
        The ``min_cluster_size`` value to use with HDBSCAN for clustering.

    min_clusters: int (optional, default = 4)
        The number of clusters to have at the highest layer; layers with fewer than this number of clusters will
        be discarded

    contamination: float (optional, default = 0.05)
        The base contamination score used for outlier detection of fine grained clusters. Larger values will
        prune out more outliers

    contamination_multiplier: float (optional, default = 1.5)
        The value to multiply the contamination score by as we increase the layers -- thus applying
        higher contamination and removing more outliers from higher layers. Larger values will prune
        more aggressively

    max_contamination: float (optional, default = 0.25)
        The maximum contamination value to use in outlier pruning -- once the multiplier increases
        contamination beyond this value the contamination used will simply be capped at this value.

    cluster_distance_threshold: float (optional, default = 0.025)
        Cluster centroid representatives from a higher layer that are within this distance of an already selected
        cluster centroid in a lower layer will be ignored (so we don't repeat clusters)

    adjust_label_locations: bool (optional, default = True)
        Whether to attempt to adjust label locations to avoid overlaps with lower layers.

    label_adjust_spring_constant: float (optional, default = 0.1)
        The "optimal" distance from the source position; larger values will allow the adjusted cluster to move farther

    label_adjust_spring_constant_multiplier: float (optional, default = 1.5)
         We can increase the spring constant for higher level layers; to do this we multiply by the
         ``spring_constant_multiplier`` as we go up a layer. Smaller values (closer to 1.0) will ensure locations
         do no stray too far; this is particularly desireable in the case where there are many layers.

    label_adjust_edge_weight: float (optional, default = 1.0)
        How strong the springs pull

    items_per_label: int (optional, default = 3)
        The number of items to use for each cluster label

    label_formatter: Function (optional, default = string_label_formatter)
        A function used for format a list of label items into a usable label (usually a single string).

    random_state: int or None (optional, default = None)
        A random state parameter which can be used to ensure fixed results for reproducibility.

     Attributes
     ----------
    labels: List of list of lists of label items
        A list of layers; each layer is a list of labels; each label is a list of label ``items_per_label`` many items

    location_layers: List of Arrays of shape (n_cluster_in_layer, n_map_features)
        A list of layers; each layer is an array of locations in the map representation to place the labels of that layer

    labels_for_display: List of list of labels
        A list of layers; each layer is a list of labels; each label is formatted for display use by ``label_formatter``
    """

    def __init__(
        self,
        source_vectors: npt.ArrayLike,
        map_representation: npt.ArrayLike,
        sparse_metadata: spmatrix,
        feature_name_dictionary: Dict[int, str],
        *,
        vector_metric: str = "cosine",
        cluster_map_representation: bool = False,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        hdbscan_min_samples: int = 10,
        hdbscan_min_cluster_size: int = 20,
        min_clusters_in_layer: int = 4,
        contamination: float = 0.05,
        contamination_multiplier: float = 1.5,
        max_contamination: float = 0.25,
        cluster_distance_threshold: float = 0.025,
        adjust_label_locations: bool = True,
        label_adjust_spring_constant: float = 0.1,
        label_adjust_spring_constant_multiplier: float = 1.5,
        label_adjust_edge_weight: float = 1.0,
        items_per_label: int = 3,
        label_formatter: Callable[[List[Any]], Any] = string_label_formatter,
        random_state: Optional[int] = None,
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
            cluster_map_representation=cluster_map_representation,
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

        self.labels = text_labels_from_sparse_metadata(
            self.pointset_layers,
            sparse_metadata,
            feature_name_dictionary,
            items_per_label=items_per_label,
        )
        self.label_formatter = label_formatter

    @property
    def labels_for_display(self):
        return [
            [self.label_formatter(label) for label in label_layer]
            for label_layer in self.labels
        ]
