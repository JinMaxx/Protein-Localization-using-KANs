"""
This module provides a framework for visualizing characteristics of sequence datasets.

It defines the `DataFiguresCollection` class,
which serves as a manager for creating and updating a variety of plots.
These visualizations are essential for exploratory
data analysis, helping to understand data distributions, class balance,
embedding space structures, and other key properties of the dataset.
"""

import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from plotly.graph_objs import Figure
from scipy.spatial.distance import pdist
from typing_extensions import (
    Literal, TypeAlias,
    Optional, Callable,
    List, Dict,
    override, cast
)

from source.data_scripts.read_data import Label
from source.abstract_figures import AbstractFiguresCollection, _AbstractFigure, get_label_color
from source.custom_types import Sequence_Encoding_Data_Generator_T, Seq_Enc_Data_Generator_Supplier_T


class PoolingType(Enum):
    Per_Protein_Mean = 0
    Per_Protein_Max = 1


DistanceMetric: TypeAlias = Literal[
    "euclidean",
    "cosine",
    "cityblock",
    "minkowski",
    "hamming",
    "jaccard"
]



class DataFiguresCollection(AbstractFiguresCollection):
    """
    Manages a collection of figures for visualizing dataset characteristics.

    This class provides methods to create and update various plots that help
    in understanding the underlying data, such as class distributions, sequence
    length histograms, and dimensionality-reduced embedding visualizations.
    """

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def class_distribution(self) -> 'ClassDistributionFigure':
        """
        Creates and adds a class distribution bar chart figure.

        Returns:
            ClassDistributionFigure: The created figure object.
        """
        return cast(ClassDistributionFigure, self._add(ClassDistributionFigure(collection=self)))


    def raw_sequence_length_distribution(self,
            bins: int = 150,
            log_y: bool = False
    ) -> 'RawSequenceLengthDistributionFigure':
        """
        Creates and adds a raw sequence length (number of amino acids) distribution histogram.

        Args:
            bins (int): The number of bins to use for the histogram.
            log_y (bool): If True, the y-axis will be on a logarithmic scale.

        Returns:
            RawSequenceLengthDistributionFigure: The created figure object.
        """
        return cast(RawSequenceLengthDistributionFigure, self._add(RawSequenceLengthDistributionFigure(
            bins = bins,
            log_y = log_y,
            collection = self
        )))


    def embedding_length_distribution(self,
            bins: int = 150,
            log_y: bool = False
    ) -> 'EmbeddingLengthDistributionFigure':
        """
        Creates and adds an embedding length distribution histogram.

        Args:
            bins (int): The number of bins to use for the histogram.
            log_y (bool): If True, the y-axis will be on a logarithmic scale.

        Returns:
            EmbeddingLengthDistributionFigure: The created figure object.
        """
        return cast(EmbeddingLengthDistributionFigure, self._add(EmbeddingLengthDistributionFigure(
            bins = bins,
            log_y = log_y,
            collection = self
        )))


    def pairwise_distance_distribution(self,
            bins: int = 150,
            distance_metric: DistanceMetric = "euclidean",
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean
    ) -> 'PairwiseDistanceDistributionFigure':
        """
        Creates and adds a histogram of pairwise distances between sequence embeddings.

        Args:
            bins (int): The number of bins for the histogram.
            distance_metric (str): The distance metric to use (e.g., "euclidean", "cosine").
            pooling_type (PoolingType): The pooling strategy to apply to embeddings before calculating distances.

        Returns:
            PairwiseDistanceDistributionFigure: The created figure object.
        """
        return cast(PairwiseDistanceDistributionFigure, self._add(PairwiseDistanceDistributionFigure(
            bins = bins,
            distance_metric = distance_metric,
            pooling_type = pooling_type,
            collection = self
        )))


    def pca_embedding(self,
            n_components: int = 2,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean
    ) -> 'PCAFigure':
        """
        Creates and adds a 2D PCA scatter plot of sequence embeddings.

        Args:
            n_components (int): The number of principal components to compute.
            random_state (Optional[int]): A seed for the random number generator for reproducibility.
            pooling_type (PoolingType): The pooling strategy to apply to embeddings before applying PCA.

        Returns:
            PCAFigure: The created figure object.
        """
        return cast(PCAFigure, self._add(PCAFigure(
            n_components = n_components,
            random_state = random_state,
            pooling_type = pooling_type,
            collection = self
        )))


    def tsne_embedding(self,
            n_components: int = 2,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean
    ) -> 'TSNEFigure':
        """
        Creates and adds a 2D t-SNE scatter plot of sequence embeddings.

        Args:
            n_components (int): The number of dimensions for the embedded space (typically 2 or 3).
            random_state (Optional[int]): A seed for the random number generator for reproducibility.
            pooling_type (PoolingType): The pooling strategy to apply to embeddings before applying t-SNE.

        Returns:
            TSNEFigure: The created figure object.
        """
        return cast(TSNEFigure, self._add(TSNEFigure(
            n_components = n_components,
            random_state = random_state,
            pooling_type = pooling_type,
            collection = self
        )))


    def umap_embedding(self,
            n_components: int = 2,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean
    ) -> 'UMAPFigure':
        """
        Creates and adds a 2D UMAP scatter plot of sequence embeddings.

        Args:
            n_components (int): The dimension of the space to embed into.
            random_state (Optional[int]): A seed for the random number generator for reproducibility.
            pooling_type (PoolingType): The pooling strategy to apply to embeddings before applying UMAP.

        Returns:
            UMAPFigure: The created figure object.
        """
        return cast(UMAPFigure, self._add(UMAPFigure(
            n_components = n_components,
            random_state = random_state,
            pooling_type = pooling_type,
            collection = self
        )))


    @override
    def update(self,
            identifier: Optional[str] = None,
            file_name: Optional[str] = None,
            encoding_model: Optional[str] = None,
            seq_enc_data_generator_supplier: Optional[Seq_Enc_Data_Generator_Supplier_T] = None,
            label_count: Optional[Dict[Label, int]] = None,
            max_samples: Optional[int] = None,
            clear: bool = True,
            **kwargs
    ) -> None:
        """
        Updates all managed figures with new data.

        Args:
            identifier (Optional[str]): An identifier for saving the figure to a file.
            file_name (Optional[str]): The name of the data file, to be used in the figure title.
            encoding_model (Optional[str]): The name of the embedding model, for the figure title.
            seq_enc_data_generator_supplier (Optional[Seq_Enc_Data_Generator_Supplier_T]): A function that supplies a data generator.
            label_count (Optional[Dict[Label, int]]): A dictionary mapping labels to their counts.
            max_samples (Optional[int]): The maximum number of samples to use for plotting.
            clear (bool): If True, clears the output before displaying updates.
        """
        super().update(
            identifier = identifier,
            file_name = file_name,
            encoding_model = encoding_model,
            seq_enc_data_generator_supplier = seq_enc_data_generator_supplier,
            label_count = label_count,
            max_samples = max_samples,
            clear = clear,
            **kwargs
        )


# -------------------------------- Figures --------------------------------


class _AbstractDataFigure(_AbstractFigure, ABC):
    """Abstract base class for figures that visualize dataset properties."""

    def __init__(self, collection: AbstractFiguresCollection):  # identifier changes in the use case
        super().__init__(collection=collection, figure=Figure())



class ClassDistributionFigure(_AbstractDataFigure):
    """
    Displays a bar chart showing the number of samples for each class.
    This plot is essential for identifying class imbalance in the dataset,
    which can significantly influence model training and evaluation.
    """

    def __init__(self, collection: AbstractFiguresCollection):
        super().__init__(collection=collection)
        self._fig.update_layout(
            title="Class Distribution",
            xaxis_title="Class Label",
            yaxis_title="Count",
            yaxis=dict(tickformat=',')
        )

    @override
    def update(self,
             label_count: Dict[Label, int],
             identifier: Optional[str] = None,
             file_name: Optional[str] = None,
             **kwargs):

        self._fig.data = []

        if identifier is not None: self._identifier = identifier

        title_prefix = f"{file_name}: " if file_name else ""
        total_sequences = sum(label_count.values())
        self._fig.update_layout(title_text=f"{title_prefix}Class Distribution (Total Sequences: {total_sequences:,})")

        label_names: List[str] = []
        counts: List[int] = []
        colors: List[str] = []
        text_labels: List[str] = []

        for label, count in label_count.items():
            label_names.append(label.name)
            counts.append(count)
            colors.append(get_label_color(label))
            # Calculate percentage and create the text label
            percentage = (count / total_sequences) * 100 if total_sequences > 0 else 0
            text_labels.append(f'{percentage:.1f}%')

        self._fig.add_bar(
            x = label_names,
            y = counts,
            marker_color = colors,
            text = text_labels,
            textposition = 'outside'
        )



class _AbstractLengthDistributionFigure(_AbstractDataFigure, ABC):
    """An abstract base class for figures that display a histogram of lengths."""

    def __init__(self,
            collection: AbstractFiguresCollection,
            base_title: str,
            bins: int = 150,
            log_y: bool = False,
    ):
        super().__init__(collection=collection)
        self._bins = bins
        self._log_y = log_y
        self._base_title = base_title
        self._fig.update_layout(
            title_text = base_title,
            yaxis_title = "log Count" if self._log_y else "Count",
            xaxis = dict(rangemode='tozero'),  # Forces the x-axis to start at 0
            yaxis = dict(type='log' if self._log_y else 'linear', tickformat=',')
        )


    @abstractmethod
    def _get_lengths(self, seq_enc_data_generator: Sequence_Encoding_Data_Generator_T) -> List[int]:
        """
        Processes the data generator and returns a list of integer lengths.
        This method must be implemented by subclasses.
        """
        pass


    @override
    def update(self,
            seq_enc_data_generator_supplier: Seq_Enc_Data_Generator_Supplier_T,
            identifier: Optional[str] = None,
            file_name: Optional[str] = None,
            **kwargs):
        self._fig.data = []
        if identifier is not None: self._identifier = identifier

        title_prefix = f"{file_name}: " if file_name else ""
        self._fig.update_layout(title_text=f"{title_prefix}{self._base_title}")

        seq_enc_data_generator = seq_enc_data_generator_supplier()
        lengths = self._get_lengths(seq_enc_data_generator)

        self._fig.add_histogram(
            x=lengths,
            nbinsx=self._bins,
            marker_color="grey"
        )



class RawSequenceLengthDistributionFigure(_AbstractLengthDistributionFigure):
    """Displays a histogram of raw sequence lengths (number of amino acids)."""

    def __init__(self,
            collection: AbstractFiguresCollection,
            bins: int = 150,
            log_y: bool = False):
        super().__init__(collection=collection, bins=bins, log_y=log_y, base_title="Raw Sequence Length Distribution")
        self._fig.update_layout(xaxis_title="Sequence Length")


    @override
    def _get_lengths(self, seq_enc_data_generator: Sequence_Encoding_Data_Generator_T) -> List[int]:
        return [len(seq_data.record.seq) for seq_data in seq_enc_data_generator]



class EmbeddingLengthDistributionFigure(_AbstractLengthDistributionFigure):
    """Displays a histogram of embedding lengths, not dimension."""

    def __init__(self,
            collection: AbstractFiguresCollection,
            bins: int = 150,
            log_y: bool = False):
        super().__init__(collection=collection, bins=bins, log_y=log_y, base_title="Embedding Length Distribution")
        self._fig.update_layout(xaxis_title = "Embedding Length")


    @override
    def _get_lengths(self, seq_enc_data_generator: Sequence_Encoding_Data_Generator_T) -> List[int]:
        # get_encoding() returns a tensor of shape [seq_len, encoding_dim], so we take the size of the first dimension.
        return [seq_data.get_encoding().shape[0] for seq_data in seq_enc_data_generator]



# This is simply too much to handle (Figure simply can't be drawn)
class PairwiseDistanceDistributionFigure(_AbstractDataFigure):
    """
    Displays a histogram of pairwise distances between sequence embeddings.
    This plot provides insight into the structure of the embedding space,
    showing how similar or dissimilar sequences are to each other on average.
    A multi-modal distribution might suggest distinct clusters within the data.
    """

    def __init__(self,
            collection: AbstractFiguresCollection,
            bins: int = 150,
            distance_metric: DistanceMetric = "euclidean",
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean):
        super().__init__(collection=collection)
        self._bins: int = bins
        self._metric: DistanceMetric = distance_metric
        self._pooling_type: PoolingType = pooling_type
        self._base_title = "Pairwise Embedding Distance Distribution"
        self._fig.update_layout(
            xaxis_title=f"{distance_metric.capitalize()} Distance",
            yaxis_title="Pair Count"
        )


    @override
    def update(self,
            seq_enc_data_generator_supplier: Seq_Enc_Data_Generator_Supplier_T,
            identifier: Optional[str] = None,
            max_samples: Optional[int] = None,
            file_name: Optional[str] = None,
            encoding_model: Optional[str] = None,
            **kwargs):
        # noinspection DuplicatedCode
        self._fig.data = []
        if identifier is not None: self._identifier = identifier

        title_prefix_parts = []
        if file_name: title_prefix_parts.append(file_name)
        if encoding_model: title_prefix_parts.append(f"{encoding_model} ({self._pooling_type.name})")
        title_prefix = " - ".join(title_prefix_parts)
        self._fig.update_layout(
            title_text = f"{title_prefix if title_prefix else ''}: Pairwise Embedding Distance Distribution"
        )
        
        seq_enc_data_generator: Sequence_Encoding_Data_Generator_T = seq_enc_data_generator_supplier()

        embeddings: List[np.ndarray] = []

        for i, seq_data in enumerate(seq_enc_data_generator):
            if max_samples is not None and i >= max_samples: break
            match self._pooling_type:
                case PoolingType.Per_Protein_Mean:  embeddings.append(seq_data.get_encoding().mean(axis=0))
                case PoolingType.Per_Protein_Max: embeddings.append(seq_data.get_encoding().max(axis=0))
                # case _: raise ValueError(f"Incorrect Encoding Type: {self._pooling_type.name}")

        if not embeddings: return

        distances = pdist(np.stack(embeddings), metric=self._metric)

        self._fig.add_histogram(
            x = distances,
            nbinsx = self._bins,
            marker_color = "orchid"
        )



class _AbstractEncodingsVisualizationFigure(_AbstractDataFigure):
    """
    Abstract base class for creating 2D scatter plots from high-dimensional embeddings.
    This class handles the common logic of generating embeddings, applying a dimensionality
    reduction technique, and plotting the results, colored by class label.
    """

    def __init__(self,
            collection: AbstractFiguresCollection,
            reducer: Callable[[np.ndarray], np.ndarray],
            base_title: str,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean):
        super().__init__(collection=collection)
        self._reducer: Callable[[np.ndarray], np.ndarray] = reducer
        self._pooling_type: PoolingType = pooling_type
        self._base_title = base_title


    @override
    def update(self,
            seq_enc_data_generator_supplier: Seq_Enc_Data_Generator_Supplier_T,
            identifier: Optional[str] = None,
            file_name: Optional[str] = None,
            encoding_model: Optional[str] = None,
            **kwargs):
        # noinspection DuplicatedCode
        self._fig.data = []
        if identifier is not None: self._identifier = identifier

        title_prefix_parts = []
        if file_name: title_prefix_parts.append(file_name)
        if encoding_model: title_prefix_parts.append(f"{encoding_model} ({self._pooling_type.name})")
        title_prefix = " - ".join(title_prefix_parts)
        final_title = f"{title_prefix}: {self._base_title}" if title_prefix else f"{self._base_title} ({self._pooling_type.name})"
        self._fig.update_layout(title_text=final_title)

        seq_enc_data_generator: Sequence_Encoding_Data_Generator_T = seq_enc_data_generator_supplier()

        embeddings: List[np.ndarray] = []
        labels: List[Label] = []
        colors: List[str] = []
        for seq_data in seq_enc_data_generator:
            match self._pooling_type:
                case PoolingType.Per_Protein_Mean:  embeddings.append(seq_data.get_encoding().mean(axis=0))
                case PoolingType.Per_Protein_Max: embeddings.append(seq_data.get_encoding().max(axis=0))
                # case _: raise ValueError(f"Incorrect Encoding Type: {self._pooling_type.name}")
            labels.append(seq_data.label)
            colors.append(get_label_color(seq_data.label))

        reduced_2d = self._reducer(np.stack(embeddings))  # Reduce dimensions

        # Get unique labels
        unique_labels = list(Label)
        for label in unique_labels:
            indices = [i for i, _label in enumerate(labels) if _label == label]
            if not indices: continue  # Skip labels not present in the data
            self._fig.add_scattergl(
                x = reduced_2d[indices, 0],
                y = reduced_2d[indices, 1],
                mode = "markers",
                marker = dict(color=get_label_color(label), opacity=0.7),
                text = [label.name] * len(indices),
                hoverinfo = "text",
                name = label.name
            )



class PCAFigure(_AbstractEncodingsVisualizationFigure):
    """
    Visualizes high-dimensional sequence embeddings in 2D using PCA.
    Principal Component Analysis (PCA) provides a linear projection of the data,
    which helps to reveal the primary axes of variation and overall data structure.
    """

    def __init__(self,
            collection: AbstractFiguresCollection,
            n_components: int,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean):
        from sklearn.decomposition import PCA
        reducer: PCA = PCA(n_components=n_components, random_state=random_state)
        super().__init__(
            collection = collection,
            pooling_type = pooling_type,
            reducer = lambda x: reducer.fit_transform(x),
            base_title = "PCA Embedding"
        )
        self._fig.update_layout(
            xaxis_title = "PCA 1",
            yaxis_title = "PCA 2"
        )



class TSNEFigure(_AbstractEncodingsVisualizationFigure):
    """
    Visualizes high-dimensional sequence embeddings in 2D using t-SNE.
    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear technique
    effective at revealing local structure and clusters within the data.
    """

    def __init__(self,
            collection: AbstractFiguresCollection,
            n_components: int,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean):
        from sklearn.manifold import TSNE
        reducer: TSNE = TSNE(n_components=n_components, random_state=random_state)
        super().__init__(
            collection = collection,
            pooling_type = pooling_type,
            reducer = lambda x: reducer.fit_transform(x),
            base_title = "TSNE Embedding"
        )
        self._fig.update_layout(
            xaxis_title = "TSNE 1",
            yaxis_title = "TSNE 2"
        )



class UMAPFigure(_AbstractEncodingsVisualizationFigure):
    """
    Visualizes high-dimensional sequence embeddings in 2D using UMAP.
    Uniform Manifold Approximation and Projection (UMAP) is a non-linear technique
    known for preserving both local and global data structure, often providing
    a meaningful and scalable visualization of clusters.
    """

    def __init__(self,
            collection: AbstractFiguresCollection,
            n_components: int,
            random_state: Optional[int] = None,
            pooling_type: PoolingType = PoolingType.Per_Protein_Mean):
        from umap import UMAP
        reducer: UMAP = UMAP(n_components=n_components, random_state=random_state)
        super().__init__(
            collection = collection,
            pooling_type = pooling_type,
            reducer = lambda x: reducer.fit_transform(x),
            base_title = "UMAP Embedding"
        )
        self._fig.update_layout(
            xaxis_title = "UMAP 1",
            yaxis_title = "UMAP 2"
        )