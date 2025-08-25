"""
Provides a robust and extensible framework for creating, managing, and saving `plotly` figures,
designed for consistency and reusability in a data science workflow.

It's built around two primary abstract classes:

-   `AbstractFiguresCollection` acts as a manager for a group of figures,
    handling batch updates, display, and saving.

-   `_AbstractFigure` is a wrapper for a single `plotly` figure,
    providing a standardized interface for updates, theming, and serialization
    (saving as both a `.png` image and a `.pkl` object).

The module also includes `MultiFigure` for combining several plots into a single subplot grid,
and a utility for applying a centralized visual theme to all figures.
"""

import os
import re
import uuid
import pickle
# import kaleido #required
# print("kaleido: ", kaleido.__version__)

from math import sqrt, ceil
from abc import abstractmethod, ABC

from typing_extensions import (
    TypeVar, Any, Optional, Callable,
    Dict, Tuple, override, cast, Iterator, List
)

from colorcet import glasbey_dark
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from IPython.display import clear_output  #, display

from source.data_scripts.read_data import Label


color_palette = glasbey_dark

def get_label_color(label: Label) -> str:
    """Returns a consistent color for a given data label."""
    return color_palette[label.value % len(color_palette)]



CustomFigureT: TypeVar = TypeVar('CustomFigureT', bound='_AbstractFigure')  # See FigureCollection.other(...)
UpdateFunctionT: TypeVar = Callable[[Optional[Figure], Tuple[Any, ...], Dict[str, Any]], Optional[Figure]]


def set_centralized_theme(fig: Figure):
    """Applies a consistent, clean visual style to a Plotly figure."""
    fig.update_layout(
        font = dict(
            family = "Arial, sans-serif",  # Minimalist font
            size = 12,
            color = "black"  # Font color
        ),
        legend = dict(
            title = dict(text="Legend", font=dict(size=12)),
            font = dict(size=10),
            bgcolor="rgba(0, 0, 0, 0)",  # Transparent
        ),
        xaxis = dict(
            gridcolor = "lightgrey",
            showline = True,
            linecolor = "black",
            linewidth = 1,
            zeroline = False,
        ),
        yaxis = dict(
            gridcolor = "lightgrey",
            showline = True,
            linecolor = "black",
            linewidth = 1,
            zeroline = False,
        ),
        plot_bgcolor = "white",
        paper_bgcolor = "white",
    )



class AbstractFiguresCollection(ABC):
    """An abstract manager for a group of figures, handling batch updates and I/O."""

    def __init__(self, save_dir: Optional[str] = None):
        self.__figures: List[_AbstractFigure] = []

        self.save_dir: Optional[str] = save_dir


    @abstractmethod  # implement this by specifying all parameters and calling super(...).update(...) to make linting better.
    def update(self, **kwargs) -> None:
        """Updates all figures in the collection with the given keyword arguments."""
        clear = kwargs.pop("clear", True)
        errors: List[Tuple[BaseException, _AbstractFigure]] = []
        for figure in self.__figures:
            try: figure.update( **kwargs)
            except Exception as error:
                errors.append((error, figure))
                figure.remove()  # remove faulty figures
                raise error # for debugging
        self.display(clear=clear)
        for error, figure in errors:
            print(f"\n{error.__class__.__name__} occurred while updating figure '{figure.name()}':\n{error} \n\n")


    @staticmethod
    def __clear() -> None:
        """Clears the output used for creating 'live' updates in notebooks."""
        clear_output(wait=True)  # simulate "live" updates


    def display(self, clear: bool = True):
        """Displays all figures in the collection."""
        if clear: self.__clear()
        for figure in self.__figures: figure.display()


    def save(self, sub_dir: Optional[str] = None, **kwargs: Any) -> None:
        """
        Saves all figures in the collection to the specified save directory,
        optionally within a given subdirectory.

        Args:
            sub_dir (Optional[str]): An optional subdirectory path to append
                                     to the base save directory. It can contain
                                     multiple levels (e.g., 'model_name/model_id').
            **kwargs: Additional keyword arguments passed to each figure's save method.
        """
        if self.save_dir is not None:
            final_save_dir = os.path.join(self.save_dir, sub_dir) if sub_dir else self.save_dir
            os.makedirs(final_save_dir, exist_ok=True)
            for figure in self.__figures:
                figure.save(final_save_dir, **kwargs)


    def delete(self, figure: '_AbstractFigure') -> None:
        """Removes a figure from the collection."""
        if figure in self.__figures: self.__figures.remove(figure)


    def _add(self, figure: '_AbstractFigure') -> '_AbstractFigure':
        """Adds a figure to the collection. For internal use."""
        self.__figures.append(figure)  # Private function prevents two collections from sharing figures.
        return figure


    def __del__(self):
        """Ensures all figures are removed from the collection upon deletion."""
        for figure in self.__figures: figure.remove()


    def __str__(self) -> str:
        return f"FiguresCollection: {self.__figures}"


    def __repr__(self) -> str:
        return self.__str__()


    def __iter__(self) -> Iterator['_AbstractFigure']:
        """Allows iteration over the figures in the collection."""
        return iter(self.__figures)


    def load(self, file_path: str) -> CustomFigureT:
        """Loads a figure from a pickle file and adds it to the collection."""
        figure: _AbstractFigure = _AbstractFigure.load(file_path)
        self._add(figure)
        return figure


    # When the parameter figure is initially None, then no figure exists yet until provided by the update_function and nothing will be displayed.
    # If update_function returns a figure, it replaces the existing one.
    def other(self,
            figure: Optional[Figure],
            update_function: UpdateFunctionT,
            identifier: Optional[str] = None,
            name: Optional[str] = None) -> CustomFigureT:
        """Creates a custom figure on-the-fly from a provided update function."""

        # noinspection PyMethodParameters
        class _CustomFigure(_AbstractFigure):
            """A dynamically created figure class."""

            def __init__(_self, _collection: AbstractFiguresCollection, _figure: Optional[Figure]):
                super().__init__(collection=_collection, figure=_figure, identifier=identifier)
                if _figure is not None: set_centralized_theme(_figure)

            @override
            def update(_self, **kwargs):
                fig = update_function(_self._fig, **kwargs)
                if fig is not None:
                    _self._fig = fig
                    set_centralized_theme(_self._fig)
                    _self._fig.update_layout(title_text=_self.name())

            @override
            def name(_self) -> str:
                return name if name is not None else "CustomFigure"

        return _CustomFigure(_collection=self, _figure=figure)


    def multifigure(self, figures: List[CustomFigureT], identifier: Optional[str] = None, show_legend: bool = True) -> 'MultiFigure':
        """Combines multiple figures into a single MultiFigure with subplots."""
        return cast(MultiFigure, self._add(MultiFigure(figures=figures, identifier=identifier, show_legend=show_legend, collection=self)))



# -------------------------------- Figures --------------------------------


class _AbstractFigure(ABC):
    """
    An abstract wrapper for a single Plotly figure, providing standardized
    methods for updating, theming, and serialization.
    """

    def __init__(self, collection: AbstractFiguresCollection, figure: Optional[Figure] = None, identifier: Optional[str] = None):
        self._fig: Optional[Figure] = figure
        if self._fig is not None: set_centralized_theme(self._fig)
        self._identifier: str = identifier if identifier is not None else str(uuid.uuid4())  # used for unique file_paths
        self.__collection: AbstractFiguresCollection = collection


    @abstractmethod
    def update(self, **kwargs) -> None:
        """Abstract method to update the figure's data or layout."""
        pass


    def display(self) -> None:
        """Renders the figure, typically in a browser or IDE window."""
        if self._fig is not None: # and self.__display:
            self._fig.show()  # when running plotly outside Jupyter (e.g., in an IDE or terminal)


    def name(self) -> str:
        """Returns the name of the figure, defaulting to its class name."""
        return self.__class__.__name__


    def save(self, save_dir: str, **kwargs: Any) -> Optional[str]:
        """
        Saves the figure as a .png image and a .pkl (pickle) object.
        Requires the 'kaleido' library for image export.
        """
        if self._fig is not None:
            os.makedirs(save_dir, exist_ok=True)
            additional_info = "_".join([f"{key}={str(value)}" for key, value in kwargs.items()])
            if additional_info:
                additional_info = re.sub("[^a-zA-Z0-9_=-]", "", additional_info)
                file_path: str = f"{save_dir}/{self.name()}_{self._identifier}_{additional_info}"
            else: file_path: str = f"{save_dir}/{self.name()}_{self._identifier}"
            image_path: str = f"{file_path}.png"
            object_path = f"{file_path}.pkl"
            self._fig.write_image(image_path)  # Requires 'kaleido' installed
            with open(object_path, "wb") as object_file_handle:
                pickle.dump(self._fig, object_file_handle, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore[arg-type]
            return image_path
        else: return None


    @staticmethod
    def load(file_path: str) -> '_AbstractFigure':
        """Loads a figure from a pickle file."""
        with open(file_path, "rb") as object_file_handle: figure = pickle.load(object_file_handle)
        if not isinstance(figure, _AbstractFigure): raise TypeError("Loaded object is not an _AbstractFigure")
        return figure


    def remove(self):
        """Removes this figure from its parent collection."""
        self.__collection.delete(self)


    def __del__(self):
        """Ensures the figure is removed from its collection upon deletion."""
        self.remove()


    def __str__(self) -> str:
        return self.name()


    def __repr__(self) -> str:
        return self.__str__()



# More general version of DuoFigure with an arbitrary number of sub-figures
class MultiFigure(_AbstractFigure):
    """A figure that combines multiple sub-figures into a single grid of subplots."""

    def __init__(self,
            figures: List[_AbstractFigure],
            collection: AbstractFiguresCollection,
            identifier: Optional[str] = None,
            show_legend: bool = True):

        assert len(figures) > 0, "At least one figure must be passed to MultiFigure"

        # Store sub-figures
        self.__figures: List[_AbstractFigure] = figures
        for figure in self.__figures: figure.remove()  # Remove from FiguresCollection. Updates handled over MultFigure.

        self.__show_legend: bool = show_legend

        # Calculate grid size (rows, cols)
        self.cols = ceil(sqrt(len(self.__figures)))
        self.rows = ceil(len(self.__figures) / self.cols)

        self.coords: List[Tuple[int, int]] = \
            [((idx // self.cols) + 1, (idx % self.cols) + 1) for idx in range(len(figures))]

        fig: Figure = make_subplots(rows=self.rows, cols=self.cols)
        fig.update_layout(
            title_text = f"{self.__class__.__name__}: {' | '.join([sub_fig.name() for sub_fig in self.__figures])}",
            showlegend = self.__show_legend
        )

        super().__init__(
            collection = collection,
            figure = fig,
            identifier = identifier
        )

        self.__refresh_traces()


    @override
    def update(self, **kwargs) -> None:
        """Updates all sub-figures and refreshes the combined plot."""
        for subfigure in self.__figures:
            subfigure.update(**kwargs)
        self.__refresh_traces()


    def __add_subfigure(self, subfigure: _AbstractFigure, row: int, col: int) -> None:
        """Adds traces from a sub-figure to the main grid, handling legends."""
        if subfigure._fig is not None:
            for trace in subfigure._fig.data:
                if self.__show_legend and trace.name not in self.__seen_names:
                    self._fig.add_trace(trace, row=row, col=col)
                    self.__seen_names.add(trace.name)
                else:
                    self._fig.add_trace(trace.update(showlegend=False), row=row, col=col)


    def __update_layout(self, subfigure: _AbstractFigure, row: int, col: int):
        """Copies the axis layout from a sub-figure to the corresponding subplot."""
        # Extract axis configurations from the subfigure, if available
        if subfigure._fig is not None:
            if hasattr(subfigure._fig.layout, 'xaxis') and hasattr(subfigure._fig.layout, 'yaxis'):
                axis_config_x = subfigure._fig.layout.xaxis.to_plotly_json()
                axis_config_y = subfigure._fig.layout.yaxis.to_plotly_json()
            else:  # default axis layout
                axis_config_x = axis_config_y = {
                    "gridcolor": "lightgrey",
                    "showline": True,
                    "linecolor": "black",
                    "linewidth": 1,
                    "zeroline": False
                }

            self._fig.update_xaxes(row=row, col=col, **axis_config_x)
            self._fig.update_yaxes(row=row, col=col, **axis_config_y)


    def __refresh_traces(self) -> None:
        """Clears and redraws all traces and layouts from the sub-figures."""
        self._fig.data = []  # Clear all data
        if self.__show_legend: self.__seen_names = set()  # Reset legend tracking
        for subfigure, (row, col)  in zip(self.__figures, self.coords):
            self.__add_subfigure(subfigure, row, col)
            self.__update_layout(subfigure, row, col)


    @override
    def name(self) -> str:
        """Generates a combined name from all sub-figures."""
        return f"MultiFigure_{'_'.join([sub_fig.name() for sub_fig in self.__figures])}"