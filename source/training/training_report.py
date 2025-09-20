import os
import re
import glob

from typing import Optional, Dict, List, Tuple

from source.metrics.metrics import Metrics
from source.models.abstract import AbstractModel
from source.report_generator import generate_report



def _find_latest_figures(directory: str) -> List[str]:
    """
    Finds the latest version of each unique figure type in a directory.

    It identifies unique figures by their filename prefix and selects the one
    with the highest epoch number.

    :param directory: The directory to search for figure files.
    :return: A list of file paths for the latest version of each figure.
    """
    all_pngs = glob.glob(os.path.join(directory, "*.png"))
    latest_figures: Dict[str, Tuple[int, str]] = {}

    for file_path in all_pngs:
        filename = os.path.basename(file_path)

        # Extract the figure prefix (e.g., "EpochDualAxisFigure")
        prefix_match = re.match(r"([a-zA-Z]+)_", filename)
        if not prefix_match:
            continue
        prefix = prefix_match.group(1)

        # Extract the epoch number
        epoch_match = re.search(r"epoch=(\d+)", filename)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))

        # If we find a newer version of a figure, update it
        if prefix not in latest_figures or epoch > latest_figures[prefix][0]:
            latest_figures[prefix] = (epoch, file_path)

    # Return only the file paths from our dictionary
    return [path for _, path in latest_figures.values()]



def generate_training_report(
        model: AbstractModel,
        best_metrics: Optional[Metrics],
        figures_save_dir: str
):
    """
    Gathers the latest training artifacts and generates a consolidated PDF report.

    :param model: The trained model object.
    :param best_metrics: The Metrics object from the best epoch (can be None on interrupt).
    :param figures_save_dir: The root directory where figures are saved.
    """
    report_figures_dir = os.path.join(figures_save_dir, model.name(), model.id())

    report_title = f"Training Report:<br/>{model.name()}<br/>{model.uuid}"
    config_summary = "\n".join([f"- {key}: {value}" for key, value in model.get_init_params().items()])

    if best_metrics:
        metrics_summary = best_metrics.summary()
        text_sections = [("Best Epoch Performance Metrics", metrics_summary)]
    else: text_sections = [("No Final Metrics", "Training was interrupted before a best model was established.")]

    text_sections.append(("Model Configuration", config_summary))
    text_sections.append(("Memory Size:", f"{model.memory_size():.3f} MB"))

    image_paths = sorted(_find_latest_figures(report_figures_dir))

    if not image_paths: print(f"Warning: No figures found in {report_figures_dir}. Report will be text-only.")
    else: print(f"Found {len(image_paths)} figures for the report.")

    report_output_path = os.path.join(report_figures_dir, f"training_report_{model.id()}.pdf")
    generate_report(
        output_path = report_output_path,
        title = report_title,
        texts = text_sections,
        image_paths = image_paths
    )
