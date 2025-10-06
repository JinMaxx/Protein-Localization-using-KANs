# evaluation_reporter.py

import os
import glob

from source.models.abstract import AbstractModel
from source.metrics.metrics import Metrics
from source.metrics.metrics_sampled import MetricsSampled
from source.report_generator import generate_report, merge_pdfs


def generate_evaluation_report(
        model: AbstractModel,
        metrics: Metrics,
        metrics_sampled: MetricsSampled,
        figures_save_dir: str
):
    """
    Gathers final evaluation artifacts for a single model and generates a PDF report.

    This function finds all generated figures for the model's evaluation run,
    formats the standard and sampled metrics summaries, and calls the generic
    report generator to create the final document.

    :param model: The evaluated model object.
    :param metrics: The standard Metrics object from the full test set evaluation.
    :param metrics_sampled: The MetricsSampled object from the statistical evaluation.
    :param figures_save_dir: The root directory where all figures are saved.

    :return: The file path of the generated evaluation report.
    """
    report_title = f"Final Evaluation Report: {model.name()}<br/>{model.uuid}"

    text_sections = [
        ("Test Set Performance", metrics.summary()),
        ("Test Set Performance (Sampled Statistics)", metrics_sampled.summary()),
    ]

    eval_model_figures_dir = os.path.join(figures_save_dir, "evaluation", model.name(), model.id())

    search_pattern = os.path.join(eval_model_figures_dir, "*.png")
    image_paths = sorted(glob.glob(search_pattern))

    if not image_paths: print(f"  - Warning: No figures found in {eval_model_figures_dir}. Report will be text-only.")
    else: print(f"  - Found {len(image_paths)} figures for the evaluation report.")

    report_output_path = os.path.join(eval_model_figures_dir, f"evaluation_report_{model.id()}.pdf")
    generate_report(
        output_path = report_output_path,
        title = report_title,
        texts = text_sections,
        image_paths = image_paths
    )

    training_report_path = os.path.join(
        figures_save_dir,
        "training",
        model.name(),
        model.id(),
        f"training_report_{model.id()}.pdf"
    )

    if os.path.exists(training_report_path) and os.path.exists(report_output_path):
        print("  - Both reports found. Merging...")
        merged_report_path = os.path.join(eval_model_figures_dir, f"full_report_{model.id()}.pdf")
        merge_pdfs(
            output_path = merged_report_path,
            input_paths = [training_report_path, report_output_path]
        )
    else: print("  - Warning: Could not find both reports. Skipping merge.")