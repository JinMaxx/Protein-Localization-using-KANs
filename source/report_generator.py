from typing import List, Tuple, Dict, Optional

from pypdf import PdfWriter
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color, gray
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable, Flowable


def _setup_styles() -> Dict:
    """Creates and returns a dictionary of custom paragraph styles."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name = 'Preformatted',
        parent = styles['Normal'],
        fontName = 'Courier',
        fontSize = 9,
        leading = 12,
        leftIndent = 6,
        rightIndent = 6,
        borderPadding = 10,
        backColor = Color(0.95, 0.95, 0.95),
        borderColor = Color(0.85, 0.85, 0.85),
        borderWidth = 1,
        borderRadius = 2,
        spaceBefore = 10,
        spaceAfter = 10,
    ))
    return styles


def _build_title_story(styles: Dict, title: str) -> List[Flowable]:
    """Builds the main title section of the report."""
    return [
        Paragraph(
            text = title,
            style = styles['h1']
        ),
        HRFlowable(
            width = "100%",
            thickness = 1,
            color = gray,
            spaceAfter = 12
        ),
        Spacer(
            width = 1,
            height = 0.5 * cm
        )
    ]


def _build_texts_story(styles: Dict, texts: List[Tuple[Optional[str], str]]) -> List[Flowable]:
    """Builds the text sections from the provided list of tuples."""
    story: List[Flowable] = []
    for header, content in texts:
        if header: story.append(
            Paragraph(
                text = header,
                style = styles['h2']
            )
        )
        story.append(
            Paragraph(
                text = content.strip().replace('\n', '<br/>'),
                style = styles['Preformatted']
            )
        )
        story.append(
            Spacer(
                width = 1,
                height = 0.5 * cm
            )
        )
    return story


def _build_images_story(image_paths: List[str])-> List[Flowable]:
    """Builds the figures section, scaling and centering images."""
    if not image_paths: return []

    styles = _setup_styles()  # Get styles for the header
    story: List[Flowable] = [
        PageBreak(),
        Paragraph(
            text = "Figures & Visualizations",
            style = styles['h1']
        ),
        HRFlowable(
            width = "100%",
            thickness = 1,
            color = gray,
            spaceAfter = 12
        ),
        Spacer(
            width = 1,
            height = 0.5 * cm
        )
    ]

    page_width, _ = A4
    margin_size = 2.5 * cm
    max_width = page_width - (2 * margin_size)

    for fig_path in image_paths:
        image = Image(
            filename = fig_path,
            hAlign = 'CENTER'
        )
        if image.imageWidth > max_width:
            ratio = max_width / image.imageWidth
            image.drawWidth = max_width
            image.drawHeight = image.imageHeight * ratio

        story.append(image)
        story.append(
            Spacer(
                width = 1,
                height = 0.5 * cm
            )
        )

    return story



def _build_story(title: str, texts: List[Tuple[Optional[str], str]], image_paths: List[str])-> List[Flowable]:
    """Assembles the report's content by calling all helper functions."""
    styles = _setup_styles()
    story: List[Flowable] = _build_title_story(styles, title)
    story.extend(_build_texts_story(styles, texts))
    story.extend(_build_images_story(image_paths))
    return story




def generate_report(output_path: str, title: str, texts: List[Tuple[Optional[str], str]], image_paths: List[str]):
    """
    Generates a professional PDF report from a title, text sections, and images.

    :param output_path: The path where the PDF will be saved.
    :param title: The main title of the report.
    :param texts: A list of tuples, where each is (optional_header, content_string).
    :param image_paths: A list of paths to .png images to include.
    """
    margin = 2.5 * cm
    doc = SimpleDocTemplate(
        output_path,
        pagesize = A4,
        leftMargin = margin, rightMargin = margin,
        topMargin = margin, bottomMargin = margin
    )

    story = _build_story(title, texts, image_paths)
    doc.build(story)
    print(f"Report successfully generated at: {output_path}")



def merge_pdfs(output_path: str, input_paths: list[str]):
    """Merges multiple PDF files into a single document."""
    pdf_writer = PdfWriter()

    for path in input_paths:
        pdf_writer.append(path)

    with open(output_path, "wb") as f_out:
        pdf_writer.write(f_out)
    print(f"Merged PDF created at: {output_path}")