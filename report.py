from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import datetime

def draw_centered_string(cnv, text, font, size, x, y):
    cnv.setFont(font, size)
    width = cnv.stringWidth(text, font, size)
    cnv.drawString(x - width / 2, y, text)

def create_pdf(title, image_path, caption, body_text, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Constants for layout
    MARGIN = inch
    TITLE_SIZE = 16
    SUBTITLE_SIZE = 12
    CAPTION_SIZE = 10
    SPACING = 10

    # Title and Date
    draw_centered_string(c, title, "Helvetica-Bold", TITLE_SIZE, width / 2, height - MARGIN - TITLE_SIZE)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    draw_centered_string(c, date_str, "Helvetica", SUBTITLE_SIZE, width / 2, height - MARGIN - TITLE_SIZE - SUBTITLE_SIZE - SPACING)

    # Image
    img = Image(image_path)
    img_width, img_height = img.wrap(0, 0)
    scale_factor = 1
    img_width /= scale_factor
    img_height /= scale_factor
    img_x = (width - img_width) / 2
    img_y = height - 2 * MARGIN - img_height - TITLE_SIZE - SUBTITLE_SIZE - 2 * SPACING
    img.drawOn(c, img_x, img_y)

    # Caption (centered with the image)
    caption_y = img_y - CAPTION_SIZE - SPACING
    draw_centered_string(c, caption, "Times-Italic", CAPTION_SIZE, width / 2, caption_y)

    # Body Text
    styles = getSampleStyleSheet()
    text = Paragraph(body_text, style=styles["Normal"])
    text_height = text.wrap(width - 2 * MARGIN, caption_y - MARGIN)[1]
    text.drawOn(c, MARGIN, caption_y - text_height - SPACING)

    # Save PDF
    c.save()





