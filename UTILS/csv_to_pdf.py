import csv
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.lib import utils
from reportlab.lib.colors import magenta, pink, blue, green
from pdfrw import PdfReader, PdfWriter, PdfDict, PdfName, PdfArray


def parse_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames

        label_col = next((col for col in columns if "LABEL" in col), None)
        short_q_col = next((col for col in columns if "SHORT Q" in col), None)
        long_q_cols = [col for col in columns if col.startswith("LONG Q")]
        units_col = next((col for col in columns if "UNITS" in col), None)
        quest_type_col = next((col for col in columns if "QUEST_TYPE" in col),
                              None)
        image_col = next((col for col in columns if "IMAGE" in col), None)

        if not all(
            [label_col, short_q_col, long_q_cols, units_col, quest_type_col]):
            raise ValueError("Missing required columns in the CSV file")

        if len(long_q_cols) > 1:
            languages = [
                col.split("LONG Q_")[-1] for col in long_q_cols if "_" in col
            ]
            print(f"Available languages: {', '.join(languages)}")
            chosen_lang = input("Enter the language code to use (e.g., ENG): "
                                ).strip().upper()
            long_q_col = next(
                (col
                 for col in long_q_cols if col.endswith(f"_{chosen_lang}")),
                None)
            if not long_q_col:
                raise ValueError(
                    f"Selected language '{chosen_lang}' not found in the CSV file"
                )
        else:
            long_q_col = long_q_cols[0]

        questions = []

        for row in reader:
            question = {
                "label": row[label_col],
                "short_q": row[short_q_col],
                "long_q": row[long_q_col],
                "units": row[units_col],
                "quest_type": row[quest_type_col],
                "image": row[image_col] if image_col and row[image_col].strip() else None
            }
            questions.append(question)

        return questions


def create_pdf(questions, output_pdf):
    tmp_pdf = "temp.pdf"
    c = canvas.Canvas(tmp_pdf, pagesize=letter)
    form = c.acroForm

    width, height = letter

    y_position = height - 50
    field_names = []

    current_category = None
    for question in questions:
        if question['quest_type'] != current_category:

            if question['quest_type'] == 'target':

                c.showPage()
                y_position = height - 50

            current_category = question['quest_type']
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2, y_position,
                                current_category.upper())
            y_position -= 30

        if y_position < 100:
            c.showPage()
            y_position = height - 50

        # Add image if available
        if question['image']:
            image_path = os.path.join("images", question['image'])
            if os.path.exists(image_path):
                img = utils.ImageReader(image_path)
                iw, ih = img.getSize()
                aspect = iw / ih
                new_width = 400
                new_height = new_width / aspect
                c.drawImage(image_path, 70, y_position - new_height - 10, width=new_width, height=new_height)
                y_position -= new_height + 20

        y_position -= 20

        if y_position < 100:
            c.showPage()
            y_position = height - 50


        title = f"{question['label']}. {question['short_q']}"
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, title)
        y_position -= 20

        c.setFont("Helvetica", 10)
        lines = simpleSplit(f"{question['long_q']} ({question['units']})",
                            "Helvetica", 10, width - 100)
        for line in lines:
            c.drawString(70, y_position, line)
            y_position -= 15

        if y_position < 100:
            c.showPage()
            y_position = height - 50



        y_position -= 25
        c.setFont("Helvetica", 10)
        x_start = 100
        labels = ["5%ile", "50%ile", "95%ile"]
        rects = []
        for i, perc in enumerate(labels):
            field_name = f"{question['label']}_{perc}"
            rect_x = x_start + i * 170
            #c.rect(rect_x, y_position, 100, 20)
            #rects.append((field_name,
            #              [rect_x, y_position, rect_x + 100, y_position + 20]))

            form.textfield(name=field_name, tooltip=perc,
                   x=rect_x, y=y_position, borderStyle='inset',
                   borderColor=green, fillColor=None, 
                   width=100,
                   height=25,
                   textColor=blue, forceBorder=True)

        y_position -= 15
        for i, perc in enumerate(labels):
            c.drawString(x_start + i * 170 + 35, y_position, perc)
        y_position -= 15

        field_names.extend(rects)

    c.save()

    # Add fillable fields
    writer = PdfWriter()
    reader = PdfReader(tmp_pdf)
    for page in reader.pages:
        annotations = []
        for field_name, rect in field_names:
            annotation = PdfDict(Subtype=PdfName.Widget,
                                 T=field_name,
                                 Rect=rect,
                                 FT=PdfName.Tx,
                                 Ff=1,
                                 DA="/Helv 0 Tf 0 g")
            annotations.append(annotation)

        if annotations:
            page.Annots = PdfArray(annotations)
        writer.addpage(page)

    writer.write(output_pdf)
    os.remove(tmp_pdf)


if __name__ == "__main__":
    csv_file = "questionnaire.csv"
    output_pdf = "questionnaire_form.pdf"
    questions = parse_csv(csv_file)
    create_pdf(questions, output_pdf)
    print(f"PDF form created: {output_pdf}")
