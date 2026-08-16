from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(uploaded_file):
    """
    Extrait le texte d'un fichier PDF.
    """
    pdf_reader = PdfReader(uploaded_file)

    content = ""

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            content += text + "\n"

    return content


def extract_text_from_txt(uploaded_file):
    """
    Extrait le texte d'un fichier TXT.
    """
    content = uploaded_file.read()

    return content.decode("utf-8")


def extract_text_from_docx(uploaded_file):
    """
    Extrait le texte d'un fichier DOCX.
    """
    document = Document(uploaded_file)

    content = ""

    for paragraph in document.paragraphs:
        if paragraph.text:
            content += paragraph.text + "\n"

    return content


def extract_text(uploaded_file):
    """
    Détermine le type du fichier et extrait son contenu.
    """

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        return extract_text_from_pdf(uploaded_file)

    elif file_type == "txt":
        return extract_text_from_txt(uploaded_file)

    elif file_type == "docx":
        return extract_text_from_docx(uploaded_file)

    else:
        raise ValueError("Format de fichier non supporté.")