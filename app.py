import streamlit as st
import joblib
import re
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import pdfplumber

# =========================
# SET TESSERACT PATH (Windows)
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# LOAD MODEL
# =========================
model = joblib.load("resume_classifier_model.pkl")

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# =========================
# PDF TEXT EXTRACTION (OCR ENABLED)
# =========================
def extract_text_from_pdf(uploaded_file):
    text = ""

    # Try normal PDF text extraction first
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    # If text is too short, use OCR
    if len(text.strip()) < 50:
        uploaded_file.seek(0)
        images = convert_from_bytes(uploaded_file.read())
        for img in images:
            text += pytesseract.image_to_string(img)

    return text.strip()

# =========================
# STREAMLIT UI
# =========================
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
POPPLER_PATH = "/usr/bin"
st.set_page_config(page_title="Resume Classifier", layout="centered")

st.title("📄 Resume Category Predictor")
st.write("Upload a resume (PDF) or paste text to predict job category.")

option = st.radio("Choose input method:", ("Paste Resume Text", "Upload Resume PDF"))

resume_text = ""

# ---- Text Input ----
if option == "Paste Resume Text":
    resume_text = st.text_area("Paste resume content here:")

# ---- PDF Upload ----
elif option == "Upload Resume PDF":
    uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:
            st.success("PDF processed and text extracted!")
        else:
            st.warning("No readable text found — try a clearer resume PDF.")

# =========================
# PREDICTION
# =========================
if st.button("Predict Category"):
    if resume_text.strip() != "":
        cleaned = clean_text(resume_text)
        prediction = model.predict([cleaned])[0]
        st.success(f"Predicted Resume Category: **{prediction}**")
    else:
        st.error("No resume text available for prediction.")
