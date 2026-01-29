import streamlit as st
import joblib
import re
import base64
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from streamlit_lottie import st_lottie
import json
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="📄",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("resume_classifier_model.pkl")

# =========================
# BACKGROUND IMAGE
# =========================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{data}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("background.jpg")   # <-- put image in same folder

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.glass {
    background: rgba(20, 20, 20, 0.75);
    backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.5);
    animation: slideUp 1s ease;
}

@keyframes slideUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

button {
    transition: 0.3s ease;
}
button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #4CAF50;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Resume", "About Project"])

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# =========================
# PDF EXTRACTION
# =========================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t
    if len(text.strip()) < 50:
        images = convert_from_bytes(uploaded_file.read())
        for img in images:
            text += pytesseract.image_to_string(img)
    return text

# =========================
# LOTTIE ANIMATION
# =========================
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

try:
    lottie_resume = load_lottie("resume.json")
except FileNotFoundError:
    lottie_resume = None


# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if lottie_resume:
        st_lottie(lottie_resume, height=250)


    st.markdown("""
    <h1 style='text-align:center;color:white;'>📄 Resume Category Predictor</h1>
    <p style='text-align:center;color:gray;'>AI-powered resume classification using NLP & ML</p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UPLOAD PAGE
# =========================
elif page == "Upload Resume":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    option = st.radio("Choose input method:", ["Paste Resume Text", "Upload Resume PDF"])
    resume_text = ""

    if option == "Paste Resume Text":
        resume_text = st.text_area("Paste resume text here")

    else:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting text..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                time.sleep(1)
            st.success("Text extracted successfully!")

    if resume_text.strip():
        if st.button("🔍 Predict Resume Category"):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            cleaned = clean_text(resume_text)
            prediction = model.predict([cleaned])[0]

            st.success(f"🎯 Predicted Category: **{prediction}**")
            st.balloons()

            with st.expander("📄 View Extracted Resume Text"):
                st.write(resume_text)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ABOUT PAGE
# =========================
else:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("""
    ## 📌 About Project

    This project uses **Natural Language Processing (NLP)** and **Machine Learning**
    to classify resumes into job categories automatically.

    **Tech Stack:**
    - Python
    - Streamlit
    - Scikit-learn
    - NLP (TF-IDF)
    - OCR (Tesseract)
    - PDF Processing

    Built by **Atmika Khandelwal** 🚀
    """)
    st.markdown("</div>", unsafe_allow_html=True)
