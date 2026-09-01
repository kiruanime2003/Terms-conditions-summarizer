import os
import sys
import re
import pathlib
from collections import defaultdict
import streamlit as st
import docx
import pdfplumber
from fastai.text.all import load_learner


if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


st.set_page_config(
    page_title="Legal Clause Scanner",
    page_icon="⚖️",
    layout="wide"
)


MODEL_PATH = "legal_terms_conditions_model.pkl"

@st.cache_resource
def load_legal_model():
    if os.path.exists(MODEL_PATH):
        return load_learner(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found!")
        return None

learn = load_legal_model()


RED_CATEGORIES = [
    'Renewal Term', 'Notice Period to Terminate Renewal', 'Most Favored Nation', 
    'Non-Compete', 'Exclusivity', 'No-Solicit of Customers', 
    'Competitive Restriction Exception', 'No-Solicit of Employees', 
    'Non-Disparagement', 'Termination for Convenience', 'Rofr/Rofo/Rofn', 
    'Change of Control', 'Anti-Assignment', 'Revenue/Profit Sharing', 
    'Price Restrictions', 'Minimum Commitment', 'Volume Restriction', 
    'IP Ownership Assignment', 'Joint IP Ownership', 
    'Unlimited/All-You-Can-Eat-License', 'Irrevocable or Perpetual License', 
    'Post-Termination Services', 'Audit Rights', 'Uncapped Liability', 
    'Cap on Liability', 'Liquidated Damages', 'Covenant Not to Sue'
]


def split_into_valid_sentences(text):
    """
    Splits text by full stops (.), ensuring:
    1. The clause ends with a dot (.)
    2. The clause contains MORE THAN 3 words
    """
    
    raw_sentences = re.split(r'\.(?=\s|$)', text)
    valid_clauses = []

    for item in raw_sentences:
        cleaned = item.strip()
        words = cleaned.split()
        
       
        if len(words) > 3:
            formatted_sentence = cleaned if cleaned.endswith('.') else cleaned + '.'
            valid_clauses.append(formatted_sentence)

    return valid_clauses

def extract_numbered_clauses_from_text(raw_text):
    sentences = split_into_valid_sentences(raw_text)
    return [{"line_num": i + 1, "text": sentence} for i, sentence in enumerate(sentences)]

def extract_numbered_clauses_from_docx(file):
    doc = docx.Document(file)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return extract_numbered_clauses_from_text(full_text)

def extract_numbered_clauses_from_pdf(file):
    text_content = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_content.append(t)
    full_text = "\n".join(text_content)
    return extract_numbered_clauses_from_text(full_text)

def get_snippet(text, word_count=7):
    words = text.split()
    if len(words) <= word_count:
        return text
    return " ".join(words[:word_count]) + "..."

def render_grouped_results(grouped_dict, is_red=True):
    sorted_categories = sorted(grouped_dict.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, items in sorted_categories:
        count = len(items)
        
        if count == 1:
            line_num = items[0]['line_num']
            snippet_text = items[0]['snippet']
            if is_red:
                st.error(f"**Category:** {category}\n\n**Line #{line_num}:** \"{snippet_text}\"")
            else:
                st.success(f"**Category:** {category}\n\n**Line #{line_num}:** \"{snippet_text}\"")
                
        else:
            expander_title = f"{category} ({count} occurrences)"
            with st.expander(expander_title, expanded=False):
                for idx, item in enumerate(items, start=1):
                    line_num = item['line_num']
                    snippet_text = item['snippet']
                    st.markdown(f"**Match #{idx} (Line #{line_num}):**")
                    if is_red:
                        st.error(f"**Snippet:** \"{snippet_text}\"")
                    else:
                        st.success(f"**Snippet:** \"{snippet_text}\"")


st.title("⚖️ Legal Clause Scanner")

input_option = st.radio("Choose Input Method:", ["Paste Text", "Upload File"], horizontal=True)

numbered_clauses = []

if input_option == "Paste Text":
    user_text = st.text_area("Paste legal text here:", height=200, placeholder="Enter text to analyze...")
    if user_text.strip():
        numbered_clauses = extract_numbered_clauses_from_text(user_text)
else:
    uploaded_file = st.file_uploader("Upload Document (.docx, .pdf, .txt):", type=["docx", "pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".docx"):
            numbered_clauses = extract_numbered_clauses_from_docx(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            numbered_clauses = extract_numbered_clauses_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            raw_text = uploaded_file.read().decode("utf-8")
            numbered_clauses = extract_numbered_clauses_from_text(raw_text)


if st.button("🚀 Predict", type="primary"):
    if not numbered_clauses:
        st.warning("Please enter text or upload a valid document before scanning.")
    elif learn is None:
        st.error("Model is not loaded.")
    else:
        st.markdown("### 📄 Processed Document View")
        formatted_doc_text = "\n\n".join([f"[{item['line_num']}] {item['text']}" for item in numbered_clauses])
        st.text_area(
            label="Sentence-by-sentence breakdown of the parsed document:",
            value=formatted_doc_text,
            height=220,
            disabled=True
        )

        red_grouped = defaultdict(list)
        green_grouped = defaultdict(list)

        total_red = 0
        total_green = 0

        with st.spinner("Scanning document with AWD-LSTM model..."):
            for item in numbered_clauses:
                line_num = item["line_num"]
                clause_text = item["text"]
                
                category, _, _ = learn.predict(clause_text)
                cat_str = str(category)
                snippet = get_snippet(clause_text, word_count=7)

                result_payload = {"line_num": line_num, "snippet": snippet}

                if cat_str in RED_CATEGORIES:
                    red_grouped[cat_str].append(result_payload)
                    total_red += 1
                else:
                    green_grouped[cat_str].append(result_payload)
                    total_green += 1

        st.markdown("### 📊 Scan Results")
        col_red, col_green = st.columns(2)

        with col_red:
            st.subheader(f"🚩 High Risk / Red Categories ({total_red})")
            if red_grouped:
                render_grouped_results(red_grouped, is_red=True)
            else:
                st.success("No high-risk categories detected.")

        with col_green:
            st.subheader(f"✅ Standard / Green Categories ({total_green})")
            if green_grouped:
                render_grouped_results(green_grouped, is_red=False)
            else:
                st.info("No standard categories detected.")