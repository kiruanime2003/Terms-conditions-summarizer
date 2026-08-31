import os
import re
import zipfile
import pandas as pd
import streamlit as st
from fastai.text.all import load_learner

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Clause Classifier",
    page_icon="⚖️",
    layout="wide"
)

# High-risk categories definition
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

# -----------------------------------------------------------------------------
# MODEL LOADING & UNZIPPING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_fastai_model():
    model_pkl = "legal_terms_conditions_model.pkl"
    model_zip = "legal_terms_conditions_model.zip"
    
    # Extract zip automatically if pkl does not exist locally
    if not os.path.exists(model_pkl) and os.path.exists(model_zip):
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
            
    if os.path.exists(model_pkl):
        return load_learner(model_pkl)
    else:
        st.error(f"Model file '{model_pkl}' or '{model_zip}' not found!")
        return None

learn = load_fastai_model()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def split_into_clauses(raw_text):
    """Splits raw text into non-empty clause paragraphs/sentences."""
    paragraphs = raw_text.split('\n')
    clauses = []
    for p in paragraphs:
        cleaned = p.strip()
        if len(cleaned) > 10:  # Ignore empty lines or trivial headers
            clauses.append(cleaned)
    return clauses

def run_batch_inference(clauses):
    """Runs parallel batch prediction over all extracted clauses."""
    dl = learn.dls.test_dl(clauses)
    preds, _, decoded_preds = learn.get_preds(dl=dl, with_decoded=True)
    vocab = learn.dls.vocab[1]
    
    results = []
    for idx, (pred_idx, prob_dist) in enumerate(zip(decoded_preds, preds)):
        cat = vocab[pred_idx]
        conf = float(prob_dist[pred_idx]) * 100
        risk = "RED" if cat in RED_CATEGORIES else "GREEN"
        results.append({
            "Clause Number": idx + 1,
            "Clause Text": clauses[idx],
            "Predicted Category": cat,
            "Risk Level": risk,
            "Confidence (%)": round(conf, 2)
        })
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# APPLICATION UI
# -----------------------------------------------------------------------------
st.title("⚖️ Legal Contract & Terms Analyzer")
st.write("Upload large contract files or paste raw legal text to identify clause categories and flag risk areas.")

# Selection tab for input method
input_mode = st.radio("Choose Input Method:", ["📋 Paste Raw Text", "📁 Upload File"], horizontal=True)

raw_input_text = ""

if input_mode == "📋 Paste Raw Text":
    raw_input_text = st.text_area(
        "Paste contract / terms & conditions text below (No word limit):",
        height=300,
        placeholder="Paste your legal document text here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .csv)",
        type=["txt", "csv"]
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.txt'):
            raw_input_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
            # Pick first string column if text column isn't explicitly named
            text_col = 'text' if 'text' in df_upload.columns else df_upload.select_dtypes(include=['object']).columns[0]
            raw_input_text = "\n".join(df_upload[text_col].dropna().tolist())

# Process and Analyze Action
if st.button("🚀 Analyze Document", type="primary"):
    if not raw_input_text.strip():
        st.warning("Please enter or upload valid text before analyzing.")
    elif learn is None:
        st.error("Model is not loaded. Please check your model weights file.")
    else:
        with st.spinner("Extracting clauses and running batch inference..."):
            clauses = split_into_clauses(raw_input_text)
            
            if not clauses:
                st.warning("No structural clauses detected in the provided text.")
            else:
                st.success(f"Successfully extracted {len(clauses)} clauses for analysis!")
                
                # Run Model Prediction
                df_results = run_batch_inference(clauses)
                
                # Separate into RED and GREEN Risk expanders
                red_df = df_results[df_results['Risk Level'] == 'RED']
                green_df = df_results[df_results['Risk Level'] == 'GREEN']
                
                # Summary Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Clauses Analyzed", len(df_results))
                col2.metric("🚩 High Risk (RED)", len(red_df))
                col3.metric("✅ Standard (GREEN)", len(green_df))
                
                st.markdown("---")
                
                # Render RED Risk Clauses
                with st.expander(f"🚩 High-Risk / Restrictive Clauses ({len(red_df)})", expanded=True):
                    if not red_df.empty:
                        for _, row in red_df.iterrows():
                            st.error(f"**[{row['Predicted Category']}]** (Confidence: {row['Confidence (%)']}%)")
                            st.write(f"_{row['Clause Text']}_")
                            st.markdown("---")
                    else:
                        st.write("No high-risk categories detected.")
                
                # Render GREEN Standard Clauses
                with st.expander(f"✅ Standard / Informational Clauses ({len(green_df)})", expanded=False):
                    if not green_df.empty:
                        for _, row in green_df.iterrows():
                            st.success(f"**[{row['Predicted Category']}]** (Confidence: {row['Confidence (%)']}%)")
                            st.write(f"_{row['Clause Text']}_")
                            st.markdown("---")
                    else:
                        st.write("No standard categories detected.")

                # Download Results as CSV
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Analysis as CSV",
                    data=csv,
                    file_name="legal_clause_analysis.csv",
                    mime="text/csv"
                )