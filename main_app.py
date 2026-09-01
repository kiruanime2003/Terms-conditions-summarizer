import os
import sys
import pathlib
from collections import defaultdict
import streamlit as st
from fastai.text.all import load_learner

# -----------------------------------------------------------------------------
# PATHLIB PATCH FOR WINDOWS
# -----------------------------------------------------------------------------
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Clause Scanner",
    page_icon="⚖️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
MODEL_PATH = "legal_terms_conditions_model.pkl"

@st.cache_resource
def load_legal_model():
    if os.path.exists(MODEL_PATH):
        return load_learner(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found!")
        return None

learn = load_legal_model()

# -----------------------------------------------------------------------------
# HIGH-RISK CATEGORIES DEFINITION
# -----------------------------------------------------------------------------
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
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def extract_clauses(text):
    lines = text.split('\n')
    return [line.strip() for line in lines if len(line.strip()) > 10]

def get_snippet(text, word_count=7):
    words = text.split()
    if len(words) <= word_count:
        return text
    return " ".join(words[:word_count]) + "..."

def render_grouped_results(grouped_dict, is_red=True):
    """
    Sorts categories from least to most occurrences and renders single
    items directly or multiple items inside a dropdown expander.
    """
    # Sort categories by frequency ascending (least repeated to most repeated)
    sorted_categories = sorted(grouped_dict.items(), key=lambda x: len(x[1]))
    
    for category, snippets in sorted_categories:
        count = len(snippets)
        
        # SINGLE OCCURRENCE: Render directly
        if count == 1:
            snippet_text = snippets[0]
            if is_red:
                st.error(f"**Category:** {category}\n\n**Snippet:** \"{snippet_text}\"")
            else:
                st.success(f"**Category:** {category}\n\n**Snippet:** \"{snippet_text}\"")
                
        # REPEATED OCCURRENCES: Wrap in a dropdown expander
        else:
            expander_title = f"{category} ({count} occurrences)"
            with st.expander(expander_title, expanded=False):
                for idx, snippet_text in enumerate(snippets, start=1):
                    st.markdown(f"**Match #{idx}:**")
                    if is_red:
                        st.error(f"**Snippet:** \"{snippet_text}\"")
                    else:
                        st.success(f"**Snippet:** \"{snippet_text}\"")

# -----------------------------------------------------------------------------
# UI INPUT SECTION
# -----------------------------------------------------------------------------
st.title("Terms and Conditions Summarizer")

input_option = st.radio("Input:", ["Paste Text", "Upload File"], horizontal=True)

user_text = ""

if input_option == "Paste Text":
    user_text = st.text_area("Paste legal text here:", height=200, placeholder="Enter text")
else:
    uploaded_file = st.file_uploader("Upload a text document (.txt):", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")

# -----------------------------------------------------------------------------
# PREDICTION & GROUPED DISPLAY
# -----------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.warning("Please enter text or upload a file")
    elif learn is None:
        st.error("Model is not loaded.")
    else:
        clauses = extract_clauses(user_text)
        
        if not clauses:
            st.warning("No valid text clauses detected.")
        else:
            # Dictionaries to hold lists of snippets per category
            red_grouped = defaultdict(list)
            green_grouped = defaultdict(list)

            total_red = 0
            total_green = 0

            with st.spinner("Scanning document"):
                for clause in clauses:
                    category, _, _ = learn.predict(clause)
                    cat_str = str(category)
                    snippet = get_snippet(clause, word_count=7)

                    if cat_str in RED_CATEGORIES:
                        red_grouped[cat_str].append(snippet)
                        total_red += 1
                    else:
                        green_grouped[cat_str].append(snippet)
                        total_green += 1

            # -------------------------------------------------------------
            # SIDE-BY-SIDE GROUPED DISPLAY
            # -------------------------------------------------------------
            st.markdown("Scan Results")
            col_red, col_green = st.columns(2)

            # Left Column: RED High Risk Categories
            with col_red:
                st.subheader(f"🚩Risk Categories ({total_red})")
                if red_grouped:
                    render_grouped_results(red_grouped, is_red=True)
                else:
                    st.success("No high-risk categories detected.")

            # Right Column: GREEN Standard Categories
            with col_green:
                st.subheader(f"✅Green Categories ({total_green})")
                if green_grouped:
                    render_grouped_results(green_grouped, is_red=False)
                else:
                    st.info("No standard categories detected.")