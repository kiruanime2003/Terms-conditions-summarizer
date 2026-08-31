import streamlit as st

st.title("Simple Input & Predict")

# Selection tab
option = st.radio("Choose Input Type:", ["Paste Text", "Upload File"])

user_text = ""

if option == "Paste Text":
    user_text = st.text_area("Paste text here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")

# Predict Button
if st.button("Predict"):
    if user_text.strip():
        st.write("### Input Received:")
        # Your prediction logic goes here
    else:
        st.warning("Please provide some text or upload a file first.")