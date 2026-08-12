import streamlit as st

uploaded_file = st.file_uploader("Choisissez un fichier")

if uploaded_file is not None:
    st.write(uploaded_file.name)