import streamlit as st
import pandas as pd

st.title("ECM Aging Atlas")
st.write("🚧 Under development...")

st.header("Project structure")
st.write("This project includes:")
st.write("- Data processing scripts")
st.write("- Raw and processed data")
st.write("- Documentation")
st.write("- Analysis results")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section", 
                           ["Overview", "Data", "Analysis", "Documentation"])

if page == "Overview":
    st.write("Welcome to the ECM Aging Atlas project overview!")
elif page == "Data":
    st.write("Data exploration coming soon...")
elif page == "Analysis":
    st.write("Analysis results coming soon...")
elif page == "Documentation":
    st.write("Documentation coming soon...")