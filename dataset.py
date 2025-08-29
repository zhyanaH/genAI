# app.py — Streamlit + Google Gemini (no OpenAI) + CSV ingest
import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-1.5-pro

st.title("Hello, Mark")
st.write("Your AI companion for quick answers, ideas, and insights.")
st.caption("Tip: Upload a CSV, click **Ingest CSV**, then ask Gemini about it.")

# --- Session state for DataFrame ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Prompt input ---
prompt = st.text_area("Enter a prompt:", "")

# --- CSV uploader + ingest button ---
uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

col1, col2 = st.columns([1, 1])
with col1:
    ingest_clicked = st.button("📥 Ingest CSV", use_container_width=True)
with col2:
    generate_clicked = st.button("🤖 Generate", use_container_width=True)

# --- Ingest logic ---
if ingest_clicked:
    if uploaded is None:
        st.warning("Please upload a CSV first.")
    else:
        try:
            # Try to detect encoding; fall back to utf-8
            content = uploaded.read()
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin1")

            st.session_state.df = df
            st.success(f"Ingested: **{uploaded.name}** ({df.shape[0]} rows × {df.shape[1]} columns)")
            with st.expander("Preview (top 5 rows)"):
                st.dataframe(df.head(), use_container_width=True)

            with st.expander("Columns"):
                st.write(list(df.columns))
        except Exception as e:
            st.error(f"CSV ingest error: {e}")

# --- Generate (Gemini) logic ---
if generate_clicked:
    try:
        # If there is a DataFrame, craft a compact summary to keep token usage low
        csv_context = ""
        if st.session_state.df is not None:
            df = st.session_state.df
            # Small sample to give context without sending entire CSV
            head_sample = df.head(3).to_dict(orient="records")
            csv_context = (
                "You are given a dataset summary. "
                "Use it to answer the user prompt. "
                "If information is insufficient, say what else is needed.\n\n"
                f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} cols\n"
                f"Columns: {list(df.columns)}\n"
                f"Sample rows (first 3): {head_sample}\n"
            )

        effective_prompt = (
            f"{csv_context}\nUser prompt:\n{prompt or 'Give me a summary of the dataset.'}"
        )

        response = model.generate_content(
            effective_prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        )
        st.success(response.text)
    except Exception as e:
        st.error(f"Gemini error: {e}")

# --- Optional: download back the ingested CSV (e.g., after light cleaning) ---
if st.session_state.df is not None:
    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download current CSV", data=csv_bytes, file_name="dataset.csv", mime="text/csv")
