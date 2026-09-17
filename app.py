import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# DOWNLOAD MODEL FROM DRIVE - 
# ---------------------------------------------------------
folder_id = "1ZzKIKB9CsXypEvlnlOz5HJ8qJAeVejb3"
folder_link = f"https://drive.google.com/drive/folders/{folder_id}"

if not os.path.exists("model"):
    with st.spinner("Pehli baar models download ho rahe hai, 1-2 min lagega..."):
        gdown.download_folder(folder_link, quiet=False, use_cookies=False)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_toxicity_model():
    model = load_model("model/toxicity_model.keras")
    with open("model/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    with open("model/config.pkl", "rb") as file:
        config = pickle.load(file)
    return model, tokenizer, config

# ---------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------
try:
    model, tokenizer, config = load_toxicity_model()
    MAX_LEN = config["max_len"]
    THRESHOLD = config["threshold"]
    LABELS = config["labels"]
except Exception as e:
    st.error("Model files could not be loaded. Check Drive folder structure.")
    st.write("Folder me ye files hai:", os.listdir("model") if os.path.exists("model") else "model folder nahi mila")
    st.code(str(e))
    st.stop()

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_comment(comment):
    cleaned = clean_text(comment)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    return probabilities

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🛡️ Comment Toxicity Detection")
st.write("This application uses a Deep Learning model to detect toxic comments.")
st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose an option:", ["Single Comment", "Bulk CSV Prediction", "About Project"])

# =========================================================
# SINGLE COMMENT
# =========================================================
if page == "Single Comment":
    st.header("🔍 Analyze a Comment")
    comment = st.text_area("Enter your comment:", height=150, placeholder="Type a comment here...")
    if st.button("Analyze Comment", type="primary"):
        if not comment.strip():
            st.warning("Please enter a comment.")
        else:
            probabilities = predict_comment(comment)
            results = pd.DataFrame({"Category": LABELS, "Probability": probabilities})
            results["Percentage"] = results["Probability"] * 100
            results["Prediction"] = np.where(results["Probability"] >= THRESHOLD, "Toxic", "Not Toxic")
            toxic_found = (probabilities >= THRESHOLD).any()
            if toxic_found:
                st.error("⚠️ This comment may contain toxic content.")
            else:
                st.success("✅ This comment appears to be non-toxic.")
            st.subheader("Toxicity Probabilities")
            for i, label in enumerate(LABELS):
                probability = probabilities[i]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(label.replace("_", " ").title())
                    st.progress(float(probability))
                with col2:
                    st.write(f"{probability * 100:.2f}%")
            st.subheader("Detailed Results")
            st.dataframe(results, use_container_width=True)

# =========================================================
# BULK CSV PREDICTION
# =========================================================
elif page == "Bulk CSV Prediction":
    st.header("📂 Bulk CSV Prediction")
    st.write("Upload a CSV file containing a `comment_text` column.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(uploaded_df.head(), use_container_width=True)
            if "comment_text" not in uploaded_df.columns:
                st.error("CSV must contain a 'comment_text' column.")
            else:
                if st.button("Predict All Comments", type="primary"):
                    texts = uploaded_df["comment_text"].astype(str)
                    cleaned_texts = texts.apply(clean_text)
                    sequences = tokenizer.texts_to_sequences(cleaned_texts)
                    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
                    probabilities = model.predict(padded_sequences, batch_size=256, verbose=0)
                    result_df = uploaded_df.copy()
                    for i, label in enumerate(LABELS):
                        result_df[f"{label}_probability"] = probabilities[:, i]
                        result_df[label] = (probabilities[:, i] >= THRESHOLD).astype(int)
                    st.success("Predictions completed successfully!")
                    st.dataframe(result_df, use_container_width=True)
                    csv_data = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(label="⬇️ Download Predictions CSV", data=csv_data, file_name="toxicity_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Error processing CSV.")
            st.code(str(e))

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":
    st.header("📊 About the Project")
    st.write("""
        ### Deep Learning for Comment Toxicity Detection
        This project uses Natural Language Processing and Deep Learning to identify toxic comments.
        The model predicts six categories:
        - Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate
        ### Technologies Used
        - Python, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, NLP, LSTM, Streamlit
        ### Project Features
        ✅ Single comment prediction ✅ Toxicity probability ✅ Multiple toxicity categories ✅ CSV bulk prediction ✅ Download prediction results ✅ Interactive Streamlit interface
        """)
    st.info("This application is designed as a machine-learning content moderation prototype.")
