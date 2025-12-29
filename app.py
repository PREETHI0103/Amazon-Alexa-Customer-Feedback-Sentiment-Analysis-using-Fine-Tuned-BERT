import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("PREETHI0103/customer-feedback-sentiment-analyser")
    model = BertForSequenceClassification.from_pretrained("PREETHI0103/customer-feedback-sentiment-analyser")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="AI Customer Sentiment Intelligence Platform", page_icon="📊", layout="centered")

st.markdown("""
<style>
/* Background */
.main {background-color: #0f172a; color: #e5e7eb; font-family: 'Segoe UI', sans-serif;}
/* Headings */
h1 {color:#38bdf8; text-align:center; font-size:2.5rem; font-weight:bold;}
h2 {color:#facc15; text-align:center; font-size:1.5rem;}
/* Subtitle */
.subtitle {text-align:center; color:#e5e7eb; font-size:1.1rem; margin-bottom:30px;}
/* Text area */
textarea {border-radius:12px; background-color:#1e293b; color:#e5e7eb; padding:10px; font-size:1rem;}
/* Buttons */
.stButton>button {background-color:#38bdf8; color:#0f172a; font-weight:bold; border-radius:12px; padding:10px 25px;}
.stButton>button:hover {background-color:#22d3ee; color:#0f172a;}
/* Instructions box */
.instruction-box {background-color:#1e293b; padding:15px; border-radius:12px; margin-bottom:20px; color:#e5e7eb;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 AI Customer Sentiment Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Turning Customer Feedback into Actionable Insights using AI</p>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div class='instruction-box'>
<b>Instructions:</b>
<ul>
<li>✍️ Enter a customer review or feedback in the text area below.</li>
<li>🔍 Click <b>Analyze Sentiment</b> to check whether the feedback is Positive or Negative.</li>
<li>💡 Ensure the feedback is at least a few words for accurate analysis.</li>
<li>😊 Positive feedback indicates satisfied customers; 😞 Negative feedback indicates areas needing improvement.</li>
</ul>
</div>
""", unsafe_allow_html=True)

review = st.text_area("✍️ Enter Customer Feedback", height=180, placeholder="Type a product review here...")

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()

        st.divider()

        if pred == 1:
            st.success("✅ Positive Sentiment Detected — Customer is Satisfied 😊")
        else:
            st.error("❌ Negative Sentiment Detected — Improvement Needed 😞")
