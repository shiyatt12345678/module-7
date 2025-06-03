import streamlit as st
import pandas as pd
import json
import re
import string
import nltk
import docx
import PyPDF2
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

@st.cache_resource
def load_model():
    data = []
    url = "https://www.dropbox.com/scl/fi/jpgt8itg3f98n7o237ltr/News_Category_Dataset_v3.json?rlkey=10mn49ajrmq4ek4vc24oi4l9d&st=nxtegwva&dl=1"
    response = requests.get(url)
    for line in response.text.strip().split('\n'):
        data.append(json.loads(line))

    df = pd.DataFrame(data)[['headline', 'category']]

    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    # Example improved TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8)

# Train on all categories (remove filtering)
    df['cleaned'] = df['headline'].apply(preprocess_text)
    X = df['cleaned']
    y = df['category']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)


    return model, vectorizer, preprocess_text

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        return None

st.title("📰 News Headline Classifier")

st.markdown("""
Upload a **Word (.docx)** or **PDF (.pdf)** file with headlines (one per line) **or** type/paste headlines manually below.
""")

uploaded_file = st.file_uploader("📁 Upload File", type=["pdf", "docx"])

model, vectorizer, preprocess_text = load_model()

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if raw_text:
        st.success("✅ File uploaded and text extracted successfully!")
        headlines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 5]
        st.subheader("📊 Predictions from File Upload")
        for headline in headlines:
            cleaned = preprocess_text(headline)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.markdown(f"**{headline}** → _{prediction}_")
    else:
        st.error("❌ Could not extract text. Make sure the file is a valid PDF or DOCX.")

st.subheader("Or type/paste headlines below (one per line):")
user_input = st.text_area("Enter headlines here...")

if st.button("Classify Headlines"):
    if not user_input.strip():
        st.warning("Please enter at least one headline.")
    else:
        input_headlines = [line.strip() for line in user_input.strip().split('\n') if len(line.strip()) > 5]
        st.subheader("📊 Predictions from Manual Input")
        for headline in input_headlines:
            cleaned = preprocess_text(headline)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.markdown(f"**{headline}** → _{prediction}_")
