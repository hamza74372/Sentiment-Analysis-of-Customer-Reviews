# 🧠 Sentiment Analysis of Customer Reviews

This project performs **binary sentiment classification** (Positive vs. Negative) on customer reviews using **Natural Language Processing (NLP)** and **Logistic Regression**. It includes preprocessing, model training, evaluation, word cloud visualizations, and a simple interactive **Streamlit web app**.

---

## 💡 Objective

To analyze customer reviews and automatically determine whether a given review expresses a **positive** or **negative** sentiment.

---

## 📂 Project Structure

📁 sentiment-analysis-reviews/
├── app.py # Streamlit app to test custom reviews
├── sentiment_model.pkl # Trained logistic regression model
├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── sentiment_analysis.ipynb # Jupyter Notebook with EDA + training
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml

---

## 🚀 How to Run

### 🔧 1. Clone the Repository


git clone https://github.com/yourusername/sentiment-analysis-reviews.git
cd sentiment-analysis-reviews
📦 2. Install Dependencies

pip install -r requirements.txt
🧠 3. Run Streamlit App

streamlit run app.py
Then open the local or external URL displayed in the terminal.

🧠 Model & Pipeline
✔️ Data Processing
Lowercasing

Tokenization using RegexpTokenizer

Stopword removal (nltk)

TF-IDF Vectorization with Bigrams
TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))

✔️ Model
LogisticRegression(max_iter=1000)

✔️ Labeling Rule
Ratings 4 or 5 → Positive (1)

Ratings 1, 2, or 3 → Negative (0)

📊 Evaluation Metrics
Metric	Score (Example)
Accuracy	0.89
Precision	0.88
Recall	1.00
F1 Score	0.94

🔍 Visualizations
📊 Confusion Matrix to analyze predictions

☁️ Word Clouds for positive and negative reviews to visualize common keywords

✨ Example Predictions
python

"The product is amazing and works perfectly!" → 🟢 Positive
"Really disappointed, it stopped working after a week." → 🔴 Negative
"Great device for reading, love the screen quality!" → 🟢 Positive
🧪 Bonus Features
✅ WordClouds for quick visual summary of top terms

✅ Streamlit App for interactive sentiment prediction

✅ Model + TF-IDF saved with joblib for future reuse

📈 (Optional) Plot sentiment trend over time or by product

🔬 Future Improvements
Upgrade model to LSTM, BERT, or Transformers

Handle unseen vocabulary more robustly (use HashingVectorizer or deep embeddings)

Train on larger datasets like IMDb or Amazon Product Reviews

Integrate live product review scraping (e.g., from Amazon)

Add multilingual sentiment support using textblob or langdetect

📦 Dependencies
pandas, numpy

scikit-learn

nltk

matplotlib, seaborn

wordcloud

streamlit

joblib

Install them all using:


pip install -r requirements.txt
📄 License
This project is open-source and available under the MIT License.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

