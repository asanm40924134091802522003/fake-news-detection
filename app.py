
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained ML model
with open("news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load text vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def home():
    return "Fake News Detection Backend is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    news = data.get("news", "")

    if not news.strip():
        return jsonify({"error": "Please enter news"})

    # Convert news into numerical format
    news_vector = vectorizer.transform([news])

    # Predict news
    prediction = model.predict(news_vector)[0]

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
