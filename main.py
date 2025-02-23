from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("medicine_recommendation_model.pkl")

# Home route (renders the frontend)
@app.route("/")
def home():
    return render_template("index.html")

# API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form["disease"]
    prediction = model.predict([data])[0]
    return jsonify({"tablet": prediction})

if __name__ == "__main__":
    app.run(debug=True)
