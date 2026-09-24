from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "ML Model API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = data["features"]

    prediction = model.predict([features])[0]

    if prediction == 1:
        result = "Malignant"
    else:
        result = "Benign"

    return jsonify({
        "prediction": int(prediction),
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)