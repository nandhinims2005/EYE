from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_FILENAME = "logistic_regression_model.joblib"

try:
    logistic_model = joblib.load(MODEL_FILENAME)
    print(f"Model '{MODEL_FILENAME}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: {MODEL_FILENAME} not found.")
    logistic_model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Diabetic Retinopathy Prediction API is running.",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if logistic_model is None:
        return jsonify({
            "error": "Model not loaded on server."
        }), 500

    if not request.is_json:
        return jsonify({
            "error": "Request must be in JSON format."
        }), 400

    data = request.get_json()

    if "features" not in data:
        return jsonify({
            "error": "Please provide features."
        }), 400

    try:
        features = np.array(
            data["features"],
            dtype=float
        ).reshape(1, -1)

        prediction = logistic_model.predict(features)
        prediction_proba = logistic_model.predict_proba(features)[0]

        class_mapping = {
            0: "DR",
            1: "No_DR"
        }

        predicted_class = int(prediction[0])

        predicted_label = class_mapping.get(
            predicted_class,
            str(predicted_class)
        )

        probabilities = {}

        for class_value, probability in zip(
            logistic_model.classes_,
            prediction_proba
        ):
            class_value = int(class_value)

            probabilities[
                class_mapping.get(
                    class_value,
                    str(class_value)
                )
            ] = float(probability)

        return jsonify({
            "prediction": predicted_label,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
