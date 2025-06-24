from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and feature order
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict')
def predict():
    return render_template("predict.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/submit', methods=["POST"])
def submit():
    # Get input values
    input_data = []
    for feature in feature_order:
        value = request.form.get(feature)
        if value is None:
            value = 0
        input_data.append(float(value))

    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]
    prediction = round(prediction, 4)

    return render_template("submit.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
