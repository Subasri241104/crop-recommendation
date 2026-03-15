from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = model.predict(data)

        crop_dict = {
            0: 'apple',
            1: 'banana',
            2: 'blackgram',
            3: 'chickpea',
            4: 'coconut',
            5: 'coffee',
            6: 'cotton',
            7: 'grapes',
            8: 'jute',
            9: 'kidneybeans',
            10: 'lentil',
            11: 'maize',
            12: 'mango',
            13: 'mothbeans',
            14: 'mungbean',
            15: 'muskmelon',
            16: 'orange',
            17: 'papaya',
            18: 'pigeonpeas',
            19: 'pomegranate',
            20: 'rice',
            21: 'watermelon'
        }

        crop_name = crop_dict[int(prediction[0])]
        return render_template("index.html", prediction_text=f"Recommended Crop: {crop_name}")

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", prediction_text="Error occurred")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)