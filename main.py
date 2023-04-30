from flask import Flask, render_template, request
import numpy as np
import pickle
import pyttsx3


# importing model
model = pickle.load(open('model.pkl', 'rb'))

# creating flask app
app = Flask(__name__)


def speak(audio):
    """Text-to-speech conversion using pyttsx3"""
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-20)
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    NITROGEN = int(request.form['Nitrogen'])
    PHOSPHORUS = int(request.form['Phosporus'])
    POTASSIUM = int(request.form['Potassium'])
    TEMPERATURE = float(request.form['Temperature'])
    HUMIDITY = float(request.form['Humidity'])
    PH = float(request.form['pH'])
    RAINFALL = float(request.form['Rainfall'])

    feature_list = [NITROGEN, PHOSPHORUS, POTASSIUM,
                    TEMPERATURE, HUMIDITY, PH, RAINFALL]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango",
                 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        print(crop)
        result = "{} is a best crop to be cultivated".format(crop)
    else:
        result = "Sorry are not able to recommend a proper crop for this environment"
    speak(
        f"According to the data you provided, the best crop to grow is {result}")
    return render_template('index.html', prediction=result)


# python main
if __name__ == '__main__':
    app.run(debug=True)
