from flask import Flask, render_template, request
import numpy as np
import pickle


# importing model
model = pickle.load(open('model.pkl', 'rb'))

# creating flask app
app = Flask(__name__)


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

    crop_dict = {0: "Rice", 1: "Maize", 2: "Jute", 3: "Cotton", 4: "Coconut", 5: "Papaya", 6: "Orange", 7: "Apple", 8: "Muskmelon", 9: "Watermelon", 10: "Grapes", 11: "Mango",
                 12: "Banana", 13: "Pomegranate", 14: "Lentil", 15: "Blackgram", 16: "Mungbean", 17: "Mothbeans", 18: "Pigeonpeas", 19: "Kidneybeans", 20: "Chickpea", 21: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        print(crop)
        result = "{} is a best crop to be cultivated".format(crop)
    else:
        result = "Sorry are not able to recommend a proper crop for this environment"

    return render_template('index.html', prediction=result)


# python main
if __name__ == '__main__':
    app.run(debug=True)
