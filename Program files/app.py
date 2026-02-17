from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():

    step = float(request.form['step'])
    type = float(request.form['type'])
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    isFlaggedFraud = float(request.form['isFlaggedFraud'])

    data = np.array([[step, type, amount,
                      oldbalanceOrg, newbalanceOrig,
                      oldbalanceDest, newbalanceDest,isFlaggedFraud]])

    prediction = model.predict(data)
    print(prediction)

    if prediction[0] == 1:
        result = "Fraud Transaction"
    else:
        result = "Not a Fraud Transaction"

    return render_template('submit.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
