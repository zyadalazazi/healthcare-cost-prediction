import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_pickle', 'rb'))

@app.route('/') 
def home():
	return render_template('webpage.html')

@app.route('/predict', methods = ['POST'])
def predict():

	age_value = request.form['age']
	bmi_value = request.form['bmi']
	children_number = request.form['children']
	smoker_value = request.form['smoker']
	query_df = pd.DataFrame([[age_value, bmi_value, children_number, smoker_value]])

	prediction = model.predict(query_df)
	output = round(prediction[0], 2)

	return render_template('webpage.html', prediction_text = 'Predicted Charge: {} USD'.format(output))

