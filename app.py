import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('regression.pkl', 'rb'))
scalar = pickle.load(open('scalar.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)
    return render_template('home.html', prediction_text="The house prediction is {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)




