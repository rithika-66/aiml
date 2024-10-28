# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the saved model
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]).reshape(1, -1)
    
    # Predict the class of the iris flower
    prediction = model.predict(features)
    
    # Mapping from model output to actual Iris species names
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    
    result = species[prediction[0]]
    return jsonify({'species': result})

if __name__ == '__main__':
    app.run(debug=True)
