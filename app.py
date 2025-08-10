import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model using pickle
with open('models/voting_classifier.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

