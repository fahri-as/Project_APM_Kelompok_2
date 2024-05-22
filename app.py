from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load Model and Vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    text_vector = vectorizer.transform([text])
    prediction = svm_model.predict(text_vector)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
