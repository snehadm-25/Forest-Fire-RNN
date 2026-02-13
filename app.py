from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and resources once at startup
MODEL_PATH = 'models/forest_fire_rnn_optimized.keras'
SCALER_PATH = 'data/scaler.pkl'
FEATURES_PATH = 'data/feature_names.pkl'

model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Conversion to DataFrame
        df_input = pd.DataFrame([data])
        
        # 2. One-Hot Encoding
        df_encoded = pd.get_dummies(df_input, columns=['month', 'day'])
        
        # Ensure all columns from training are present
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder columns to match scaler
        df_encoded = df_encoded[feature_names]
        
        # 3. Scaling
        input_scaled = scaler.transform(df_encoded)
        
        # 4. Reshape for RNN
        input_rnn = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
        
        # 5. Predict
        prediction = model.predict(input_rnn, verbose=0)
        probability = float(prediction[0][0])
        result = "FIRE" if probability > 0.5 else "NO FIRE"
        confidence = probability if result == "FIRE" else 1 - probability
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence * 100, 2),
            'probability': round(probability, 4)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("Starting Forest Fire Classifier Web App on port 5001...")
    app.run(debug=True, port=5001)
