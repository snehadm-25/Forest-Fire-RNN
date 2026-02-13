import gradio as gr
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# Load model and resources
MODEL_PATH = 'models/forest_fire_rnn_optimized.keras'
SCALER_PATH = 'data/scaler.pkl'
FEATURES_PATH = 'data/feature_names.pkl'

# Load the model, scaler and feature names
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)

def predict_fire(x, y, month, day, ffmc, dmc, dc, isi, temp, rh, wind, rain):
    """
    Function to predict forest fire classification.
    """
    # 1. Prepare input dictionary
    input_data = {
        'X': x, 'Y': y, 'month': month, 'day': day,
        'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi,
        'temp': temp, 'RH': rh, 'wind': wind, 'rain': rain
    }
    
    # 2. Conversion to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 3. One-Hot Encoding
    df_encoded = pd.get_dummies(df_input, columns=['month', 'day'])
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reorder columns to match scaler
    df_encoded = df_encoded[feature_names]
    
    # 4. Scaling
    input_scaled = scaler.transform(df_encoded)
    
    # 5. Reshape for RNN
    input_rnn = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
    
    # 6. Predict
    prediction = model.predict(input_rnn, verbose=0)
    probability = float(prediction[0][0])
    result = "FIRE" if probability > 0.5 else "NO FIRE"
    confidence = probability if result == "FIRE" else 1 - probability
    
    return f"{result} ({confidence*100:.2f}% Confidence)"

# Define Gradio Interface
iface = gr.Interface(
    fn=predict_fire,
    inputs=[
        gr.Number(label="X Coordinate (1-9)", value=7),
        gr.Number(label="Y Coordinate (2-9)", value=5),
        gr.Dropdown(label="Month", choices=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], value="aug"),
        gr.Dropdown(label="Day", choices=["mon", "tue", "wed", "thu", "fri", "sat", "sun"], value="fri"),
        gr.Number(label="FFMC Index", value=91.0),
        gr.Number(label="DMC Index", value=166.9),
        gr.Number(label="DC Index", value=752.6),
        gr.Number(label="ISI Index", value=7.1),
        gr.Number(label="Temperature (°C)", value=25.9),
        gr.Number(label="Relative Humidity (%)", value=41),
        gr.Number(label="Wind Speed (km/h)", value=3.6),
        gr.Number(label="Rain (mm/m2)", value=0.0)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🔥 Forest Fire AI Classifier",
    description="Predict wildfire risks using meteorological data and Bidirectional LSTMs. [Dataset: UCI Forest Fires]",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()
