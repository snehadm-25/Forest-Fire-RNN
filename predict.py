import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

def predict_fire(input_data):
    """
    input_data: list or dict containing 12 features:
    X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain
    """
    # 1. Load resources
    model = tf.keras.models.load_model('models/forest_fire_rnn_optimized.keras')
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # 2. Conversion to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 3. One-Hot Encoding (matching training)
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
    result = "FIRE" if prediction[0][0] > 0.5 else "NO FIRE"
    confidence = prediction[0][0] if result == "FIRE" else 1 - prediction[0][0]
    
    return result, confidence

if __name__ == "__main__":
    print("--- Forest Fire AI Classifier: Interactive Terminal ---")
    print("Please enter the following meteorological values:")
    
    try:
        user_input = {
            'X': int(input("X Coordinate (1-9) [e.g. 7]: ") or 7),
            'Y': int(input("Y Coordinate (2-9) [e.g. 5]: ") or 5),
            'month': input("Month (jan-dec) [e.g. aug]: ") or 'aug',
            'day': input("Day (mon-sun) [e.g. fri]: ") or 'fri',
            'FFMC': float(input("FFMC Index [e.g. 91.0]: ") or 91.0),
            'DMC': float(input("DMC Index [e.g. 166.9]: ") or 166.9),
            'DC': float(input("DC Index [e.g. 752.6]: ") or 752.6),
            'ISI': float(input("ISI Index [e.g. 7.1]: ") or 7.1),
            'temp': float(input("Temperature (°C) [e.g. 25.0]: ") or 25.0),
            'RH': int(input("Relative Humidity (%) [e.g. 40]: ") or 40),
            'wind': float(input("Wind Speed (km/h) [e.g. 4.0]: ") or 4.0),
            'rain': float(input("Rain (mm/m2) [0.0]: ") or 0.0)
        }
        
        print("\nAnalyzing conditions...")
        result, conf = predict_fire(user_input)
        
        color_start = "\033[91m" if result == "FIRE" else "\033[92m"
        color_end = "\033[0m"
        
        print(f"\nPrediction: {color_start}{result}{color_end}")
        print(f"Confidence: {conf*100:.2f}%")
        
    except ValueError as e:
        print(f"Error: Invalid input. Please enter numerical values where required.")
