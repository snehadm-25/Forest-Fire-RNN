# Forest Fire RNN Classifier

An end-to-end Recurrent Neural Network (RNN) project to classify forest fire occurrence using meteorological data from the UCI Forest Fires dataset.

## Project Structure
- `preprocess.py`: Loads the raw CSV, performs One-Hot encoding, scales features, and prepares data for the RNN.
- `train.py`: Trains an optimized Bidirectional LSTM model with learning rate scheduling and early stopping.
- `predict.py`: Provides a simple interface to classify new weather data snapshots.
- `data/`: Contains preprocessed numpy arrays, the scaler, and feature names.
- `models/`: Stores the trained `.keras` model.
- `results/`: Contains performance visualizations (Accuracy, Confusion Matrix).

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data:
   ```bash
   python preprocess.py
   ```
3. Train model:
   ```bash
   python train.py
   ```
4. Predict via script:
   ```bash
   python predict.py
   ```
5. Use Web Dashboard:
   ```bash
   python app.py
   ```
   Then visit `http://127.0.0.1:5001` in your browser.

## Model Highlights
- **Architecture**: Bidirectional LSTM (128 -> 64) with Dense heads.
- **Regularization**: Dropout (0.3) and Batch Normalization.
- **Optimization**: Adam optimizer with `ReduceLROnPlateau`.
