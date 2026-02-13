import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# 1. Load data
df = pd.read_csv('forestfires.csv')
print(f"Dataset shape: {df.shape}")

# 2. Prepare target for classification
df['target'] = (df['area'] > 0).astype(int)
print(f"Target distribution:\n{df['target'].value_counts()}")

# 3. One-Hot Encoding (month, day)
df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

# 4. Feature Selection
# Drop 'area' since it's the source of 'target'
features = df.drop(['area', 'target'], axis=1)
target = df['target']

# 5. Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42, stratify=target)

# 7. Reshape for RNN (Samples, Timesteps, Features)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"X_train_rnn shape: {X_train_rnn.shape}")
print(f"X_test_rnn shape: {X_test_rnn.shape}")

# 8. Save preprocessed data
np.save('data/X_train.npy', X_train_rnn)
np.save('data/X_test.npy', X_test_rnn)
np.save('data/y_train.npy', y_train.values)
np.save('data/y_test.npy', y_test.values)

# Save scaler and feature names
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(features.columns.tolist(), f)

print("Preprocessing complete (One-Hot Encoded). Data saved in 'data/' directory.")
