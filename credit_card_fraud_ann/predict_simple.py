import pickle
import numpy as np
import tensorflow as tf

# Load Model + Scaler
model = tf.keras.models.load_model("outputs/fraud_ann_model_simple.h5")
with open("outputs/scaler_simple.pkl", "rb") as f:
    scaler = pickle.load(f)

# Interactive User Input
print("\n🔍 Fraud Detection Interactive Mode")
amount = float(input("Enter transaction amount: "))
hour = int(input("Enter transaction hour (0-23): "))

# Only 2 features
features = np.array([[amount, hour]])

# Scale + Predict
features_scaled = scaler.transform(features)
prob = model.predict(features_scaled)[0][0]

if prob > 0.7:
    print(f"\n⚠️ Fraudulent transaction detected! (Probability: {prob:.2f})")
else:
    print(f"\n✅ Legitimate transaction. (Probability: {prob:.2f})")

# =====================
# Quick Demo: Force-Test Examples
# =====================
test_cases = [
    ("Legitimate Example", [100, 14]),     # small daytime transaction
    ("Fraud Example", [200000, 2])        # huge night transaction
]

for label, vals in test_cases:
    features = np.array([vals])
    features_scaled = scaler.transform(features)
    prob = model.predict(features_scaled)[0][0]
    result = "Fraud" if prob > 0.7 else "Legit"
    print(f"\n📌 {label}: {result} (Probability: {prob:.2f})")
