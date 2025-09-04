import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

# =====================
# 1. Load Dataset
# =====================
data = pd.read_csv("data/creditcard_final.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# =====================
# 2. Balance Dataset (SMOTE)
# =====================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# =====================
# 3. Train-Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# =====================
# 4. Scale Features
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("outputs", exist_ok=True)
with open("outputs/scaler_simple.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =====================
# 5. Build ANN Model
# =====================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# =====================
# 6. Train Model
# =====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10, batch_size=32, verbose=1
)

# =====================
# 7. Evaluate Model
# =====================
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Save classification report
with open("outputs/metrics_report_simple.txt", "w") as f:
    f.write(report)

# Save trained model
model.save("outputs/fraud_ann_model_simple.h5")

# =====================
# 8. Plot Training Curves
# =====================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("outputs/training_curves_simple.png")
plt.close()
