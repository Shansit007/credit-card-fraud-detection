import pandas as pd

# Load original Kaggle dataset (you must have it as creditcard.csv inside data/)
data = pd.read_csv("data/creditcard.csv")

# Keep only useful columns for demo
final_data = data[["Amount", "Time", "Class"]].copy()

# Convert "Time" into transaction hour (0-23)
final_data["Hour"] = (final_data["Time"] // 3600) % 24
final_data = final_data.drop(columns=["Time"])

# Save cleaned dataset
final_data.to_csv("data/creditcard_final.csv", index=False)
print("✅ Final dataset created and saved as data/creditcard_final.csv")
