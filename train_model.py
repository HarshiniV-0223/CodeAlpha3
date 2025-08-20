import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
df = pd.read_csv("Advertising.csv")  # Make sure this file is in your folder

# Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("sales_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'sales_model.pkl'")
