import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("advertising.csv")  # Make sure the file is in the same folder
df = df.drop(columns=["Unnamed: 0"])

# Step 2: Preview the data
print("🔹 First 5 rows of dataset:")
print(df.head())

# Step 3: Check for missing values
print("\n🔹 Missing values:")
print(df.isnull().sum())

# Step 4: Visualize relationships
sns.pairplot(df)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()
X = df[['TV', 'Radio', 'Newspaper']]  # Features (inputs)
y = df['Sales']                       # Target (output)

# Step 6: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict on test data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Step 10: Plot Actual vs Predicted Sales
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.tight_layout()
plt.show()
