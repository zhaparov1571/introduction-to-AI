import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle


df = pd.read_csv('CarPrice_Assignment.csv')
df['fueltype'] = df['fueltype'].map({'gas': 0, 'diesel': 1})
df['aspiration'] = df['aspiration'].map({'std': 0, 'turbo': 1})

features = ['enginesize', 'horsepower', 'curbweight', 'carwidth', 'highwaympg']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

print(f"Точность: {r2_score(y_test, y_pred):.4f}")
print()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.savefig("plot.png")
print("График сохранен как plot.png")
plt.show()  
