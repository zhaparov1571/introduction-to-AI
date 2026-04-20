import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        model, features = data['model'], data['features']
    df = pd.read_csv('CarPrice_Assignment.csv')
except FileNotFoundError:
    st.error("Ошибка: Файлы не найдены!")
    st.stop()

st.title("Car Price Prediction Website")

st.header("Graph from Stage 1")
fig, ax = plt.subplots()
sns.regplot(x=df['enginesize'], y=df['price'], ax=ax)
st.pyplot(fig)

st.header("Model Accuracy")
y_pred = model.predict(df[features])
acc = r2_score(df['price'], y_pred)
st.write(f"R2 Score: {acc:.4f}")

st.header("10 Samples from Dataset")
st.dataframe(df[features + ['price']].head(10))

st.header("Predict Price")
user_inputs = {}
for f in features:
    user_inputs[f] = st.number_input(f"Enter {f}", value=float(df[f].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])
    res = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${res:.2f}")
    
