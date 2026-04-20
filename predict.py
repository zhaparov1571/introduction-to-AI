import pickle
import pandas as pd

try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        features = data['features']
except FileNotFoundError:
    print("Ошибка: Сначала запусти train.py!")
    exit()

def get_val(name):
    while True:
        res = input(f"Введите {name}: ")
        try:
            return float(res)
        except ValueError:
            print("Введите число!")

print("\n--- Расчет стоимости авто ---")
vals = [] 
for f in features:
    vals.append(get_val(f))

input_df = pd.DataFrame([vals], columns=features)
res = model.predict(input_df)[0]

print("-" * 25)

print(f"Цена: ${res:.2f}")
print("-" * 25)
