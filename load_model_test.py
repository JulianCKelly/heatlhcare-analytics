import joblib

model = joblib.load("models/random_forest_diabetes.pkl")
print(model)