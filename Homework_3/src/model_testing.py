from fastapi import FastAPI
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd

# Load the test data and the model
X_test = pd.read_csv("Homework_1\\test\\X_test_processed.csv", index_col=0)
y_test = pd.read_csv("Homework_1\\test\\y_test_enc.csv", index_col=0)

model = joblib.load("Homework_1\\linear_regression_model.pkl")
y_pred = model.predict(X_test)

loss = mean_squared_error(y_test, y_pred)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app"}

@app.get("/evaluate")
async def evaluate():
    return {"mean_squared_error": loss}

@app.post("/predict")
async def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

print(f"Initial evaluation MSE: {loss}")