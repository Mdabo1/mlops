from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd

X_test = pd.read_csv("test\X_test_processed.csv", index_col=0)
y_test = pd.read_csv("test\y_test_enc.csv", index_col=0)

model = joblib.load("linear_regression_model.pkl")
y_pred = model.predict(X_test)

loss = mean_squared_error(y_test, y_pred)

print(loss)
