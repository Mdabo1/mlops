from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

X_train = pd.read_csv("train\X_train_processed.csv", index_col=0)
y_train = pd.read_csv("train\y_train_enc.csv", index_col=0)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, "linear_regression_model.pkl")
