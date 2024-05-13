from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

X_train = pd.read_csv("X_train.csv", index_col=0)
X_test = pd.read_csv("X_test.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
y_test = pd.read_csv("y_test.csv", index_col=0)

columns_to_exclude = ["Name", "Address", "Email", "Phone Number"]
X_train = X_train.loc[:, ~X_train.columns.isin(columns_to_exclude)]
X_test = X_test.loc[:, ~X_test.columns.isin(columns_to_exclude)]


cat_columns = []
num_columns = []

for column_name in X_train.columns:
    if X_train[column_name].dtypes == object:
        cat_columns += [column_name]
    else:
        num_columns += [column_name]

print("categorical columns:\t ", cat_columns, "\n len = ", len(cat_columns))

print("numerical columns:\t ", num_columns, "\n len = ", len(num_columns))

scaler = StandardScaler()
scaler.fit(X_train[num_columns])
train_scaled = scaler.transform(X_train[num_columns])
test_scaled = scaler.transform(X_test[num_columns])

enc = OrdinalEncoder()
enc.fit(y_train)
train_enc = enc.transform(y_train)
test_enc = enc.transform(y_test)


X_train = pd.DataFrame(train_scaled, columns=num_columns)
X_test = pd.DataFrame(test_scaled, columns=num_columns)
y_train = pd.DataFrame(train_enc, columns=["target"])
y_test = pd.DataFrame(test_enc, columns=["target"])

X_train.to_csv("X_train_processed.csv")
X_test.to_csv("X_test_processed.csv")
y_train.to_csv("y_train_enc.csv")
y_test.to_csv("y_test_enc.csv")
