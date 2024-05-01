from faker import Faker
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np


fake = Faker()

num_rows = 1000

data = {
    "Name": [fake.name() for _ in range(num_rows)],
    "Address": [fake.address() for _ in range(num_rows)],
    "Email": [fake.email() for _ in range(num_rows)],
    "Phone Number": [fake.phone_number() for _ in range(num_rows)],
    "Job": [fake.job() for _ in range(num_rows)],
    "Age": np.random.randint(18, 80, size=num_rows),
    "Income": [random.choice(["High", "Low"]) for _ in range(num_rows)],
}


df = pd.DataFrame(data)

X, y = df.loc[:, df.columns != "Income"], df.loc[:, df.columns == "Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.to_csv("X_train.csv")
y_train.to_csv("y_train.csv")
X_test.to_csv("X_test.csv")
y_test.to_csv("y_test.csv")

print('Data created')