import requests
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "input": "input",
        "X": X_test[0:1].astype(np.float32).tolist(),
    },
).content.decode()

print(json.loads(response))
