import mlflow
import numpy as np
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/wine-classifier/1")
sample = np.array([[
    13.2, 1.78, 2.14, 11.2, 100,
    2.65, 2.76, 0.26, 1.28, 4.38,
    1.05, 3.40, 1050
]])
prediction = model.predict(sample)
print("Wine classification prediction:", prediction)