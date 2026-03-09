import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("shot_classifier.pkl")

initial_type = [('float_input', FloatTensorType([None, 5]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("shot_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model exported to shot_classifier.onnx")