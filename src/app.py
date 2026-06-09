import onnxruntime as ort
import json
from fastapi import FastAPI, Request


app = FastAPI()
session = ort.InferenceSession("output/model.onnx")


@app.post("/predict")
async def predict(request: Request):
    try:
        # parse the json from the incoming payload
        data = await request.json()

        # Get input and its type
        X = data["X"]
        input_type = data["input"]

        # Run inference
        y_pred = session.run(None, {input_type: X})

        return {
            # returns the predicted class label
            "response": int(y_pred[0][0])  # pyright: ignore[reportIndexIssue]
        }

    except json.JSONDecodeError as e:
        return {
            "response": f"Internal Server error: 500, Exception: {e}",
        }

    except Exception as e:
        return {
            "response": f"Unexpected error: {e}",
        }
