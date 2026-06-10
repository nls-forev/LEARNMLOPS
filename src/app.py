import onnxruntime as ort
import json
from fastapi import FastAPI, Request


app = FastAPI()

# Single-threaded session options
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1  # one thread for parallelism inside operators
sess_opts.inter_op_num_threads = 1  # one thread for parallelism across operators
sess_opts.execution_mode = (
    ort.ExecutionMode.ORT_SEQUENTIAL
)  # disable internal parallelism

session = ort.InferenceSession("output/model.onnx", sess_options=sess_opts)


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
