import onnxruntime as ort
import hashlib
import json
from fastapi import FastAPI, Request
from cache.redis_client import r

app = FastAPI()

# Single-threaded session options
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1  # one thread for parallelism inside operators
sess_opts.inter_op_num_threads = 1  # one thread for parallelism across operators
sess_opts.execution_mode = (
    ort.ExecutionMode.ORT_SEQUENTIAL
)  # disable internal parallelism

session = ort.InferenceSession("output/model.onnx", sess_options=sess_opts)


@app.post("/ttl")
async def check_ttl(request: Request):
    try:
        data = await request.json()
        X = data["X"]
        X_hash = hashlib.md5(str(X).encode()).hexdigest()

        if r.get(X_hash):
            return {"time_to_live": r.ttl(X_hash)}

        return {"response": "Key does not exist."}

    except json.JSONDecodeError as e:
        return {
            "response": f"Internal Server error: 500, Exception: {e}",
        }

    except Exception as e:
        return {
            "response": f"Unexpected error: {e}",
        }


@app.post("/predict")
async def predict(request: Request):
    try:
        # parse the json from the incoming payload
        data = await request.json()

        # Get input and its type
        X = data["X"]
        input_type = data["input"]

        X_hash = hashlib.md5(str(X).encode()).hexdigest()
        cached = r.get(X_hash)

        # get cached value if exists
        if cached:
            return {
                "response": int(cached),
                "source": "redis",
            }

        # Run inference
        output = session.run(None, {input_type: X})
        y_pred = int(output[0][0])  # pyright: ignore[reportIndexIssue]

        # set cache and expire after 1 hour.
        r.set(X_hash, y_pred, ex=3600)

        return {
            # returns the predicted class label
            "response": y_pred,
            "source": "onnx",
        }

    except json.JSONDecodeError as e:
        return {
            "response": f"Internal Server error: 500, Exception: {e}",
        }

    except Exception as e:
        return {
            "response": f"Unexpected error: {e}",
        }
