import redis

r = redis.Redis(
    "localhost",
    port=8000,
    decode_responses=True,
)
