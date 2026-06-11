import redis

r = redis.Redis(
    "localhost",
    port=6379,
    decode_responses=True,
)
