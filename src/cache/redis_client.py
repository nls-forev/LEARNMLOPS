import redis

r = redis.Redis(
    "redis",
    port=6379,
    decode_responses=True,
)
