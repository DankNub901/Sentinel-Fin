import os
import redis.asyncio as aioredis
from typing import Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Global async client instance
redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(
            REDIS_URL,
            decode_responses=True, # Automatically parses bytes to UTF-8 strings
            max_connections=20
        )
    return redis_client


async def check_redis_health() -> bool:
    try:
        client = await get_redis()
        return await client.ping()
    except Exception as e:
        print(f"⚠️ Redis Connection Failed: {e}")
        return False


async def close_redis() -> None:
    """Gracefully close the global Redis connection pool on app shutdown."""
    global redis_client
    if redis_client is not None:
        await redis_client.aclose()  # Closes connection pool asynchronously
        redis_client = None
        print("✅ Redis connection pool closed successfully.")