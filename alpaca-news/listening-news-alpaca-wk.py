import asyncio
import os
import json
from dotenv import load_dotenv
import websockets
from collections import deque

# Load API keys from .env file
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

NEWS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

async def fetch_news():
    headlines = deque(maxlen=15)
    async with websockets.connect(NEWS_URL) as ws:
        # Authenticate
        await ws.send(json.dumps({
            "action": "auth",
            "key": API_KEY,
            "secret": API_SECRET
        }))
        # Wait for authentication response
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if isinstance(data, list) and data and data[0].get("msg") == "authenticated":
                break

        # Subscribe to all news
        await ws.send(json.dumps({
            "action": "subscribe",
            "news": ["*"]
        }))

        print("Connected and subscribed. Listening for news headlines...\n")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            # News messages are lists of dicts with "T": "n"
            for item in data:
                if item.get("T") == "n":
                    headlines.appendleft(item.get("headline"))
                    print(f"NEW: {item.get('headline')}")
                    print("\nLast 15 headlines:")
                    for i, h in enumerate(headlines, 1):
                        print(f"{i}. {h}")
                    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(fetch_news())