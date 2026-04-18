import asyncio
import httpx
import time

async def check_route(client, route_url, name):
    start = time.time()
    try:
        resp = await client.get(route_url)
        elapsed = time.time() - start
        print(f"✅ {name}: {resp.status_code} (took {elapsed:.3f}s)")
        if resp.status_code != 200:
            print(f"   ↳ Output: {resp.text[:100]}")
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")

async def main():
    print("--- Testing Port 4500 ---")
    base = "http://127.0.0.1:4500/api"
    async with httpx.AsyncClient() as client:
        await check_route(client, f"{base}/health", "Health Check")
    
    print("\n--- Testing Port 8000 ---")
    base = "http://127.0.0.1:8000/api"
    async with httpx.AsyncClient() as client:
        await check_route(client, f"{base}/health", "Health Check")

asyncio.run(main())
