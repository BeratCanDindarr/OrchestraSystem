from orchestra.tui.app_v2 import OrchestraAppV2
import asyncio

async def test():
    app = OrchestraAppV2()
    try:
        async with app.run_test():
            print("Booted OK")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test())
