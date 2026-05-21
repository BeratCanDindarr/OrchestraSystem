from orchestra.service import run_task
import json

try:
    res = run_task(mode="dual", task="Say hello")
    print(json.dumps(res, indent=2))
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
