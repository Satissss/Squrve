import argparse
import json
import sys
import time
from urllib import request, error


def http_get(url: str, timeout: float = 10.0):
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = body
            return status, data
    except error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def http_post_json(url: str, payload: dict, timeout: float = 20.0):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = body
            return status, data
    except error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = body
        return e.code, data
    except Exception as e:
        return 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Simple test for Squrve backend API")
    parser.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", default="8080", help="API port (default: 8080)")
    parser.add_argument("--instance-id", default="dev_0", help="Optional instance_id to run a real task")
    parser.add_argument(
        "--task",
        action="append",
        default=["DINSQLGenerator"],
        help="Task id to include in task_lis (repeat for multiple). Example: --task SQLGenerator",
    )
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    print(f"[1/3] Checking health: {base}/healthz")
    status, data = http_get(f"{base}/healthz")
    print(f"  -> status={status}, body={data}")
    if status != 200:
        print("Health check failed.")
        sys.exit(1)

    print(f"[2/3] Negative test: POST /api/run with empty payload (expect 400)")
    status, data = http_post_json(f"{base}/api/run", {})
    print(f"  -> status={status}, body={data}")
    if status != 400:
        print("Negative test failed: expected HTTP 400 for invalid payload.")
        sys.exit(1)

    if args.instance_id and args.task:
        payload = {"instance_id": args.instance_id, "task_lis": args.task}
        print(f"[3/3] Positive test: POST /api/run with payload={payload}")
        started = time.perf_counter()
        status, data = http_post_json(f"{base}/api/run", payload)
        elapsed = time.perf_counter() - started
        print(f"  -> status={status}, elapsed={elapsed:.2f}s, body={data}")
        if status != 200:
            print("Positive test failed (non-200). Check instance_id and tasks are valid for your dataset.")
            sys.exit(1)
    else:
        print("[3/3] Skipped positive test (provide --instance-id and --task to enable).")

    print("All checks passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())


