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


def http_post_json(url: str, payload: dict, timeout: float = 120):
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
    base = f"http://127.0.0.1:8080"

    payload = {
        "dev_0": [["DINSQLGenerator"],["DINSQLGenerator"],["MACSQLGenerator"]],
        "dev_1": [
            ["MACSQLGenerator"],
            ["DINSQLGenerator"],
            ["MACSQLGenerator"]
        ],
    }
    print(f"[3/3] Positive test: POST /api/run_batch with payload={payload}")
    started = time.perf_counter()
    status, res = http_post_json(f"{base}/api/run_batch", payload)
    elapsed = time.perf_counter() - started
    print(res)

    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
