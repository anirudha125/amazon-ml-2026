"""
kaggle_exec.py -- run Python code on the Kaggle notebook server through the Jupyter proxy.

The proxy base URL (contains an auth token) is read from env KAGGLE_JUPYTER_BASE or from a file path in
env KAGGLE_JUPYTER_BASE_FILE; it is never stored in the repository.

Usage:
  python tools/kaggle_exec.py new                 -> prints a new kernel id (own kernel; user's kernel untouched)
  python tools/kaggle_exec.py run <kid> <file.py> -> executes the file's code in that kernel, streams output
  python tools/kaggle_exec.py code <kid> "<code>"
"""
import os, sys, json, uuid, time
import urllib.request
import websocket

def base():
    b = os.environ.get("KAGGLE_JUPYTER_BASE")
    if not b:
        b = open(os.environ["KAGGLE_JUPYTER_BASE_FILE"]).read().strip()
    return b.rstrip("/")

def http(method, path, body=None):
    req = urllib.request.Request(base() + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"null")

def new_kernel():
    return http("POST", "/api/kernels", {"name": "python3"})["id"]

def execute(kid, code, timeout=6 * 3600):
    url = base().replace("https://", "wss://").replace("http://", "ws://") + f"/api/kernels/{kid}/channels"
    ws = websocket.create_connection(url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": msg_id, "username": "claude", "session": uuid.uuid4().hex,
                                   "msg_type": "execute_request", "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "channel": "shell",
                        "content": {"code": code, "silent": False, "store_history": False,
                                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}}))
    status = "ok"
    while True:
        m = json.loads(ws.recv())
        if m.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        t = m["msg_type"]; c = m["content"]
        if t == "stream":
            sys.stdout.write(c["text"]); sys.stdout.flush()
        elif t in ("execute_result", "display_data"):
            print(c["data"].get("text/plain", ""))
        elif t == "error":
            status = "error"; print("\n".join(c["traceback"]))
        elif t == "status" and c["execution_state"] == "idle":
            break
    ws.close()
    return status

def upload(local, remote, chunk_mb=8):
    """Chunked upload via the Jupyter contents API (base64). remote is relative to the server root (e.g. kaggle/working/x)."""
    import base64
    size = os.path.getsize(local); n = 0; sent = 0
    with open(local, "rb") as f:
        while True:
            b = f.read(chunk_mb * 2 ** 20)
            last = sent + len(b) >= size
            n += 1
            body = {"type": "file", "format": "base64", "name": os.path.basename(remote), "path": remote,
                    "content": base64.b64encode(b).decode(), "chunk": -1 if last else n}
            http("PUT", "/api/contents/" + remote, body)
            sent += len(b)
            if last:
                break
    return sent


def download(remote, local):
    import base64
    r = http("GET", "/api/contents/" + remote + "?content=1&format=base64&type=file")
    data = base64.b64decode(r["content"])
    open(local, "wb").write(data)
    return len(data)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "new":
        print(new_kernel())
    elif cmd == "run":
        sys.exit(0 if execute(sys.argv[2], open(sys.argv[3], encoding="utf-8").read()) == "ok" else 1)
    elif cmd == "upload":
        t = time.time(); n = upload(sys.argv[2], sys.argv[3]); dt = time.time() - t
        print(f"uploaded {n/2**20:.1f} MB in {dt:.1f}s ({n/2**20/dt:.2f} MB/s)")
    elif cmd == "download":
        print(f"downloaded {download(sys.argv[2], sys.argv[3])/2**20:.1f} MB")
    elif cmd == "code":
        sys.exit(0 if execute(sys.argv[2], sys.argv[3]) == "ok" else 1)
