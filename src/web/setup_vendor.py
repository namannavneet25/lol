"""Downloads the pinned onnxruntime-web build into src/web/vendor/ so the PWA runs fully offline."""
import os
import urllib.request

VERSION = "1.23.2"
FILES = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "onnxruntime-web")

if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    for name in FILES:
        path = os.path.join(DEST, name)
        if os.path.exists(path):
            print(f"exists: {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, path)
    print(f"onnxruntime-web {VERSION} ready in {DEST}")
