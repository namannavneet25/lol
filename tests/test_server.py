"""Security checks for src/web/server.py: served paths, calibration upload validation."""
import http.client
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.web.server as server


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)          # uploads go to a temp calibration.json
    httpd = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.CalibrationRequestHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()
    httpd.server_close()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


@pytest.mark.parametrize("path", ["/.certs/key.pem", "/nav/../.certs/key.pem", "/%2e%63erts/key.pem",
                                  "/server.py", "/export_replay.py", "/nav/js/", "/calibration/../../../etc/passwd",
                                  "/../../../etc/passwd", "/nav/index.html%00.pem", "/setup_vendor.py"])
def test_forbidden_paths(srv, path):
    port, _ = srv
    status, body = _req(port, "GET", path)
    assert status == 404, (path, status)
    assert b"PRIVATE KEY" not in body and b"import " not in body


def test_assets_are_served(srv):
    port, _ = srv
    assert _req(port, "GET", "/nav/index.html")[0] == 200
    assert _req(port, "GET", "/nav/js/ekf.js")[0] == 200


VALID = {"is_calibrated": True, "resting_gravity_vector": {"x": 0.1, "y": 0.2, "z": 9.8},
         "gyro_bias_rad_s": {"gx": 0.001, "gy": 0.0, "gz": -0.002}}


def _post(port, payload, origin=None, ctype="application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    headers = {"Content-Type": ctype, "Content-Length": str(len(body))}
    if origin:
        headers["Origin"] = origin
    return _req(port, "POST", "/api/save_calibration", body=body, headers=headers)[0]


def test_calibration_upload_accepts_same_origin_json(srv):
    port, root = srv
    assert _post(port, VALID, origin=f"https://127.0.0.1:{port}") == 200
    assert json.loads(Path(root, "calibration.json").read_text())["gyro_bias_rad_s"]["gz"] == -0.002


@pytest.mark.parametrize("payload,origin,ctype,code", [
    (VALID, "https://evil.example", "application/json", 403),                                    # cross-origin page
    (VALID, None, "text/plain", 415),                                                            # "simple" CORS request
    ({**VALID, "resting_gravity_vector": {"x": 0, "y": 0, "z": 0}}, None, "application/json", 400),   # NaN source
    ({**VALID, "gyro_bias_rad_s": {"gx": "a", "gy": 0, "gz": 0}}, None, "application/json", 400),
    (b"x" * (server.MAX_BODY_BYTES + 1), None, "application/json", 413),                         # oversized body
    (b"not json", None, "application/json", 400),
    (b'{"resting_gravity_vector": {"x": NaN, "y": 0, "z": 9.8}}', None, "application/json", 400),
    (b'{"is_calibrated": true, "offset": Infinity}', None, "application/json", 400),
    (b'{"gyro_bias_rad_s": {"gx": 1' + b"0" * 400 + b', "gy": 0, "gz": 0}}', None, "application/json", 400),
    (b"[" * 20000 + b"]" * 20000, None, "application/json", 400),                                # deep nesting
    ({**VALID, "resting_gravity_vector": {"x": True, "y": 0, "z": 9.8}}, None, "application/json", 400),
    ({**VALID, "is_calibrated": "no"}, None, "application/json", 400),
], ids=["cross-origin", "text-plain", "zero-gravity", "string-bias", "oversized", "not-json", "nan", "infinity",
        "huge-number", "deep-nesting", "bool-gravity", "string-flag"])   # short ids: Windows caps env vars at 32767 chars
def test_calibration_upload_rejects_bad_requests(srv, payload, origin, ctype, code):
    port, root = srv
    assert _post(port, payload, origin=origin, ctype=ctype) == code
    assert not Path(root, "calibration.json").exists()
