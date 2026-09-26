#!/usr/bin/env python3
"""
IDR Mobile HTTPS Server
Serves the web apps over HTTPS so mobile Chrome / Safari can access DeviceMotionEvent
(gyroscope/accelerometer) and Geolocation APIs:
  /nav/          IDR Navigator PWA (live navigation + IO-VNBD replay)
  /calibration/  pre-drive stand calibrator
  GET  /api/calibration       current calibration.json
  POST /api/save_calibration  store calibration.json
"""

import argparse
import datetime
import http.server
import ipaddress
import json
import math
import os
import socket
import ssl
import sys
import threading
from pathlib import Path

# Paths
WEB_DIR = Path(__file__).resolve().parent
CERTS_DIR = WEB_DIR / ".certs"
PROJECT_ROOT = WEB_DIR.parent.parent


def get_lan_ip() -> str:
    """Detects local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable, just triggers routing resolution
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_self_signed_cert(cert_path: Path, key_path: Path, ip_addr: str):
    """
    Generates a 2048-bit RSA self-signed SSL certificate with SAN
    for localhost, 127.0.0.1, and the detected LAN IP.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    print(f"[*] Generating self-signed SSL certificate for {ip_addr}...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IDR Navigation"),
        x509.NameAttribute(NameOID.COMMON_NAME, ip_addr),
    ])

    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]

    try:
        if ip_addr not in ("127.0.0.1", "localhost"):
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(ip_addr)))
    except Exception:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[+] SSL Certificate created at {cert_path}")


MAX_BODY_BYTES = 64 * 1024
SERVED_SUFFIXES = {".html", ".js", ".mjs", ".css", ".json", ".geojson", ".onnx", ".wasm", ".svg", ".png", ".ico",
                   ".webmanifest", ".txt", ".md"}


def validate_calibration(data):
    """
    Checks an uploaded calibration before it is written to calibration.json (which the app and the
    Python pipeline load by default). Raises ValueError with a reason if it is malformed.
    """
    def number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))

    if not isinstance(data, dict):
        raise ValueError("calibration must be a JSON object")
    if "is_calibrated" in data and not isinstance(data["is_calibrated"], bool):
        raise ValueError("is_calibrated must be true or false")
    g = data.get("resting_gravity_vector")
    if g is not None:
        vec = [g.get(k) for k in ("x", "y", "z")] if isinstance(g, dict) else None
        if not vec or not all(number(v) for v in vec):
            raise ValueError("resting_gravity_vector needs finite x, y, z")
        if not 5.0 <= math.sqrt(sum(v * v for v in vec)) <= 15.0:
            raise ValueError("resting_gravity_vector magnitude must be 5-15 m/s^2")
    b = data.get("gyro_bias_rad_s")
    if b is not None:
        vals = [b.get(k) for k in ("gx", "gy", "gz")] if isinstance(b, dict) else None
        if not vals or not all(number(v) and abs(v) < 0.5 for v in vals):
            raise ValueError("gyro_bias_rad_s needs finite gx, gy, gz below 0.5 rad/s")
    return data


class CalibrationRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serves the web apps from src/web/ and handles calibration download/upload.

    Hardening: only files inside src/web/ with web-asset suffixes are served (no directory listings,
    no .py/.pem, no symlink escapes, never the TLS key directory); the calibration upload accepts only
    same-origin JSON of bounded size that passes validate_calibration(). There is no CORS: the apps are
    served from this same origin. Note: any client on the LAN that can reach the port can still POST
    (there is no login), so run the server only on trusted networks.
    """

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
                      ".geojson": "application/geo+json", ".json": "application/json"}
    timeout = 30  # seconds per connection, so a stalled client cannot hold a worker forever

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/nav/")
            self.end_headers()
            return
        if self.path == "/api/calibration":
            target = PROJECT_ROOT / "calibration.json"
            body = target.read_bytes() if target.exists() else b'{"is_calibrated": false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    def translate_path(self, path):
        """Maps a URL to a file inside src/web/ only; anything else resolves to a non-existent path (404)."""
        forbidden = str(WEB_DIR / "__forbidden__")
        translated = super().translate_path(path)
        if "\x00" in translated:                              # NUL byte: newer Pythons no longer raise on it in resolve()
            return forbidden
        try:
            resolved = Path(translated).resolve()
        except (ValueError, OSError):
            return forbidden
        web_root, certs = WEB_DIR.resolve(), CERTS_DIR.resolve()
        if resolved != web_root and web_root not in resolved.parents:
            return forbidden                                  # outside the web root (e.g. via a symlink)
        if resolved == certs or certs in resolved.parents:
            return forbidden                                  # TLS key/cert, however the URL is spelled
        if resolved.is_file() and resolved.suffix.lower() not in SERVED_SUFFIXES:
            return forbidden                                  # server sources, keys, other non-assets
        return str(resolved)

    def do_POST(self):
        """Saves calibration JSON sent by the calibrator page on this same origin."""
        if self.path != "/api/save_calibration":
            self.send_error(404, "Not found")
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin is not None and origin != f"https://{host}":
            return self._json(403, {"status": "error", "error": "cross-origin request refused"})
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return self._json(415, {"status": "error", "error": "Content-Type must be application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY_BYTES:
            return self._json(413, {"status": "error", "error": f"body must be 1-{MAX_BODY_BYTES} bytes"})
        def reject_constant(name):
            raise ValueError(f"{name} is not allowed")

        try:
            data = validate_calibration(json.loads(self.rfile.read(length).decode("utf-8"),
                                                   parse_constant=reject_constant))
        except (ValueError, UnicodeDecodeError, RecursionError, OverflowError, TypeError) as e:
            return self._json(400, {"status": "error", "error": str(e)[:200]})
        target_file = PROJECT_ROOT / "calibration.json"
        tmp = target_file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:                               # atomic replace: readers never see a partial file
            json.dump(data, f, indent=2, allow_nan=False)
        os.replace(tmp, target_file)
        print(f"\n[+] Calibration successfully saved to {target_file}")
        print(f"    Gyro Bias: {data.get('gyro_bias_rad_s')}")
        print(f"    Resting Gravity: {data.get('resting_gravity_vector')}")
        self._json(200, {"status": "ok", "message": "Calibration saved to calibration.json"})

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def print_banner(ip_addr: str, port: int):
    """Prints clear connection instructions."""
    url = f"https://{ip_addr}:{port}"
    local_url = f"https://localhost:{port}"

    print("=" * 66)
    print("      IDR NAVIGATOR & CALIBRATION SERVER (HTTPS)")
    print("=" * 66)
    print(f"  Navigator (mobile): \033[1;32m{url}/nav/\033[0m")
    print(f"  Calibrator        : \033[1;32m{url}/calibration/\033[0m")
    print(f"  Local desktop     : \033[1;36m{local_url}/nav/\033[0m")
    print("-" * 66)
    print("  📱 HOW TO CONNECT FROM YOUR SMARTPHONE:")
    print(f"  1. Ensure your phone is connected to the same Wi-Fi network.")
    print(f"  2. Open Chrome or Safari on your phone and navigate to:")
    print(f"     👉  \033[1;32m{url}/calibration/\033[0m  (then {url}/nav/)")
    print(f"  3. You will see a self-signed SSL warning ('Your connection is not private').")
    print(f"     Tap 'Advanced' -> 'Proceed to {ip_addr} (unsafe)'.")
    print(f"  4. Tap 'Enable Sensors' and begin testing:")
    print(f"     - Test 1 (Wobble/Tilt Immunity): Wobble phone, watch True Yaw stay flat.")
    print(f"     - Test 2 (Gravity Removal): Total |a| ~9.81, Horiz |a| ~0.")
    print(f"     - Test 3 (Bias Calibration): Keep stationary for 3s, tap 'Calibrate'.")
    print("=" * 66)
    print("  Press Ctrl+C to stop the server.\n")


def run_server(port: int = 8443, ip_addr: str = None):
    if not ip_addr:
        ip_addr = get_lan_ip()

    cert_path = CERTS_DIR / "cert.pem"
    key_path = CERTS_DIR / "key.pem"

    if not cert_path.exists() or not key_path.exists():
        generate_self_signed_cert(cert_path, key_path, ip_addr)

    # Setup SSL Context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    server_address = ("0.0.0.0", port)
    httpd = http.server.ThreadingHTTPServer(server_address, CalibrationRequestHandler)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print_banner(ip_addr, port)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down calibration server...")
    finally:
        httpd.server_close()
        print("[+] Server stopped.")


if __name__ == "__main__":
    # The banner uses emoji; on Windows, redirected output defaults to cp1252 and would crash on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="IDR Orientation Calibration HTTPS Server")
    parser.add_argument("--port", type=int, default=8443, help="Port to listen on (default: 8443)")
    parser.add_argument("--ip", type=str, default=None, help="LAN IP address to bind/display (auto-detected if omitted)")
    args = parser.parse_args()

    run_server(port=args.port, ip_addr=args.ip)
