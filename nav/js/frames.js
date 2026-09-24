// Coordinate conventions - JS port of src/navigation/frames.py (keep in sync).
// ENU local tangent plane in metres; psi = heading CCW from East (rad); compass = deg clockwise from North.

const WGS84_A = 6378137.0;
const WGS84_E2 = 6.69437999014e-3;
const DEG = Math.PI / 180;

export function wrapPi(a) {
  const r = (a + Math.PI) % (2 * Math.PI);
  return (r < 0 ? r + 2 * Math.PI : r) - Math.PI;
}

export const compassToPsi = (courseDeg) => wrapPi(Math.PI / 2 - courseDeg * DEG);

export function psiToCompass(psi) {
  const d = (Math.PI / 2 - psi) / DEG;
  return ((d % 360) + 360) % 360;
}

function ecef(latDeg, lonDeg, h) {
  const lat = latDeg * DEG, lon = lonDeg * DEG;
  const sl = Math.sin(lat), cl = Math.cos(lat);
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sl * sl);
  return [(n + h) * cl * Math.cos(lon), (n + h) * cl * Math.sin(lon), (n * (1 - WGS84_E2) + h) * sl];
}

export class LocalTangentPlane {
  constructor(lat0, lon0, h0 = 0) {
    this.lat0 = lat0; this.lon0 = lon0; this.h0 = h0;
    const lat = lat0 * DEG, lon = lon0 * DEG;
    this.sl = Math.sin(lat); this.cl = Math.cos(lat);
    this.so = Math.sin(lon); this.co = Math.cos(lon);
    [this.x0, this.y0, this.z0] = ecef(lat0, lon0, h0);
  }

  toEnu(lat, lon) {
    const [x, y, z] = ecef(lat, lon, this.h0);
    const dx = x - this.x0, dy = y - this.y0, dz = z - this.z0;
    const e = -this.so * dx + this.co * dy;
    const n = -this.sl * this.co * dx - this.sl * this.so * dy + this.cl * dz;
    return [e, n];
  }

  toGeodetic(e, n) {
    const x = this.x0 - this.so * e - this.sl * this.co * n;
    const y = this.y0 + this.co * e - this.sl * this.so * n;
    const z = this.z0 + this.cl * n;
    const lon = Math.atan2(y, x);
    const p = Math.hypot(x, y);
    let lat = Math.atan2(z, p * (1 - WGS84_E2));
    for (let i = 0; i < 3; i++) {
      const nn = WGS84_A / Math.sqrt(1 - WGS84_E2 * Math.sin(lat) ** 2);
      const h = p / Math.cos(lat) - nn;
      lat = Math.atan2(z, p * (1 - WGS84_E2 * nn / (nn + h)));
    }
    return [lat / DEG, lon / DEG];
  }
}
