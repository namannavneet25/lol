"""
Coordinate-frame conventions shared by every IDR module.

Navigation frame : local East-North-Up (ENU) tangent plane, metres.
Heading psi      : radians, counter-clockwise from East (the ENU yaw angle).
Compass course   : degrees, clockwise from North, in [0, 360) - what GNSS receivers,
                   OSM bearings and the NavigationState output use.

    psi = pi/2 - radians(course)        course = degrees(pi/2 - psi) mod 360
"""
import math

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def wrap_pi(angle):
    """Wraps an angle (scalar or array) to [-pi, pi)."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi if isinstance(angle, np.ndarray) \
        else (angle + math.pi) % (2.0 * math.pi) - math.pi


def compass_to_psi(course_deg):
    return wrap_pi(math.pi / 2.0 - np.radians(course_deg)) if isinstance(course_deg, np.ndarray) \
        else wrap_pi(math.pi / 2.0 - math.radians(course_deg))


def psi_to_compass(psi_rad):
    if isinstance(psi_rad, np.ndarray):
        return np.degrees(math.pi / 2.0 - psi_rad) % 360.0
    return math.degrees(math.pi / 2.0 - psi_rad) % 360.0


class LocalTangentPlane:
    """
    WGS84 geodetic <-> local ENU converter anchored at (lat0, lon0).
    Uses the exact ECEF rotation, so it stays accurate over the 50-100 km extents
    of the IO-VNBD trips (the flat 111320 m/deg approximation does not).
    """

    def __init__(self, lat0_deg, lon0_deg, h0=0.0):
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        self.h0 = float(h0)
        lat, lon = math.radians(self.lat0), math.radians(self.lon0)
        self._sl, self._cl = math.sin(lat), math.cos(lat)
        self._so, self._co = math.sin(lon), math.cos(lon)
        self._x0, self._y0, self._z0 = self._ecef(self.lat0, self.lon0, self.h0)

    @staticmethod
    def _ecef(lat_deg, lon_deg, h):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        return (n + h) * cl * np.cos(lon), (n + h) * cl * np.sin(lon), (n * (1.0 - WGS84_E2) + h) * sl

    def to_enu(self, lat_deg, lon_deg, h=None):
        """Returns (east, north) in metres. Accepts scalars or arrays."""
        h = self.h0 if h is None else h
        x, y, z = self._ecef(lat_deg, lon_deg, h)
        dx, dy, dz = x - self._x0, y - self._y0, z - self._z0
        e = -self._so * dx + self._co * dy
        n = -self._sl * self._co * dx - self._sl * self._so * dy + self._cl * dz
        if np.ndim(e) == 0:
            return float(e), float(n)
        return e, n

    def to_geodetic(self, east, north):
        """Inverse of to_enu for the horizontal plane (up = 0). Accurate to mm over 100 km."""
        dx = -self._so * east - self._sl * self._co * north
        dy = self._co * east - self._sl * self._so * north
        dz = self._cl * north
        x, y, z = self._x0 + dx, self._y0 + dy, self._z0 + dz
        lon = np.arctan2(y, x)
        p = np.hypot(x, y)
        lat = np.arctan2(z, p * (1.0 - WGS84_E2))
        for _ in range(3):
            n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
            h = p / np.cos(lat) - n
            lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        lat, lon = np.degrees(lat), np.degrees(lon)
        if np.ndim(lat) == 0:
            return float(lat), float(lon)
        return lat, lon
