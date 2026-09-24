"""
Online Hidden-Markov-Model map matcher over an OpenStreetMap road graph
(after Newson & Krumm, "Hidden Markov Map Matching Through Noise and Sparseness", 2009).

Hidden state : which road segment the vehicle is on (plus the projection onto it).
Emission     : Gaussian in the distance from the filtered position to the segment and in the
               difference between vehicle heading and road direction (one-way aware).
Transition   : the road-network distance between successive candidates should match the
               distance the filter says the vehicle travelled (penalises impossible jumps
               between parallel roads and makes the match follow the graph through junctions).

The matcher runs at ~1 Hz. Its output feeds back into the EKF as a *cross-track* pseudo-
measurement: it removes sideways drift, and after a turn the cross-track direction of the new
road is the along-track direction of the old one, so turns also correct along-track drift.
"""
import heapq
import json
import math
import os
from collections import defaultdict

import numpy as np

from src.navigation.frames import wrap_pi

LOG_ZERO = -1e9


def oneway_direction(props):
    """+1 one-way along the geometry, -1 against it, 0 two-way (OSM tagging rules incl. implied one-ways)."""
    tag = str(props.get("oneway") or "").lower()
    if tag in ("yes", "true", "1"):
        return 1
    if tag == "-1":
        return -1
    implied = props.get("junction") in ("roundabout", "circular") or props.get("highway") in ("motorway",
                                                                                             "motorway_link")
    return 1 if implied and tag not in ("no", "false", "0") else 0


class RoadNetwork:
    def __init__(self, cell_m=50.0):
        self.cell = cell_m
        self.p1 = np.zeros((0, 2))
        self.p2 = np.zeros((0, 2))
        self.length = np.zeros(0)
        self.psi = np.zeros(0)
        self.oneway = np.zeros(0, dtype=np.int8)     # 0 both ways, +1 along p1->p2, -1 against
        self.n1 = np.zeros(0, dtype=np.int64)
        self.n2 = np.zeros(0, dtype=np.int64)
        self.grid = defaultdict(list)
        self.node_segs = defaultdict(list)

    @classmethod
    def from_geojson(cls, path_or_dict, ltp, cell_m=50.0):
        data = path_or_dict
        if isinstance(path_or_dict, (str, os.PathLike)):
            with open(path_or_dict) as f:
                data = json.load(f)
        net = cls(cell_m)
        node_ids = {}
        p1, p2, n1, n2, ow = [], [], [], [], []
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            lines = [geom.get("coordinates", [])] if geom.get("type") == "LineString" else \
                geom.get("coordinates", []) if geom.get("type") == "MultiLineString" else []
            props = feat.get("properties") or {}
            oneway = oneway_direction(props)
            for coords in lines:
                if len(coords) < 2:
                    continue
                lon = np.array([c[0] for c in coords])
                lat = np.array([c[1] for c in coords])
                e, n = ltp.to_enu(lat, lon)
                ids = []
                for lo, la in zip(lon, lat):
                    key = (round(lo, 7), round(la, 7))
                    ids.append(node_ids.setdefault(key, len(node_ids)))
                for i in range(len(coords) - 1):
                    if ids[i] == ids[i + 1]:
                        continue
                    p1.append((e[i], n[i])); p2.append((e[i + 1], n[i + 1]))
                    n1.append(ids[i]); n2.append(ids[i + 1]); ow.append(oneway)
        net.p1, net.p2 = np.asarray(p1, dtype=float).reshape(-1, 2), np.asarray(p2, dtype=float).reshape(-1, 2)
        net.n1, net.n2 = np.asarray(n1, dtype=np.int64), np.asarray(n2, dtype=np.int64)
        net.oneway = np.asarray(ow, dtype=np.int8)
        d = net.p2 - net.p1
        net.length = np.hypot(d[:, 0], d[:, 1])
        net.psi = np.arctan2(d[:, 1], d[:, 0])
        net._index()
        return net

    @classmethod
    def from_segments(cls, segments, cell_m=50.0, snap_m=0.5):
        """segments: iterable of ((e1, n1), (e2, n2)[, oneway]) in local ENU metres (for tests)."""
        rows = []
        for seg in segments:
            (a, b), ow = seg[:2], (seg[2] if len(seg) > 2 else 0)
            rows.append((a, b, ow))
        net = cls(cell_m)
        node_ids = {}

        def nid(p):
            return node_ids.setdefault((round(p[0] / snap_m), round(p[1] / snap_m)), len(node_ids))

        net.p1 = np.array([r[0] for r in rows], dtype=float)
        net.p2 = np.array([r[1] for r in rows], dtype=float)
        net.n1 = np.array([nid(r[0]) for r in rows])
        net.n2 = np.array([nid(r[1]) for r in rows])
        net.oneway = np.array([r[2] for r in rows], dtype=np.int8)
        d = net.p2 - net.p1
        net.length = np.hypot(d[:, 0], d[:, 1])
        net.psi = np.arctan2(d[:, 1], d[:, 0])
        net._index()
        return net

    def _index(self):
        self.grid = defaultdict(list)
        self.node_segs = defaultdict(list)
        c = self.cell
        for i in range(len(self.length)):
            lo = np.minimum(self.p1[i], self.p2[i]) // c
            hi = np.maximum(self.p1[i], self.p2[i]) // c
            for gx in range(int(lo[0]), int(hi[0]) + 1):
                for gy in range(int(lo[1]), int(hi[1]) + 1):
                    self.grid[(gx, gy)].append(i)
            self.node_segs[int(self.n1[i])].append(i)
            self.node_segs[int(self.n2[i])].append(i)

    def __len__(self):
        return len(self.length)

    def nearby(self, e, n, radius):
        """Returns arrays (seg_ids, t, proj_e, proj_n, dist) for segments within radius."""
        c = self.cell
        r = int(math.ceil(radius / c))
        gx, gy = int(e // c), int(n // c)
        ids = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                ids.update(self.grid.get((gx + dx, gy + dy), ()))
        if not ids:
            return (np.zeros(0, dtype=np.int64),) + tuple(np.zeros(0) for _ in range(4))
        ids = np.fromiter(ids, dtype=np.int64)
        a, b = self.p1[ids], self.p2[ids]
        d = b - a
        L2 = np.maximum(np.sum(d * d, axis=1), 1e-9)
        t = np.clip(((e - a[:, 0]) * d[:, 0] + (n - a[:, 1]) * d[:, 1]) / L2, 0.0, 1.0)
        pe, pn = a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1]
        dist = np.hypot(e - pe, n - pn)
        keep = dist <= radius
        return ids[keep], t[keep], pe[keep], pn[keep], dist[keep]

    def network_distances(self, seg, t, cutoff):
        """Bounded Dijkstra from a point on `seg` (fraction t). Returns {node_id: distance}."""
        L = self.length[seg]
        dist = {}
        heap = [(t * L, int(self.n1[seg])), ((1.0 - t) * L, int(self.n2[seg]))]
        while heap:
            d, node = heapq.heappop(heap)
            if node in dist or d > cutoff:
                continue
            dist[node] = d
            for s in self.node_segs[node]:
                other = int(self.n2[s]) if int(self.n1[s]) == node else int(self.n1[s])
                if other not in dist:
                    heapq.heappush(heap, (d + self.length[s], other))
        return dist


class MatchResult:
    __slots__ = ("e", "n", "road_psi", "confidence", "segment", "distance")

    def __init__(self, e, n, road_psi, confidence, segment, distance):
        self.e, self.n, self.road_psi = e, n, road_psi
        self.confidence, self.segment, self.distance = confidence, segment, distance


class HMMMapMatcher:
    def __init__(self, network=None, sigma_heading_deg=25.0, min_sigma_dist=4.0, max_candidates=12,
                 beta_m=8.0):
        self.net = network
        self.sigma_h = math.radians(sigma_heading_deg)
        self.min_sigma_d = min_sigma_dist
        self.max_candidates = max_candidates
        self.beta = beta_m
        self.reset()

    def reset(self):
        self.cands = None        # dict seg -> (t, score)

    @property
    def has_map(self):
        return self.net is not None and len(self.net) > 0

    def _road_heading(self, seg, psi):
        """Road direction (as travelled) closest to the vehicle heading, respecting one-way streets."""
        fwd = self.net.psi[seg]
        ow = self.net.oneway[seg]
        if ow == 1:
            return fwd
        if ow == -1:
            return wrap_pi(fwd + math.pi)
        rev = wrap_pi(fwd + math.pi)
        return fwd if abs(wrap_pi(psi - fwd)) <= abs(wrap_pi(psi - rev)) else rev

    def step(self, e, n, psi, speed, travelled_m, pos_std_m):
        """One HMM step. Returns MatchResult or None when no road is nearby."""
        if not self.has_map:
            return None
        sigma_d = max(self.min_sigma_d, pos_std_m)
        radius = min(max(3.0 * sigma_d, 25.0), 120.0)
        ids, t, pe, pn, dist = self.net.nearby(e, n, radius)
        if len(ids) == 0:
            self.reset()
            return None
        order = np.argsort(dist, kind="stable")[:self.max_candidates]
        ids, t, pe, pn, dist = ids[order], t[order], pe[order], pn[order], dist[order]

        emis = -0.5 * (dist / sigma_d) ** 2
        road_psi = np.array([self._road_heading(int(s), psi) for s in ids])
        if speed > 2.0:
            dpsi = np.abs(wrap_pi(psi - road_psi))
            emis = emis - 0.5 * (dpsi / self.sigma_h) ** 2

        scores = emis.copy()
        if self.cands:
            beta = self.beta + 0.25 * travelled_m
            cutoff = 2.0 * travelled_m + 60.0
            best_prev = np.full(len(ids), LOG_ZERO)
            for pseg, (pt, pscore) in self.cands.items():
                nd = self.net.network_distances(pseg, pt, cutoff)
                for j, s in enumerate(ids):
                    s = int(s)
                    if s == pseg:
                        route = abs(t[j] - pt) * self.net.length[s]
                    else:
                        L = self.net.length[s]
                        d1 = nd.get(int(self.net.n1[s]), math.inf) + t[j] * L
                        d2 = nd.get(int(self.net.n2[s]), math.inf) + (1.0 - t[j]) * L
                        route = min(d1, d2)
                    if math.isinf(route):
                        continue
                    trans = -abs(route - travelled_m) / beta
                    best_prev[j] = max(best_prev[j], pscore + trans)
            if np.all(best_prev <= LOG_ZERO / 2):
                scores = emis          # the chain broke (e.g. left the mapped corridor): restart
            else:
                scores = emis + best_prev

        scores = scores - np.max(scores)
        post = np.exp(scores)
        post /= post.sum()
        best = int(np.argmax(scores))
        self.cands = {int(s): (float(tt), float(sc)) for s, tt, sc in zip(ids, t, scores)}
        # Consecutive pieces of the same road meet at shared nodes; count them as one hypothesis.
        same = (np.hypot(pe - pe[best], pn - pn[best]) < 3.0) & \
               (np.abs(wrap_pi(road_psi - road_psi[best])) < math.radians(20.0))
        confidence = float(post[same].sum())
        return MatchResult(float(pe[best]), float(pn[best]), float(road_psi[best]), confidence,
                           int(ids[best]), float(dist[best]))
