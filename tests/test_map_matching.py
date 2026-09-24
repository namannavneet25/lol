"""Map matching on a GeoJSON road network (the original PS26168 snapping scenarios, now asserted)."""
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.navigation.frames import LocalTangentPlane, compass_to_psi
from src.navigation.map_matching import HMMMapMatcher, RoadNetwork

GEOJSON = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"id": "Main_Street_North"},
     "geometry": {"type": "LineString", "coordinates": [[91.736, 26.144], [91.736, 26.145]]}},
    {"type": "Feature", "properties": {"id": "Cross_Street_East"},
     "geometry": {"type": "LineString", "coordinates": [[91.736, 26.145], [91.737, 26.145]]}},
]}


def _matcher():
    ltp = LocalTangentPlane(26.144, 91.736)
    net = RoadNetwork.from_geojson(GEOJSON, ltp)
    assert len(net) == 2
    return HMMMapMatcher(net), net


def test_geojson_segments_in_local_metres():
    _, net = _matcher()
    assert abs(net.length[0] - 110.7) < 1.0          # 0.001 deg latitude
    assert abs(net.length[1] - 100.0) < 0.5          # 0.001 deg longitude at 26.1 deg
    assert net.n2[0] == net.n1[1]                    # junction node shared -> routable


def test_snapping_scenarios():
    cases = [  # (east, north, compass heading, expected snapped east, expected snapped north)
        (0.0, 10.0, 0.0, 0.0, 10.0),      # on the centreline
        (6.5, 30.0, 5.0, 0.0, 30.0),      # drifted 6.5 m east
        (15.2, 70.0, 12.0, 0.0, 70.0),    # heavy tunnel drift
    ]
    for e, n, hdg, ee, en in cases:
        m, _ = _matcher()
        res = m.step(e, n, compass_to_psi(hdg), 10.0, 0.0, 8.0)
        assert res is not None and math.hypot(res.e - ee, res.n - en) < 0.5, (e, n, res.e, res.n)


def test_turn_at_junction_follows_graph():
    m, net = _matcher()
    res = None
    for n in range(20, 111, 10):
        res = m.step(3.0, float(n), compass_to_psi(0.0), 10.0, 10.0, 6.0)
    for e in range(10, 60, 10):
        res = m.step(float(e), 118.0, compass_to_psi(90.0), 10.0, 10.0, 6.0)
    assert res.segment == 1 and abs(res.n - net.p1[1][1]) < 0.5
