"""
GNSS quality monitor & navigation-mode state machine (the "Seamless GNSS Deficit Handler").

Modes
  GNSS_AIDED      fixes arriving, accurate, consistent with the filter -> fused
  DEGRADED        fixes arriving but poor (accuracy > degraded_accuracy_m) -> fused with inflated R
  DEAD_RECKONING  no usable fix for > timeout -> INS + AI speed + map only
  REACQUISITION   fixes are back after an outage but not yet confirmed

The EKF runs continuously in every mode, so switching modes never interrupts the position
output: dead reckoning simply stops receiving GNSS updates. The switch to DEAD_RECKONING
happens at the first frame after `timeout_factor` x the observed fix interval (1.5 s at 1 Hz).

Reacquisition trap fix: after dead reckoning the accumulated drift makes the first fixes look
like outliers (large NIS). Instead of rejecting them, two consecutive fixes that are
consistent with *each other* (displacement matches the fix speeds) trigger a re-anchor of
the filter onto GNSS.
"""
import math


MAX_PAIR_GAP_S = 60.0          # s; a pending re-anchor fix older than this is discarded
MAX_PLAUSIBLE_SPEED = 60.0     # m/s; hard cap in the sparse-pair plausibility check
SPARSE_INTERVAL_S = 3.0        # s; above this estimated fix interval the receiver is treated as sparse


class NavigationStateMode:
    GNSS_AIDED = "GNSS_AIDED"
    DEGRADED = "DEGRADED"
    DEAD_RECKONING = "DEAD_RECKONING"
    REACQUISITION = "REACQUISITION"


class GNSSDecision:
    FUSE = "fuse"
    FUSE_INFLATED = "fuse_inflated"
    REANCHOR = "reanchor"
    REJECT = "reject"
    NONE = "none"


class GNSSQualityMonitor:
    def __init__(self, max_accuracy_m=25.0, degraded_accuracy_m=12.0, timeout_factor=1.5,
                 nominal_interval_s=1.0, nis_gate=13.82, max_consecutive_rejects=3):
        self.mode = NavigationStateMode.GNSS_AIDED
        self.gnss_quality = 0.0
        self.max_accuracy_m = max_accuracy_m
        self.degraded_accuracy_m = degraded_accuracy_m
        self.timeout_factor = timeout_factor
        self.interval = nominal_interval_s
        self.nis_gate = nis_gate
        self.max_consecutive_rejects = max_consecutive_rejects
        self.last_fix_time = None
        self.last_fix = None
        self.consecutive_rejects = 0
        self.pending_fix = None
        self.transitions = []          # (timestamp, from_mode, to_mode)
        self.initialized = False

    def _set_mode(self, t, mode):
        if mode != self.mode:
            self.transitions.append((t, self.mode, mode))
            self.mode = mode

    def timeout_s(self):
        return self.timeout_factor * max(self.interval, 0.1)

    def tick(self, t):
        """Called every frame (with or without a fix). Handles the GNSS-loss timeout."""
        if self.last_fix_time is None or t - self.last_fix_time > self.timeout_s():
            if self.mode != NavigationStateMode.DEAD_RECKONING:
                self._set_mode(t, NavigationStateMode.DEAD_RECKONING)
            self.gnss_quality = 0.0
        return self.mode

    @staticmethod
    def _consistent(a, b, filter_speed=0.0):
        """
        True if two fixes (t, e, n, speed, acc) agree with each other.
        Fixes up to 5 s apart must match the distance implied by their speeds. Sparse receivers (e.g. the
        IO-VNBD phone GPS, one fix every ~9 s) report poor speeds, and speed changes a lot between such fixes,
        so the jump is checked against the fastest of the fix speeds and the filter's own speed estimate
        (x1.5 + 3 m/s, capped at 60 m/s). Without this the filter never re-anchors after an outage; the bound
        stops a single far-off fix from being accepted.
        """
        dt = b[0] - a[0]
        if dt <= 0 or dt > MAX_PAIR_GAP_S:
            return False
        moved = math.hypot(b[1] - a[1], b[2] - a[2])
        if dt <= 5.0:
            expected = 0.5 * (a[3] + b[3]) * dt
            return abs(moved - expected) < 3.0 * math.hypot(a[4], b[4]) + 2.0 * dt
        v = max(filter_speed or 0.0, a[3] or 0.0, b[3] or 0.0)
        limit = min(MAX_PLAUSIBLE_SPEED, 1.5 * v + 3.0) * dt
        return moved <= limit + 3.0 * math.hypot(a[4], b[4]) + 10.0

    def on_fix(self, t, e, n, speed, accuracy_m, nis, filter_speed=0.0):
        """
        Decides what the pipeline should do with a new fix.
        nis: normalized innovation squared of the fix against the current filter state.
        filter_speed: the filter's current speed estimate (m/s), used to bound sparse fix pairs.
        """
        if accuracy_m is None or accuracy_m > self.max_accuracy_m:
            return GNSSDecision.REJECT
        if self.last_fix_time is not None:
            # A single outage gap must not inflate the interval of a normal (~1 Hz) receiver, so gaps are
            # capped at 5 s; only once the receiver is evidently sparse may they count up to 15 s.
            cap = 15.0 if self.interval > SPARSE_INTERVAL_S else 5.0
            self.interval = 0.8 * self.interval + 0.2 * min(max(t - self.last_fix_time, 0.1), cap)
        self.last_fix_time = t
        fix = (t, e, n, speed, accuracy_m)
        base_q = max(0.1, min(1.0, 1.0 - (accuracy_m - 3.0) / 25.0))

        if not self.initialized:
            self.initialized = True
            self.last_fix = fix
            self.gnss_quality = base_q
            self._set_mode(t, NavigationStateMode.GNSS_AIDED)
            return GNSSDecision.REANCHOR

        if self.mode in (NavigationStateMode.DEAD_RECKONING, NavigationStateMode.REACQUISITION):
            if nis <= self.nis_gate:          # drift was small: just resume fusing
                self.pending_fix = None
                self._set_mode(t, NavigationStateMode.GNSS_AIDED)
                self.gnss_quality = base_q
                self.last_fix = fix
                return GNSSDecision.FUSE
            if self.pending_fix is not None and self._consistent(self.pending_fix, fix, filter_speed):
                self.pending_fix = None
                self.consecutive_rejects = 0
                self._set_mode(t, NavigationStateMode.GNSS_AIDED)
                self.gnss_quality = base_q
                self.last_fix = fix
                return GNSSDecision.REANCHOR
            self.pending_fix = fix
            self._set_mode(t, NavigationStateMode.REACQUISITION)
            self.gnss_quality = 0.5 * base_q
            return GNSSDecision.REJECT

        # GNSS_AIDED / DEGRADED: outlier gating
        if nis > self.nis_gate:
            self.consecutive_rejects += 1
            self.gnss_quality = max(0.0, base_q - 0.2 * self.consecutive_rejects)
            if self.consecutive_rejects >= self.max_consecutive_rejects and \
                    self._consistent(self.last_fix, fix, filter_speed):
                # GNSS is self-consistent but disagrees with us: the filter has drifted.
                self.consecutive_rejects = 0
                self.last_fix = fix
                return GNSSDecision.REANCHOR
            self.last_fix = fix
            return GNSSDecision.REJECT
        self.consecutive_rejects = 0
        self.last_fix = fix
        self.gnss_quality = base_q
        if accuracy_m > self.degraded_accuracy_m:
            self._set_mode(t, NavigationStateMode.DEGRADED)
            return GNSSDecision.FUSE_INFLATED
        self._set_mode(t, NavigationStateMode.GNSS_AIDED)
        return GNSSDecision.FUSE
