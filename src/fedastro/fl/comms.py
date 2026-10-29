"""Simulated deep-space links: delay, jitter, bandwidth, loss of signal (LOS).

Time here is SIMULATED seconds, a modelled quantity. It is never slept (except by
--real-sleep for the live demo) and is reported in ``sim_wallclock_s``, separate from
the real elapsed time in manifest.json. Never plot the two on one unlabelled axis.

Transfer model (one message over one link):
- The sender may start transmitting only while the link is up. A transmission needs
  ``bytes * 8 / bandwidth`` seconds of uninterrupted link; if an outage starts during
  it, the transmission restarts when the link returns (store-and-forward, as in
  delay-tolerant deep-space networking: nothing is lost, it waits).
- Arrival = end of transmission + one-way light time + |N(0, jitter_s)|.
- Propagation is not interrupted by an outage that starts after the signal has left.

Sender buffer (one per link direction), CommsConfig.supersede / ttl_s / cache_bytes:
- supersede: handing a newer message to a direction whose previous message is still
  waiting in the buffer (not yet transmitting) drops the previous one.
- expiry: a message that would arrive more than ``ttl_s`` after hand-over is dropped
  (it is lost: the transfer returns ``inf``).
- cache cap: a message that has to wait (link down at hand-over) and is larger than
  ``cache_bytes`` is dropped on arrival at the buffer (drop-tail). Off by default.
  With supersede on, the buffer holds at most one message per direction, so the cap
  only matters if it is smaller than one message.
Every drop is logged in ``Link.drops`` with its reason.

Outages come from the silo's ``los_profile`` (CommsConfig.los_profiles), generated
lazily and deterministically from the run seed and the silo's topology index, plus an
optional forced "dark" window (CommsConfig.dark). An outage with no end makes every
later transfer return ``inf``: the silo is unreachable. A dark window given by round
(``dark.from_round``) is resolved to a time by the server and set before any transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fedastro.config import CommsConfig, ExperimentConfig, LOSProfile, SiloConfig
from fedastro.training import derive_seed

COMMS_ID = 5  # derive_seed namespace; ARM_IDS uses 1-3, the attack 4
MOON_ONE_WAY_S = 1.3  # Earth-Moon light time ~1.2-1.3 s (CLAUDE.md section 4)
INF = math.inf


def resolve_latency(silo: SiloConfig, preset: str) -> float:
    """One-way light time for ``silo`` under ``preset``. Only silos that define
    ``latency_presets_s`` (the deep-space silo) move; the others keep their own."""
    if not silo.latency_presets_s:
        return silo.one_way_latency_s
    if preset == "earth":
        return 0.0
    if preset == "moon":
        return MOON_ONE_WAY_S
    key = preset.removeprefix("mars_")
    if key not in silo.latency_presets_s:
        raise ValueError(f"silo {silo.name} has no latency preset {key!r}")
    return silo.latency_presets_s[key]


@dataclass
class Link:
    """One silo's bidirectional link (same model both ways, separate jitter draws)."""

    silo: str
    index: int
    one_way_s: float
    bandwidth_bps: float
    jitter_s: float
    profile: LOSProfile
    los_rng: np.random.Generator  # outage schedule only: independent of traffic
    jitter_rng: np.random.Generator
    forced: tuple[float, float] | None = None  # (start, end), end may be inf
    supersede: bool = True
    ttl_s: float = 0.0  # 0 = never expire
    cache_bytes: int = 0  # 0 = unlimited
    drops: list[dict] = field(default_factory=list)
    _waiting: dict[str, dict] = field(default_factory=dict)  # direction -> last buffered message
    _outages: list[tuple[float, float]] = field(default_factory=list)
    _generated_to: float = 0.0
    _next_onset: float | None = None

    # -- outages ---------------------------------------------------------------
    def _generate(self, until: float) -> None:
        p = self.profile
        if p.kind == "poisson":
            while self._generated_to < until:
                if self._next_onset is None:
                    self._next_onset = float(self.los_rng.exponential(p.mean_interval_s))
                start = self._next_onset
                self._outages.append((start, start + p.duration_s))
                self._next_onset = start + p.duration_s + float(self.los_rng.exponential(p.mean_interval_s))
                self._generated_to = self._next_onset
        elif p.kind == "periodic":
            while self._generated_to < until:
                k = len(self._outages)
                start = p.phase_s + k * p.period_s
                self._outages.append((start, start + p.duration_s))
                self._generated_to = start + p.period_s
        else:
            self._generated_to = max(self._generated_to, until)

    def outages(self, until: float) -> list[tuple[float, float]]:
        """All outages (profile + forced) that start before ``until``, sorted."""
        self._generate(until)
        out = [o for o in self._outages if o[0] < until]
        if self.forced is not None and self.forced[0] < until:
            out.append(self.forced)
        return sorted(out)

    def _blocking(self, t: float, length: float) -> tuple[float, float] | None:
        """First outage overlapping [t, t + length), if any."""
        for start, end in self.outages(t + length + 1e-9):
            if start < t + length and end > t:
                return start, end
        return None

    # -- transfers -------------------------------------------------------------
    def transmit_s(self, nbytes: int) -> float:
        return nbytes * 8.0 / self.bandwidth_bps

    def transfer(self, t: float, nbytes: int, direction: str = "down") -> float:
        """Simulated arrival time of a message handed to this link at time ``t``, or
        ``inf`` if it is never delivered (unreachable, expired or dropped)."""
        prev = self._waiting.get(direction)
        if self.supersede and prev is not None and not prev["dropped"] and prev["leaves_s"] > t:
            prev["dropped"] = True
            self._drop(prev, "superseded", t)
        tx = self.transmit_s(nbytes)
        start = t
        while (block := self._blocking(start, tx)) is not None:
            start = block[1]
            if start == INF:
                break
        msg = {"direction": direction, "handed_s": t, "bytes": nbytes, "leaves_s": start, "dropped": False}
        self._waiting[direction] = msg
        if start > t and self.cache_bytes and nbytes > self.cache_bytes:
            msg.update(dropped=True, leaves_s=t)
            self._drop(msg, "cache_full", t)
            return INF
        arrival = INF
        if start < INF:
            jitter = abs(float(self.jitter_rng.normal(0.0, self.jitter_s))) if self.jitter_s > 0 else 0.0
            arrival = start + tx + self.one_way_s + jitter
        if self.ttl_s and arrival - t > self.ttl_s:
            # it would still be waiting in the buffer at expiry, or arrive too late
            msg.update(dropped=True, leaves_s=min(start, t + self.ttl_s))
            self._drop(msg, "expired", t + self.ttl_s)
            return INF
        return arrival

    def _drop(self, msg: dict, reason: str, at: float) -> None:
        self.drops.append({"silo": self.silo, "direction": msg["direction"], "reason": reason,
                           "handed_s": msg["handed_s"], "dropped_s": at, "bytes": msg["bytes"]})


def dark_window(c: CommsConfig, from_s: float) -> tuple[float, float]:
    """(start, end) of the forced outage starting at ``from_s``; end may be inf."""
    return from_s, INF if c.dark.duration_s == 0 else from_s + c.dark.duration_s


def build_links(cfg: ExperimentConfig, seed: int) -> dict[str, Link]:
    """One link per silo. With ``comms.dark.from_round`` the forced window is left
    unset: the server resolves the round to a time and sets it (fl/server.py)."""
    c: CommsConfig = cfg.comms
    links = {}
    for i, silo in enumerate(cfg.topology.silos):
        profile = c.los_profiles[silo.los_profile] if c.los else LOSProfile()
        forced = None
        if c.dark.silo == silo.name and not c.dark.from_round:
            forced = dark_window(c, c.dark.from_s)
        links[silo.name] = Link(
            silo=silo.name, index=i, one_way_s=resolve_latency(silo, c.preset),
            bandwidth_bps=c.bandwidth_bps, jitter_s=c.jitter_s, profile=profile,
            los_rng=np.random.default_rng(derive_seed(seed, COMMS_ID, i, 1)),
            jitter_rng=np.random.default_rng(derive_seed(seed, COMMS_ID, i, 2)), forced=forced,
            supersede=c.supersede, ttl_s=c.ttl_s, cache_bytes=c.cache_bytes,
        )
    return links


def compute_s(cfg: ExperimentConfig, n_train: int) -> float:
    """Simulated local training time for one dispatch."""
    return cfg.comms.compute_s_per_example * n_train * cfg.fl.local_epochs


def los_events(links: dict[str, Link], until: float) -> list[dict]:
    """Every outage that began before ``until`` (the end of the simulated run)."""
    events = []
    for link in links.values():
        for start, end in link.outages(until):
            events.append({"silo": link.silo, "start_s": start,
                           "end_s": None if end == INF else end,
                           "forced": link.forced is not None and (start, end) == link.forced})
    return sorted(events, key=lambda e: (e["start_s"], e["silo"]))


def link_drops(links: dict[str, Link], until: float) -> list[dict]:
    """Every buffered message dropped before ``until``, in time order."""
    drops = [d for l in links.values() for d in l.drops if d["dropped_s"] <= until]
    return sorted(drops, key=lambda d: (d["dropped_s"], d["silo"], d["direction"]))


def describe_links(links: dict[str, Link]) -> dict[str, dict]:
    return {n: {"one_way_s": l.one_way_s, "bandwidth_bps": l.bandwidth_bps, "jitter_s": l.jitter_s,
                "los": {"kind": l.profile.kind, "mean_interval_s": l.profile.mean_interval_s,
                        "period_s": l.profile.period_s, "phase_s": l.profile.phase_s,
                        "duration_s": l.profile.duration_s},
                "forced_dark": None if l.forced is None else
                {"from_s": l.forced[0], "until_s": None if l.forced[1] == INF else l.forced[1]}}
            for n, l in links.items()}
