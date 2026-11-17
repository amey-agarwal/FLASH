"""Simulated pairwise-mask secure aggregation (CLAUDE.md Phase H4).

**This is a protocol SIMULATION, not a cryptographic implementation.** The masks come from a
seeded PRG (``derive_seed``), not from key agreement; there are no public keys, no
encryption and no Shamir secret sharing, and anyone who knows the run seed can remove every
mask. What it reproduces is the protocol's *data flow and cost*: the aggregator receives only
masked vectors, the masks cancel in the sum, a member that drops out after the masks are set
has its masks rebuilt from the survivors (dropout recovery), and every extra protocol message
is charged in bytes and simulated time. It gives no formal privacy guarantee, and the final
model can still leak through membership inference.

Protocol modelled (pairwise masks of Bonawitz et al. 2017, without the self-mask), per
aggregation at one level (``regional``: nodes -> regional aggregator; ``global``: regional
aggregators -> global aggregator):
1. Advertise keys: the aggregator asks, each member sends its public keys (one round trip).
2. Share keys: the aggregator sends the key list, each member sends encrypted shares of its
   seed for every other member (one round trip). The members that complete this form the
   mask group. Below the group minimum the aggregation does not run.
3. Masked input: the aggregator forwards the shares (with the model, at the regional level);
   each member uploads ``encode(n * w) + sum_{j > me} m_ij - sum_{j < me} m_ij`` mod 2**64,
   with its example count ``n`` appended (masked too).
4. Recovery (only if a group member's upload is missing): the aggregator sends the dropout
   list to the survivors, each answers with its shares of the dropped members' seeds (one
   round trip); the aggregator rebuilds the dropped members' pairwise masks and removes them.
   If fewer than the group minimum survive, nothing is unmasked.

Values are fixed-point integers (``frac_bits`` fractional bits) in the ring Z_(2**64), so the
masks cancel exactly; the only difference from plain FedAvg is the quantisation,
<= 2**-(frac_bits+1) per summed term before the division by the total count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fedastro.training import derive_seed

SECAGG_ID = 7  # derive_seed namespace (ARM_IDS 1-3, attack 4, COMMS_ID 5, HIER_LINK_ID 6)
LEVELS = ("regional", "global")
KEY_AGREEMENT = ("per_round",)

# Message sizes: scenario assumptions, not measured (CLAUDE.md rule 5).
REQUEST_BYTES = 64  # a protocol request or a dropout list
KEY_BYTES = 64  # one member's two public keys (2 x 32-byte X25519 keys)
SHARE_BYTES = 96  # one encrypted share pair for one peer (2 x 32 B + nonce/tag)
WORD_BYTES = 8  # one masked ring element (uint64)

_HEADROOM_BITS = 62  # |sum of encoded values| must stay below 2**62 (sign + 1 bit spare)


class SecAggError(RuntimeError):
    pass


def masked_bytes(weights: list[np.ndarray]) -> int:
    """Size of one masked upload: every parameter plus the example count, 8 B each."""
    return WORD_BYTES * (int(sum(a.size for a in weights)) + 1)


def pair_mask(seed: int, level: str, rnd: int, a: int, b: int, length: int) -> np.ndarray:
    """The mask shared by members ``a`` and ``b`` (order-free) for one aggregation."""
    lo, hi = sorted((a, b))
    if lo == hi:
        raise ValueError("a member has no pairwise mask with itself")
    gen = np.random.PCG64(derive_seed(seed, SECAGG_ID, LEVELS.index(level), rnd, lo, hi))
    return gen.random_raw(length).astype(np.uint64)


def encode(arrays: list[np.ndarray], n: int, frac_bits: int, group_size: int) -> np.ndarray:
    """Fixed-point ring encoding of ``n * w`` with ``n`` appended."""
    flat = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays]) * float(n)
    scaled = np.rint(flat * 2.0 ** frac_bits)
    bound = 2.0 ** _HEADROOM_BITS / max(group_size, 1)
    if not np.all(np.isfinite(scaled)) or (scaled.size and np.max(np.abs(scaled)) >= bound):
        raise SecAggError(f"secagg encoding overflow: |n * w| * 2**{frac_bits} must stay below "
                          f"2**{_HEADROOM_BITS} / group size; lower secagg.frac_bits")
    out = np.empty(flat.size + 1, dtype=np.uint64)
    out[:-1] = scaled.astype(np.int64).view(np.uint64)
    out[-1] = np.uint64(n)
    return out


def mask_update(arrays: list[np.ndarray], n: int, me: int, group: Iterable[int], seed: int,
                level: str, rnd: int, frac_bits: int) -> np.ndarray:
    """What one member sends: its encoded, pairwise-masked update. Runs on the member."""
    group = sorted(set(group))
    if me not in group:
        raise ValueError(f"member {me} is not in its mask group {group}")
    vec = encode(arrays, n, frac_bits, len(group))
    for peer in group:
        if peer == me:
            continue
        m = pair_mask(seed, level, rnd, me, peer, vec.size)
        vec = vec + m if me < peer else vec - m  # uint64 arithmetic wraps: the ring Z_(2**64)
    return vec


@dataclass(frozen=True)
class Unmasked:
    weights: list[np.ndarray]  # example-weighted mean, the FedAvg aggregate
    total_examples: int
    recovered: tuple[int, ...]  # dropped members whose masks were rebuilt


def unmask_sum(masked: dict[int, np.ndarray], group: Iterable[int], seed: int, level: str, rnd: int,
               frac_bits: int, template: list[np.ndarray], min_group: int) -> Unmasked:
    """The aggregator's side: sum the masked vectors, remove the masks of members that dropped
    after the masks were set (simulated recovery), decode. Only the sum is ever decoded."""
    group = sorted(set(group))
    survivors = sorted(masked)
    if set(survivors) - set(group):
        raise ValueError("masked vectors from outside the mask group")
    if len(survivors) < min_group:
        raise SecAggError(f"{len(survivors)} masked inputs, fewer than the group minimum {min_group}")
    dropped = [m for m in group if m not in masked]
    total = np.zeros_like(masked[survivors[0]])
    for s in survivors:
        total = total + masked[s]
    for d in dropped:  # each survivor's mask with d is left uncancelled; take it back out
        for s in survivors:
            m = pair_mask(seed, level, rnd, s, d, total.size)
            total = total - m if s < d else total + m
    count = int(total[-1])
    if count <= 0:
        raise SecAggError("secagg: decoded example count is not positive")
    values = total[:-1].view(np.int64).astype(np.float64) / 2.0 ** frac_bits / count
    out, k = [], 0
    for ref in template:
        out.append(values[k:k + ref.size].reshape(ref.shape).astype(ref.dtype))
        k += ref.size
    if k != values.size:
        raise SecAggError("secagg: decoded length does not match the model")
    return Unmasked(out, count, tuple(dropped))


@dataclass
class Cost:
    """Extra bytes and simulated seconds of the protocol at one aggregation."""

    bytes: int = 0
    sim_s: float | None = None

    def add(self, nbytes: int) -> None:
        self.bytes += int(nbytes)


def key_bytes(group_size: int, contacted: int) -> int:
    """Advertise keys (request + keys, every contacted member) plus share keys (key list down,
    shares up, and the shares forwarded down to their recipients, every group member)."""
    advertise = contacted * (REQUEST_BYTES + KEY_BYTES)
    share = group_size * (KEY_BYTES * group_size + 2 * SHARE_BYTES * (group_size - 1))
    return advertise + share


def recovery_bytes(n_survivors: int, n_dropped: int) -> int:
    return n_survivors * (REQUEST_BYTES + SHARE_BYTES * n_dropped) if n_dropped else 0
