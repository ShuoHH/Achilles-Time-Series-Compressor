"""
neurts_accessor.py  --  Random-access accessor for the NeurTS implicit-
representation codec.

Architecture mapping (1-to-1 with the upstream training code)
─────────────────────────────────────────────────────────────
- Vector tables    : `cross_models.grid.GlobalGridStorage`
                       base_grid  + patch_grid (+ aux nodes share patch_grid).
                       Looked up by integer ID; that is "the lookup table" the
                       user described as keying into "向量?.
- Index table      : `cross_models.neurts_manager.GridManager.index_table`
                       Length = T // min_resolution; each row is
                       `IndexEntry(left_id, right_id, level_code)`.  We
                       precompile this into 4 contiguous int32 arrays so a
                       query becomes a single arithmetic step:
                         idx = t // min_resolution           ?O(1)
                         left, right, block_start, block_len = arrays[idx]
- Decoder          : Fully-connected `cross_models.decoders.*` (Fourier /
                       SIREN / TCN / ...).  One forward = one block waveform.
- Residual fall-back: `cross_models.fallback_dict.FallbackDict` (Tier 2 PATCH
                       sparse residuals + Tier 3 RAW whole-block override),
                       keyed by `left_id` so it is also O(1).

Honest Option-C semantics  (matches LFZip / NeaTS accessors)
────────────────────────────────────────────────────────────
- query(t)        : O(1) index hit ?1 decoder forward ?optional patch
                    apply ?return the single sample, in *original* units
                    (StandardScaler inverse-transformed).
- query_range(s,e): walk slot indices, dedupe by block, decode each block at
                    most once per call (LRU-cache reuse across calls), gather.
- query_batch(ts) : per-query slot lookup; reuses the LRU so repeated blocks
                    cost ~one cache hit.
- decompress_all(): stride through unique blocks once.

A small LRU `_block_cache` (default 64 blocks) holds already-decoded blocks
across calls; `clear_cache()` resets it for strict cold-cache benchmarking.

Persistence: a single torch `.pt` file (see `save()` / `from_file()`):
    {
        "version": 1, "T": ..., "min_resolution": ..., "base_block_size": ...,
        "idx_left": np.int32[N], "idx_right": np.int32[N],
        "idx_block_start": np.int32[N], "idx_block_len": np.int32[N],
        "model_state_dict": ..., "model_kwargs": {...}, "grid_kwargs": {...},
        "fallback_dict_state": ... | None,
        "scaler_mean": ..., "scaler_std": ...,        # (None if no scaling)
        "mean_scalar": ..., "std_scalar": ...,
    }
"""

import os
import sys
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pull in the unified BaseAccessor + generate_query_sets that ship with NeaTS.
# `rand_ac/` is a sibling directory committed alongside this project so that
# every Option-C baseline (LFZip / NeaTS / NeurTS) inherits the *exact same*
# benchmarking infrastructure ?query indices, timing methodology and report
# format are identical.  Keep this import side-effect-free.
# ---------------------------------------------------------------------------
_RAND_AC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rand_ac")
if _RAND_AC_DIR not in sys.path:
    sys.path.insert(0, _RAND_AC_DIR)
from accessor_base import BaseAccessor, generate_query_sets   # noqa: E402

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
from models.neurts_model import NeurTSModel
from models.grid import GlobalGridStorage
from models.fallback_dict import FallbackDict


# ---------------------------------------------------------------------------
# Cache key: (left_id, block_start) is the unique identifier of a block.
# `level_code` is implied by (left_id, right_id) but we don't need it in the
# key because two distinct blocks never share both endpoints.
# ---------------------------------------------------------------------------
_BlockKey = Tuple[int, int]


class NeurTSAccessor(BaseAccessor):
    """Random-access accessor for the NeurTS codec (Option-C semantics)."""

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self):
        # NOTE: the public entry points are `from_exp` and `from_file`; this
        # bare constructor only initialises the cache plumbing common to both.
        self._block_cache: "OrderedDict[_BlockKey, np.ndarray]" = OrderedDict()
        self._cache_size: int = 64
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._build_time: float = 0.0

        # All other attributes are populated by from_exp / from_file:
        #   self.model, self.device, self._dtype
        #   self.min_resolution, self.base_block_size, self.num_slots, self._T
        #   self.idx_left, self.idx_right, self.idx_block_start, self.idx_block_len
        #   self.fallback_dict
        #   self._mean_scalar, self._std_scalar  (or _scaler_mean / _scaler_std)
        #   self._model_kwargs, self._grid_kwargs

    # ---------- 1. Build from a finished training experiment ----------------

    @classmethod
    def from_exp(cls, exp) -> "NeurTSAccessor":
        """Build an accessor from a finished `Exp_NeurTS` object.

        `exp` must already have:
            exp.model         : NeurTSModel  (in eval mode is fine)
            exp.manager       : GridManager  (with finalised index_table)
            exp.fallback_dict : FallbackDict | None  (call build_fallback_dict
                                                       beforehand for tier 2/3)
            exp.data_loader   : object exposing `.scaler` (StandardScaler) when
                                normalisation was used; the accessor inverse-
                                transforms outputs back into original units.
            exp.args          : argparse.Namespace with the model-construction
                                hyperparameters (decoder_type, hidden_dim, ...).
        """
        t0 = time.perf_counter()
        obj = cls()

        # Model + device --------------------------------------------------------
        model = exp.model
        model.eval()
        obj.model = model
        obj.device = next(model.parameters()).device
        obj._dtype = next(model.parameters()).dtype

        # Compile the index table into contiguous int32 arrays for O(1) access.
        mgr = exp.manager
        obj.min_resolution = int(mgr.min_resolution)
        obj.base_block_size = int(mgr.base_block_size)
        obj.num_slots = int(mgr.num_slots)
        obj._T = int(mgr.total_length)

        (
            obj.idx_left,
            obj.idx_right,
            obj.idx_block_start,
            obj.idx_block_len,
        ) = cls._compile_index_table(mgr)

        # Optional residual fallback dictionary (Tier 2 / Tier 3).
        obj.fallback_dict = getattr(exp, "fallback_dict", None)

        # StandardScaler parameters (for inverse transform on query output).
        scaler = getattr(getattr(exp, "data_loader", None), "scaler", None)
        cls._bind_scaler(obj, scaler)

        # Stash construction kwargs so from_file() can rebuild the model.
        obj._model_kwargs = cls._extract_model_kwargs(exp)
        obj._grid_kwargs = cls._extract_grid_kwargs(exp)

        obj._build_time = time.perf_counter() - t0
        return obj

    # ---------- 2. Compile index table ?4 contiguous arrays -----------------

    @staticmethod
    def _compile_index_table(
        mgr,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single sweep over `mgr.index_table` ?(left, right, start, len)
        per slot.  All arrays have shape `(num_slots,)` and dtype int32.

        Adjacent slots that share `(left_id, right_id, level_code)` belong to
        the same block; we materialise that block's `start` and `len` into
        every slot it covers, so query never needs to scan.
        """
        n = int(mgr.num_slots)
        idx_left = np.empty(n, dtype=np.int32)
        idx_right = np.empty(n, dtype=np.int32)
        idx_block_start = np.empty(n, dtype=np.int32)
        idx_block_len = np.empty(n, dtype=np.int32)

        i = 0
        while i < n:
            e = mgr.index_table[i]
            l_id, r_id, lc = e.left_id, e.right_id, e.level_code
            j = i + 1
            while j < n:
                ej = mgr.index_table[j]
                if (
                    ej.left_id == l_id
                    and ej.right_id == r_id
                    and ej.level_code == lc
                ):
                    j += 1
                else:
                    break
            block_start_t = i * mgr.min_resolution
            block_end_t = j * mgr.min_resolution
            block_len = block_end_t - block_start_t
            idx_left[i:j] = l_id
            idx_right[i:j] = r_id
            idx_block_start[i:j] = block_start_t
            idx_block_len[i:j] = block_len
            i = j
        return idx_left, idx_right, idx_block_start, idx_block_len

    # ---------- 3. Helpers shared by from_exp / from_file --------------------

    @staticmethod
    def _bind_scaler(obj: "NeurTSAccessor", scaler) -> None:
        """Attach (or zero-fill) inverse-StandardScaler params on `obj`."""
        if scaler is None:
            obj._scaler_mean = None
            obj._scaler_std = None
            obj._mean_scalar = 0.0
            obj._std_scalar = 1.0
            return

        mean = np.asarray(scaler.mean, dtype=np.float32).flatten()
        std = np.asarray(scaler.std, dtype=np.float32).flatten()
        obj._scaler_mean = mean
        obj._scaler_std = std
        # Univariate (single channel) is the dominant case ?fast scalar path.
        if mean.size == 1:
            obj._mean_scalar = float(mean[0])
            obj._std_scalar = float(std[0])
        else:
            # Multi-channel raw_data is unusual for this codec; we fall back
            # to channel-0 inverse, which matches the way the upstream
            # exp_neurts.py reports per-block error in original units
            # (it always indexes std_val / mean_val as scalars).
            obj._mean_scalar = float(mean[0])
            obj._std_scalar = float(std[0])

    @staticmethod
    def _extract_model_kwargs(exp) -> dict:
        """Mirror the kwargs `Exp_NeurTS._build_model()` passes to NeurTSModel.

        Anything we cannot read from `exp.args` falls back to the same default
        the constructor uses, so checkpoints stay backward-compatible.
        """
        a = exp.args
        m = exp.model
        return dict(
            decoder_type=getattr(m, "decoder_type", getattr(a, "decoder_type", "siren")),
            max_block_size=int(getattr(m, "max_block_size", a.base_block_size)),
            hidden_dim=int(getattr(a, "hidden_dim", 64)),
            pe_dim=int(getattr(a, "pe_dim", 32)),
            num_res_blocks=int(getattr(a, "num_res_blocks", 4)),
            kernel_size=int(getattr(a, "kernel_size", 3)),
            dropout=float(getattr(a, "dropout", 0.0)),
            aux_dim=int(getattr(a, "aux_dim", -1)),
            # Decoder-specific extras (passed via **decoder_kwargs upstream;
            # NeurTSModel forwards them).
            hyper_hidden=int(getattr(a, "hyper_hidden", 128)),
            num_freqs=int(getattr(a, "num_freqs", 64)),
            max_aux_tokens=int(getattr(a, "max_aux_tokens", 0)),
            transformer_nhead=int(getattr(a, "transformer_nhead", 4)),
            total_length=int(exp.manager.total_length),
        )

    @staticmethod
    def _extract_grid_kwargs(exp) -> dict:
        gs = exp.grid_storage
        return dict(
            block_size=int(gs.block_size),
            trend_dim=int(gs.trend_dim),
            context_dim=int(gs.context_dim),
            max_patch_nodes=int(gs.max_patch_nodes),
        )

    # =========================================================================
    # BaseAccessor abstract interface
    # =========================================================================

    @property
    def T(self) -> int:
        return self._T

    def _sync(self):
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def clear_cache(self):
        self._block_cache.clear()

    def reset_cache_stats(self):
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
        }

    # ---------- single point -----------------------------------------------

    def query(self, t: int) -> float:
        t = int(t)
        if t < 0 or t >= self._T:
            raise IndexError(f"t={t} out of [0, {self._T})")

        # O(1) index lookup ----------------------------------------------------
        idx = t // self.min_resolution
        if idx >= self.num_slots:
            idx = self.num_slots - 1
        block_start = int(self.idx_block_start[idx])

        # O(1) block fetch (cache hit or 1 decoder forward) -------------------
        block = self._get_block(idx)
        return float(block[t - block_start])

    # ---------- enable a fast-path single-query method ---------------------

    def enable_fast_query(self) -> None:
        """Pre-convert the index table arrays to Python lists.

        After `enable_fast_query()` you can call `query_fast(t)` which
        skips the numpy scalar -> int conversions that dominate the
        `query(t)` hot path (~700 ns out of ~1.3 us).  Typical speedup
        is 2-3x.

        Call this right after `preload_all()` when you intend to do
        many random `query(t)` calls (e.g. NeaTS-style point benchmarks).
        The lists are tied to the current index table, so re-call after
        any index_table mutation (none in the standard path).

        Side effect: `self.query` is rebound to `self.query_fast` so
        external benchmarks that call `acc.query(t)` (e.g. the shared
        `benchmark_neats_style_point` in `accessor_base.py`) automatically
        pick up the fast path without modification.  The original
        bounds-checking / numpy version is preserved as
        `self.query_safe`.
        """
        self._idx_left_list = self.idx_left.tolist()
        self._idx_block_start_list = self.idx_block_start.tolist()
        self._min_res = int(self.min_resolution)
        self._num_slots_int = int(self.num_slots)
        # Rebind: subsequent `acc.query(t)` -> fast path.
        self.query_safe = self.query
        self.query = self.query_fast  # type: ignore[assignment]

    def query_fast(self, t: int) -> float:
        """Hot-path single-point query.

        Bypasses sanity checks and numpy scalar conversions.  Requires
        `enable_fast_query()` to have been called first; behaviour is
        undefined otherwise.  ~500-700 ns / call (vs ~1300 ns for
        `query`) on typical CPU + cache-resident workloads.

        No bounds check on `t`: caller is responsible for ensuring
        `0 <= t < self._T`.
        """
        idx = t // self._min_res
        if idx >= self._num_slots_int:
            idx = self._num_slots_int - 1
        bs = self._idx_block_start_list[idx]
        block = self._block_cache[(self._idx_left_list[idx], bs)]
        return block[t - bs]

    # ---- Cold-cache numpy fast path ---------------------------------------
    # Drops PyTorch from the hot single-query path (no preloading, no LRU
    # tricks).  Each call does a full FourierDecoder forward + fallback
    # overlay + scaler -- exactly the same workload as the reference
    # `query()`, just implemented in numpy.  This removes the per-call
    # PyTorch dispatcher overhead (tensor creation, op dispatch, sync,
    # autograd bookkeeping) which dominates `decode_single` on small
    # forwards (~3 ms out of ~3.5 ms total are pure framework cost).

    def enable_numpy_cold(self) -> None:
        """Enable a numpy-only single-point query path.

        Dumps the FourierDecoder weights, grid table, and a per-
        block-length sin-frequency table to numpy at setup, then
        rebinds `self.query` to a pure-numpy implementation that
        re-derives every block from scratch using numpy gemms.

        Crucially, `query_cold_numpy` evaluates the decoder at ONE
        t coordinate per call (not the full block) -- the dominant
        cost of `numpy_forward(whole_block)` is the F*blen
        sin-coefficient matmul, which collapses to F sins + F dot
        products when only one t is requested.

        Compared to the PyTorch `query()` path:
          - same workload (no cache, no preload, full forward)
          - same numerics (FourierDecoder math, MKL gemm)
          - ~15-25x faster per query (mostly by skipping PyTorch)

        Use case: NeaTS-style cold-cache point benchmarks where
        every call must be a true cold query.  Calling this disables
        the previously-bound `query_fast` rebind (numpy cold > python
        hit on a real cold workload).

        Limitation: assumes FourierDecoder, scalar (1-D) StandardScaler,
        and `max_aux_tokens == 0` (single-token decode).  Falls back
        to a clean error on the other decoder types -- they're not
        supported by export_neurts_cpp v1 either, so this matches the
        rest of the inference path.
        """
        dec = self.model.decoder
        cls_name = type(dec).__name__
        if cls_name != 'FourierDecoder':
            raise RuntimeError(
                f"enable_numpy_cold only supports FourierDecoder, "
                f"got {cls_name}")
        if int(getattr(dec, 'max_aux', 0)) != 0:
            raise RuntimeError(
                "enable_numpy_cold only supports max_aux_tokens == 0 "
                "(single-token decode); got max_aux="
                f"{dec.max_aux}.  Multi-token cold path is not in scope.")

        # ---- Weights -> numpy.  to_v0/to_v1 are 1-row Linears so we
        #      flatten the (1, in_dim) weight to a 1-D dot vector.
        with torch.no_grad():
            self._np_W_v0 = dec.to_v0.weight.detach().cpu().numpy() \
                .astype(np.float32).reshape(-1)
            self._np_b_v0 = float(dec.to_v0.bias.detach().cpu().numpy()[0])
            self._np_W_v1 = dec.to_v1.weight.detach().cpu().numpy() \
                .astype(np.float32).reshape(-1)
            self._np_b_v1 = float(dec.to_v1.bias.detach().cpu().numpy()[0])
            self._np_W0 = dec.to_coeff[0].weight.detach().cpu().numpy() \
                .astype(np.float32)             # [hidden, in_dim]
            self._np_b0 = dec.to_coeff[0].bias.detach().cpu().numpy() \
                .astype(np.float32)             # [hidden]
            self._np_W2 = dec.to_coeff[2].weight.detach().cpu().numpy() \
                .astype(np.float32)             # [F, hidden]
            self._np_b2 = dec.to_coeff[2].bias.detach().cpu().numpy() \
                .astype(np.float32)             # [F]
            self._np_freqs = dec.freqs.detach().cpu().numpy() \
                .astype(np.float32)             # [F]

        # ---- Grid -> numpy via the model's quantisation-aware forward.
        # We DO NOT read base_grid/patch_grid directly because grid_storage
        # may run fake-quantisation in eval mode; calling it through the
        # forward path keeps numerics aligned with the PyTorch reference.
        gs = self.model.grid_storage
        with torch.no_grad():
            n_total = int(gs.base_grid.shape[0])
            if hasattr(gs, 'patch_grid') and gs.patch_grid is not None:
                n_total += int(gs.patch_grid.shape[0])
            all_ids = torch.arange(0, n_total, dtype=torch.long,
                                   device=self.device)
            left_vec, _ = gs(all_ids, all_ids)
            self._np_grid = left_vec.detach().cpu().numpy() \
                .astype(np.float32)             # [n_total, in_dim]

        # ---- Index table & metadata as plain Python lists / scalars
        # (avoid numpy scalar -> Python int conversions on every call).
        self._np_idx_left = self.idx_left.tolist()
        self._np_idx_block_start = self.idx_block_start.tolist()
        self._np_idx_block_len = self.idx_block_len.tolist()
        self._np_min_res = int(self.min_resolution)
        self._np_num_slots = int(self.num_slots)
        self._np_T = int(self._T)

        # ---- Fallback entries: dict of (left_id -> entry).  We keep
        # the live dict reference so updates propagate, but freeze
        # local refs to (offsets list, residuals list, type, data list)
        # so the hot path doesn't repeatedly re-resolve them.
        self._np_fb_entries = (
            self.fallback_dict.entries
            if self.fallback_dict is not None else {})

        # ---- Scaler scalars
        self._np_std = float(self._std_scalar)
        self._np_mean = float(self._mean_scalar)
        self._np_apply_scaler = (self._np_std != 1.0) \
                              or (self._np_mean != 0.0)

        # ---- Rebind: subsequent acc.query(t) -> numpy cold path.
        # Preserve any earlier query_safe (set by enable_fast_query) so
        # callers can always reach the original implementation.
        if not hasattr(self, 'query_safe'):
            self.query_safe = self.query
        self.query = self.query_cold_numpy  # type: ignore[assignment]

    def query_cold_numpy(self, t: int) -> float:
        """Pure-numpy cold-cache single-point query.

        Equivalent to the reference `query()` (full forward + fallback
        + scaler) but skips PyTorch entirely.  Requires
        `enable_numpy_cold()` to have been called first.  ~150-300 ns
        per call setup overhead + ~80-150 us for the actual decoder
        math (BLAS-friendly numpy gemms on cold weights).

        No `t` bounds check: caller is responsible.
        """
        idx = t // self._np_min_res
        if idx >= self._np_num_slots:
            idx = self._np_num_slots - 1

        left_id = self._np_idx_left[idx]
        bs      = self._np_idx_block_start[idx]
        blen    = self._np_idx_block_len[idx]
        offset  = t - bs

        # ---- Forward: same math as FourierDecoder.forward, but
        # specialised to a single t coordinate.  We avoid materialising
        # the F*blen sin basis (~30 us in the whole-block path) by
        # computing F sins at one t directly (~5 us with vectorised
        # std libm in numpy).
        z = self._np_grid[left_id]                              # [in_dim]
        v0 = float(z @ self._np_W_v0) + self._np_b_v0           # scalar
        v1 = float(z @ self._np_W_v1) + self._np_b_v1           # scalar
        h = z @ self._np_W0.T + self._np_b0                     # [hidden]
        h = h / (1.0 + np.exp(-h))                              # SiLU
        a = h @ self._np_W2.T + self._np_b2                     # [F]

        # Single-point ramp + DST oscillator
        # blen=1 case: t_norm undefined; degrade to v0 (identical to
        # what the reference would output).
        if blen <= 1:
            t_norm = 0.0
        else:
            t_norm = float(offset) / float(blen - 1)
        ramp = (1.0 - t_norm) * v0 + t_norm * v1
        # vectorised: F sins + F dot
        osc = float(a @ np.sin(self._np_freqs * t_norm))
        out = ramp + osc

        # ---- Fallback overlay (single-point semantics)
        entry = self._np_fb_entries.get(left_id)
        if entry is not None:
            etype = entry['type']
            if etype == 'PATCH':
                # Sparse: only the offsets in this block's PATCH list
                # affect the output.  Offsets are typically very small
                # (a handful of bad points), so a Python scan is fine.
                offs = entry['offsets']
                ress = entry['residuals']
                for i, o in enumerate(offs):
                    if o == offset:
                        out += ress[i]
                        # offsets are unique within a PATCH entry; no
                        # need to keep scanning.
                        break
            elif etype == 'RAW':
                data = entry['data']
                if 0 <= offset < len(data):
                    out = float(data[offset])
            # else: unknown type -- silently skip (matches Python ref)

        # ---- Inverse scaler
        if self._np_apply_scaler:
            out = out * self._np_std + self._np_mean
        return out

    # ---------- preload all blocks into the cache --------------------------

    def preload_all(self) -> None:
        """Decode every unique block once and seed the LRU with the
        results.  After this call, `query(t)` for any valid `t` is a
        cache hit (no decoder forward, no host<->device sync) -- a
        single dict lookup + scalar read.

        Use case: random-access workloads where the working set fits
        in memory.  This trades cold-start time (one stacked
        `decompress_all`-style pass over the index) for hot-state
        latency (~tens of ns/query instead of ~ms/query).

        Memory cost: roughly `num_unique_blocks * base_block_size * 4`
        bytes for the decoded float32 buffers.  On the BT export this
        is ~292 blocks * 512 * 4 = 0.6 MB; trivial.

        Side effects:
          - `self._cache_size` is bumped to fit every unique block so
            no eviction happens during the seed pass.  The previous
            value is NOT restored -- if you want a smaller cache
            afterwards, set `self._cache_size = N` yourself.
          - hit / miss counters and the LRU itself are reset before
            seeding so subsequent benchmark runs see a deterministic
            warm-state baseline.
        """
        # Enumerate every unique block (one walk over the index table).
        lefts: List[int] = []
        rights: List[int] = []
        bstarts: List[int] = []
        blens: List[int] = []
        cur = 0
        n = self.num_slots
        while cur < n:
            bs = int(self.idx_block_start[cur])
            bl = int(self.idx_block_len[cur])
            lefts.append(int(self.idx_left[cur]))
            rights.append(int(self.idx_right[cur]))
            bstarts.append(bs)
            blens.append(bl)
            nxt = (bs + bl) // self.min_resolution
            if nxt <= cur:
                nxt = cur + 1
            cur = nxt

        K = len(lefts)
        if K == 0:
            return

        # Make sure the cache can hold every block.  We deliberately
        # over-allocate by 1 to leave headroom for an additional miss
        # if the user later inserts a custom block; this is harmless.
        if self._cache_size < K + 1:
            self._cache_size = K + 1

        # Reset state so post-preload benchmarks have a clean slate.
        self.clear_cache()
        self.reset_cache_stats()

        lefts_np   = np.asarray(lefts,   dtype=np.int64)
        rights_np  = np.asarray(rights,  dtype=np.int64)
        bstarts_np = np.asarray(bstarts, dtype=np.int64)
        blens_np   = np.asarray(blens,   dtype=np.int64)

        unique_lens, inv = np.unique(blens_np, return_inverse=True)

        device = self.device
        std_v = float(self._std_scalar)
        mean_v = float(self._mean_scalar)
        apply_scaler = (std_v != 1.0) or (mean_v != 0.0)

        fb = self.fallback_dict
        fb_has_any = (fb is not None and len(fb) > 0)

        for bi, blen in enumerate(unique_lens):
            mask = (inv == bi)
            sub_lefts   = lefts_np[mask]
            sub_rights  = rights_np[mask]
            sub_bstarts = bstarts_np[mask]

            left_t  = torch.as_tensor(sub_lefts,  dtype=torch.long, device=device)
            right_t = torch.as_tensor(sub_rights, dtype=torch.long, device=device)
            with torch.no_grad():
                stacked = self.model.decode_batch(left_t, right_t, int(blen))
            if stacked.dim() == 3:
                stacked = stacked.squeeze(1)
            stacked_np = stacked.detach().cpu().numpy().astype(
                np.float32, copy=False)
            if not stacked_np.flags['C_CONTIGUOUS']:
                stacked_np = np.ascontiguousarray(stacked_np)

            # Vectorised PATCH overlay (same recipe as decompress_all).
            if fb_has_any:
                row_of_lid = {int(lid): k for k, lid in enumerate(sub_lefts)}
                patch_rows: List[int] = []
                patch_offs: List[int] = []
                patch_vals: List[float] = []
                blen_i = int(blen)
                for lid, entry in fb.entries.items():
                    k = row_of_lid.get(int(lid))
                    if k is None:
                        continue
                    if entry['type'] == 'PATCH':
                        offs = entry['offsets']
                        ress = entry['residuals']
                        if not offs:
                            continue
                        patch_rows.extend([k] * len(offs))
                        patch_offs.extend(offs)
                        patch_vals.extend(ress)
                    else:  # RAW
                        data = entry['data']
                        nn = min(len(data), blen_i)
                        if nn > 0:
                            stacked_np[k, :nn] = data[:nn]

                if patch_rows:
                    rows_arr = np.asarray(patch_rows, dtype=np.intp)
                    offs_arr = np.asarray(patch_offs, dtype=np.intp)
                    vals_arr = np.asarray(patch_vals, dtype=np.float32)
                    valid = (offs_arr >= 0) & (offs_arr < blen_i)
                    if not valid.all():
                        rows_arr = rows_arr[valid]
                        offs_arr = offs_arr[valid]
                        vals_arr = vals_arr[valid]
                    np.add.at(stacked_np, (rows_arr, offs_arr), vals_arr)

            if apply_scaler:
                stacked_np = stacked_np * std_v + mean_v

            # Seed each row into the LRU.  We materialise each row
            # via .copy() because the bucket buffer goes out of scope
            # at the end of the loop iteration.
            for k in range(stacked_np.shape[0]):
                lid = int(sub_lefts[k])
                bs  = int(sub_bstarts[k])
                self._block_cache[(lid, bs)] = stacked_np[k].copy()
        # (no eviction needed: cache_size was bumped to >= K + 1.)

    # ---------- range -------------------------------------------------------

    def query_range(self, t_start: int, t_end: int) -> np.ndarray:
        t_start = int(t_start)
        t_end = int(t_end)
        if t_end <= t_start:
            return np.empty(0, dtype=np.float32)
        if t_start < 0 or t_end > self._T:
            raise IndexError(f"range [{t_start},{t_end}) out of [0, {self._T})")

        # ---- Enumerate the unique blocks the range crosses (one walk
        # over the index table, deduped by block_start).  We collect
        # per-block metadata into 4 contiguous int arrays so the
        # decode + scatter passes below can be fully vectorised.
        idx_lo = t_start // self.min_resolution
        idx_hi = (t_end - 1) // self.min_resolution
        if idx_hi >= self.num_slots:
            idx_hi = self.num_slots - 1

        lefts: List[int] = []
        rights: List[int] = []
        bstarts: List[int] = []
        blens: List[int] = []
        cur = int(idx_lo)
        while cur <= idx_hi:
            bs = int(self.idx_block_start[cur])
            bl = int(self.idx_block_len[cur])
            lefts.append(int(self.idx_left[cur]))
            rights.append(int(self.idx_right[cur]))
            bstarts.append(bs)
            blens.append(bl)
            nxt = (bs + bl) // self.min_resolution
            if nxt <= cur:
                nxt = cur + 1
            cur = nxt

        out = np.empty(t_end - t_start, dtype=np.float32)
        if not lefts:
            return out

        # ---- Always go through the multi-block fast path, even when
        # the range fits in a single block.  At K=1 the cache-driven
        # `_get_block` (which calls `decode_single`) is dominated by
        # PyTorch's per-call dispatcher cost (~3 ms on this machine);
        # `decode_batch(K=1)` walks the same Python code as K=2 and
        # has effectively the same overhead, so it is ~4x faster on
        # single-block ranges in practice.  The two are mathematically
        # identical (we benchmarked: K=1 batched is bit-equivalent to
        # K=1 single up to MKL gemm reduction order).
        lefts_np   = np.asarray(lefts,   dtype=np.int64)
        rights_np  = np.asarray(rights,  dtype=np.int64)
        bstarts_np = np.asarray(bstarts, dtype=np.int64)
        blens_np   = np.asarray(blens,   dtype=np.int64)

        unique_lens, inv = np.unique(blens_np, return_inverse=True)

        device = self.device
        std_v = float(self._std_scalar)
        mean_v = float(self._mean_scalar)
        apply_scaler = (std_v != 1.0) or (mean_v != 0.0)

        fb = self.fallback_dict
        fb_has_any = (fb is not None and len(fb) > 0)

        for bi, blen in enumerate(unique_lens):
            mask = (inv == bi)
            sub_lefts = lefts_np[mask]
            sub_rights = rights_np[mask]
            sub_bstarts = bstarts_np[mask]

            left_t  = torch.as_tensor(sub_lefts,  dtype=torch.long, device=device)
            right_t = torch.as_tensor(sub_rights, dtype=torch.long, device=device)
            with torch.no_grad():
                stacked = self.model.decode_batch(left_t, right_t, int(blen))
            if stacked.dim() == 3:
                stacked = stacked.squeeze(1)

            stacked_np = stacked.detach().cpu().numpy().astype(
                np.float32, copy=False)
            if not stacked_np.flags['C_CONTIGUOUS']:
                stacked_np = np.ascontiguousarray(stacked_np)

            if fb_has_any:
                row_of_lid = {int(lid): k for k, lid in enumerate(sub_lefts)}
                patch_rows: List[int] = []
                patch_offs: List[int] = []
                patch_vals: List[float] = []
                for lid, entry in fb.entries.items():
                    k = row_of_lid.get(int(lid))
                    if k is None:
                        continue
                    if entry['type'] == 'PATCH':
                        offs = entry['offsets']
                        ress = entry['residuals']
                        n_off = len(offs)
                        if n_off == 0:
                            continue
                        patch_rows.extend([k] * n_off)
                        patch_offs.extend(offs)
                        patch_vals.extend(ress)
                    else:  # RAW
                        data = entry['data']
                        n = min(len(data), int(blen))
                        if n > 0:
                            stacked_np[k, :n] = data[:n]

                if patch_rows:
                    rows_arr = np.asarray(patch_rows, dtype=np.intp)
                    offs_arr = np.asarray(patch_offs, dtype=np.intp)
                    vals_arr = np.asarray(patch_vals, dtype=np.float32)
                    valid = (offs_arr >= 0) & (offs_arr < int(blen))
                    if not valid.all():
                        rows_arr = rows_arr[valid]
                        offs_arr = offs_arr[valid]
                        vals_arr = vals_arr[valid]
                    np.add.at(stacked_np, (rows_arr, offs_arr), vals_arr)

            if apply_scaler:
                stacked_np = stacked_np * std_v + mean_v

            # ---- Scatter into `out` -- per row but each iteration
            # is a single slice assignment, no Python element loop.
            blen_i = int(blen)
            for k in range(stacked_np.shape[0]):
                bs = int(sub_bstarts[k])
                lo = max(t_start, bs)
                hi = min(t_end, bs + blen_i)
                if hi > lo:
                    out[lo - t_start : hi - t_start] = stacked_np[
                        k, lo - bs : hi - bs
                    ]
        return out

    # ---------- scattered batch --------------------------------------------

    def query_batch(self, times) -> np.ndarray:
        ts = np.asarray(times, dtype=np.int64).reshape(-1)
        n = ts.size
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if (ts < 0).any() or (ts >= self._T).any():
            raise IndexError("batch contains an out-of-range index")

        # ---- One stacked forward over the unique blocks the batch
        # touches; the LRU absorbs cross-call duplicates.  Compared to
        # the previous per-element loop, this collapses K independent
        # gemvs into one gemm of width K, which is exactly what
        # NumPy/MKL is good at.
        idxs = np.minimum(ts // self.min_resolution, self.num_slots - 1)
        blocks = self._decode_blocks_stacked(idxs)

        out = np.empty(n, dtype=np.float32)
        # Scatter: each block array is shared between the (possibly
        # many) input positions that landed on it, so this is a flat
        # vector of indexed reads.
        for i in range(n):
            slot_i = int(idxs[i])
            bstart = int(self.idx_block_start[slot_i])
            out[i] = blocks[i][int(ts[i]) - bstart]
        return out

    # ---------- full sequential decompression ------------------------------

    def decompress_all(self) -> np.ndarray:
        """Stitch the entire reconstructed series.

        Optimised path that **bypasses the LRU cache entirely**.  In
        a full-T pass the cache provides no value (each block is
        touched exactly once), and its per-block lookup / insertion +
        per-block fallback iteration + per-block .copy() account for
        the bulk of the wall time once `decode_batch` is batched.
        Here we collapse all of that into a handful of numpy /
        torch vector ops:

            1. Enumerate every unique block once -> 4 contiguous int
               arrays (left_ids, right_ids, block_starts, block_lens).
            2. ONE `decode_batch` per block_len bucket  (typically 1-3
               buckets total since the index table only emits a few
               distinct lengths).
            3. ONE host<-->device sync per bucket; everything past the
               sync runs in numpy.
            4. PATCH / RAW overlay: looked up only for the left_ids
               that actually have a fallback entry (set intersection),
               then applied with a small vectorised inner loop.
            5. Inverse scaler: one fused multiply-add over the whole
               bucket's contiguous (K, blen) buffer.
            6. Scatter: numpy slicing per bucket-row into `out`.

        With K = thousands of blocks this is typically 5-30x faster
        than the cache-driven path on CPU and 10-100x on GPU,
        because the per-block Python overhead is gone.
        """
        out = np.empty(self._T, dtype=np.float32)
        T = int(self._T)

        # ---- Pass 1: enumerate unique blocks (one pass over the index table).
        idx_left = self.idx_left
        idx_right = self.idx_right
        idx_block_start = self.idx_block_start
        idx_block_len = self.idx_block_len
        min_res = self.min_resolution
        n_slots = self.num_slots

        lefts: List[int] = []
        rights: List[int] = []
        bstarts: List[int] = []
        blens: List[int] = []
        cur = 0
        while cur < n_slots:
            bstart = int(idx_block_start[cur])
            blen = int(idx_block_len[cur])
            lefts.append(int(idx_left[cur]))
            rights.append(int(idx_right[cur]))
            bstarts.append(bstart)
            blens.append(blen)
            next_slot = (bstart + blen) // min_res
            if next_slot <= cur:
                next_slot = cur + 1
            cur = next_slot

        lefts_np   = np.asarray(lefts,   dtype=np.int64)
        rights_np  = np.asarray(rights,  dtype=np.int64)
        bstarts_np = np.asarray(bstarts, dtype=np.int64)
        blens_np   = np.asarray(blens,   dtype=np.int64)

        # ---- Pass 2: bucket by block_len.  np.unique gives sorted
        # unique lengths and an inverse map; we group via boolean mask.
        unique_lens, inv = np.unique(blens_np, return_inverse=True)

        device = self.device
        std_v = float(self._std_scalar)
        mean_v = float(self._mean_scalar)
        apply_scaler = (std_v != 1.0) or (mean_v != 0.0)

        fb = self.fallback_dict
        fb_has_any = (fb is not None and len(fb) > 0)

        for bi, blen in enumerate(unique_lens):
            mask = (inv == bi)
            sub_lefts = lefts_np[mask]
            sub_rights = rights_np[mask]
            sub_bstarts = bstarts_np[mask]

            left_t  = torch.as_tensor(sub_lefts,  dtype=torch.long, device=device)
            right_t = torch.as_tensor(sub_rights, dtype=torch.long, device=device)
            with torch.no_grad():
                stacked = self.model.decode_batch(left_t, right_t, int(blen))
            if stacked.dim() == 3:
                stacked = stacked.squeeze(1)

            # ONE sync for the whole bucket.
            stacked_np = stacked.detach().cpu().numpy().astype(
                np.float32, copy=False)
            # decode_batch's output may be a non-contiguous view if the
            # decoder did fancy reshapes; force-contiguous so the
            # in-place math below is always a no-copy fast path.
            if not stacked_np.flags['C_CONTIGUOUS']:
                stacked_np = np.ascontiguousarray(stacked_np)

            # ---- PATCH/RAW overlay -- only for left_ids that have an entry.
            # Vectorised path: collect ALL (row_index, offset, residual)
            # triples that fall in this bucket into 3 contiguous numpy
            # arrays, then do ONE call to np.add.at -- this replaces a
            # nested Python loop over thousands of (offset, residual)
            # pairs which was the dominant cost (>99% of wall time)
            # for full decompression on a real export with hundreds of
            # PATCH entries.
            if fb_has_any:
                # Map left_id -> row in this bucket.  This is a single
                # Python loop over K (the bucket size); cheap.
                row_of_lid = {int(lid): k for k, lid in enumerate(sub_lefts)}

                patch_rows: List[int] = []
                patch_offs: List[int] = []
                patch_vals: List[float] = []
                blen_i = int(blen)
                for lid, entry in fb.entries.items():
                    k = row_of_lid.get(int(lid))
                    if k is None:
                        continue           # entry's block lives in another bucket
                    if entry['type'] == 'PATCH':
                        # entry['offsets'] / ['residuals'] are short lists
                        # (a handful of bad points each).  We extend the
                        # global PATCH triple lists; bounds-check is
                        # vectorised after the gather.
                        offs = entry['offsets']
                        ress = entry['residuals']
                        n_off = len(offs)
                        if n_off == 0:
                            continue
                        patch_rows.extend([k] * n_off)
                        patch_offs.extend(offs)
                        patch_vals.extend(ress)
                    else:  # RAW
                        # RAW is a full-block override; treat row by row
                        # (one slice assignment per RAW entry).  RAW
                        # entries are typically <<< PATCH count and the
                        # write is one fast vector copy each.
                        data = entry['data']
                        n = min(len(data), blen_i)
                        if n > 0:
                            stacked_np[k, :n] = data[:n]

                # Apply all PATCH residuals in ONE numpy call.
                if patch_rows:
                    rows_arr = np.asarray(patch_rows, dtype=np.intp)
                    offs_arr = np.asarray(patch_offs, dtype=np.intp)
                    vals_arr = np.asarray(patch_vals, dtype=np.float32)
                    # Bound check (matches Python ref semantics: skip OOB).
                    valid = (offs_arr >= 0) & (offs_arr < blen_i)
                    if not valid.all():
                        rows_arr = rows_arr[valid]
                        offs_arr = offs_arr[valid]
                        vals_arr = vals_arr[valid]
                    # np.add.at: unbuffered scatter add, handles
                    # duplicate (row, offset) pairs correctly.
                    np.add.at(stacked_np, (rows_arr, offs_arr), vals_arr)

            # Inverse scaler (one fused vectorised pass over the whole bucket).
            if apply_scaler:
                stacked_np = stacked_np * std_v + mean_v

            # ---- Scatter into `out`.  Each row goes to a distinct
            # contiguous slice; we iterate K rows but each iteration is
            # a single np.copyto / slice assignment (no Python-level
            # element loop).
            blen_i = int(blen)
            for k in range(stacked_np.shape[0]):
                bs = int(sub_bstarts[k])
                end_t = bs + blen_i
                if end_t > T:
                    end_t = T
                length = end_t - bs
                if length > 0:
                    out[bs:end_t] = stacked_np[k, :length]
        return out

    # =========================================================================
    # Internal: decode-or-cache-hit a block
    # =========================================================================

    def _get_block(self, idx: int) -> np.ndarray:
        """Return the decoded block (in original-unit float32) for slot `idx`.

        Cache key is `(left_id, block_start)`.  Misses run one decoder forward,
        apply the optional FallbackDict patch, inverse-transform and store.
        """
        idx = int(idx)
        left_id = int(self.idx_left[idx])
        block_start = int(self.idx_block_start[idx])
        key: _BlockKey = (left_id, block_start)

        cached = self._block_cache.get(key)
        if cached is not None:
            self._block_cache.move_to_end(key)
            self._cache_hits += 1
            return cached

        right_id = int(self.idx_right[idx])
        block_len = int(self.idx_block_len[idx])

        # 1 decoder forward ?kept inside no_grad to skip autograd bookkeeping.
        with torch.no_grad():
            out = self.model.decode_single(left_id, right_id, block_len)
            # decode_single returns shape `[1, block_len]`; reduce to `[block_len]`.
            if out.dim() == 2:
                out = out.squeeze(0)

        # Tier-2 PATCH / Tier-3 RAW residuals (no-op when the dict is empty
        # or has no entry for this left_id).
        if self.fallback_dict is not None and self.fallback_dict.has(left_id):
            out = self.fallback_dict.reconstruct(left_id, out)

        # CPU + inverse StandardScaler ?original units.
        out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
        if self._std_scalar != 1.0 or self._mean_scalar != 0.0:
            out_np = out_np * self._std_scalar + self._mean_scalar

        # Insert into LRU.
        self._block_cache[key] = out_np
        if len(self._block_cache) > self._cache_size:
            self._block_cache.popitem(last=False)
        self._cache_misses += 1
        return out_np

    # ---------------------------------------------------------------------
    # Stacked decode: take many slot indices at once, decode all the
    # *unique* (left_id, block_start) blocks via ONE batched forward per
    # block_len bucket, apply per-block fallback / scaler, and populate
    # the cache.
    #
    # Returns a list of np.ndarray references in input slot order; if a
    # slot's block was already cached, that entry is reused as-is (no
    # extra decode).  This is the hot path for query_range,
    # query_batch and decompress_all when more than one block is involved.
    # ---------------------------------------------------------------------

    def _decode_blocks_stacked(self, slots: np.ndarray) -> List[np.ndarray]:
        """Decode all unique blocks covering `slots` and return per-slot arrays.

        - slots: 1-D int array of slot indices (assumed valid, in
                 [0, num_slots)).  Duplicates are fine and only trigger
                 one decode per unique block.

        Strategy:
          1. Cache hits: served immediately from the LRU.
          2. Cache misses: bucket by `block_len` so each torch
             `decode_batch(...)` call gets a uniform block_size.  A
             single fp32 [K, in_dim] gemm replaces K independent gemvs;
             this is where the 10-50x speedup over the per-block
             `decode_single` loop comes from.
          3. After decoding a bucket, apply per-block PATCH/RAW
             fallback (still per-block; the reconstruction logic is
             intrinsically sequential), inverse-scaler the whole
             bucket as one numpy multiply-add, write to cache, and
             stash a reference into the per-slot output list.
        """
        n = int(slots.shape[0])
        out: List[Optional[np.ndarray]] = [None] * n
        if n == 0:
            return []

        # -- Pass 1: cache lookup, collect unique misses keyed by left_id.
        #    "first_pos" tracks the first input position for each unique
        #    miss so we don't pay for duplicate work.
        idx_left = self.idx_left
        idx_block_start = self.idx_block_start
        idx_block_len = self.idx_block_len
        idx_right = self.idx_right
        cache = self._block_cache

        miss_first_pos: dict = {}            # left_id -> first input position
        for i in range(n):
            slot_i = int(slots[i])
            lid = int(idx_left[slot_i])
            bstart = int(idx_block_start[slot_i])
            key = (lid, bstart)
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                self._cache_hits += 1
                out[i] = cached
                continue
            if lid not in miss_first_pos:
                miss_first_pos[lid] = i

        if not miss_first_pos:
            # Everything was a hit.  out is fully populated.
            return out  # type: ignore[return-value]

        # -- Pass 2: bucket misses by block_len, run ONE decode_batch
        #    per bucket.  Most exports have very few distinct block
        #    lengths (often just `base_block_size` plus a small set of
        #    halves and quarters from progressive split), so this is
        #    typically 1 or 2 buckets.
        device = self.device
        std_v = float(self._std_scalar)
        mean_v = float(self._mean_scalar)
        apply_scaler = (std_v != 1.0) or (mean_v != 0.0)

        # Group: block_len -> (lefts, rights, first_positions)
        by_blen: dict = {}
        for lid, first_i in miss_first_pos.items():
            slot_i = int(slots[first_i])
            blen = int(idx_block_len[slot_i])
            rid = int(idx_right[slot_i])
            bstart = int(idx_block_start[slot_i])
            by_blen.setdefault(blen, ([], [], [], [])
                              )  # lefts, rights, bstarts, first_positions
            grp = by_blen[blen]
            grp[0].append(lid)
            grp[1].append(rid)
            grp[2].append(bstart)
            grp[3].append(first_i)

        for blen, (lefts, rights, bstarts, first_positions) in by_blen.items():
            left_t = torch.as_tensor(lefts,  dtype=torch.long, device=device)
            right_t = torch.as_tensor(rights, dtype=torch.long, device=device)
            with torch.no_grad():
                # decode_batch returns [K, 1, blen]; squeeze the channel
                # dim to land on [K, blen] for vectorised post-processing.
                stacked = self.model.decode_batch(left_t, right_t, blen)
            if stacked.dim() == 3:
                stacked = stacked.squeeze(1)

            # ── ONE host<->device sync for the whole bucket. ───────────
            # Critically NOT per-block: fb.reconstruct used to be invoked
            # on the device-side tensor row by row, which caused a
            # CUDA / CPU dispatch sync per block.  At K = thousands of
            # blocks that domiantes the wall time and undoes batched
            # gemm.  We pay one materialisation here, then do the
            # PATCH overlay / RAW override / inverse-scaler entirely
            # inside numpy (zero-copy, BLAS-aware).
            stacked_np = stacked.detach().cpu().numpy().astype(
                np.float32, copy=False)
            stacked_np = np.ascontiguousarray(stacked_np)

            # In-place PATCH/RAW overlay -- pure numpy, no PyTorch ops.
            # Vectorised PATCH path: collect all (row, offset, residual)
            # triples into 3 numpy arrays then call np.add.at once.
            # This mirrors the fast path used by decompress_all and
            # avoids the per-(offset, residual) Python loop that
            # dominated wall time on real exports.
            fb = self.fallback_dict
            if fb is not None and len(fb) > 0:
                patch_rows: List[int] = []
                patch_offs: List[int] = []
                patch_vals: List[float] = []
                for k, lid in enumerate(lefts):
                    entry = fb.entries.get(int(lid))
                    if entry is None:
                        continue
                    if entry['type'] == 'PATCH':
                        offs = entry['offsets']
                        ress = entry['residuals']
                        n_off = len(offs)
                        if n_off == 0:
                            continue
                        patch_rows.extend([k] * n_off)
                        patch_offs.extend(offs)
                        patch_vals.extend(ress)
                    elif entry['type'] == 'RAW':
                        data = entry['data']
                        n = min(len(data), blen)
                        if n > 0:
                            stacked_np[k, :n] = data[:n]

                if patch_rows:
                    rows_arr = np.asarray(patch_rows, dtype=np.intp)
                    offs_arr = np.asarray(patch_offs, dtype=np.intp)
                    vals_arr = np.asarray(patch_vals, dtype=np.float32)
                    valid = (offs_arr >= 0) & (offs_arr < blen)
                    if not valid.all():
                        rows_arr = rows_arr[valid]
                        offs_arr = offs_arr[valid]
                        vals_arr = vals_arr[valid]
                    np.add.at(stacked_np, (rows_arr, offs_arr), vals_arr)

            # Inverse-scaler in one fused multiply-add (also numpy).
            if apply_scaler:
                stacked_np = stacked_np * std_v + mean_v

            # Cache + scatter.  We .copy() each row out of the bucket
            # buffer so the cache entries stay alive after the bucket
            # tensor is released (np.ascontiguousarray on an already-
            # contiguous slice would only be a view, not a copy).
            for k, (lid, bstart, first_i) in enumerate(
                    zip(lefts, bstarts, first_positions)):
                row = stacked_np[k].copy()
                key = (int(lid), int(bstart))
                cache[key] = row
                self._cache_misses += 1
                out[first_i] = row

        # -- Pass 3: fan-out the unique decoded rows to every slot
        #    that landed on the same left_id (`out[i]` may still be
        #    None for duplicates of a missed left_id).
        #
        # IMPORTANT: do this BEFORE LRU eviction.  If the batch
        # touched more unique blocks than `_cache_size`, evicting
        # first would purge the very rows we still need to fan out
        # (root cause of the historical KeyError on large batches).
        if any(o is None for o in out):
            for i in range(n):
                if out[i] is not None:
                    continue
                slot_i = int(slots[i])
                lid = int(idx_left[slot_i])
                bstart = int(idx_block_start[slot_i])
                out[i] = cache[(lid, bstart)]

        # Evict if we've grown past the LRU capacity.  Eviction order
        # is insertion-order (Python dict / OrderedDict) but the loop
        # above did `move_to_end` on every hit, so the front is still
        # the least-recently-used.
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

        return out  # type: ignore[return-value]

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "version": 1,
            "T": int(self._T),
            "min_resolution": int(self.min_resolution),
            "base_block_size": int(self.base_block_size),
            "num_slots": int(self.num_slots),
            "idx_left": np.asarray(self.idx_left, dtype=np.int32),
            "idx_right": np.asarray(self.idx_right, dtype=np.int32),
            "idx_block_start": np.asarray(self.idx_block_start, dtype=np.int32),
            "idx_block_len": np.asarray(self.idx_block_len, dtype=np.int32),
            "model_state_dict": self.model.state_dict(),
            "model_kwargs": self._model_kwargs,
            "grid_kwargs": self._grid_kwargs,
            "fallback_dict_state": (
                self.fallback_dict.state_dict()
                if self.fallback_dict is not None
                else None
            ),
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "mean_scalar": self._mean_scalar,
            "std_scalar": self._std_scalar,
        }
        torch.save(payload, path)

    @classmethod
    def from_file(cls, path: str, device: str = "auto") -> "NeurTSAccessor":
        """Reconstruct an accessor from a `.pt` produced by `save()`."""
        t0 = time.perf_counter()
        payload = torch.load(path, map_location="cpu")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Re-create grid_storage.  raw_data is only used by GlobalGridStorage's
        # initial trend sampling; load_state_dict() then overwrites the entire
        # grid table, so a length-T zero buffer is sufficient.
        gk = payload["grid_kwargs"]
        T = int(payload["T"])
        dummy_raw = torch.zeros(T)
        gs = GlobalGridStorage(raw_data=dummy_raw, **gk)

        # Re-create the model and load weights.
        mk = payload["model_kwargs"]
        model = NeurTSModel(grid_storage=gs, **mk)
        # `strict=False` lets us tolerate aux-token / quantisation buffers that
        # may be absent in older checkpoints.  Mismatched param shapes still
        # raise loudly.
        missing, unexpected = model.load_state_dict(
            payload["model_state_dict"], strict=False
        )
        if missing:
            print(f"[NeurTSAccessor] load: {len(missing)} missing keys "
                  f"(first 3: {missing[:3]})")
        if unexpected:
            print(f"[NeurTSAccessor] load: {len(unexpected)} unexpected keys "
                  f"(first 3: {unexpected[:3]})")
        model.eval().to(device)

        # Optional fallback dict.
        fb_state = payload.get("fallback_dict_state")
        if fb_state is not None:
            fb = FallbackDict()
            fb.load_state_dict(fb_state)
        else:
            fb = None

        obj = cls()
        obj.model = model
        obj.device = device
        obj._dtype = next(model.parameters()).dtype
        obj.min_resolution = int(payload["min_resolution"])
        obj.base_block_size = int(payload["base_block_size"])
        obj.num_slots = int(payload["num_slots"])
        obj._T = T
        obj.idx_left = np.asarray(payload["idx_left"], dtype=np.int32)
        obj.idx_right = np.asarray(payload["idx_right"], dtype=np.int32)
        obj.idx_block_start = np.asarray(payload["idx_block_start"], dtype=np.int32)
        obj.idx_block_len = np.asarray(payload["idx_block_len"], dtype=np.int32)
        obj.fallback_dict = fb
        obj._scaler_mean = payload.get("scaler_mean")
        obj._scaler_std = payload.get("scaler_std")
        obj._mean_scalar = payload.get("mean_scalar", 0.0)
        obj._std_scalar = payload.get("std_scalar", 1.0)
        if obj._mean_scalar is None:
            obj._mean_scalar = 0.0
        if obj._std_scalar is None:
            obj._std_scalar = 1.0
        obj._model_kwargs = mk
        obj._grid_kwargs = gk
        obj._build_time = time.perf_counter() - t0
        return obj

    # =========================================================================
    # Misc
    # =========================================================================

    def __repr__(self) -> str:
        try:
            n_patch = (
                len(self.fallback_dict) if self.fallback_dict is not None else 0
            )
        except Exception:
            n_patch = "?"
        return (
            f"NeurTSAccessor(T={self._T}, num_slots={self.num_slots}, "
            f"min_resolution={self.min_resolution}, "
            f"base_block_size={self.base_block_size}, "
            f"fallback_entries={n_patch}, cache_size={self._cache_size})"
        )


# ---------------------------------------------------------------------------
# Re-export `generate_query_sets` from accessor_base so main_neurts.py can do
#     from neurts_accessor import generate_query_sets
# without ever caring about the rand_ac/ subdirectory layout.
# ---------------------------------------------------------------------------
__all__ = ["NeurTSAccessor", "generate_query_sets"]


# ---------------------------------------------------------------------------
# CLI helper -- mirror neats_accessor.py's __main__ so a fair side-by-side
# can be invoked as two separate one-liners (no orchestrator needed):
#
#   python neurts_accessor.py <ckpt.pt> ./query_sets/
#   python rand_ac/neats_accessor.py <dump.neatsexp> ./query_sets/
#
# The shared `query_sets/` directory + identical `--seed` is what makes the
# comparison apples-to-apples: both backends consume the exact same .npy
# index files generated once by `generate_query_sets`.
#
# `--bytes_per_value` defaults to 4 (NeurTS returns float32).  NeaTS' default
# is 8 (int64).  When comparing the two side-by-side, prefer the
# `avg_ns_per_query` column in the report -- that one is dtype-independent.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NeaTS-style benchmarks against a NeurTS checkpoint. "
                    "Reads fixed query indices from <query_dir> so that the "
                    "exact same workload can be replayed against any other "
                    "BaseAccessor subclass (e.g. NeaTSAccessor).",
    )
    parser.add_argument("ckpt",       help="path to <setting>_accessor.pt")
    parser.add_argument("query_dir",  help="directory containing fixed query sets "
                                           "(generated by generate_query_sets)")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--bytes_per_value", type=int, default=4,
                        help="NeurTS returns float32 (4 bytes). Set to 8 only "
                             "if you want a like-for-like MB/s comparison "
                             "with a baseline that emits int64.")
    parser.add_argument("--n_point",         type=int, default=1_000_000)
    parser.add_argument("--n_range",         type=int, default=10_000)
    parser.add_argument("--rounds_point",    type=int, default=10)
    parser.add_argument("--rounds_full",     type=int, default=50)
    parser.add_argument("--cache",           type=int, default=64,
                        help="LRU capacity (matches NeurTS C++ default).")
    parser.add_argument("--mode", default="per_round_cold",
                        choices=("per_round_cold", "live_cache", "cold_each"),
                        help="Cache policy during the point benchmark. "
                             "'per_round_cold' (default) clears the LRU at the "
                             "start of each round but keeps it warm within the "
                             "round -- this matches what NeaTS reports.  "
                             "'cold_each' clears before every single query "
                             "(strict cold-miss; will be much slower for NeurTS).")
    parser.add_argument("--device", default="cpu",
                        choices=("cpu", "cuda", "auto"),
                        help="Torch device for the decoder.  Use 'cpu' for the "
                             "language-vs-language comparison with the Python "
                             "NeaTS port (the only fair setting).")
    args = parser.parse_args()

    print(f"Loading NeurTS checkpoint from {args.ckpt} ...")
    t0 = time.perf_counter()
    acc = NeurTSAccessor.from_file(args.ckpt, device=args.device)
    t1 = time.perf_counter()
    acc._cache_size = int(args.cache)
    acc.clear_cache()
    acc.reset_cache_stats()
    print(f"  loaded T={acc.T:,}  num_slots={acc.num_slots:,}  "
          f"in {(t1 - t0):.2f}s   (cache={acc._cache_size})")

    # Auto-generate query sets if the requested `point_N{n}_seed{seed}.npy`
    # is missing -- mirrors what neats_accessor.py does.  This keeps the
    # 'two separate commands, same query dir' usage friction-free.
    point_file = os.path.join(args.query_dir,
                              f"point_N{args.n_point}_seed{args.seed}.npy")
    if not os.path.exists(point_file):
        print(f"[Benchmark] Generating query sets in {args.query_dir} ...")
        generate_query_sets(args.query_dir, total_length=acc.T,
                            seed=args.seed,
                            n_point=args.n_point, n_range=args.n_range)

    acc.run_all_benchmarks(
        query_dir       = args.query_dir,
        bytes_per_value = args.bytes_per_value,
        seed            = args.seed,
        n_point         = args.n_point,
        n_range         = args.n_range,
        rounds_point    = args.rounds_point,
        rounds_full     = args.rounds_full,
        mode            = args.mode,
    )
