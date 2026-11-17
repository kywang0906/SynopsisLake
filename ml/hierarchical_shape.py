"""Budgeted native-cell refinement with four child moment distributions.

Each selected native R08 cell stores three exact uint32 quadrant counts and
four five-byte moment summaries in child-local coordinates. Geometry is
inherited from the native grid. This classical hierarchical moment basis is
an empirical candidate, with no accuracy or novelty claim. Child quarters are
source partitions: zero-area intersections with adjacent quarters are excluded.
Closed raw queries through split-plane atoms can therefore differ from stored
partition quotas, as in the original quadrant-raking control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib

import numpy as np

from . import bounded_sidecar as bounded
from . import local_shape as local
from ._distribution_math import boundary_probability

MAGIC = b"SLHIE001"
VERSION = 1
HEADER = local.HEADER
SOURCE = local.SOURCE
HEADER_BYTES = HEADER.size
SOURCE_BYTES = SOURCE.size
SELECTED_CELL_BYTES = 32
RECORD_DTYPE = np.dtype([("counts", "<u4", (3,)), ("child_encoded", "u1", (4, 5))], align=False)
MODES = ("hierarchical_r08", "empirical_uniform")
QUADRANTS = local.QUADRANTS
UINT32_MAX = local.UINT32_MAX
load_native = local.load_native


def accumulate_child_moments(points, hist):
    """Return additive Nx4 counts and Nx4x5 child-coordinate raw sums.

    Parent ownership and local clipping exactly follow the native arithmetic
    construction. Sums are ``u, v, u*u, v*v, u*v`` within the assigned quarter.
    This helper is construction-only; no runtime API accepts raw points.
    """
    from .sidecar_workflow import _sufficient
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 source points")
    _sufficient(points, hist)
    _, bx, by, counts = hist
    bx, by = np.asarray(bx), np.asarray(by)
    nx, ny = np.asarray(counts).shape
    ix = ((points[:, 0]-bx[0])/((bx[-1]-bx[0])/nx)).astype(np.int64)
    iy = ((points[:, 1]-by[0])/((by[-1]-by[0])/ny)).astype(np.int64)
    u = np.clip((points[:, 0]-bx[ix])/np.diff(bx)[ix], 0., 1.)
    v = np.clip((points[:, 1]-by[iy])/np.diff(by)[iy], 0., 1.)
    high_x, high_y = (u >= .5).astype(np.int64), (v >= .5).astype(np.int64)
    child = 4*(ix*ny+iy)+2*high_x+high_y
    u, v = np.clip(2*u-high_x, 0., 1.), np.clip(2*v-high_y, 0., 1.)
    n = 4*nx*ny
    count = np.bincount(child, minlength=n).astype(np.uint64).reshape(-1, 4)
    sums = np.column_stack([np.bincount(child, weights=value, minlength=n)
                            for value in (u, v, u*u, v*v, u*v)]).reshape(-1, 4, 5)
    return count, sums


def encode_child_moments(counts, sums):
    """Quantize additive child statistics using the unchanged R08 codec."""
    from .sidecar_workflow import _moments
    counts, sums = np.asarray(counts), np.asarray(sums, float)
    if counts.ndim != 2 or counts.shape[1] != 4 or sums.shape != counts.shape+(5,):
        raise ValueError("Expected Nx4 child counts and Nx4x5 additive sums")
    mass = bounded._masses(counts.ravel(), counts.size, positive=False)
    flat = sums.reshape(-1, 5)
    if (not np.isfinite(flat).all() or (flat < -1e-10).any()
            or (flat > mass[:, None]+1e-10).any()
            or ((mass == 0)[:, None] & (flat != 0)).any()):
        raise ValueError("Invalid normalized child sufficient statistics")
    return bounded.encode_moments(_moments(mass, flat)).reshape(counts.shape+(5,))


def _source_state(hists, sizes, mass, quadrant_counts, child_encoded):
    counts, encoded = [], []
    quadrant_counts, child_encoded = list(quadrant_counts), list(child_encoded)
    if len(quadrant_counts) != len(hists) or len(child_encoded) != len(hists):
        raise ValueError("Expected counts and child moment arrays for every source")
    for hist, size, q, e in zip(hists, sizes, quadrant_counts, child_encoded):
        q, e = np.asarray(q), np.asarray(e)
        if q.shape not in ((int(size), 4), hist[3].shape+(4,)):
            raise ValueError("Expected four quadrant counts per native cell")
        q = bounded._masses(q.reshape(-1), 4*int(size), positive=False).reshape(-1, 4)
        if (q > UINT32_MAX).any():
            raise OverflowError("Source quadrant count exceeds uint32")
        if e.dtype != np.uint8 or e.shape not in ((int(size), 4, 5), hist[3].shape+(4, 5)):
            raise ValueError("Expected four encoded five-byte child moments per native cell")
        e = e.reshape(-1, 4, 5)
        bounded.decode_moments(e.reshape(-1, 5))
        counts.append(q)
        encoded.append(e)
    counts, encoded = np.concatenate(counts), np.concatenate(encoded)
    if not np.array_equal(counts.sum(axis=1), mass):
        raise ValueError("Child counts must sum to each exact native parent count")
    return counts, encoded


def _priority(mass, native_encoded, counts, priority):
    if priority is None:
        probability = local._quadrant_probabilities(np.concatenate(native_encoded))
        return .5*np.abs(counts.astype(float)-mass[:, None]*probability).sum(axis=1)
    try:
        values = np.asarray(priority)
    except ValueError:
        values = np.concatenate([np.asarray(p).reshape(-1) for p in priority])
    if values.shape != mass.shape and values.ndim > 1:
        values = values.reshape(-1)
    if values.shape != mass.shape or values.dtype.kind == "b":
        raise ValueError("Priority must supply one numeric score per native cell")
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Priority scores must be finite and nonnegative")
    return values


def build_selection(native_histograms, encoded_cells, quadrant_counts, child_encoded,
                    total_extra_budget, priority=None, *, mode="hierarchical_r08"):
    """Serialize the highest-priority nonempty native cells within the byte cap.

    Default priority is parent-mass-weighted quadrant total variation versus
    native R08. Optional ``priority`` overrides this score with a fixed vector
    derived by the caller from source summaries and TRAIN geometry only. The
    core has no query label input. Zero scores are omitted; ties use native ID.
    """
    budget = local._integer(total_extra_budget, "total_extra_budget")
    if mode not in MODES:
        raise ValueError("Unknown hierarchical shape mode")
    hists, encoded, mass, sizes, identity = local._native(native_histograms, encoded_cells)
    counts, child = _source_state(hists, sizes, mass, quadrant_counts, child_encoded)
    score = _priority(mass, encoded, counts, priority)
    bitmap_bytes = (len(mass)+7)//8
    fixed = HEADER_BYTES+SOURCE_BYTES*len(hists)+bitmap_bytes
    if fixed > budget:
        raise ValueError("Incremental budget cannot hold header, source table and cell bitmap")
    active = np.flatnonzero((mass > 0) & (score > 0))
    ranked = active[np.lexsort((active, -score[active]))]
    selected = np.sort(ranked[:(budget-fixed)//SELECTED_CELL_BYTES])
    bits = np.zeros(len(mass), np.uint8)
    bits[selected] = 1
    offsets = np.r_[0, np.cumsum(sizes)]
    table = b"".join(SOURCE.pack(int(n), int(((selected >= a) & (selected < b)).sum()))
                     for n, a, b in zip(sizes, offsets[:-1], offsets[1:]))
    records = np.empty(len(selected), RECORD_DTYPE)
    records["counts"], records["child_encoded"] = counts[selected, :3], child[selected]
    body = table+np.packbits(bits, bitorder="little").tobytes()+records.tobytes()
    header = HEADER.pack(MAGIC, VERSION, MODES.index(mode), len(hists), len(mass),
                         len(selected), bitmap_bytes, budget, identity, bytes(12), 0)
    return header[:-4]+struct.pack("<I", zlib.crc32(header[:-4]+body))+body


@dataclass(frozen=True)
class Selection:
    selected_ids: np.ndarray
    quadrant_counts: np.ndarray
    child_encoded: np.ndarray
    mode: str
    budget_bytes: int
    payload_bytes: int
    source_sizes: tuple


def decode_selection(native_histograms, encoded_cells, payload):
    """Validate the complete native identity, record counts and moment codec."""
    _, _, mass, sizes, identity = local._native(native_histograms, encoded_cells)
    payload = bytes(payload)
    if len(payload) < HEADER_BYTES:
        raise ValueError("Truncated hierarchical shape header")
    magic, version, mode, sources, cells, n, nb, budget, fingerprint, reserved, crc = HEADER.unpack_from(payload)
    expected = HEADER_BYTES+SOURCE_BYTES*sources+nb+SELECTED_CELL_BYTES*n
    if (magic != MAGIC or version != VERSION or mode >= len(MODES) or reserved != bytes(12)
            or sources != len(sizes) or cells != len(mass) or fingerprint != identity
            or nb != (cells+7)//8 or expected != len(payload) or len(payload) > budget
            or zlib.crc32(payload[:HEADER_BYTES-4]+payload[HEADER_BYTES:]) != crc):
        raise ValueError("Invalid hierarchical packet identity, version, length, budget or CRC")
    table = [SOURCE.unpack_from(payload, HEADER_BYTES+i*SOURCE_BYTES) for i in range(sources)]
    offset = HEADER_BYTES+SOURCE_BYTES*sources
    bits = np.unpackbits(np.frombuffer(payload, np.uint8, count=nb, offset=offset), bitorder="little")
    selected = np.flatnonzero(bits[:cells])
    if bits[cells:].any() or sum(b for _, b in table) != n or len(selected) != n or (mass[selected] == 0).any():
        raise ValueError("Invalid hierarchical source count or selected cell bitmap")
    offsets = np.r_[0, np.cumsum(sizes)]
    for (size, count), actual, a, b in zip(table, sizes, offsets[:-1], offsets[1:]):
        if size != actual or count != ((selected >= a) & (selected < b)).sum():
            raise ValueError("Hierarchical source table disagrees with native cells")
    records = np.frombuffer(payload, RECORD_DTYPE, count=n, offset=offset+nb)
    first = records["counts"].astype(np.uint64)
    partial = first.sum(axis=1)
    if (partial > mass[selected]).any():
        raise ValueError("Stored child counts exceed native parent count")
    fourth = mass[selected]-partial
    if (fourth > UINT32_MAX).any():
        raise ValueError("Implicit fourth child count exceeds uint32")
    child = records["child_encoded"]
    bounded.decode_moments(child.reshape(-1, 5))
    return Selection(bounded._immutable(selected, np.int64),
                     bounded._immutable(np.column_stack((first, fourth)), np.uint64),
                     bounded._immutable(child, np.uint8), MODES[mode], budget,
                     len(payload), tuple(map(int, sizes)))


class HierarchicalShapeRuntime(local.LocalShapeRuntime):
    """Native R08 plus selected quarter-local distributions, without raw data."""

    def __init__(self, native_histograms, encoded_cells, payload):
        hists, encoded, mass, _, _ = local._native(native_histograms, encoded_cells)
        self.selection = decode_selection(hists, encoded, payload)
        self.payload = bytes(payload)
        self.histograms, self.encoded_cells = hists, encoded
        parts = [bounded.components_from_histogram_moments(h, e) for h, e in zip(hists, encoded)]
        self.components = bounded.Components(np.concatenate([p.bounds for p in parts]),
                                             np.concatenate([p.mass for p in parts]),
                                             np.concatenate([p.encoded for p in parts]))
        self._moments = self.components.moments.astype(float)
        active = np.flatnonzero(mass)
        self._selected_index = np.full(len(active), -1, np.int64)
        self._selected_index[np.searchsorted(active, self.selection.selected_ids)] = np.arange(len(self.selection.selected_ids))
        self._child_moments = bounded.decode_moments(self.selection.child_encoded.reshape(-1, 5)).astype(float).reshape(-1, 4, 5)
        probability = local._quadrant_probabilities(np.concatenate(encoded)[self.selection.selected_ids])
        self._disagreement = .5*np.abs(self.selection.quadrant_counts.astype(float)
                                       -mass[self.selection.selected_ids, None]*probability).sum(axis=1)
        self.native_bytes = sum(24+8*(len(h[1])+len(h[2])+h[3].size)+64+5*h[3].size for h in hists)
        self.auxiliary_bytes = self.native_bytes+len(self.payload)
        self.nbytes = self.auxiliary_bytes
        self.original_histogram = None

    def _corrected(self, local_rectangles, selected, *, conditional=True):
        r, q = local_rectangles[:, None, :], QUADRANTS[None, :, :]
        inter = np.stack((np.maximum(r[:, :, 0], q[:, :, 0]), np.minimum(r[:, :, 1], q[:, :, 1]),
                          np.maximum(r[:, :, 2], q[:, :, 2]), np.minimum(r[:, :, 3], q[:, :, 3])), axis=2)
        valid = (inter[:, :, 0] < inter[:, :, 1]) & (inter[:, :, 2] < inter[:, :, 3])
        full = ((r[:, :, 0] <= q[:, :, 0]) & (r[:, :, 1] >= q[:, :, 1])
                & (r[:, :, 2] <= q[:, :, 2]) & (r[:, :, 3] >= q[:, :, 3]))
        inter[~valid] = 0.
        child_rect = np.clip((inter-q)*2, 0., 1.)
        # Subtract each axis's lower endpoint, rather than the upper endpoint.
        child_rect[:, :, 1] = np.clip((inter[:, :, 1]-q[:, :, 0])*2, 0., 1.)
        child_rect[:, :, 3] = np.clip((inter[:, :, 3]-q[:, :, 2])*2, 0., 1.)
        child_rect = child_rect.astype(np.float32).astype(float)
        uniform = (child_rect[:, :, 1]-child_rect[:, :, 0])*(child_rect[:, :, 3]-child_rect[:, :, 2])
        uniform[~valid], uniform[full] = 0., 1.
        counts = self.selection.quadrant_counts[selected].astype(float)
        uniform_count = (counts*uniform).sum(axis=1)
        if not conditional:
            return uniform_count, uniform_count, np.zeros(len(selected))
        probability = full.astype(float)
        partial = valid & ~full & (counts > 0)
        if partial.any():
            probability[partial] = boundary_probability(child_rect[partial], self._child_moments[selected][partial],
                                                        "beta_gaussian_copula")
        return (counts*probability).sum(axis=1), uniform_count, np.zeros(len(selected))

    def batch_reference(self, rectangles, *, mode=None):
        mode = self.selection.mode if mode is None else mode
        if mode == "conditional_r08":
            mode = MODES[0]
        if mode not in MODES:
            raise ValueError("Unknown hierarchical shape mode")
        return super().batch_reference(rectangles, mode="conditional_r08" if mode == MODES[0] else MODES[1])


def load_runtime(native_histograms, encoded_cells, payload):
    return HierarchicalShapeRuntime(native_histograms, encoded_cells, payload)


def reference_count(runtime, rectangle, *, mode=None):
    return runtime.reference_count(rectangle, mode=mode)


def batch_reference(runtime, rectangles, *, mode=None):
    return runtime.batch_reference(rectangles, mode=mode)


def selection_audit(native_histograms, encoded_cells, quadrant_counts, child_encoded, payload, priority=None):
    """Return source/geometry-only ranking evidence for offline persistence."""
    hists, encoded, mass, sizes, _ = local._native(native_histograms, encoded_cells)
    counts, child = _source_state(hists, sizes, mass, quadrant_counts, child_encoded)
    selection = decode_selection(hists, encoded, payload)
    expected = build_selection(hists, encoded, quadrant_counts, child_encoded,
                               selection.budget_bytes, priority=priority, mode=selection.mode)
    if expected != bytes(payload):
        raise ValueError("Packet does not match the stated fixed source/geometry priority")
    score = _priority(mass, encoded, counts, priority)
    active = np.flatnonzero((mass > 0) & (score > 0))
    order = active[np.lexsort((active, -score[active]))]
    rank = np.full(len(mass), -1, np.int64)
    rank[order] = np.arange(len(order))
    selected = np.zeros(len(mass), bool)
    selected[selection.selected_ids] = True
    result = dict(source_index=np.repeat(np.arange(len(sizes)), sizes),
                  cell_id=np.concatenate([np.arange(size) for size in sizes]),
                  global_cell_id=np.arange(len(mass)), parent_count=mass,
                  priority=score, source_rank=rank, selected=selected)
    for j in range(4):
        result[f"quadrant_count_{j}"] = counts[:, j]
    return result


def save_snapshot(directory, native_histograms, encoded_cells, payload, *, original_histogram=None):
    from . import sidecar_workflow as workflow
    runtime = HierarchicalShapeRuntime(native_histograms, encoded_cells, payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Snapshot directory must be empty")
    for sid, (hist, encoded) in enumerate(zip(runtime.histograms, runtime.encoded_cells)):
        workflow.save_histogram(directory/f"src_{sid}.hist", hist)
        _, bx, by, counts = hist
        moment_payload = workflow._MOMENT_HEADER.pack(b"SLMOM001", 1, 5, 0, sid, *counts.shape,
                                                     bx[0], bx[-1], by[0], by[-1])+encoded.tobytes()
        (directory/f"src_{sid}-moments.bin").write_bytes(moment_payload)
    (directory/"hierarchical.bin").write_bytes(runtime.payload)
    if original_histogram is not None:
        workflow.save_histogram(directory/"original_H.hist", original_histogram)
    return load_snapshot(directory).nbytes


def load_snapshot(directory):
    from .sidecar_workflow import load_histogram
    directory = Path(directory)
    hists, encoded = load_native(directory)
    runtime = HierarchicalShapeRuntime(hists, encoded, (directory/"hierarchical.bin").read_bytes())
    original = directory/"original_H.hist"
    if original.exists():
        runtime.original_histogram = load_histogram(original)
    if any(not path.is_file() for path in directory.iterdir()):
        raise ValueError("Snapshot contains unexpected subdirectories")
    runtime.nbytes = sum(path.stat().st_size for path in directory.iterdir())
    if runtime.nbytes != runtime.auxiliary_bytes+(original.stat().st_size if original.exists() else 0):
        raise ValueError("Snapshot has uncharged or unexpected files")
    return runtime


def protocol():
    return dict(method="classical hierarchical child moment distributions", modes=list(MODES),
                source_construction="same arithmetic native parent ownership; midpoint belongs high; child coordinates normalized to quarter",
                default_selection="parent-mass-weighted quadrant TV against native R08; native ID tie break",
                optional_priority="caller-provided nonnegative cell priority from source metadata and TRAIN geometry only, overriding default TV",
                bytes="64-byte header +8/source +ceil(all native cells/8) bitmap +32/selected (three uint32 masses and four five-byte child moments)",
                split_plane_atoms="zero-area adjacent child intersections excluded; stored partition quota is not closed raw truth at split-plane atoms",
                inference_raw_rows=0, query_truth_at_inference=False, accuracy_guarantee=False, novelty_claim=False)
