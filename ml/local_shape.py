"""Source-only, budgeted quadrant raking of the native R08 cell density.

Selected cells retain their decoded R08 beta/copula shape inside each quarter,
rescaled to its exact source count. This is classical density raking, not an
accuracy guarantee. Selection uses source quadrant disagreement only. Query
evaluation reads the native synopsis and this packet, never raw points/labels.
The counts describe half-open source partitions. A closed query ending at a
split plane can also include raw atoms assigned to the adjacent partition;
stored quarter mass therefore does not promise exact closed-query truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np

from . import bounded_sidecar as bounded
from ._distribution_math import boundary_probability

MAGIC = b"SLSHP001"
VERSION = 1
HEADER = struct.Struct("<8sHHIIIII16s12sI")
SOURCE = struct.Struct("<II")
HEADER_BYTES = HEADER.size
SOURCE_BYTES = SOURCE.size
SELECTED_CELL_BYTES = 12
DENOMINATOR_TOLERANCE = 1e-12
MODES = ("conditional_r08", "empirical_uniform")
# The high-half bit is first x, then y; raw midpoint points belong to high.
QUADRANTS = np.array([[0., .5, 0., .5], [0., .5, .5, 1.],
                      [.5, 1., 0., .5], [.5, 1., .5, 1.]])
UINT32_MAX = (1 << 32) - 1


def _integer(value, name, minimum=0):
    return bounded._integer(value, name, minimum, UINT32_MAX)


def _native(histograms, encoded_cells):
    histograms, encoded_cells = list(histograms), list(encoded_cells)
    if not histograms or len(histograms) != len(encoded_cells):
        raise ValueError("Expected matching nonempty native histogram/moment lists")
    hists, encoded, masses, sizes = [], [], [], []
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", len(histograms)))
    for hist, values in zip(histograms, encoded_cells):
        # This validator preserves native grid order and exact parent masses.
        bounded.components_from_histogram_moments(hist, values)
        rate, bx, by, counts = hist
        bx, by = np.asarray(bx, float), np.asarray(by, float)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Invalid native histogram rate")
        mass = bounded._masses(np.asarray(counts).ravel(), np.asarray(counts).size,
                               positive=False)
        values = np.asarray(values)
        if values.dtype != np.uint8 or values.shape not in ((len(mass), 5), np.asarray(counts).shape + (5,)):
            raise ValueError("Expected the native five-byte encoded moments for every cell")
        values = values.reshape(-1, 5)
        bounded.decode_moments(values)
        h = (float(rate), bx.copy(), by.copy(), mass.reshape(np.asarray(counts).shape))
        hists.append(h)
        encoded.append(values.copy())
        masses.append(mass)
        sizes.append(len(mass))
        digest.update(struct.pack("<IId", *h[3].shape, float(rate)))
        for array, dtype in ((bx, "<f8"), (by, "<f8"), (mass, "<u8"), (values, "u1")):
            digest.update(np.asarray(array, dtype=dtype).tobytes())
    if sum(sizes) > UINT32_MAX:
        raise OverflowError("Native cell identities exceed uint32")
    return hists, encoded, np.concatenate(masses), np.array(sizes), digest.digest()[:16]


def quadrant_counts_from_points(points, hist):
    """One raw chunk -> Nx4 counts, matching R09 arithmetic parent ownership.

    This is a construction helper only. Add returned uint64 arrays across raw
    chunks, then pass them to :func:`build_selection`. Midpoint equality uses
    the high half after the same local-coordinate clipping as ``_sufficient``.
    """
    from .sidecar_workflow import _sufficient
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 source points")
    _sufficient(points, hist)  # Owns the original arithmetic and roundoff guards.
    _, bx, by, counts = hist
    bx, by = np.asarray(bx), np.asarray(by)
    nx, ny = np.asarray(counts).shape
    ix = ((points[:, 0]-bx[0])/((bx[-1]-bx[0])/nx)).astype(np.int64)
    iy = ((points[:, 1]-by[0])/((by[-1]-by[0])/ny)).astype(np.int64)
    u = np.clip((points[:, 0]-bx[ix])/np.diff(bx)[ix], 0., 1.)
    v = np.clip((points[:, 1]-by[iy])/np.diff(by)[iy], 0., 1.)
    quarter = 2*(u >= .5).astype(np.int64)+(v >= .5)
    return np.bincount(4*(ix*ny+iy)+quarter, minlength=4*nx*ny).reshape(-1, 4).astype(np.uint64)


def _quadrant_probabilities(encoded):
    moments = bounded.decode_moments(encoded).astype(float)
    return boundary_probability(np.tile(QUADRANTS, (len(moments), 1)),
                                np.repeat(moments, 4, axis=0),
                                "beta_gaussian_copula").reshape(-1, 4)


def build_selection(native_histograms, encoded_cells, quadrant_counts,
                    total_extra_budget, *, mode="conditional_r08"):
    """Return a complete correction packet within the incremental byte cap.

    Rank nonempty cells by ``N * .5 * sum(abs(empirical_fraction-R08_fraction))``.
    Ties use source order then row-major native cell id. This single fixed rule
    has no query geometry or truth input. Header, source table, cell bitmap, and
    all three uint32 counts per selected cell are included in the returned bytes.
    """
    budget = _integer(total_extra_budget, "total_extra_budget")
    if mode not in MODES:
        raise ValueError("Unknown local shape mode")
    hists, encoded, mass, sizes, identity = _native(native_histograms, encoded_cells)
    arrays = list(quadrant_counts)
    if len(arrays) != len(hists):
        raise ValueError("Expected source quadrant counts for every native histogram")
    counts = []
    for values, size, hist in zip(arrays, sizes, hists):
        values = np.asarray(values)
        if values.shape not in ((int(size), 4), hist[3].shape + (4,)):
            raise ValueError("Expected four quadrant counts per native cell")
        q = bounded._masses(values.reshape(-1), 4*int(size), positive=False).reshape(-1, 4)
        if (q > UINT32_MAX).any():
            raise OverflowError("Source quadrant count exceeds uint32")
        counts.append(q)
    counts = np.concatenate(counts)
    if not np.array_equal(counts.sum(axis=1), mass):
        raise ValueError("Quadrant counts must sum to each exact native parent count")
    bitmap_bytes = (len(mass)+7)//8
    fixed = HEADER_BYTES + SOURCE_BYTES*len(hists) + bitmap_bytes
    if fixed > budget:
        raise ValueError("Incremental budget cannot hold the header, source table and cell bitmap")
    active = np.flatnonzero(mass)
    probability = _quadrant_probabilities(np.concatenate(encoded)[active])
    score = .5*np.abs(counts[active].astype(float)-mass[active, None]*probability).sum(axis=1)
    positive = score > 0
    ranked = active[positive][np.lexsort((active[positive], -score[positive]))]
    selected = np.sort(ranked[:(budget-fixed)//SELECTED_CELL_BYTES])
    bitmap = np.zeros(len(mass), np.uint8)
    bitmap[selected] = 1
    bitmap = np.packbits(bitmap, bitorder="little").tobytes()
    offsets = np.r_[0, np.cumsum(sizes)]
    table = b"".join(SOURCE.pack(int(n), int(((selected >= a) & (selected < b)).sum()))
                     for n, a, b in zip(sizes, offsets[:-1], offsets[1:]))
    body = table + bitmap + counts[selected, :3].astype("<u4").tobytes()
    header = HEADER.pack(MAGIC, VERSION, MODES.index(mode), len(hists), len(mass),
                         len(selected), bitmap_bytes, budget, identity, bytes(12), 0)
    return header[:-4]+struct.pack("<I", zlib.crc32(header[:-4]+body))+body


@dataclass(frozen=True)
class Selection:
    selected_ids: np.ndarray
    quadrant_counts: np.ndarray
    mode: str
    budget_bytes: int
    payload_bytes: int
    source_sizes: tuple


def decode_selection(native_histograms, encoded_cells, payload):
    """Validate all packet identities, lengths, padding, CRC and exact counts."""
    _, _, mass, sizes, identity = _native(native_histograms, encoded_cells)
    payload = bytes(payload)
    if len(payload) < HEADER_BYTES:
        raise ValueError("Truncated local shape header")
    magic, version, mode, sources, cells, n, nb, budget, fingerprint, reserved, crc = HEADER.unpack_from(payload)
    expected = HEADER_BYTES+SOURCE_BYTES*sources+nb+SELECTED_CELL_BYTES*n
    if (magic != MAGIC or version != VERSION or mode >= len(MODES) or reserved != bytes(12)
            or sources != len(sizes) or cells != len(mass) or fingerprint != identity
            or nb != (cells+7)//8 or expected != len(payload) or len(payload) > budget
            or zlib.crc32(payload[:HEADER_BYTES-4]+payload[HEADER_BYTES:]) != crc):
        raise ValueError("Invalid local shape packet identity, version, length, budget or CRC")
    table = [SOURCE.unpack_from(payload, HEADER_BYTES+i*SOURCE_BYTES) for i in range(sources)]
    offset = HEADER_BYTES+SOURCE_BYTES*sources
    bits = np.unpackbits(np.frombuffer(payload, np.uint8, count=nb, offset=offset), bitorder="little")
    if bits[cells:].any() or sum(b for _, b in table) != n:
        raise ValueError("Noncanonical local shape bitmap or source counts")
    selected = np.flatnonzero(bits[:cells])
    if len(selected) != n or (mass[selected] == 0).any():
        raise ValueError("Invalid selected cell bitmap")
    offsets = np.r_[0, np.cumsum(sizes)]
    for (size, count), actual, a, b in zip(table, sizes, offsets[:-1], offsets[1:]):
        if size != actual or count != ((selected >= a) & (selected < b)).sum():
            raise ValueError("Local shape source table does not match bitmap/native cells")
    first = np.frombuffer(payload, dtype="<u4", count=3*n, offset=offset+nb).reshape(-1, 3).astype(np.uint64)
    partial = first.sum(axis=1)
    if (partial > mass[selected]).any():
        raise ValueError("Stored quadrant counts exceed their native parent count")
    fourth = mass[selected]-partial
    if (fourth > UINT32_MAX).any():
        raise ValueError("Implicit fourth quadrant count exceeds uint32")
    counts = np.column_stack((first, fourth))
    return Selection(bounded._immutable(selected, np.int64), bounded._immutable(counts, np.uint64),
                     MODES[mode], budget, len(payload), tuple(map(int, sizes)))


def selection_audit(native_histograms, encoded_cells, quadrant_counts, payload):
    """Offline column arrays for freezing the source-only ranking before labels.

    These diagnostics are construction evidence, not required inference state.
    The packet must match the fixed source-only selection rule exactly.
    """
    hists, encoded, mass, sizes, _ = _native(native_histograms, encoded_cells)
    counts = list(quadrant_counts)
    selection = decode_selection(hists, encoded, payload)
    expected = build_selection(hists, encoded, counts, selection.budget_bytes, mode=selection.mode)
    if bytes(payload) != expected:
        raise ValueError("Packet does not match the fixed source-only selection rule")
    counts = np.concatenate([np.asarray(values).reshape(-1, 4) for values in counts])
    probability = _quadrant_probabilities(np.concatenate(encoded))
    score = .5*np.abs(counts.astype(float)-mass[:, None]*probability).sum(axis=1)
    ids = np.arange(len(mass))
    candidates = np.flatnonzero((mass > 0) & (score > 0))
    order = candidates[np.lexsort((candidates, -score[candidates]))]
    rank = np.full(len(mass), -1, np.int64)
    rank[order] = np.arange(len(order))
    selected = np.zeros(len(mass), bool)
    selected[selection.selected_ids] = True
    result = dict(source_index=np.repeat(np.arange(len(sizes)), sizes),
                  cell_id=np.concatenate([np.arange(size) for size in sizes]),
                  global_cell_id=ids, parent_count=mass, priority=score,
                  source_rank=rank, selected=selected)
    for j in range(4):
        result[f"quadrant_count_{j}"] = counts[:, j]
        result[f"r08_quadrant_probability_{j}"] = probability[:, j]
    return result


class LocalShapeRuntime:
    """Native synopsis plus immutable selected-cell counts; no raw-data handle."""

    def __init__(self, native_histograms, encoded_cells, payload):
        hists, encoded, mass, _, _ = _native(native_histograms, encoded_cells)
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
        self._denominator = _quadrant_probabilities(np.concatenate(encoded)[self.selection.selected_ids])
        self._disagreement = .5*np.abs(self.selection.quadrant_counts.astype(float)
                                       -mass[self.selection.selected_ids, None]*self._denominator).sum(axis=1)
        self.native_bytes = sum(24+8*(len(h[1])+len(h[2])+h[3].size)+64+5*h[3].size for h in hists)
        self.auxiliary_bytes = self.native_bytes+len(self.payload)
        self.nbytes = self.auxiliary_bytes
        self.original_histogram = None

    @property
    def total_mass(self):
        return self.components.total_mass

    def _corrected(self, local, selected, *, conditional=True):
        """Vectorized conditional and uniform contributions for selected rows."""
        q = QUADRANTS[None, :, :]
        r = local[:, None, :]
        inter = np.stack((np.maximum(r[:, :, 0], q[:, :, 0]), np.minimum(r[:, :, 1], q[:, :, 1]),
                          np.maximum(r[:, :, 2], q[:, :, 2]), np.minimum(r[:, :, 3], q[:, :, 3])), axis=2)
        valid = (inter[:, :, 0] <= inter[:, :, 1]) & (inter[:, :, 2] <= inter[:, :, 3])
        full = ((r[:, :, 0] <= q[:, :, 0]) & (r[:, :, 1] >= q[:, :, 1])
                & (r[:, :, 2] <= q[:, :, 2]) & (r[:, :, 3] >= q[:, :, 3]))
        inter[~valid] = 0.
        uniform = np.maximum(inter[:, :, 1]-inter[:, :, 0], 0.)*np.maximum(inter[:, :, 3]-inter[:, :, 2], 0.)*4
        uniform[full] = 1.
        counts = self.selection.quadrant_counts[selected].astype(float)
        uniform_count = (counts*uniform).sum(axis=1)
        if not conditional:
            return uniform_count, uniform_count, np.zeros(len(local))
        denominator = self._denominator[selected]
        moments = self._moments[np.flatnonzero(self._selected_index >= 0)[selected]]
        numerator = boundary_probability(inter.reshape(-1, 4), np.repeat(moments, 4, axis=0),
                                         "beta_gaussian_copula").reshape(-1, 4)
        conditional = np.divide(numerator, denominator, out=uniform.copy(),
                                where=denominator > DENOMINATOR_TOLERANCE)
        conditional = np.where(valid, np.clip(conditional, 0., 1.), 0.)
        # Exact containment wins even for an analytic-zero quadrant.
        conditional[full] = 1.
        fallback_mass = (counts*((denominator <= DENOMINATOR_TOLERANCE) & valid & ~full)).sum(axis=1)
        return (counts*conditional).sum(axis=1), uniform_count, fallback_mass

    def _query_blocks(self, queries):
        bounds, mass = self.components.bounds, self.components.mass
        block = max(1, min(256, 262144//max(1, len(mass))))
        for start in range(0, len(queries), block):
            stop = min(start+block, len(queries))
            b, q = bounds[:, None, :], queries[None, start:stop, :]
            touched = ((b[:, :, 0] <= q[:, :, 1]) & (b[:, :, 1] >= q[:, :, 0])
                       & (b[:, :, 2] <= q[:, :, 3]) & (b[:, :, 3] >= q[:, :, 2]))
            full = ((q[:, :, 0] <= b[:, :, 0]) & (b[:, :, 1] <= q[:, :, 1])
                    & (q[:, :, 2] <= b[:, :, 2]) & (b[:, :, 3] <= q[:, :, 3]))
            ci, qi = np.nonzero(touched & ~full)
            local = np.column_stack(((queries[start+qi, 0]-bounds[ci, 0])/(bounds[ci, 1]-bounds[ci, 0]),
                                     (queries[start+qi, 1]-bounds[ci, 0])/(bounds[ci, 1]-bounds[ci, 0]),
                                     (queries[start+qi, 2]-bounds[ci, 2])/(bounds[ci, 3]-bounds[ci, 2]),
                                     (queries[start+qi, 3]-bounds[ci, 2])/(bounds[ci, 3]-bounds[ci, 2])))
            local = np.clip(local, 0., 1.).astype(np.float32).astype(float)
            yield start, stop, (full*mass[:, None]).sum(axis=0), ci, qi, local

    def query_features(self, rectangles, histogram=None):
        """Return source/query-only arrays; optional H uses the fixed original H."""
        queries = bounded._rectangles(rectangles)
        names = ("reference_base", "reference_hybrid", "reference_uniform", "full_mass", "boundary_mass",
                 "corrected_boundary_mass", "corrected_boundary_estimate_base", "corrected_boundary_estimate_hybrid",
                 "selected_boundary_cells", "source_disagreement_mass", "zero_cdf_boundary_mass")
        result = {name: np.zeros(len(queries), float) for name in names}
        mass = self.components.mass
        for start, stop, full_mass, ci, qi, local in self._query_blocks(queries):
            result["full_mass"][start:stop] = full_mass
            base = mass[ci].astype(float)*boundary_probability(local, self._moments[ci], "beta_gaussian_copula")
            hybrid, uniform = base.copy(), base.copy()
            selected = self._selected_index[ci]
            chosen = selected >= 0
            k = stop-start
            def add(name, values, indices=qi):
                result[name][start:stop] += np.bincount(indices, weights=values, minlength=k)
            add("boundary_mass", mass[ci].astype(float))
            if chosen.any():
                hybrid[chosen], uniform[chosen], fallback = self._corrected(local[chosen], selected[chosen])
                j = qi[chosen]
                add("corrected_boundary_mass", mass[ci[chosen]].astype(float), j)
                add("corrected_boundary_estimate_base", base[chosen], j)
                add("corrected_boundary_estimate_hybrid", hybrid[chosen], j)
                add("selected_boundary_cells", np.ones(len(j)), j)
                add("source_disagreement_mass", self._disagreement[selected[chosen]], j)
                add("zero_cdf_boundary_mass", fallback, j)
            for name, values in (("reference_base", base), ("reference_hybrid", hybrid), ("reference_uniform", uniform)):
                add(name, values)
                result[name][start:stop] += result["full_mass"][start:stop]
        result["correction_delta"] = result["reference_hybrid"]-result["reference_base"]
        result["correction_abs_delta"] = np.abs(result["correction_delta"])
        result["corrected_boundary_fraction"] = np.divide(result["corrected_boundary_mass"], result["boundary_mass"],
                                                            out=np.zeros(len(queries)), where=result["boundary_mass"] > 0)
        histogram = self.original_histogram if histogram is None else histogram
        if histogram is not None:
            from .sidecar_workflow import histogram_count
            result["H"] = np.array([histogram_count(rect, histogram) for rect in queries])
        return result

    def batch_reference(self, rectangles, *, mode=None):
        mode = self.selection.mode if mode is None else mode
        if mode not in MODES:
            raise ValueError("Unknown local shape mode")
        queries = bounded._rectangles(rectangles)
        result = np.zeros(len(queries), float)
        mass = self.components.mass
        for start, stop, full_mass, ci, qi, local in self._query_blocks(queries):
            selected = self._selected_index[ci]
            chosen = selected >= 0
            values = np.zeros(len(ci))
            if (~chosen).any():
                values[~chosen] = mass[ci[~chosen]].astype(float)*boundary_probability(
                    local[~chosen], self._moments[ci[~chosen]], "beta_gaussian_copula")
            if chosen.any():
                values[chosen] = self._corrected(local[chosen], selected[chosen],
                                                conditional=mode == MODES[0])[0]
            result[start:stop] = full_mass+np.bincount(qi, weights=values, minlength=stop-start)
        return result

    def reference_count(self, rectangle, *, mode=None):
        rectangle = np.asarray(rectangle, float)
        if rectangle.shape != (4,):
            raise ValueError("Expected one finite query rectangle")
        return float(self.batch_reference(rectangle[None, :], mode=mode)[0])


def load_runtime(native_histograms, encoded_cells, payload):
    return LocalShapeRuntime(native_histograms, encoded_cells, payload)


def reference_count(runtime, rectangle, *, mode=None):
    return runtime.reference_count(rectangle, mode=mode)


def batch_reference(runtime, rectangles, *, mode=None):
    return runtime.batch_reference(rectangles, mode=mode)


def load_native(directory):
    """Read/validate the persistent native R08 geometry and moment codec."""
    from . import sidecar_workflow as workflow
    workflow.load_structured_reference(directory)
    paths = sorted(Path(directory).glob("src_*.hist"), key=lambda p: int(p.stem.split("_")[-1]))
    histograms, encoded = [], []
    for path in paths:
        sid = int(path.stem.split("_")[-1])
        hist = workflow.load_histogram(path)
        histograms.append(hist)
        payload = path.with_name(f"src_{sid}-moments.bin").read_bytes()
        encoded.append(np.frombuffer(payload, np.uint8, offset=workflow._MOMENT_HEADER.size).reshape(-1, 5).copy())
    return histograms, encoded


def save_snapshot(directory, native_histograms, encoded_cells, payload, *, original_histogram=None):
    """Persist only native codecs + correction, returning every saved byte."""
    from . import sidecar_workflow as workflow
    runtime = LocalShapeRuntime(native_histograms, encoded_cells, payload)
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
    (directory/"correction.bin").write_bytes(runtime.payload)
    if original_histogram is not None:
        workflow.save_histogram(directory/"original_H.hist", original_histogram)
    loaded = load_snapshot(directory)
    return loaded.nbytes


def load_snapshot(directory):
    from .sidecar_workflow import load_histogram
    directory = Path(directory)
    histograms, encoded = load_native(directory)
    runtime = LocalShapeRuntime(histograms, encoded, (directory/"correction.bin").read_bytes())
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
    return dict(method="classical source-quadrant density raking", mode=list(MODES),
                selection="descending exact-parent-mass weighted total variation of empirical and native R08 quadrant fractions; source/cell tie break",
                quadrant_order="low-x/low-y, low-x/high-y, high-x/low-y, high-x/high-y; raw midpoint equality belongs high",
                split_plane_atoms="stored partition mass is not closed-query raw truth when raw atoms lie on split planes; decoded byte means cannot equal the .5 split",
                denominator_tolerance=DENOMINATOR_TOLERANCE,
                zero_denominator_policy="uniform inside that quarter; full quarter returns its exact count",
                counts="three uint32 source counts, fourth is exact native parent count minus their sum",
                bytes="64-byte header + 8/source + ceil(all native cells/8) bitmap + 12/selected cell; native geometry/moments charged separately",
                inference_raw_rows=0, query_truth_at_selection=False, query_truth_at_inference=False,
                accuracy_guarantee=False, novelty_claim=False)
