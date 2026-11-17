"""Hard-budget, child-only mixtures using the frozen R08 distribution reference.

The persisted state is a 64-byte versioned header and 45 bytes per component:
four float64 bounds, an exact uint64 mass, and five quantized local moments.
Every pooled component is quantized immediately. Merging reads only serialized
children and optional frozen training rectangles, never raw rows or query labels.

Compression is a heuristic over adjacent components in a deterministic Morton
ordering. ``geometry`` uses weighted mean separation; ``workload`` minimizes
local reference-count distortion on a bounded training workload. It preserves
mass, not the distribution, and is deliberately not claimed to be associative.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import numbers
import struct

import numpy as np

from ._distribution_math import boundary_probability

MAGIC = b"SLMIX001"
VERSION = 1
HEADER = struct.Struct("<8sHHIQQ4d")
RECORD_DTYPE = np.dtype([("bounds", "<f8", (4,)), ("mass", "<u8"),
                         ("encoded", "u1", (5,))], align=False)
HEADER_BYTES = HEADER.size
COMPONENT_BYTES = RECORD_DTYPE.itemsize
UINT64_MAX = (1 << 64) - 1
DEFAULT_MAX_TRAINING_QUERIES = 64


def _integer(value, name, minimum=0, maximum=UINT64_MAX):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(name + " outside supported range")
    return value


def _immutable(array, dtype, shape=None):
    value = np.asarray(array, dtype=dtype)
    if shape is not None and value.shape != shape:
        raise ValueError("Unexpected array shape")
    # A bytes-backed array cannot be made writeable again by its caller.
    return np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)


def _masses(values, n, *, positive=True):
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype.kind == "b":
        raise ValueError("Expected one exact integer mass per component")
    if array.dtype.kind in "iu":
        if array.dtype.kind == "i" and (array < 0).any():
            raise ValueError("Negative component mass")
        result = np.asarray(array, dtype=np.uint64)
    elif array.dtype.kind == "f":
        if (not np.isfinite(array).all() or (array < 0).any()
                or (array > 2 ** 53).any() or (array != np.floor(array)).any()):
            raise ValueError("Floating masses must be exact nonnegative integers <=2**53")
        result = array.astype(np.uint64)
    else:
        exact = [_integer(value, "component mass") for value in array.tolist()]
        result = np.asarray(exact, dtype=np.uint64)
    if positive and (result == 0).any():
        raise ValueError("Stored components must have positive mass")
    total = sum(map(int, result))
    if total > UINT64_MAX:
        raise OverflowError("Total component mass exceeds uint64")
    return result


def encode_moments(moments):
    """Original R08 five-byte codec: means/255, variances/1020, cov/508."""
    values = np.asarray(moments, float)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Expected finite N by 5 moments")
    tolerance = 1e-10
    if (((values[:, :2] < -tolerance) | (values[:, :2] > 1 + tolerance)).any()
            or ((values[:, 2:4] < -tolerance) | (values[:, 2:4] > .25 + tolerance)).any()
            or (np.abs(values[:, 4]) > .25 + tolerance).any()):
        raise ValueError("Normalized moments outside physical ranges")
    result = np.empty(values.shape, np.uint8)
    result[:, :2] = np.rint(255 * np.clip(values[:, :2], 0, 1)).astype(np.uint8)
    result[:, 2:4] = np.rint(1020 * np.clip(values[:, 2:4], 0, .25)).astype(np.uint8)
    covariance = np.rint(508 * np.clip(values[:, 4], -.25, .25)).astype(np.int8)
    result[:, 4] = covariance.view(np.uint8)
    return result


def decode_moments(encoded):
    """Decode using R08's float32 feature contract and reserved covariance byte."""
    values = np.asarray(encoded)
    if values.ndim != 2 or values.shape[1] != 5 or values.dtype != np.uint8:
        raise ValueError("Expected N by 5 uint8 moment bytes")
    covariance = np.ascontiguousarray(values[:, 4]).view(np.int8)
    if (covariance == -128).any():
        raise ValueError("Reserved covariance byte -128")
    return np.column_stack((values[:, :2].astype(float) / 255,
                            values[:, 2:4].astype(float) / 1020,
                            covariance.astype(float) / 508)).astype(np.float32)


@dataclass(frozen=True, eq=False)
class Components:
    bounds: np.ndarray
    mass: np.ndarray
    encoded: np.ndarray

    def __post_init__(self):
        bounds = np.asarray(self.bounds, float)
        if (bounds.ndim != 2 or bounds.shape[1] != 4 or not np.isfinite(bounds).all()
                or (bounds[:, 0] >= bounds[:, 1]).any()
                or (bounds[:, 2] >= bounds[:, 3]).any()
                or not np.isfinite(bounds[:, 1] - bounds[:, 0]).all()
                or not np.isfinite(bounds[:, 3] - bounds[:, 2]).all()):
            raise ValueError("Expected finite, positive component bounds")
        n = len(bounds)
        # Signed zero denotes the same boundary and must not provide an
        # input-order-dependent tie between otherwise identical records.
        bounds = np.where(bounds == 0., 0., bounds)
        mass = _masses(self.mass, n)
        encoded = np.asarray(self.encoded)
        if encoded.shape != (n, 5):
            raise ValueError("Moment bytes do not match component count")
        decode_moments(encoded)
        object.__setattr__(self, "bounds", _immutable(bounds, np.float64))
        object.__setattr__(self, "mass", _immutable(mass, np.uint64))
        object.__setattr__(self, "encoded", _immutable(encoded, np.uint8))
        if n:
            _root(self)

    def __len__(self):
        return len(self.mass)

    @property
    def moments(self):
        return _immutable(decode_moments(self.encoded), np.float32)

    @property
    def total_mass(self):
        return sum(map(int, self.mass))


@dataclass(frozen=True, eq=False)
class Sidecar:
    components: Components
    budget_bytes: int
    payload_bytes: int

    def __post_init__(self):
        if not isinstance(self.components, Components):
            raise TypeError("Sidecar components must be immutable Components")
        budget = _integer(self.budget_bytes, "budget_bytes", HEADER_BYTES)
        size = _integer(self.payload_bytes, "payload_bytes", HEADER_BYTES)
        if size != HEADER_BYTES + COMPONENT_BYTES * len(self.components) or size > budget:
            raise ValueError("Sidecar metadata does not match its charged components")

    @property
    def total_mass(self):
        return self.components.total_mass

    def __len__(self):
        return len(self.components)


def _root(components):
    if not len(components):
        return (0., 0., 0., 0.)
    b = components.bounds
    root = (float(b[:, 0].min()), float(b[:, 1].max()),
            float(b[:, 2].min()), float(b[:, 3].max()))
    if not np.isfinite([root[1] - root[0], root[3] - root[2]]).all():
        raise ValueError("Component union has nonfinite extent")
    return root


def _canonical_order(components):
    b, m, e = components.bounds, components.mass, components.encoded
    return np.lexsort((e[:, 4], e[:, 3], e[:, 2], e[:, 1], e[:, 0], m,
                       b[:, 3], b[:, 2], b[:, 1], b[:, 0]))


def _take(components, ids):
    return Components(components.bounds[ids], components.mass[ids], components.encoded[ids])


def _components(value):
    if isinstance(value, Components):
        return value
    if isinstance(value, Sidecar):
        return value.components
    if isinstance(value, (bytes, bytearray, memoryview)):
        return decode(value).components
    raise TypeError("Expected Components, Sidecar, or serialized bytes")


def components_from_histogram_moments(hist, moments, variance=None, covariance=None):
    """Build source components without raw access from exact histograms/moments.

    ``hist`` is the existing ``(rate, bx, by, counts)`` tuple. ``moments`` is
    Nx5 (or nx×ny×5) local moments, or Nx2 means followed by Nx2 variance and
    N covariance arguments. An Nx5 uint8 array is already encoded R08 state.
    Empty histogram cells consume no mixture component.
    """
    try:
        _, bx, by, counts = hist
    except (TypeError, ValueError) as error:
        raise ValueError("Expected (rate,bx,by,counts) histogram") from error
    bx, by, counts = np.asarray(bx, float), np.asarray(by, float), np.asarray(counts)
    if (bx.ndim != 1 or by.ndim != 1 or len(bx) < 2 or len(by) < 2
            or not np.isfinite(bx).all() or not np.isfinite(by).all()
            or not (np.diff(bx) > 0).all() or not (np.diff(by) > 0).all()
            or counts.shape != (len(bx) - 1, len(by) - 1)):
        raise ValueError("Invalid histogram grid/count shape")
    mass = _masses(counts.ravel(), counts.size, positive=False)
    values = np.asarray(moments)
    if variance is not None or covariance is not None:
        if variance is None or covariance is None:
            raise ValueError("Both variance and covariance are required with means")
        values = np.column_stack((values.reshape(-1, 2),
                                  np.asarray(variance).reshape(-1, 2),
                                  np.asarray(covariance).reshape(-1)))
    else:
        if values.shape not in ((counts.size, 5), counts.shape + (5,)):
            raise ValueError("Expected five local moments per histogram cell")
        values = values.reshape(-1, 5)
    if values.shape != (counts.size, 5):
        raise ValueError("Moment arrays do not match histogram cells")
    cells = np.flatnonzero(mass)
    ix, iy = cells // counts.shape[1], cells % counts.shape[1]
    bounds = np.column_stack((bx[ix], bx[ix + 1], by[iy], by[iy + 1]))
    encoded = values[cells] if values.dtype == np.uint8 else encode_moments(values[cells])
    return Components(bounds, mass[cells], encoded)


def encode(components, budget_bytes):
    """Serialize complete components; refuse to hide excess state beyond the cap."""
    components = _components(components)
    budget = _integer(budget_bytes, "budget_bytes", HEADER_BYTES)
    size = HEADER_BYTES + COMPONENT_BYTES * len(components)
    if len(components) >= 2 ** 32:
        raise OverflowError("Component count exceeds uint32")
    if size > budget:
        raise ValueError("Components exceed serialized byte budget")
    order = _canonical_order(components)
    records = np.empty(len(components), RECORD_DTYPE)
    records["bounds"] = components.bounds[order]
    records["mass"] = components.mass[order]
    records["encoded"] = components.encoded[order]
    return HEADER.pack(MAGIC, VERSION, 0, len(components), budget,
                       components.total_mass, *_root(components)) + records.tobytes()


def decode(payload):
    """Read and validate the entire standalone codec, including charged bounds."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Serialized sidecar must be bytes")
    payload = bytes(payload)
    if len(payload) < HEADER_BYTES:
        raise ValueError("Truncated sidecar header")
    magic, version, flags, n, budget, total, *root = HEADER.unpack_from(payload)
    size = HEADER_BYTES + COMPONENT_BYTES * n
    if (magic != MAGIC or version != VERSION or flags or budget < HEADER_BYTES
            or len(payload) != size or size > budget):
        raise ValueError("Invalid sidecar header, version, flags, length or budget")
    records = np.frombuffer(payload, RECORD_DTYPE, offset=HEADER_BYTES, count=n)
    components = Components(records["bounds"], records["mass"], records["encoded"])
    if components.total_mass != total or tuple(root) != _root(components):
        raise ValueError("Sidecar mass or bounding root mismatch")
    if not np.array_equal(_canonical_order(components), np.arange(n)):
        raise ValueError("Noncanonical component order")
    return Sidecar(components, int(budget), size)


def _rectangles(rectangles):
    values = np.asarray(rectangles, float)
    if values.size == 0:
        values = np.empty((0, 4), float)
    if (values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all()
            or (values[:, 0] > values[:, 1]).any()
            or (values[:, 2] > values[:, 3]).any()):
        raise ValueError("Expected finite closed query rectangles Nx4")
    return values


def _contributions(bounds, mass, moments, rectangles):
    """Vectorized component/query counts; intersections match R08 float32."""
    n, qn = len(bounds), len(rectangles)
    result = np.zeros((n, qn), float)
    if not n or not qn:
        return result
    b, q = bounds[:, None, :], rectangles[None, :, :]
    touched = ((b[:, :, 0] <= q[:, :, 1]) & (b[:, :, 1] >= q[:, :, 0])
               & (b[:, :, 2] <= q[:, :, 3]) & (b[:, :, 3] >= q[:, :, 2]))
    full = ((q[:, :, 0] <= b[:, :, 0]) & (b[:, :, 1] <= q[:, :, 1])
            & (q[:, :, 2] <= b[:, :, 2]) & (b[:, :, 3] <= q[:, :, 3]))
    i, j = np.nonzero(full)
    result[i, j] = mass[i].astype(float)
    i, j = np.nonzero(touched & ~full)
    if len(i):
        cell, query = bounds[i], rectangles[j]
        local = np.column_stack(((query[:, 0] - cell[:, 0]) / (cell[:, 1] - cell[:, 0]),
                                 (query[:, 1] - cell[:, 0]) / (cell[:, 1] - cell[:, 0]),
                                 (query[:, 2] - cell[:, 2]) / (cell[:, 3] - cell[:, 2]),
                                 (query[:, 3] - cell[:, 2]) / (cell[:, 3] - cell[:, 2])))
        local = np.clip(local, 0., 1.).astype(np.float32).astype(float)
        probability = boundary_probability(local, moments[i], "beta_gaussian_copula")
        result[i, j] = mass[i].astype(float) * probability
    return result


def batch_reference(summary, rectangles):
    components, queries = _components(summary), _rectangles(rectangles)
    result = np.zeros(len(queries), float)
    moments = components.moments.astype(float)
    # Bound temporary component×query arrays for arbitrary external workloads.
    block = max(1, min(256, 262144 // max(1, len(components))))
    for start in range(0, len(queries), block):
        stop = start + block
        result[start:stop] = _contributions(components.bounds, components.mass,
                                             moments, queries[start:stop]).sum(axis=0)
    return result


def reference_count(summary, rectangle):
    query = np.asarray(rectangle, float)
    if query.shape != (4,):
        raise ValueError("Expected one finite query rectangle")
    return float(batch_reference(summary, query[None, :])[0])


def _training_queries(rectangles, maximum):
    maximum = _integer(maximum, "max_training_queries", 1, 4096)
    if rectangles is None:
        raise ValueError("Workload compression requires frozen training rectangles")
    values = _rectangles(rectangles)
    if not len(values):
        raise ValueError("Workload compression requires nonempty training geometry")
    # Geometry-only deduplication/order makes selection independent of row order.
    values = np.unique(values, axis=0)
    if len(values) > maximum:
        ids = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
        values = values[ids]
    return values


def _morton_order(components):
    root = _root(components)
    b = components.bounds
    x = ((b[:, 0] - root[0]) + .5 * (b[:, 1] - b[:, 0])) / (root[1] - root[0])
    y = ((b[:, 2] - root[2]) + .5 * (b[:, 3] - b[:, 2])) / (root[3] - root[2])
    def spread(values):
        values = np.rint(np.clip(values, 0, 1) * 65535).astype(np.uint32)
        values = (values | (values << 8)) & np.uint32(0x00FF00FF)
        values = (values | (values << 4)) & np.uint32(0x0F0F0F0F)
        values = (values | (values << 2)) & np.uint32(0x33333333)
        return (values | (values << 1)) & np.uint32(0x55555555)
    z = spread(x) | (spread(y) << 1)
    canonical = _canonical_order(components)
    return canonical[np.argsort(z[canonical], kind="stable")]


def _pool(bound_a, mass_a, moment_a, bound_b, mass_b, moment_b):
    """Pool decoded child moments in new local coordinates, then re-encode."""
    bound = np.array((min(bound_a[0], bound_b[0]), max(bound_a[1], bound_b[1]),
                      min(bound_a[2], bound_b[2]), max(bound_a[3], bound_b[3])))
    total = int(mass_a) + int(mass_b)
    if total > UINT64_MAX:
        raise OverflowError("Pooled mass exceeds uint64")
    width = bound[[1, 3]] - bound[[0, 2]]
    def local(child, moment):
        scale = (child[[1, 3]] - child[[0, 2]]) / width
        mean = (child[[0, 2]] - bound[[0, 2]]) / width + scale * moment[:2]
        return mean, scale * scale * moment[2:4], float(scale.prod() * moment[4])
    ma, va, ca = local(bound_a, moment_a)
    mb, vb, cb = local(bound_b, moment_b)
    wa, wb = int(mass_a) / total, int(mass_b) / total
    mean = wa * ma + wb * mb
    da, db = ma - mean, mb - mean
    variance = wa * (va + da * da) + wb * (vb + db * db)
    covariance = wa * (ca + da[0] * da[1]) + wb * (cb + db[0] * db[1])
    pooled = np.r_[mean, variance, covariance][None, :]
    # Independent quantization can violate the bounded-variable moment relation
    # by a rounding step. Keep the original physical codec ranges and clamps.
    pooled[:, :2] = np.clip(pooled[:, :2], 0., 1.)
    pooled[:, 2:4] = np.clip(pooled[:, 2:4], 0., .25)
    pooled[:, 4] = np.clip(pooled[:, 4], -.25, .25)
    encoded = encode_moments(pooled)[0]
    return bound, total, encoded, decode_moments(encoded[None, :])[0].astype(float)


def compress(components, budget_bytes, training_rectangles=None, mode="workload",
             *, max_training_queries=DEFAULT_MAX_TRAINING_QUERIES):
    """Return a standalone bounded payload using only supplied summary state.

    The neighbor graph begins as a Morton-ordered chain. A pooled node replaces
    its two neighbors and only its two incident edge costs are recomputed.
    Workload edge loss is mean squared local reference difference divided by
    ``max(initial_child_mixture_reference,1)`` squared. Normalizers are frozen
    before compression, and no truth or H is accepted by this interface.
    """
    components = _components(components)
    budget = _integer(budget_bytes, "budget_bytes", HEADER_BYTES)
    if mode == "query_distortion":
        mode = "workload"
    if mode not in ("geometry", "workload"):
        raise ValueError("Unknown compression mode")
    # Validate optional inputs even when a summary already fits, so an Nx5
    # labelled table is never silently accepted as a geometry argument.
    if training_rectangles is not None:
        training_rectangles = _rectangles(training_rectangles)
    max_training_queries = _integer(max_training_queries, "max_training_queries", 1, 4096)
    capacity = (budget - HEADER_BYTES) // COMPONENT_BYTES
    if len(components) and capacity < 1:
        raise ValueError("Budget cannot store one nonempty component")
    if len(components) <= capacity:
        return encode(components, budget)
    queries = (_training_queries(training_rectangles, max_training_queries)
               if mode == "workload" else np.empty((0, 4), float))
    components = _take(components, _morton_order(components))
    n = len(components)
    allocated = 2 * n - 1
    bounds = np.empty((allocated, 4), float)
    mass = np.empty(allocated, np.uint64)
    encoded = np.empty((allocated, 5), np.uint8)
    moments = np.empty((allocated, 5), float)
    bounds[:n], mass[:n], encoded[:n] = components.bounds, components.mass, components.encoded
    moments[:n] = components.moments
    active = np.zeros(allocated, bool)
    active[:n] = True
    previous = np.arange(allocated, dtype=np.int64) - 1
    following = np.arange(allocated, dtype=np.int64) + 1
    following[n - 1] = -1
    contributions = np.empty((allocated, len(queries)), float)
    contributions[:n] = _contributions(bounds[:n], mass[:n], moments[:n], queries)
    normalizer = np.maximum(contributions[:n].sum(axis=0), 1.)
    root = np.asarray(_root(components))
    extent = root[[1, 3]] - root[[0, 2]]
    heap, candidates = [], {}

    def add_edge(a, b):
        if a < 0 or b < 0:
            return
        candidate = _pool(bounds[a], mass[a], moments[a], bounds[b], mass[b], moments[b])
        bd, weight, code, moment = candidate
        # Ward separation in global-root-normalized coordinates; all operations
        # remain stable for large offsets in the original coordinate system.
        mua = ((bounds[a, [0, 2]] - root[[0, 2]])
               + (bounds[a, [1, 3]] - bounds[a, [0, 2]]) * moments[a, :2]) / extent
        mub = ((bounds[b, [0, 2]] - root[[0, 2]])
               + (bounds[b, [1, 3]] - bounds[b, [0, 2]]) * moments[b, :2]) / extent
        geometry = (int(mass[a]) / weight) * int(mass[b]) * float(np.sum((mua - mub) ** 2))
        counts = _contributions(bd[None, :], np.asarray([weight], np.uint64),
                                moment[None, :], queries)[0]
        if mode == "workload":
            delta = (counts - contributions[a] - contributions[b]) / normalizer
            cost = float(np.mean(delta * delta))
        else:
            cost = geometry
        if not np.isfinite(cost) or not np.isfinite(geometry):
            raise ValueError("Nonfinite compression cost")
        candidates[(a, b)] = candidate + (counts,)
        heapq.heappush(heap, (cost, geometry, a, b))

    for a in range(n - 1):
        add_edge(a, a + 1)
    remaining, next_id = n, n
    while remaining > capacity:
        while heap:
            _, _, a, b = heapq.heappop(heap)
            if active[a] and active[b] and following[a] == b:
                break
        else:
            raise RuntimeError("Compression neighbor graph exhausted")
        candidate = candidates.pop((a, b))
        pa, nb = int(previous[a]), int(following[b])
        candidates.pop((pa, a), None)
        candidates.pop((b, nb), None)
        node = next_id
        next_id += 1
        bounds[node], mass[node], encoded[node], moments[node], contributions[node] = candidate
        active[a] = active[b] = False
        active[node] = True
        previous[node], following[node] = pa, nb
        if pa >= 0:
            following[pa] = node
        if nb >= 0:
            previous[nb] = node
        add_edge(pa, node)
        add_edge(node, nb)
        remaining -= 1
    ids = np.flatnonzero(active[:next_id])
    result = Components(bounds[ids], mass[ids], encoded[ids])
    if result.total_mass != components.total_mass:
        raise RuntimeError("Compression failed exact mass conservation")
    return encode(result, budget)


def build_leaf(components, budget_bytes, training_rectangles=None, mode="workload",
               *, max_training_queries=DEFAULT_MAX_TRAINING_QUERIES):
    """Compress source summary inputs; this function never accepts a raw path."""
    return compress(components, budget_bytes, training_rectangles, mode,
                    max_training_queries=max_training_queries)


def merge_bytes(left, right, budget_bytes, training_rectangles=None, mode="workload",
                *, max_training_queries=DEFAULT_MAX_TRAINING_QUERIES):
    """Decode two persisted children and emit a bounded, child-only parent."""
    # Require bytes even though other public reference helpers accept Components.
    a, b = decode(left).components, decode(right).components
    combined = Components(np.concatenate((a.bounds, b.bounds)),
                          np.concatenate((a.mass, b.mass)),
                          np.concatenate((a.encoded, b.encoded)))
    return compress(combined, budget_bytes, training_rectangles, mode,
                    max_training_queries=max_training_queries)


def protocol():
    """Small JSON-compatible description for experiment manifests."""
    return dict(codec="SLMIX001-v1", header_bytes=HEADER_BYTES,
                component_bytes=COMPONENT_BYTES, mass="exact uint64",
                moments="R08 five-byte normalized mean/variance/covariance codec",
                reference="R08 beta marginals + Gaussian copula, fixed16 quadrature",
                intersection_precision="R08 float32 normalized query intersections",
                neighbor_graph="deterministic Morton chain, local pooled-node replacement",
                workload_loss="mean squared normalized local reference-count distortion",
                workload_normalizer="max(initial decoded child-mixture reference,1)",
                training_selection="unique lexicographic geometry, evenly spaced <=maximum",
                default_max_training_queries=DEFAULT_MAX_TRAINING_QUERIES,
                count_conservation=True, truth_input=False, H_input=False,
                child_only_merge=True, associative=False, accuracy_guarantee=False)
