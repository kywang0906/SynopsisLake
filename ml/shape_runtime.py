"""Indexed execution of the frozen native and hierarchical R08 references.

Only execution changes: native identity and packet checks remain intact, and
parent/child float32 intersections and distribution math are reused unchanged.
Source grid lookup avoids scanning unrelated cells. Exact counts sum covered
cells, with prefix totals for completely covered sources. No raw rows, labels, query lookup table or diagnostic
parent-quadrant CDF precomputation is needed. Derived RAM is reported separately
from the unchanged serialized synopsis bytes. This is a runtime optimization,
not a new density model or an effectiveness claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import bounded_sidecar as bounded
from . import hierarchical_shape as hierarchy
from . import local_shape as local
from ._distribution_math import boundary_probability


class FastNativeRuntime:
    """Native structured R08, indexed by each source's shared grid axes."""

    def __init__(self, native_histograms, encoded_cells):
        self._initialize(native_histograms, encoded_cells, None)

    def _initialize(self, native_histograms, encoded_cells, payload):
        hists, encoded, mass, sizes, _ = local._native(native_histograms, encoded_cells)
        self.histograms, self.encoded_cells = hists, encoded
        self.total_mass = sum(map(int, mass))
        if self.total_mass > bounded.UINT64_MAX:
            raise OverflowError("Total native mass exceeds uint64")
        self._offsets = np.r_[0, np.cumsum(sizes)]
        self._moments = bounded.decode_moments(np.concatenate(encoded)).astype(float)
        self._prefix = []
        for hist in hists:
            prefix = np.zeros(tuple(n+1 for n in hist[3].shape), np.uint64)
            prefix[1:, 1:] = np.cumsum(np.cumsum(hist[3], axis=0, dtype=np.uint64), axis=1, dtype=np.uint64)
            self._prefix.append(prefix)
        self.payload = b"" if payload is None else bytes(payload)
        self.selection = None if payload is None else hierarchy.decode_selection(hists, encoded, payload)
        self._selected_index = np.full(len(mass), -1, np.int32)
        self._child_moments = np.empty((0, 4, 5), float)
        if self.selection is not None:
            self._selected_index[self.selection.selected_ids] = np.arange(len(self.selection.selected_ids))
            self._child_moments = bounded.decode_moments(self.selection.child_encoded.reshape(-1, 5)).astype(float).reshape(-1, 4, 5)
        self.native_bytes = sum(24+8*(len(h[1])+len(h[2])+h[3].size)+64+5*h[3].size for h in hists)
        self.auxiliary_bytes = self.native_bytes+len(self.payload)
        self.nbytes = self.auxiliary_bytes
        self.original_histogram = None
        derived = [self._offsets, self._moments, self._selected_index, self._child_moments, *self._prefix]
        self.derived_ram_bytes = sum(array.nbytes for array in derived)
        resident = [array for hist in hists for array in hist[1:]]+encoded
        if self.selection is not None:
            resident += [self.selection.selected_ids, self.selection.quadrant_counts, self.selection.child_encoded]
        self.ram_bytes = self.derived_ram_bytes+sum(array.nbytes for array in resident)+len(self.payload)

    @staticmethod
    def _span(axis, lo, hi):
        n = len(axis)-1
        return (max(0, min(n, int(np.searchsorted(axis, lo, "left"))-1)),
                max(0, min(n, int(np.searchsorted(axis, hi, "right")))))

    @staticmethod
    def _boundary_sum(values):
        # Frozen hierarchy uses bincount's sequential row-order accumulation.
        return float(np.bincount(np.zeros(len(values), np.intp), weights=values, minlength=1)[0])

    def _intersections(self, rect):
        """Return exact covered mass and only nonempty partial cell rows."""
        full_mass, all_ids, all_local, all_mass = 0, [], [], []
        for source, hist in enumerate(self.histograms):
            _, bx, by, counts = hist
            if rect[1] < bx[0] or rect[0] > bx[-1] or rect[3] < by[0] or rect[2] > by[-1]:
                continue
            if rect[0] <= bx[0] and rect[1] >= bx[-1] and rect[2] <= by[0] and rect[3] >= by[-1]:
                full_mass += int(self._prefix[source][-1, -1])
                continue
            xa, xb = self._span(bx, rect[0], rect[1])
            ya, yb = self._span(by, rect[2], rect[3])
            if xa >= xb or ya >= yb:
                continue
            ids = (np.arange(xa, xb)[:, None]*counts.shape[1]+np.arange(ya, yb)[None, :]).ravel()
            ids = ids[counts.ravel()[ids] > 0]
            if not len(ids):
                continue
            ix, iy = ids//counts.shape[1], ids % counts.shape[1]
            full = ((rect[0] <= bx[ix]) & (bx[ix+1] <= rect[1])
                    & (rect[2] <= by[iy]) & (by[iy+1] <= rect[3]))
            full_mass += int(counts.ravel()[ids[full]].sum(dtype=np.uint64))
            ids, ix, iy = ids[~full], ix[~full], iy[~full]
            if not len(ids):
                continue
            # Same parent-coordinate arithmetic and float32 round trip as R08.
            local_rect = np.column_stack(((rect[0]-bx[ix])/(bx[ix+1]-bx[ix]),
                                          (rect[1]-bx[ix])/(bx[ix+1]-bx[ix]),
                                          (rect[2]-by[iy])/(by[iy+1]-by[iy]),
                                          (rect[3]-by[iy])/(by[iy+1]-by[iy])))
            all_ids.append(ids+self._offsets[source])
            all_local.append(np.clip(local_rect, 0., 1.).astype(np.float32).astype(float))
            all_mass.append(counts.ravel()[ids].astype(float))
        if not all_ids:
            return float(full_mass), np.empty(0, np.int64), np.empty((0, 4)), np.empty(0)
        return float(full_mass), np.concatenate(all_ids), np.concatenate(all_local), np.concatenate(all_mass)

    def _corrected(self, local_rectangles, selected, *, conditional=True):
        return hierarchy.HierarchicalShapeRuntime._corrected(self, local_rectangles, selected, conditional=conditional)

    def _mode(self, mode):
        if mode is None:
            return hierarchy.MODES[0] if self.selection is None else self.selection.mode
        if mode == "conditional_r08":
            mode = hierarchy.MODES[0]
        if mode not in hierarchy.MODES:
            raise ValueError("Unknown indexed shape mode")
        return mode

    def batch_reference(self, rectangles, *, mode=None):
        mode, queries = self._mode(mode), bounded._rectangles(rectangles)
        result = np.zeros(len(queries))
        for qi, rect in enumerate(queries):
            full, ids, local_rect, mass = self._intersections(rect)
            selected = self._selected_index[ids]
            chosen = selected >= 0
            values = np.zeros(len(ids))
            if (~chosen).any():
                values[~chosen] = mass[~chosen]*boundary_probability(local_rect[~chosen], self._moments[ids[~chosen]], "beta_gaussian_copula")
            if chosen.any():
                values[chosen] = self._corrected(local_rect[chosen], selected[chosen], conditional=mode == hierarchy.MODES[0])[0]
            result[qi] = full+(values.sum() if self.selection is None else self._boundary_sum(values))
        return result

    def reference_count(self, rectangle, *, mode=None):
        rectangle = np.asarray(rectangle, float)
        if rectangle.shape != (4,):
            raise ValueError("Expected one finite query rectangle")
        return float(self.batch_reference(rectangle[None, :], mode=mode)[0])

    def query_features(self, rectangles, histogram=None):
        """Required frozen ML features; expensive source-TV diagnostics omitted."""
        queries = bounded._rectangles(rectangles)
        names = ("reference_base", "reference_hybrid", "reference_uniform", "full_mass", "boundary_mass",
                 "corrected_boundary_mass", "corrected_boundary_estimate_base", "corrected_boundary_estimate_hybrid",
                 "selected_boundary_cells", "zero_cdf_boundary_mass")
        result = {name: np.zeros(len(queries)) for name in names}
        for qi, rect in enumerate(queries):
            full, ids, local_rect, mass = self._intersections(rect)
            base = mass*boundary_probability(local_rect, self._moments[ids], "beta_gaussian_copula")
            hybrid, uniform = base.copy(), base.copy()
            selected = self._selected_index[ids]
            chosen = selected >= 0
            if chosen.any():
                hybrid[chosen], uniform[chosen], _ = self._corrected(local_rect[chosen], selected[chosen])
                result["corrected_boundary_mass"][qi] = mass[chosen].sum()
                result["corrected_boundary_estimate_base"][qi] = self._boundary_sum(base[chosen])
                result["corrected_boundary_estimate_hybrid"][qi] = self._boundary_sum(hybrid[chosen])
                result["selected_boundary_cells"][qi] = chosen.sum()
            for name, values in (("reference_base", base), ("reference_hybrid", hybrid), ("reference_uniform", uniform)):
                result[name][qi] = full+self._boundary_sum(values)
            result["full_mass"][qi], result["boundary_mass"][qi] = full, mass.sum()
        result["correction_delta"] = result["reference_hybrid"]-result["reference_base"]
        result["correction_abs_delta"] = np.abs(result["correction_delta"])
        result["corrected_boundary_fraction"] = np.divide(result["corrected_boundary_mass"], result["boundary_mass"],
                                                           out=np.zeros(len(queries)), where=result["boundary_mass"] > 0)
        histogram = self.original_histogram if histogram is None else histogram
        if histogram is not None:
            from .sidecar_workflow import histogram_count
            result["H"] = np.array([histogram_count(rect, histogram) for rect in queries])
        return result


class IndexedHierarchicalRuntime(FastNativeRuntime):
    def __init__(self, native_histograms, encoded_cells, payload):
        self._initialize(native_histograms, encoded_cells, payload)


def load_indexed(native_dir, packet_path=None):
    """Validate native files and optional packet; derive grid indexes in RAM."""
    hists, encoded = local.load_native(native_dir)
    runtime = (FastNativeRuntime(hists, encoded) if packet_path is None else
               IndexedHierarchicalRuntime(hists, encoded, Path(packet_path).read_bytes()))
    return runtime


def protocol():
    return dict(model_changed=False, native_identity_and_packet_validation=True,
                indexing="source-axis searchsorted; exact covered cell counts and whole-source prefix totals; native row ordering",
                probability="unchanged native float32 intersections and frozen hierarchical child evaluator",
                omitted_diagnostics=["source_disagreement_mass", "parent quadrant CDF precomputation"],
                bytes="same native/packet serialized bytes; ram_bytes and derived_ram_bytes reported separately",
                raw_rows_at_inference=0, query_lookup=False, novelty_claim=False)
