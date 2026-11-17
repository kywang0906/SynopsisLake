"""Optional compiled execution of the frozen native/hierarchical estimator.

C++ receives validated synopsis arrays and the current SciPy C API pointers.
It stores no observations, query tables or fitted parameters. Compilation and
library setup are reported separately from source loading; cache identity binds
both sources, compiler, Python ABI and SciPy. Missing support raises explicitly.
The Python feature path remains available with the same frozen feature math.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

import numpy as np
import scipy
from scipy.special import cython_special

from . import _distribution_math as distribution
from . import bounded_sidecar as bounded
from . import hierarchical_shape as hierarchy
from . import local_shape as local
from . import shape_runtime as indexed

SCORE_ROUNDING_ENVELOPE = 1e-9
_STATE = None
_SETUP_LOCK = threading.Lock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capsule(name, signature):
    capsule = cython_special.__pyx_capi__.get(name)
    if capsule is None:
        raise RuntimeError(f"SciPy C API capsule is unavailable: {name}")
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = [ctypes.py_object], ctypes.c_char_p
    actual = get_name(capsule)
    if actual != signature:
        raise RuntimeError(f"Unsupported SciPy C API signature for {name}: {actual!r}")
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes, get_pointer.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_void_p
    return get_pointer(capsule, actual)


def _kernel_setup():
    """Compile/load once per process; return cache/build/setup metadata."""
    global _STATE
    if _STATE is not None:
        return _STATE
    started = time.perf_counter()
    beta = _capsule("__pyx_fuse_0betainc", b"double (double, double, double, int __pyx_skip_dispatch)")
    ndtri = _capsule("ndtri", b"double (double, int __pyx_skip_dispatch)")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Compiled shape kernel requires g++; use the explicitly selected Python runtime otherwise")
    source = Path(__file__).with_suffix(".cc")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    flags = ["-std=c++17", "-O3", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off"]
    identity = dict(source_sha256=_sha(source), wrapper_sha256=_sha(__file__), compiler=compiler,
                    compiler_version=compiler_version, flags=flags, python_abi=sysconfig.get_config_var("SOABI"),
                    python_implementation=sys.implementation.cache_tag, scipy_version=scipy.__version__,
                    platform=platform.platform(), pointer_bytes=ctypes.sizeof(ctypes.c_void_p))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(tempfile.gettempdir())/"synopsislake-shape-kernel"/key
    cache.mkdir(parents=True, exist_ok=True)
    library_path, manifest_path = cache/"shape_kernel.so", cache/"build.json"
    compilation_seconds = 0.
    if library_path.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity or saved["library_sha256"] != _sha(library_path):
            raise RuntimeError("Compiled shape kernel cache identity/hash changed")
    else:
        temporary = cache/f"kernel-{os.getpid()}.so"
        tick = time.perf_counter()
        process = subprocess.run([compiler, *flags, str(source), "-o", str(temporary)], capture_output=True, text=True)
        compilation_seconds = time.perf_counter()-tick
        if process.returncode:
            raise RuntimeError("Shape kernel compilation failed: "+process.stderr.strip())
        os.replace(temporary, library_path)
        manifest_path.write_text(json.dumps(dict(identity=identity, library_sha256=_sha(library_path)), indent=2)+"\n")
    tick = time.perf_counter()
    library = ctypes.PyDLL(str(library_path))  # Keep the GIL while calling SciPy's Cython C API.
    double_ptr = ctypes.POINTER(ctypes.c_double)
    library.shape_kernel_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, double_ptr, double_ptr]
    library.shape_kernel_init.restype = ctypes.c_int
    nodes = np.ascontiguousarray(distribution._GL_NODES, np.float64)
    weights = np.ascontiguousarray(distribution._GL_WEIGHTS, np.float64)
    if library.shape_kernel_init(beta, ndtri, nodes.ctypes.data_as(double_ptr), weights.ctypes.data_as(double_ptr)):
        raise RuntimeError("Shape kernel SciPy C API initialization failed")
    library.shape_kernel_batch.argtypes = [ctypes.c_size_t, double_ptr, ctypes.POINTER(ctypes.c_uint64),
        double_ptr, ctypes.POINTER(ctypes.c_int32), double_ptr, double_ptr, ctypes.c_size_t, double_ptr,
        ctypes.c_int, double_ptr]
    library.shape_kernel_batch.restype = ctypes.c_int
    library_load_seconds = time.perf_counter()-tick
    _STATE = dict(library=library, cache_key=key, cache_dir=str(cache), identity=identity,
                  compilation_seconds=compilation_seconds, library_load_seconds=library_load_seconds,
                  setup_seconds=time.perf_counter()-started, library_sha256=_sha(library_path))
    return _STATE


def kernel_setup():
    """Thread-safe process initialization; no data/query state is global."""
    if _STATE is not None:
        return _STATE
    with _SETUP_LOCK:
        return _kernel_setup()


class CompiledRuntime(indexed.FastNativeRuntime):
    """Validated native/child metadata with C++ count and Python feature paths."""
    backend = "compiled_scipy_capi"

    def __init__(self, native_histograms, encoded_cells, payload=None):
        setup = kernel_setup()
        self.compilation_seconds = setup["compilation_seconds"]
        self.library_load_seconds = setup["library_load_seconds"]
        self.kernel_cache_key = setup["cache_key"]
        self.kernel_library_sha256 = setup["library_sha256"]
        self._library = setup["library"]
        tick = time.perf_counter()
        self._initialize(native_histograms, encoded_cells, payload)
        parts = [bounded.components_from_histogram_moments(h,e) for h,e in zip(self.histograms,self.encoded_cells)]
        self._kernel_bounds = np.ascontiguousarray(np.concatenate([p.bounds for p in parts]), np.float64)
        self._kernel_mass = np.ascontiguousarray(np.concatenate([p.mass for p in parts]), np.uint64)
        self._kernel_encoded = np.ascontiguousarray(np.concatenate([p.encoded for p in parts]), np.uint8)
        active = np.flatnonzero(np.concatenate([h[3].ravel() for h in self.histograms]))
        self._kernel_moments = np.ascontiguousarray(self._moments[active], np.float64)
        self._kernel_selected = np.ascontiguousarray(self._selected_index[active], np.int32)
        self._kernel_counts = np.ascontiguousarray(self.selection.quadrant_counts if self.selection is not None else np.empty((0,4)), np.float64)
        self._kernel_children = np.ascontiguousarray(self._child_moments, np.float64)
        arrays = (self._kernel_bounds,self._kernel_mass,self._kernel_encoded,self._kernel_moments,
                  self._kernel_selected,self._kernel_counts)
        added = sum(a.nbytes for a in arrays)
        # Child moments are shared with the validated Python feature fallback.
        self.derived_ram_bytes += added
        self.ram_bytes += added
        self.source_initialization_seconds = time.perf_counter()-tick
        self._canonical = None

    def batch_reference(self, rectangles, *, mode=None):
        mode = self._mode(mode)
        queries = np.ascontiguousarray(bounded._rectangles(rectangles), np.float64)
        result = np.empty(len(queries), np.float64)
        double_ptr = ctypes.POINTER(ctypes.c_double)
        def doubles(array): return array.ctypes.data_as(double_ptr)
        status = self._library.shape_kernel_batch(len(self._kernel_mass), doubles(self._kernel_bounds),
            self._kernel_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), doubles(self._kernel_moments),
            self._kernel_selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), doubles(self._kernel_counts),
            doubles(self._kernel_children), len(queries), doubles(queries), mode == hierarchy.MODES[1], doubles(result))
        if status:
            raise RuntimeError(f"Compiled shape reference failed with status {status}")
        return result

    def canonical_reference_count(self, rectangle, *, mode=None):
        """Exact frozen accumulator for explicit near-cutoff rounding guards."""
        if self._canonical is None:
            components = bounded.Components(self._kernel_bounds,self._kernel_mass,self._kernel_encoded)
            added = components.bounds.nbytes+components.mass.nbytes+components.encoded.nbytes
            self.derived_ram_bytes += added
            self.ram_bytes += added
            if self.selection is None:
                self._canonical = components
            else:
                runtime = hierarchy.HierarchicalShapeRuntime.__new__(hierarchy.HierarchicalShapeRuntime)
                runtime.components, runtime.selection = components, self.selection
                runtime._moments, runtime._selected_index = self._kernel_moments, self._kernel_selected
                runtime._child_moments = self._kernel_children
                self._canonical = runtime
        if self.selection is None:
            return bounded.reference_count(self._canonical,rectangle)
        return self._canonical.reference_count(rectangle,mode=mode)

    def reference_and_score(self, rectangle, answer, *, cutoff=None, mode=None):
        """Return (reference, proxy), preserving frozen near-cutoff decisions.

        A fixed 1e-9 score envelope triggers the canonical numerical path. The
        decision uses the existing cutoff, never labels or measured outcomes.
        """
        reference = self.reference_count(rectangle,mode=mode)
        score = abs(float(answer)-reference)/max(reference,1.)
        if cutoff is not None and np.isfinite(cutoff) and abs(score-float(cutoff)) <= SCORE_ROUNDING_ENVELOPE:
            reference = self.canonical_reference_count(rectangle,mode=mode)
            score = abs(float(answer)-reference)/max(reference,1.)
        return reference,score


def load_compiled(native_dir, packet_path=None):
    """Load and validate all native files/packet before deriving kernel arrays."""
    kernel_setup()  # Compilation/setup is separately reported, never hidden in query timing.
    tick = time.perf_counter()
    hists, encoded = local.load_native(native_dir)
    runtime = CompiledRuntime(hists,encoded,None if packet_path is None else Path(packet_path).read_bytes())
    runtime.source_load_seconds = time.perf_counter()-tick
    return runtime


def protocol():
    return dict(backend=CompiledRuntime.backend, numerical_policy="frozen R08 beta/copula with original SciPy C API special functions and 16 GL nodes",
                parent_and_child_geometry="original float32 round trips", fast_math=False,
                compilation="explicit O3 C++ shared library; source/compiler/ABI/SciPy cache identity; setup separate",
                near_cutoff_policy=f"canonical numerical fallback within fixed {SCORE_ROUNDING_ENVELOPE} score envelope",
                features="unchanged Python indexed evaluator; no fitted model changes",
                serialization="unchanged native/packet bytes; derived ndarray RAM separately reported",
                raw_rows_at_inference=0, query_lookup=False, novelty_claim=False)
