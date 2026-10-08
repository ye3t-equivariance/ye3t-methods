"""ASE calculator for a strict PACE/YACE model using the native CPU evaluator."""

import ctypes
import ctypes.util
import os
import weakref
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ye3t_methods.atomistic.tagged_cauchy_native import (
    _TaggedCauchyNativeRuntime, _bundled_native_library_path,
)


class _YACENativeRuntime(_TaggedCauchyNativeRuntime):
    """Load one YACE model and reuse the tagged adapter's neighbor geometry."""

    def __init__(self, path, *, library_path=None, neighbor_skin=0.3,
                 neighbors="auto"):
        if neighbors not in {"auto", "ase", "matscipy"}:
            raise ValueError("Native YACE neighbors must be auto, ase, or matscipy.")
        self.neighbors = neighbors
        self._matscipy_neighbor_list = None
        if neighbors == "matscipy":
            try:
                from matscipy.neighbours import neighbour_list
            except ImportError as exc:
                raise ImportError(
                    "neighbors='matscipy' requires the optional matscipy dependency."
                ) from exc
            self._matscipy_neighbor_list = neighbour_list
        candidate = library_path or os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
        if candidate is None:
            candidate = _bundled_native_library_path()
        if candidate is None:
            candidate = ctypes.util.find_library("ye3t_tagged_c_api")
        if not candidate:
            raise RuntimeError(
                "The native YACE CPU library is unavailable. Build the bundled "
                "native/ CMake target and pass native_library=... "
                "or set YE3T_TAGGED_C_API_LIBRARY."
            )
        self.library = ctypes.CDLL(str(candidate))
        try:
            self.library.ye3t_yace_open
        except AttributeError as exc:
            raise RuntimeError(
                "This native library lacks the YACE C API; rebuild the bundled "
                "native/ CMake target."
            ) from exc
        self.library.ye3t_yace_open.argtypes = (
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
        )
        self.library.ye3t_yace_open.restype = ctypes.c_void_p
        self.library.ye3t_yace_close.argtypes = (ctypes.c_void_p,)
        self.library.ye3t_yace_close.restype = None
        self.library.ye3t_yace_species_count.argtypes = (ctypes.c_void_p,)
        self.library.ye3t_yace_species_count.restype = ctypes.c_int
        self.library.ye3t_yace_species_name.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.library.ye3t_yace_species_name.restype = ctypes.c_char_p
        self.library.ye3t_yace_maximum_cutoff.argtypes = (ctypes.c_void_p,)
        self.library.ye3t_yace_maximum_cutoff.restype = ctypes.c_double
        self.library.ye3t_yace_evaluate.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_size_t,
        )
        self.library.ye3t_yace_evaluate.restype = ctypes.c_int
        error = ctypes.create_string_buffer(2048)
        handle = self.library.ye3t_yace_open(os.fsencode(path), error, len(error))
        if not handle:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        self.handle = handle
        self._finalizer = weakref.finalize(self, self.library.ye3t_yace_close, handle)
        count = self.library.ye3t_yace_species_count(handle)
        names = tuple(self.library.ye3t_yace_species_name(handle, index).decode("utf-8")
                      for index in range(count))
        if count <= 0 or len(set(names)) != count:
            self.close()
            raise RuntimeError("Native YACE model contains invalid species metadata.")
        self.type_map = {name: index for index, name in enumerate(names)}
        self.cutoff = float(self.library.ye3t_yace_maximum_cutoff(handle))
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            self.close()
            raise RuntimeError("Native YACE model contains an invalid cutoff.")
        self.neighbor_skin = float(neighbor_skin)
        if not np.isfinite(self.neighbor_skin) or self.neighbor_skin < 0.0:
            self.close()
            raise ValueError("neighbor_skin must be finite and nonnegative.")
        self._topology = None
        self.topology_rebuilds = 0
        self.last_neighbor_backend = None
        self.selected_policy = "yace_direct"

    def close(self):
        self._finalizer()

    def evaluate_atoms(self, atoms):
        if not self._finalizer.alive:
            raise RuntimeError("Native YACE calculator has been closed.")
        atom_types, src, dst, displacement, offsets, neighbor_types = self._geometry(atoms)
        atomic = np.empty(len(atoms), dtype=np.float64)
        edge_gradient = np.empty((len(src), 3), dtype=np.float64)
        error = ctypes.create_string_buffer(2048)
        status = self.library.ye3t_yace_evaluate(
            self.handle, len(atoms),
            atom_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            neighbor_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            displacement.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            atomic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            edge_gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            error, len(error),
        )
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        np.add.at(forces, src, edge_gradient)
        np.add.at(forces, dst, -edge_gradient)
        tensor = -np.einsum("ei,ej->ij", displacement, edge_gradient)
        virial = np.array((tensor[0, 0], tensor[1, 1], tensor[2, 2],
                           tensor[0, 1], tensor[0, 2], tensor[1, 2]))
        return float(np.sum(atomic)), forces, virial, atomic


class YE3TYACENativeCalculator(Calculator):
    """ASE energy, atomic energy, force, and stress calculator for a YACE file."""

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, path, *, native_library=None, neighbor_skin=0.3,
                 neighbors="auto"):
        super().__init__()
        self.native_runtime = _YACENativeRuntime(
            path, library_path=native_library, neighbor_skin=neighbor_skin,
            neighbors=neighbors,
        )
        self._temporary = None

    @classmethod
    def from_artifact(cls, path, *, native_library=None, neighbor_skin=0.3,
                      neighbors="auto"):
        """Open a strict YACE file with an explicit native neighbor policy."""
        return cls(path, native_library=native_library, neighbor_skin=neighbor_skin,
                   neighbors=neighbors)

    def close(self):
        self.native_runtime.close()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces, virial, atomic = self.native_runtime.evaluate_atoms(self.atoms)
        self.results = {
            "energy": energy, "free_energy": energy,
            "energies": atomic, "forces": forces,
        }
        volume = float(self.atoms.get_volume()) if self.atoms.cell.rank == 3 else 0.0
        if volume > 0.0:
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / volume
        elif "stress" in properties:
            raise ValueError("Stress requires a cell with positive volume.")
