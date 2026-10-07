"""Optional ASE adapter for native tagged and tagged-plus-ACE linear models."""

import ctypes
import ctypes.util
import json
import os
import tempfile
import weakref
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

from ye3t_methods.atomistic.tagged_cauchy_image import export_tagged_cauchy_image_model


class _TaggedCauchyNativeRuntime:
    """Keep one hash-verified native model and its compiled schedules resident."""

    def __init__(self, model, library_path=None, execution_policy="direct", neighbors="auto"):
        if execution_policy not in {"auto", "direct"}:
            raise ValueError("Native tagged execution_policy must be auto or direct.")
        if neighbors not in {"auto", "ase", "matscipy"}:
            raise ValueError("Native tagged neighbors must be auto, ase, or matscipy.")
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
            suffix = ".dll" if os.name == "nt" else (".dylib" if os.sys.platform == "darwin" else ".so")
            bundled = Path(__file__).resolve().parent / f"libye3t_tagged_c_api{suffix}"
            candidate = str(bundled) if bundled.is_file() else None
        if candidate is None:
            candidate = ctypes.util.find_library("ye3t_tagged_c_api")
        if not candidate:
            raise RuntimeError(
                "The native tagged CPU library is unavailable. Build the "
                "bundled native/ CMake target and pass "
                "native_library=... or set YE3T_TAGGED_C_API_LIBRARY."
            )
        self.library = ctypes.CDLL(str(candidate))
        self.library.ye3t_tagged_open.argtypes = (
            ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
        )
        self.library.ye3t_tagged_open.restype = ctypes.c_void_p
        self.library.ye3t_tagged_close.argtypes = (ctypes.c_void_p,)
        self.library.ye3t_tagged_close.restype = None
        self.library.ye3t_tagged_selected_policy.argtypes = (ctypes.c_void_p,)
        self.library.ye3t_tagged_selected_policy.restype = ctypes.c_char_p
        self.library.ye3t_tagged_evaluate.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_size_t,
        )
        self.library.ye3t_tagged_evaluate.restype = ctypes.c_int
        self.library.ye3t_tagged_evaluate_with_features.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p, ctypes.c_size_t,
        )
        self.library.ye3t_tagged_evaluate_with_features.restype = ctypes.c_int
        self.model = model if not isinstance(model, (str, Path)) else None
        temporary = None
        if self.model is None:
            path = Path(model)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("schema") == "ye3t_tagged_cauchy_composite_v1":
                component = path.parent / metadata["tagged_component"]["path"]
                tagged_metadata = json.loads(component.read_text(encoding="utf-8"))
                species_order = metadata["species_order"]
                cutoff = tagged_metadata["cutoff"]
                feature_count = tagged_metadata["real_moment_program"]["feature_count"]
            elif "readout_binding" in metadata:
                tagged_metadata = metadata
                species_order = metadata["readout_binding"]["payload"]["species_order"]
                cutoff = metadata["source_binding"]["payload"]["cutoff"]
                feature_count = metadata["readout_binding"]["payload"]["feature_count"]
            else:
                tagged_metadata = metadata
                species_order = metadata["species_order"]
                cutoff = tagged_metadata["cutoff"]
                feature_count = tagged_metadata["real_moment_program"]["feature_count"]
            self.type_map = {name: index for index, name in enumerate(species_order)}
            self.cutoff = float(cutoff)
            self.feature_count = int(feature_count)
            self.has_ordinary_backbone = tagged_metadata is not metadata
        else:
            temporary = tempfile.TemporaryDirectory(prefix="ye3t_tagged_native_")
            path = Path(temporary.name) / "model.ye3t.json"
            export_tagged_cauchy_image_model(path, model)
            self.type_map = dict(model.evaluator.type_map)
            self.cutoff = float(model.evaluator.cutoff)
            self.feature_count = int(model.evaluator.feature_count)
            self.has_ordinary_backbone = False
        error = ctypes.create_string_buffer(2048)
        handle = self.library.ye3t_tagged_open(
            os.fsencode(path), int(execution_policy == "auto"),
            error, len(error),
        )
        if not handle:
            if temporary is not None:
                temporary.cleanup()
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        self.handle = handle
        self.selected_policy = self.library.ye3t_tagged_selected_policy(handle).decode("ascii")
        self._finalizer = weakref.finalize(self, self._release, self.library, handle, temporary)
        self.neighbor_skin = 0.3
        self._topology = None
        self.topology_rebuilds = 0
        self.last_neighbor_backend = None

    @staticmethod
    def _release(library, handle, temporary):
        library.ye3t_tagged_close(handle)
        if temporary is not None:
            temporary.cleanup()

    def close(self):
        self._finalizer()

    def _geometry(self, atoms):
        symbols = tuple(atoms.get_chemical_symbols())
        unknown = sorted(set(symbols) - set(self.type_map))
        if unknown:
            raise ValueError(f"Native tagged model contains unknown species: {unknown}.")
        atom_types = np.ascontiguousarray([self.type_map[name] for name in symbols], dtype=np.int32)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cached = self._topology
        rebuild = (cached is None or cached["symbols"] != symbols or
                   not np.array_equal(cached["cell"], cell) or
                   not np.array_equal(cached["pbc"], pbc) or
                   len(cached["positions"]) != len(positions) or
                   np.max(np.linalg.norm(positions - cached["positions"], axis=1), initial=0.0) >
                   self.neighbor_skin / 2)
        if rebuild:
            radius = self.cutoff + self.neighbor_skin
            periodic_box = (bool(np.all(pbc)) and np.allclose(cell,
                np.diag(np.diag(cell)), atol=1e-12, rtol=0) and
                np.all(np.diag(cell) > 2 * radius))
            fast_pairs = self.neighbors == "auto" and (
                periodic_box or not bool(np.any(pbc))
            )
            if fast_pairs:
                try:
                    from scipy.spatial import cKDTree
                except ImportError:
                    fast_pairs = False
            if fast_pairs:
                box = np.diag(cell) if periodic_box else None
                coordinates = np.mod(positions, box) if periodic_box else positions
                pairs = np.asarray(cKDTree(coordinates, boxsize=box).query_pairs(
                    radius, output_type="ndarray"), dtype=np.int64).reshape(-1, 2)
                src = np.concatenate((pairs[:, 0], pairs[:, 1]))
                dst = np.concatenate((pairs[:, 1], pairs[:, 0]))
                if periodic_box:
                    first_shifts = -np.rint((positions[pairs[:, 1]] -
                                             positions[pairs[:, 0]]) / box).astype(np.int64)
                    shifts = np.concatenate((first_shifts, -first_shifts))
                else:
                    shifts = np.zeros((len(src), 3), dtype=np.int64)
                self.last_neighbor_backend = "scipy_ckdtree_periodic" if periodic_box else "scipy_ckdtree_nonperiodic"
            else:
                fast_neighbor_list = self._matscipy_neighbor_list
                if self.neighbors == "auto":
                    try:
                        from matscipy.neighbours import neighbour_list as fast_neighbor_list
                    except ImportError:
                        fast_neighbor_list = None
                if fast_neighbor_list is None:
                    src, dst, shifts = neighbor_list("ijS", atoms, radius, self_interaction=False)
                    self.last_neighbor_backend = "ase_neighbor_list"
                else:
                    src, dst, shifts = fast_neighbor_list("ijS", atoms, radius)
                    self.last_neighbor_backend = "matscipy_neighbor_list"
            order = np.argsort(src, kind="stable")
            src, dst, shifts = src[order], dst[order], shifts[order]
            offsets = np.zeros(len(atoms) + 1, dtype=np.uintp)
            offsets[1:] = np.cumsum(np.bincount(src, minlength=len(atoms)))
            cached = {"symbols": symbols, "cell": cell.copy(), "pbc": pbc.copy(),
                      "positions": positions.copy(), "src": src, "dst": dst,
                      "shifts": shifts, "offsets": offsets}
            self._topology = cached
            self.topology_rebuilds += 1
        src, dst, shifts, offsets = (cached[name] for name in ("src", "dst", "shifts", "offsets"))
        displacement = np.ascontiguousarray(
            positions[dst] - positions[src] + shifts @ cell,
            dtype=np.float64,
        )
        neighbor_types = np.ascontiguousarray(atom_types[dst], dtype=np.int32)
        return atom_types, src, dst, displacement, offsets, neighbor_types

    def evaluate_atoms(self, atoms, *, return_features=False):
        """Return native energy, forces, virial, atomic energy, and optional features."""

        if return_features and self.has_ordinary_backbone:
            raise ValueError("Composite feature export needs both ordinary ACE and tagged columns.")
        atom_types, src, dst, displacement, offsets, neighbor_types = self._geometry(atoms)
        atomic = np.empty(len(atoms), dtype=np.float64)
        edge_gradient = np.empty((len(src), 3), dtype=np.float64)
        features = (np.empty((len(atoms), self.feature_count), dtype=np.float64)
                    if return_features else None)
        error = ctypes.create_string_buffer(2048)
        arguments = (
            self.handle, len(atoms),
            atom_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            neighbor_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            displacement.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            atomic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            edge_gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if return_features:
            status = self.library.ye3t_tagged_evaluate_with_features(
                *arguments, features.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                error, len(error),
            )
        else:
            status = self.library.ye3t_tagged_evaluate(*arguments, error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        np.add.at(forces, src, edge_gradient)
        np.add.at(forces, dst, -edge_gradient)
        tensor = -np.einsum("ei,ej->ij", displacement, edge_gradient)
        virial = np.array((tensor[0, 0], tensor[1, 1], tensor[2, 2],
                           tensor[0, 1], tensor[0, 2], tensor[1, 2]))
        if self.model is not None and self.model.reference_terms:
            reference_atomic, reference_forces, reference_virial = self.model._reference_values(atoms)
            atomic += reference_atomic
            forces += reference_forces
            virial += reference_virial
        result = (float(np.sum(atomic)), forces, virial, atomic)
        return (*result, features) if return_features else result
