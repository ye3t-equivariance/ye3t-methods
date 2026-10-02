"""Reference-potential targets kept separate from descriptor geometry caches."""

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_hash(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _zbl_pair_cutoffs(config, species_order):
    """Canonical, complete symmetric pair switches; never mix missing pairs."""
    supplied = config.get("pair_cutoffs_A")
    pairs = {}
    if supplied is None:
        inner, outer = float(config["inner_cutoff_A"]), float(config["outer_cutoff_A"])
        supplied = {"-".join(sorted((left, right))): [inner, outer]
                    for left in species_order for right in species_order}
    for key, values in supplied.items():
        elements = str(key).split("-")
        if len(elements) != 2 or any(value not in species_order for value in elements):
            raise ValueError("ZBL pair_cutoffs_A contains an unknown species pair.")
        if len(values) != 2:
            raise ValueError("Each ZBL pair needs [inner_cutoff_A, outer_cutoff_A].")
        inner, outer = map(float, values)
        if not np.isfinite(inner) or not np.isfinite(outer) or not 0 < inner < outer:
            raise ValueError("ZBL cutoffs must satisfy 0 < inner_cutoff_A < outer_cutoff_A.")
        canonical = "-".join(sorted(elements))
        if canonical in pairs and pairs[canonical] != [inner, outer]:
            raise ValueError("ZBL pair cutoffs must be symmetric under species exchange.")
        pairs[canonical] = [inner, outer]
    expected = {"-".join(sorted((left, right))) for left in species_order for right in species_order}
    if set(pairs) != expected:
        raise ValueError("ZBL pair_cutoffs_A must cover every unordered species pair.")
    return dict(sorted(pairs.items()))


def lammps_zbl_reference_config(config, species_order):
    """Validate and normalize one LAMMPS ``zbl`` reference component."""

    config = dict(config)
    species_order = tuple(str(value) for value in species_order)
    if not species_order or len(set(species_order)) != len(species_order):
        raise ValueError("species_order must be nonempty and unique.")
    if str(config.get("engine", "lammps_executable")) != "lammps_executable":
        raise ValueError("The ZBL reference engine must be lammps_executable.")
    if str(config.get("pair_style", "zbl")) != "zbl":
        raise ValueError("This reference component requires pair_style zbl.")
    if str(config.get("units", "metal")) != "metal":
        raise ValueError("This reference component requires LAMMPS metal units.")
    if str(config.get("atom_style", "atomic")) != "atomic":
        raise ValueError("This reference component requires atom_style atomic.")
    pairs = _zbl_pair_cutoffs(config, species_order)
    switches = sorted(set(tuple(value) for value in pairs.values()))
    pair_specific = "pair_cutoffs_A" in config
    atomic_numbers = {
        str(key): int(value)
        for key, value in dict(config["atomic_numbers"]).items()
    }
    if set(atomic_numbers) != set(species_order) or any(
        value <= 0 for value in atomic_numbers.values()
    ):
        raise ValueError(
            "atomic_numbers must contain one positive value for every species."
        )
    standalone_pair_coeff = []
    overlay_pair_coeff = []
    for left, left_species in enumerate(species_order, start=1):
        for right in range(left, len(species_order) + 1):
            right_species = species_order[right - 1]
            switch = tuple(pairs["-".join(sorted((left_species, right_species)))])
            substyle = "zbl " + (str(switches.index(switch)+1)+" " if len(switches) > 1 else "")
            standalone_pair_coeff.append(
                f"{left} {right} {substyle if len(switches) > 1 else ''}{atomic_numbers[left_species]} "
                f"{atomic_numbers[right_species]}"
            )
            overlay_pair_coeff.append(
                f"{left} {right} {substyle}{atomic_numbers[left_species]} "
                f"{atomic_numbers[right_species]}"
            )
    timeout = int(config.get("timeout_seconds", 300))
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive.")
    body = {
        "schema": "ye3t_lammps_reference_potential_v2",
        "engine": "lammps_executable",
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "zbl",
        "type_order": list(species_order),
        "atomic_numbers": atomic_numbers,
        "standalone_pair_coeff": standalone_pair_coeff,
        "pair_coeff": overlay_pair_coeff,
        "target_operation": "ab_initio_minus_reference",
    }
    if pair_specific:
        arguments = " ".join(f"zbl {inner:.17g} {outer:.17g}" for inner, outer in switches)
        body.update(pair_cutoffs_A=pairs, overlay_pair_style_arguments=arguments,
                    standalone_pair_style=("hybrid/overlay " if len(switches) > 1 else "")+arguments)
    else:
        body.update(inner_cutoff_A=switches[0][0], outer_cutoff_A=switches[0][1])
    return {
        **body,
        "semantic_sha256": _semantic_hash(body),
        "executable": str(config.get("executable", "lmp")),
        "timeout_seconds": timeout,
    }


def _validated_lammps_zbl_reference_config(reference_config):
    reference_config = dict(reference_config)
    supplied_hash = str(reference_config.pop("semantic_sha256", ""))
    semantic = {
        key: value
        for key, value in reference_config.items()
        if key not in {"executable", "timeout_seconds"}
    }
    if not supplied_hash or supplied_hash != _semantic_hash(semantic):
        raise ValueError("LAMMPS reference-potential semantic hash mismatch.")
    if reference_config.get("schema") != "ye3t_lammps_reference_potential_v2":
        raise ValueError("Unsupported LAMMPS reference-potential schema.")
    normalized = lammps_zbl_reference_config(
        reference_config, reference_config.get("type_order", ())
    )
    supplied = {**reference_config, "semantic_sha256": supplied_hash}
    if json.dumps(normalized, sort_keys=True) != json.dumps(
        supplied, sort_keys=True
    ):
        raise ValueError("LAMMPS reference-potential record is not canonical.")
    return normalized


def evaluate_lammps_zbl_reference(structures, reference_config):
    """Evaluate reference energies/forces without changing input structures."""

    from ase.calculators.lammps.coordinatetransform import Prism
    from ase.io.lammpsdata import write_lammps_data

    structures = list(structures)
    if not structures:
        raise ValueError("Reference evaluation requires at least one structure.")
    reference_config = _validated_lammps_zbl_reference_config(reference_config)
    executable = shutil.which(str(reference_config["executable"]))
    if executable is None:
        raise FileNotFoundError(
            "LAMMPS reference executable not found: "
            f"{reference_config['executable']!r}."
        )
    energies = []
    forces = []
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        input_lines = []
        prisms = []
        for index, atoms in enumerate(structures):
            prism = Prism(atoms.cell, atoms.pbc)
            prisms.append(prism)
            boundary = " ".join(
                "p" if periodic else "f" for periodic in atoms.pbc
            )
            data_path = temporary / f"structure_{index:04d}.data"
            energy_path = temporary / f"energy_{index:04d}.txt"
            force_path = temporary / f"forces_{index:04d}.dump"
            with data_path.open("w", encoding="utf-8") as handle:
                write_lammps_data(
                    handle,
                    atoms,
                    specorder=reference_config["type_order"],
                    prismobj=prism,
                    masses=True,
                    units="metal",
                    atom_style="atomic",
                )
            input_lines.extend(
                (
                    "clear",
                    "units metal",
                    "atom_style atomic",
                    f"boundary {boundary}",
                    "atom_modify map array sort 0 0",
                    f"read_data {data_path}",
                    "neighbor 0.3 bin",
                    "neigh_modify delay 0 every 1 check yes",
                    "newton on",
                    "pair_style " + (reference_config["standalone_pair_style"]
                        if "standalone_pair_style" in reference_config else
                        f"zbl {reference_config['inner_cutoff_A']:.17g} {reference_config['outer_cutoff_A']:.17g}"),
                )
            )
            input_lines.extend(
                f"pair_coeff {value}"
                for value in reference_config["standalone_pair_coeff"]
            )
            input_lines.extend(
                (
                    "thermo_style custom step pe",
                    "run 0",
                    f'print "$(pe:%.17g)" file {energy_path} screen no',
                    f"write_dump all custom {force_path} id fx fy fz "
                    "modify sort id format float %.17g",
                )
            )
        input_path = temporary / "reference.in"
        input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [executable, "-screen", "none", "-log", "none", "-in", str(input_path)],
            check=True,
            timeout=int(reference_config["timeout_seconds"]),
        )
        for index, (atoms, prism) in enumerate(
            zip(structures, prisms, strict=True)
        ):
            energy = float(
                (temporary / f"energy_{index:04d}.txt").read_text(
                    encoding="utf-8"
                )
            )
            lines = (temporary / f"forces_{index:04d}.dump").read_text(
                encoding="utf-8"
            ).splitlines()
            atom_line = next(
                row
                for row, line in enumerate(lines)
                if line.startswith("ITEM: ATOMS")
            )
            fields = lines[atom_line].split()[2:]
            rows = [
                dict(zip(fields, map(float, line.split())))
                for line in lines[atom_line + 1 : atom_line + 1 + len(atoms)]
            ]
            rows.sort(key=lambda row: int(row["id"]))
            force = prism.vector_to_ase(
                np.asarray(
                    [
                        [row[axis] for axis in ("fx", "fy", "fz")]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
            )
            if not np.isfinite(energy) or not np.all(np.isfinite(force)):
                raise FloatingPointError(
                    "LAMMPS reference potential returned non-finite targets."
                )
            energies.append(energy)
            forces.append(force)
            digest.update(np.asarray([energy], dtype="<f8").tobytes())
            digest.update(np.asarray(force, dtype="<f8").tobytes())
    metadata = {
        **reference_config,
        "structure_count": len(structures),
        "evaluation_sha256": digest.hexdigest(),
        "executable_path": str(Path(executable).resolve()),
        "executable_sha256": _sha256(executable),
        "minimum_energy_eV_per_atom": float(
            min(energy / len(atoms) for energy, atoms in zip(energies, structures))
        ),
        "maximum_energy_eV_per_atom": float(
            max(energy / len(atoms) for energy, atoms in zip(energies, structures))
        ),
        "maximum_absolute_force_eV_per_A": float(
            max(np.max(np.abs(value)) for value in forces)
        ),
    }
    return {
        "reference_energies": np.asarray(energies, dtype=np.float64),
        "reference_forces": tuple(np.asarray(value) for value in forces),
        "metadata": metadata,
    }


def _portable_zbl_metadata(config):
    """Bind the portable numerical convention independently of geometry."""
    species = tuple(sorted(config["atomic_numbers"]))
    normalized = lammps_zbl_reference_config({**config, "engine": "lammps_executable"}, species)
    body = {"schema": "ye3t_portable_zbl_reference_v1", "engine": "numpy",
            "pair_style": "zbl", "units": "metal", "atomic_numbers": normalized["atomic_numbers"],
            "coulomb_constant_eV_A": 14.399645, "switch": "additive_gromacs_C2"}
    for key, value in body.items():
        if key in config and config[key] != value:
            raise ValueError(f"Unsupported portable ZBL convention for {key}.")
    if "pair_cutoffs_A" in config:
        body["pair_cutoffs_A"] = normalized["pair_cutoffs_A"]
    else:
        body.update(inner_cutoff_A=normalized["inner_cutoff_A"], outer_cutoff_A=normalized["outer_cutoff_A"])
    digest = _semantic_hash(body)
    if "semantic_sha256" in config and config["semantic_sha256"] != digest:
        raise ValueError("Portable ZBL semantic hash mismatch.")
    return {**body, "semantic_sha256": digest}


def evaluate_zbl_reference(structures, config):
    """Evaluate portable ZBL energies and analytic forces in eV and Angstrom.

    Mathematical contract:
        LAMMPS ZBL screening and the documented GROMACS C2 switching polynomial.
        Directed periodic neighbors carry half the pair energy and full force.
    Inputs/outputs:
        ASE structures and visible cutoffs/atomic numbers; the same E/F result
        fields as ``evaluate_lammps_zbl_reference``.
    Does not:
        Fit parameters, modify structures, or require a LAMMPS installation.

    Independent implementation from the mathematical specifications at
    https://docs.lammps.org/pair_zbl.html and pair_gromacs.html; no external
    implementation was copied or translated.
    """
    from ase.neighborlist import neighbor_list

    structures = list(structures)
    if not structures:
        raise ValueError("Reference evaluation requires structures.")
    metadata = _portable_zbl_metadata(config)
    species = tuple(sorted(config["atomic_numbers"]))
    normalized = lammps_zbl_reference_config({**config, "engine": "lammps_executable"}, species)
    pairs = _zbl_pair_cutoffs(normalized, species)
    maximum_outer = max(value[1] for value in pairs.values())
    if config.get("engine", "numpy") != "numpy":
        raise ValueError("Portable ZBL requires engine='numpy'.")
    numbers = normalized["atomic_numbers"]
    screening_weights = np.array([0.18175, 0.50986, 0.28022, 0.02817])
    screening_rates = np.array([3.19980, 0.94229, 0.40290, 0.20162])
    energies, forces, atomic_energies, virials = [], [], [], []
    for atoms in structures:
        symbols = atoms.get_chemical_symbols()
        if set(symbols)-set(numbers):
            raise ValueError("ZBL atomic numbers do not cover the structure species.")
        z = np.array([numbers[symbol] for symbol in symbols], dtype=np.float64)
        centers, neighbors, displacement, distance = neighbor_list("ijDd", atoms, maximum_outer, self_interaction=False)
        switches = np.asarray([pairs["-".join(sorted((symbols[i], symbols[j])))]
                               for i, j in zip(centers, neighbors)], dtype=np.float64).reshape(-1, 2)
        active = distance < switches[:, 1]
        centers, neighbors = centers[active], neighbors[active]
        displacement, distance = displacement[active], distance[active]
        inner, outer = switches[active, 0], switches[active, 1]
        if np.any(distance <= 0):
            raise ValueError("ZBL rejects coincident nuclei.")
        screening_length = 0.46850/(z[centers]**0.23+z[neighbors]**0.23)
        rates = screening_rates[None, :]/screening_length[:, None]
        # LAMMPS 'metal' electrostatic conversion, in eV Angstrom.
        amplitude = 14.399645*z[centers]*z[neighbors]

        def unswitched(radius):
            radius = np.broadcast_to(radius, distance.shape)
            terms = screening_weights[None, :]*np.exp(-rates*radius[:, None])
            phi = terms.sum(axis=1)
            first = -(rates*terms).sum(axis=1)
            second = (rates*rates*terms).sum(axis=1)
            return (amplitude*phi/radius,
                    amplitude*(first/radius-phi/radius**2),
                    amplitude*(second/radius-2*first/radius**2+2*phi/radius**3))

        value, derivative, _second = unswitched(distance)
        endpoint, endpoint_first, endpoint_second = unswitched(outer)
        width = outer-inner
        cubic = (-3*endpoint_first+width*endpoint_second)/width**2
        quartic = (2*endpoint_first-width*endpoint_second)/width**3
        constant = -endpoint+0.5*width*endpoint_first-width**2*endpoint_second/12
        shift = np.maximum(distance-inner, 0.0)
        value += constant+cubic*shift**3/3+quartic*shift**4/4
        derivative += cubic*shift**2+quartic*shift**3
        force = np.zeros((len(atoms), 3), dtype=np.float64)
        gradient = derivative[:, None]*displacement/distance[:, None]
        np.add.at(force, centers, gradient)
        atomic = np.zeros(len(atoms), dtype=np.float64)
        np.add.at(atomic, centers, 0.5*value)
        virial_matrix = -0.5*displacement.T @ gradient
        virial = virial_matrix[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
        if not np.all(np.isfinite(atomic)) or not np.all(np.isfinite(force)):
            raise FloatingPointError("Portable ZBL returned non-finite values.")
        energies.append(float(0.5*value.sum()))
        forces.append(force)
        atomic_energies.append(atomic)
        virials.append(virial)
    metadata["structure_count"] = len(structures)
    return {"reference_energies": np.asarray(energies), "reference_forces": tuple(forces),
            "reference_atomic_energies": tuple(atomic_energies),
            "reference_virials": np.asarray(virials), "metadata": metadata}


class YE3TZBLCalculator(Calculator):
    """ASE energy, force, atomic energy, and stress adapter for portable ZBL."""

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.reference_config = _portable_zbl_metadata(config)

    @classmethod
    def from_model_manifest(cls, path, **kwargs):
        """Read and verify a promoted paper model's LAMMPS ZBL record."""
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        reference = _validated_lammps_zbl_reference_config(manifest["reference_potential"])
        if reference.get("target_operation") != "ab_initio_minus_reference":
            raise ValueError("The model manifest does not declare a residual ZBL target.")
        config = {"engine": "numpy", "atomic_numbers": reference["atomic_numbers"]}
        if "pair_cutoffs_A" in reference:
            config["pair_cutoffs_A"] = reference["pair_cutoffs_A"]
        else:
            config["inner_cutoff_A"] = reference["inner_cutoff_A"]
            config["outer_cutoff_A"] = reference["outer_cutoff_A"]
        return cls(config, **kwargs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        record = evaluate_zbl_reference((self.atoms,), self.reference_config)
        energy = float(record["reference_energies"][0])
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "energies": record["reference_atomic_energies"][0],
            "forces": record["reference_forces"][0],
        }
        volume = float(self.atoms.get_volume()) if self.atoms.cell.rank == 3 else 0.0
        if volume > 0.0:
            virial = record["reference_virials"][0]
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / volume
        elif "stress" in properties:
            raise ValueError("ZBL stress requires a cell with positive volume.")


__all__ = [
    "YE3TZBLCalculator",
    "evaluate_zbl_reference",
    "evaluate_lammps_zbl_reference",
    "lammps_zbl_reference_config",
]
