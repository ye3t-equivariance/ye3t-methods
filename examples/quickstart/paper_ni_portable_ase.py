"""Load the vetted Ni-127 portable model and evaluate its complete ASE energy."""

from pathlib import Path

from ase.build import bulk

from ye3t_methods import LinearModel


paper = Path(__file__).resolve().parents[1] / "publication" / "cost_comparison"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_portable_ase",
        "status": "stable",
        "system": {
            "element": "Ni", "lattice": "fcc", "lattice_parameter_A": 3.508,
            "repeat": (2, 2, 2),
            "first_atom_displacement_A": (0.08, -0.05, 0.04),
        },
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "torch", "neighbors": "ase"},
    "model": {
        "artifact": paper / "portable_models" / "Ni_ye3t_tagged_127.ye3t",
        "feature_count": 127,
    },
    "targets": {"properties": ("energy", "forces", "stress")},
    "validation": {
        "expected_parent_L": 0,
        "lammps_step_zero_energy_eV": -184.89616285503024,
        "energy_tolerance_eV": 1e-8,
    },
}

system = config["metadata"]["system"]
atoms = bulk(
    system["element"], system["lattice"],
    a=system["lattice_parameter_A"], cubic=True,
).repeat(system["repeat"])
atoms.positions[0] += system["first_atom_displacement_A"]

model = LinearModel.read(config["model"]["artifact"])
basis = model.basis
rows = basis.create(atoms)
assert rows.shape == (len(atoms), config["model"]["feature_count"])
assert all(label.as_dict()["L"] == config["validation"]["expected_parent_L"]
           for label in basis.labels)

atoms.calc = model.ase_calculator(
    evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"],
)
energy = atoms.get_potential_energy()
print("descriptor_rows", rows.shape)
print("energy_eV", energy)
print("maximum_force_eV_per_A", abs(atoms.get_forces()).max())
print("stress_eV_per_A3", atoms.get_stress())
assert abs(energy - config["validation"]["lammps_step_zero_energy_eV"]) < (
    config["validation"]["energy_tolerance_eV"]
)
