"""Evaluate the saved Ni paper model, including its fitted ZBL reference."""
from pathlib import Path
from ase.build import bulk
from ye3t_methods import LinearModel

paper = Path(__file__).resolve().parent
config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "paper_ni_native", "status": "publication",
                 "system": {"element": "Ni", "lattice": "fcc", "lattice_parameter_A": 3.508,
                            "repeat": [2,2,2], "first_atom_displacement_A": [.08,-.05,.04]}},
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "neighbors": "ase", "native_library": None,
                "execution_policy": "direct", "device": "cpu", "dtype": "float64"},
    "model": {"artifact": str(paper / "lammps/Ni/models/ye3t_tagged_127/model.ye3t.json")},
    "targets": {"properties": ["energy", "forces", "stress"]},
    "validation": {"reference_energy_eV": -184.89616285503024, "energy_tolerance_eV": 1e-8},
}

system = config["metadata"]["system"]
atoms = bulk(system["element"], system["lattice"], a=system["lattice_parameter_A"], cubic=True).repeat(system["repeat"])
atoms.positions[0] += system["first_atom_displacement_A"]
model = LinearModel.read(config["model"]["artifact"])
atoms.calc = model.ase_calculator(evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"], native_library=config["runtime"]["native_library"],
    execution_policy=config["runtime"]["execution_policy"])
print("energy / eV", atoms.get_potential_energy())
print("largest force component / eV per Angstrom", abs(atoms.get_forces()).max())
print("stress / eV per cubic Angstrom", atoms.get_stress())
