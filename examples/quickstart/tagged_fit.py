"""Fit and evaluate a rank-four tagged Cauchy image model."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {
        "name": "tagged_fit", "structures": fixtures / "ta3_training.extxyz",
    },
    "basis": {
        "elements": ["Ta"], "source": "tagged_cauchy_image", "cutoff": 4.8,
        "tensor_order": 4, "tag_counts": (0, 2), "radial_degrees": (0,),
        "angular_degree": 1, "backend": "reference",
    },
    "representation": {"parent_young": "trivial", "parent_L": 0},
    "runtime": {
        "ase_backend": "native_cpu", "native_library": None,
        "execution_policy": "direct",
        "output_path": output_root / "ta_tagged.ye3t.json",
    },
    "model": {"type": "linear", "regularization": 1e-8,
              "energy_weight": 1.0, "force_weight": 1.0, "stress_weight": 0.1},
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"evaluate_structure": fixtures / "ta3_structure.extxyz"},
}
if config["representation"] != {"parent_young": "trivial", "parent_L": 0}:
    raise ValueError("This linear tagged readout supports only a scalar invariant parent.")

structures = read(config["metadata"]["structures"], index=":")
basis = Basis(**config["basis"])
model = LinearModel(basis).fit(
    structures, regularization=config["model"]["regularization"],
    energy_weight=config["model"]["energy_weight"],
    force_weight=config["model"]["force_weight"],
    stress_weight=config["model"]["stress_weight"],
    energy_key=config["targets"]["energy"],
    force_key=config["targets"]["forces"],
    stress_key=config["targets"]["stress"],
)
config["runtime"]["output_path"].parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(config["runtime"]["output_path"])
restored = LinearModel.read(artifact)
# Choose "reference", "native_polynomial", or "native_cpu" for ASE evaluation.
atoms = read(config["validation"]["evaluate_structure"])
atoms.calc = restored.ase_calculator(
    backend=config["runtime"]["ase_backend"],
    native_library=config["runtime"]["native_library"],
    execution_policy=config["runtime"]["execution_policy"],
)
two_tag = next(label for label in basis.labels if any(
    tuple(raw["tag_kappa"]) == (1, 1)
    for raw in label.as_dict()["compiler_raw_opportunities"]
))
print(basis)
print("training_structures", len(structures))
print("two_tag_coordinate", two_tag.feature_index)
print("tag_kappa", sorted({
    tuple(raw["tag_kappa"])
    for raw in two_tag.as_dict()["compiler_raw_opportunities"]
}))
print("role_kappa", sorted({
    tuple(raw["role_kappa"])
    for raw in two_tag.as_dict()["compiler_raw_opportunities"]
}))
print("roundtrip_features", len(restored.labels))
print("saved_model", artifact)
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
print("stress_eV_per_A3", atoms.get_stress())
