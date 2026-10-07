"""Fit explicit pair and three-site star descriptors to ASE structures."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel
from ye3t_methods.atomistic.cluster_phi import MotifTemplate, PhiMotifSpec, PhiSlotChannel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
channel = PhiSlotChannel(n=1, l=0, m=0, neighbor_type=0)
motifs = (
    PhiMotifSpec(MotifTemplate("pair", 2, ()), (channel, channel)),
    PhiMotifSpec(MotifTemplate("star3", 3, ((0, 1), (0, 2))), (channel,) * 3),
)
config = {
    "metadata": {"name": "phi_fit", "structures": fixtures / "h3_training.extxyz"},
    "basis": {
        "elements": ["H"], "source": "bar_phi", "cutoff": 3.0,
        "channels": (channel,), "motif_specs": motifs,
        "edge_basis_backend": "simple", "backend": "pytorch",
    },
    "representation": {"source": "explicit_bar_phi", "parent_L": 0},
    "runtime": {"ase_backend": "pytorch",
                "output_path": output_root / "h3.phi.pt"},
    "model": {"type": "linear", "regularization": 1e-8,
              "energy_weight": 1.0, "force_weight": 1.0, "stress_weight": 0.1},
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"evaluate_structure": fixtures / "h3_structure.extxyz"},
}
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
atoms = read(config["validation"]["evaluate_structure"])
atoms.calc = restored.ase_calculator(backend=config["runtime"]["ase_backend"])
print(basis)
print("training_structures", len(structures))
print(restored.describe(0))
print("saved_model", artifact)
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
print("stress_eV_per_A3", atoms.get_stress())
