"""Fit a fixed-feature explicit-cluster barPhi model to manufactured H3 data."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel
from ye3t_ace.cluster_phi import MotifTemplate, PhiMotifSpec, PhiSlotChannel


fixtures = Path(__file__).with_name("fixtures")
structures = read(fixtures / "h3_training.extxyz", index=":")
channel = PhiSlotChannel(n=1, l=0, m=0, neighbor_type=0)
motifs = (
    PhiMotifSpec(MotifTemplate("pair", 2, ()), (channel, channel)),
    PhiMotifSpec(MotifTemplate("star3", 3, ((0, 1), (0, 2))), (channel,) * 3),
)
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"name": "phi_fit_quickstart", "status": "stable"},
    "basis": {"elements": ["H"], "source": "bar_phi", "cutoff": 3.0,
              "channels": (channel,), "motif_specs": motifs,
              "edge_basis_backend": "simple", "backend": "pytorch"},
    "representation": {"carrier": "bar_phi", "target": {"permutation": "trivial", "L": 0},
                       "coupling": {"source": "ye3t.couplings"}},
    "runtime": {"ase_backend": "pytorch", "output_path": output_root / "h3_fitted.phi.pt"},
    "model": {"type": "linear", "regularization": 1e-12, "stress_weight": 0.1},
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"fixture": "manufactured H3", "checks": ["energy", "forces", "stress"]},
}
basis = Basis(**config["basis"])
model = LinearModel(basis).fit(
    structures, regularization=config["model"]["regularization"],
    stress_weight=config["model"]["stress_weight"],
)
config["runtime"]["output_path"].parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(config["runtime"]["output_path"])
restored = LinearModel.read(artifact)
atoms = read(fixtures / "h3_structure.extxyz")
atoms.calc = restored.ase_calculator(backend=config["runtime"]["ase_backend"])
print(basis)
print(restored.describe(0))
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
print("stress_eV_per_A3", atoms.get_stress())
