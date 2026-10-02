"""Build an explicit four-vertex motif descriptor on a depth-two tree."""

from ase import Atoms
from ye3t_methods import Basis
from ye3t_ace.cluster_phi import MotifTemplate, PhiMotifSpec, PhiSlotChannel


config = {
    "metadata": {
        "name": "phi_depth2", "formula": "H4",
        "positions_A": [[0, 0, 0], [1.0, 0, 0], [1.8, 0.6, 0], [1.7, -0.7, 0.2]],
        "cell_A": [8, 8, 8], "pbc": False,
    },
    "basis": {"elements": ["H"], "source": "bar_phi", "cutoff": 3.0,
              "channel": {"n": 1, "l": 0, "m": 0, "neighbor_type": 0},
              "edge_basis_backend": "simple", "backend": "pytorch"},
    "representation": {"motif": "depth2_tree", "slot_count": 4,
                       "edges": ((0, 1), (1, 2), (1, 3))},
    "runtime": {"device": "cpu"},
    "model": None,
    "targets": {"descriptor": True},
    "validation": {"check_coupling": True},
}
atoms = Atoms(
    config["metadata"]["formula"], positions=config["metadata"]["positions_A"],
    cell=config["metadata"]["cell_A"], pbc=config["metadata"]["pbc"],
)
channel = PhiSlotChannel(**config["basis"]["channel"])
motif = PhiMotifSpec(
    MotifTemplate(config["representation"]["motif"],
                  config["representation"]["slot_count"],
                  config["representation"]["edges"]),
    (channel,) * config["representation"]["slot_count"],
)
basis_settings = {key: value for key, value in config["basis"].items() if key != "channel"}
basis = Basis(**basis_settings, channels=(channel,), motif_specs=(motif,))
print("descriptor_shape", basis.create(atoms).shape)
label = basis.labels[0].as_dict()
print("motif", label["motif_template"])
print("coupling_validated", label["compiler_coupling_plan"]["validation_report"]["passed"])
