"""Build an explicit four-vertex motif descriptor on a depth-two tree."""

from ase import Atoms
from ye3t_methods import Basis
from ye3t_ace.cluster_phi import MotifTemplate, PhiMotifSpec, PhiSlotChannel


atoms = Atoms(
    "H4", positions=[[0, 0, 0], [1.0, 0, 0], [1.8, 0.6, 0], [1.7, -0.7, 0.2]],
    cell=[8, 8, 8], pbc=False,
)
channel = PhiSlotChannel(n=1, l=0, m=0, neighbor_type=0)
motif = PhiMotifSpec(
    MotifTemplate("depth2_tree", 4, ((0, 1), (1, 2), (1, 3))),
    (channel,) * 4,
)
basis = Basis(
    elements=["H"], source="bar_phi", cutoff=3.0,
    channels=(channel,), motif_specs=(motif,), edge_basis_backend="simple",
    backend="pytorch",
)
print("descriptor_shape", basis.create(atoms).shape)
label = basis.labels[0].as_dict()
print("motif", label["motif_template"])
print("coupling_validated", label["compiler_coupling_plan"]["validation_report"]["passed"])
