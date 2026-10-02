"""Inspect one-hot chemical channels and a fixed in-memory embedding kernel."""

import torch

from ye3t_methods import Basis
from ye3t_ace.equivariant_calc.labeling import SingleChannelLabel
from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisConfig, SiteBasisV2


config = {
    "metadata": {"name": "chemical_encoding", "status": "low_level_source_example"},
    "basis": {
        "elements": ["Li", "Na"],
        "cutoff_A": 4.0,
        "max_rank": 1,
        "nmax": 1,
        "lmax": 0,
        "one_hot_chemical_basis": "delta",
        "fixed_embedding_rows": [[1.0, 0.0], [0.5, 0.8660254037844386]],
    },
    "representation": {"source": "ordinary_density", "target_L": 0},
    "runtime": {"backend": "torch", "device": "cpu", "dtype": "float64"},
    "model": {"type": "linear_source_probe"},
    "targets": {"energy": None, "forces": None},
    "validation": {"compare_embedded_to_one_hot_transform": True},
}


class FixedChemicalKernel:
    """Evaluate the Gram kernel of a fixed species embedding at an edge."""

    def __init__(self, embedding):
        rows = torch.as_tensor(embedding, dtype=torch.float64)
        if rows.ndim != 2 or rows.shape[0] < 1 or not torch.isfinite(rows).all():
            raise ValueError("fixed_embedding_rows must be a finite species-by-channel matrix")
        self.kernel = rows @ rows.T

    def __call__(self, *, mu0_edge, mu_edge, mu0, mu, evaluator):
        kernel = self.kernel.to(device=mu0_edge.device, dtype=evaluator.cfg.dtype)
        return kernel[mu0_edge, mu0] * kernel[mu_edge, mu]


elements = config["basis"]["elements"]
basis = Basis(
    elements=elements,
    cutoff=config["basis"]["cutoff_A"],
    max_rank=config["basis"]["max_rank"],
    nmax=config["basis"]["nmax"],
    lmax=config["basis"]["lmax"],
)
print("one-hot descriptor columns", len(basis.labels))
for label in basis.labels:
    print(label.as_dict()["one_factor_channels"])

site_config = SiteBasisConfig(
    rc=[config["basis"]["cutoff_A"]],
    lmbda=[0.25],
    nradmax=1,
    lmax=0,
    possible_types=(0, 1),
    chemical_basis="delta",
    charge_mode="none",
    atomic_base_normalization="none",
    factor_normalization="none",
    source_backend=config["runtime"]["backend"],
    dtype=torch.float64,
    complex_dtype=torch.complex128,
)
channels = tuple(
    SingleChannelLabel(mu0=0, mu=neighbor_type, kappa0=0, kappa=0,
                       n=1, l=0, m=0)
    for neighbor_type in (0, 1)
)
vectors = torch.tensor([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=torch.float64)
edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
types = torch.tensor([0, 0, 1], dtype=torch.long)

one_hot = SiteBasisV2(site_config)
embedding = FixedChemicalKernel(config["basis"]["fixed_embedding_rows"])
embedded = SiteBasisV2(site_config, chemical_provider=embedding)
identity = SiteBasisV2(
    site_config, chemical_provider=FixedChemicalKernel(torch.eye(len(elements))),
)
_, one_hot_values = one_hot.compute_site_basis(
    vectors, edges, types, channels, real_output=True,
)
_, embedded_values = embedded.compute_site_basis(
    vectors, edges, types, channels, real_output=True,
)
_, identity_values = identity.compute_site_basis(
    vectors, edges, types, channels, real_output=True,
)
kernel = embedding.kernel
torch.testing.assert_close(identity_values, one_hot_values)
torch.testing.assert_close(embedded_values[0], one_hot_values[0] @ kernel)
_, one_hot_edges, one_hot_dx = one_hot.compute_channel_edges_with_dx(
    vectors, edges, types, channels, real_output=True,
)
_, embedded_edges, embedded_dx = embedded.compute_channel_edges_with_dx(
    vectors, edges, types, channels, real_output=True,
)
torch.testing.assert_close(embedded_edges, one_hot_edges @ kernel)
torch.testing.assert_close(
    embedded_dx, torch.einsum("eac,ab->ebc", one_hot_dx, kernel),
)
print("one-hot source", one_hot_values[0].tolist())
print("fixed-kernel source", embedded_values[0].tolist())
