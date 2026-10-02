"""Concrete ACE descriptor adapters for the ye3t ACE boundary."""

import torch

from ye3t_ace.equivariant_calc.ace_eval_v2 import ACECovariantEvaluator
from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisV2


class AtomicBaseAdapter:
    """Interface for objects that provide ACE atomic base channels.

    The interface lives on the ACE side of the package split so the public
    `ye3t` core does not need an ACE-specific implementation module.
    """

    def compute_atomic_base(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges=None,
        aux_tensor_basis=None,
    ):
        """Return channel labels and the corresponding atomic base tensor."""
        raise NotImplementedError


class ACEAtomicBaseAdapter:
    """ACE ``A_i,nlm``/``phi_nlm`` evaluator behind the generic adapter."""

    def __init__(self, site_basis):
        self.site_basis = site_basis

    @classmethod
    def from_config(cls, config):
        return cls(SiteBasisV2(config))

    def compute_atomic_base(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges=None,
        aux_tensor_basis=None,
    ):
        """Evaluate ACE atomic base channels for one local environment batch."""
        return self.site_basis.compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )


class ACEDescriptorEvaluationAdapter:
    """Evaluate ACE descriptors while keeping ACE implementation at the edge."""

    def __init__(
        self,
        basis_config,
        *,
        atomic_base=None,
        backend="pytorch",
        strict_backend=False,
        validate_backend=True,
    ):
        self.atomic_base = atomic_base or ACEAtomicBaseAdapter.from_config(basis_config)
        self.evaluator = ACECovariantEvaluator(
            basis_config,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
        )
        if isinstance(self.atomic_base, ACEAtomicBaseAdapter):
            self.evaluator.site_basis = self.atomic_base.site_basis

    def evaluate(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        descriptors,
        charges=None,
        aux_tensor_basis=None,
        real_if_scalar=True,
        imag_tol=1e-12,
    ):
        """Evaluate descriptors using either the native ACE base or an adapter."""
        if not isinstance(self.atomic_base, ACEAtomicBaseAdapter):
            return self._evaluate_with_atomic_base(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                descriptors=descriptors,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
                real_if_scalar=real_if_scalar,
                imag_tol=imag_tol,
            )
        return self.evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=descriptors,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
            imag_tol=imag_tol,
        )

    __call__ = evaluate

    def _evaluate_with_atomic_base(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        descriptors,
        charges,
        aux_tensor_basis,
        real_if_scalar,
        imag_tol,
    ):
        """Evaluate descriptors from an externally supplied atomic base."""
        if len(descriptors) == 0:
            return torch.zeros((atom_types.shape[0], 0), dtype=self.evaluator.site_basis.cfg.complex_dtype, device=x_ij.device)
        compiled = self.evaluator._compile_descriptors(descriptors)
        _, atomic_base = self.atomic_base.compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        return self.evaluator._contract_atomic_products(
            compiled,
            atomic_base,
            descriptor_count=len(descriptors),
            real_if_scalar=real_if_scalar,
            imag_tol=imag_tol,
        )


__all__ = [
    "ACEAtomicBaseAdapter",
    "ACEDescriptorEvaluationAdapter",
]
