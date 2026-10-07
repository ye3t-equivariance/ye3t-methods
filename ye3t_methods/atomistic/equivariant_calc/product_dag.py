
from dataclasses import field
import os

import torch
from ye3t_methods.atomistic._record import recordclass

try:
    from ye3t_methods.atomistic._runtime import configure_runtime_environment
except Exception:  # pragma: no cover - local-source fallback
    from _runtime import configure_runtime_environment

configure_runtime_environment(force_local_temp=True)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None

from .ace_eval_v2 import CompiledDescriptorBatch


def _ace_env(name, default=None):
    return os.environ.get("YE3T_ACE_" + str(name), os.environ.get("gne3_ace_" + str(name), default))


def _triton_disabled():
    return os.environ.get("GNE3_ACE_DISABLE_TRITON") == "1" or _ace_env("DISABLE_TRITON") == "1"


@recordclass(('left', 'right'), frozen = True)
class ProductDAGNode:
    pass


@recordclass(('descriptor_index', 'node_index', 'coeff'), frozen = True)
class ProductDAGOutputTerm:
    pass


@recordclass(('n_roots', 'nodes', 'output_terms', 'descriptor_count', '_tensor_schedule_cache', '_reverse_chunk_schedule_cache', '_reverse_depth_schedule_cache'), frozen = True)
class ProductDAG:
    """Internal binary product expansion schedule over atomic-base channels.

    Root nodes ``0..n_roots-1`` correspond to atomic-base channels. Interior
    nodes store ordered binary products. The reverse adjoint pass follows the
    product rule and returns adjoints with respect to root channels.
    This helper is intentionally narrow: it records descriptor-product
    dependencies and tensor schedules, not a general runtime graph executor.
    """
    _tensor_schedule_cache = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _reverse_chunk_schedule_cache = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _reverse_depth_schedule_cache = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def n_nodes_total(self):
        return int(self.n_roots + len(self.nodes))

    @classmethod
    def from_compiled_descriptors(cls, compiled, *, descriptor_count):
        product_node_by_key = {}
        nodes = []
        output_terms = []

        def get_product_node(left, right):
            key = (int(left), int(right))
            existing = product_node_by_key.get(key)
            if existing is not None:
                return existing
            node_index = int(len(compiled.channels) + len(nodes))
            product_node_by_key[key] = node_index
            nodes.append(ProductDAGNode(left=int(left), right=int(right)))
            return node_index

        for descriptor_index, (rows, coeffs) in enumerate(zip(compiled.channel_rows_cpu, compiled.coeffs_cpu)):
            for row, coeff in zip(rows.tolist(), coeffs.tolist()):
                if not row:
                    continue
                current = int(row[0])
                for channel_index in row[1:]:
                    current = get_product_node(current, int(channel_index))
                output_terms.append(
                    ProductDAGOutputTerm(
                        descriptor_index=int(descriptor_index),
                        node_index=int(current),
                        coeff=complex(coeff),
                    )
                )
        return cls(
            n_roots=len(compiled.channels),
            nodes=tuple(nodes),
            output_terms=tuple(output_terms),
            descriptor_count=int(descriptor_count),
        )


@recordclass(('output_descriptor_indices', 'output_node_indices', 'output_coeffs', 'node_left_indices', 'node_right_indices'), frozen = True)
class ProductDAGTensorSchedule:
    """Packed tensor view of ProductDAG output wiring for one device/dtype."""


@recordclass(('node_offsets', 'chunk_size', 'chunk_count', 'policy'), frozen = True)
class ProductDAGReverseChunkTensorSchedule:
    """Packed reverse-node chunks for scheduled ProductDAG traversal."""


@recordclass(('node_offsets', 'level_starts', 'level_counts', 'level_depths'), frozen = True)
class ProductDAGDepthTensorSchedule:
    """Packed reverse ProductDAG nodes grouped by product depth."""

    @property
    def level_count(self):
        return int(len(self.level_counts))


@recordclass(('all_node_values', 'descriptor_values'), frozen = True)
class ProductDAGValues:
    pass


def product_dag_from_compiled_cached(compiled, *, descriptor_count):
    """Return a cached internal product expansion schedule."""

    cache = getattr(compiled, "_product_dag_cache", None)
    if cache is None:
        cache = {}
        setattr(compiled, "_product_dag_cache", cache)
    key = int(descriptor_count)
    dag = cache.get(key)
    if dag is None:
        dag = ProductDAG.from_compiled_descriptors(compiled, descriptor_count=descriptor_count)
        cache[key] = dag
    return dag


ProductExpansionSchedule = ProductDAG
ProductExpansionNode = ProductDAGNode
ProductExpansionOutputTerm = ProductDAGOutputTerm
product_expansion_schedule_from_compiled_cached = product_dag_from_compiled_cached


def product_dag_coefficients_are_real(dag, *, tol = 1.0e-14):
    return all(abs(complex(term.coeff).imag) <= float(tol) for term in dag.output_terms)


def _tensor_schedule(dag, *, device, dtype):
    key = (str(device), str(dtype))
    cached = dag._tensor_schedule_cache.get(key)
    if cached is not None:
        return cached
    if dtype in (torch.float32, torch.float64):
        if not product_dag_coefficients_are_real(dag):
            raise ValueError("Cannot materialize complex ProductDAG coefficients into a real dtype schedule.")
        coeff_values = [float(complex(term.coeff).real) for term in dag.output_terms]
    else:
        coeff_values = [term.coeff for term in dag.output_terms]
    cached = ProductDAGTensorSchedule(
        output_descriptor_indices=torch.tensor(
            [term.descriptor_index for term in dag.output_terms],
            dtype=torch.long,
            device=device,
        ),
        output_node_indices=torch.tensor(
            [term.node_index for term in dag.output_terms],
            dtype=torch.long,
            device=device,
        ),
        output_coeffs=torch.tensor(
            coeff_values,
            dtype=dtype,
            device=device,
        ),
        node_left_indices=torch.tensor(
            [node.left for node in dag.nodes],
            dtype=torch.long,
            device=device,
        ),
        node_right_indices=torch.tensor(
            [node.right for node in dag.nodes],
            dtype=torch.long,
            device=device,
        ),
    )
    dag._tensor_schedule_cache[key] = cached
    return cached


def _product_dag_node_depths(dag):
    depths = [0 for _ in range(dag.n_nodes_total)]
    for offset, node in enumerate(dag.nodes):
        depths[dag.n_roots + offset] = max(depths[node.left], depths[node.right]) + 1
    return depths[dag.n_roots :]


def _product_dag_node_reuse_scores(dag):
    child_use = [0 for _ in range(dag.n_nodes_total)]
    output_use = [0 for _ in range(dag.n_nodes_total)]
    for node in dag.nodes:
        child_use[node.left] += 1
        child_use[node.right] += 1
    for term in dag.output_terms:
        output_use[term.node_index] += 1
    return [
        int(2 * child_use[dag.n_roots + offset] + 4 * output_use[dag.n_roots + offset])
        for offset in range(len(dag.nodes))
    ]


def _reverse_chunk_node_offsets(dag, *, chunk_size, policy):
    chunk_size = max(1, int(chunk_size))
    policy = str(policy).strip().lower().replace("-", "_")
    if policy == "flat":
        chunks = []
        last_start = ((len(dag.nodes) - 1) // chunk_size) * chunk_size if dag.nodes else 0
        for chunk_start in range(last_start, -1, -chunk_size):
            nodes = tuple(
                offset
                for offset in range(chunk_start + chunk_size - 1, chunk_start - 1, -1)
                if offset < len(dag.nodes)
            )
            if nodes:
                chunks.append(nodes)
        return tuple(chunks)
    if policy not in {"subtree", "depth", "subtree_depth"}:
        raise ValueError("ProductDAG reverse chunk policy must be flat or subtree.")

    depths = _product_dag_node_depths(dag)
    reuse_scores = _product_dag_node_reuse_scores(dag)
    # Decreasing depth is reverse topological: every product node has strictly
    # greater depth than its children. Reuse score keeps shared subtree roots
    # near one another inside each depth band.
    ordered = sorted(
        range(len(dag.nodes)),
        key=lambda offset: (-depths[offset], -reuse_scores[offset], -offset),
    )
    return tuple(
        tuple(ordered[start : start + chunk_size])
        for start in range(0, len(ordered), chunk_size)
        if ordered[start : start + chunk_size]
    )


def _reverse_chunk_schedule(
    dag,
    *,
    device,
    chunk_size,
    policy,
):
    chunk_size = max(1, int(chunk_size))
    policy = str(policy).strip().lower().replace("-", "_")
    key = (str(device), chunk_size, policy)
    cached = dag._reverse_chunk_schedule_cache.get(key)
    if cached is not None:
        return cached
    chunks = _reverse_chunk_node_offsets(dag, chunk_size=chunk_size, policy=policy)
    padded = []
    for chunk in chunks:
        padded.extend(int(offset) for offset in chunk)
        padded.extend([-1] * (chunk_size - len(chunk)))
    node_offsets = torch.tensor(padded, dtype=torch.long, device=device)
    cached = ProductDAGReverseChunkTensorSchedule(
        node_offsets=node_offsets,
        chunk_size=int(chunk_size),
        chunk_count=int(len(chunks)),
        policy=policy,
    )
    dag._reverse_chunk_schedule_cache[key] = cached
    return cached


def _reverse_depth_schedule(dag, *, device):
    key = str(device)
    cached = dag._reverse_depth_schedule_cache.get(key)
    if cached is not None:
        return cached
    depths = _product_dag_node_depths(dag)
    by_depth = {}
    reuse_scores = _product_dag_node_reuse_scores(dag)
    for offset, depth in enumerate(depths):
        by_depth.setdefault(int(depth), []).append(int(offset))
    ordered_depths = tuple(sorted(by_depth.keys(), reverse=True))
    flat_offsets = []
    starts = []
    counts = []
    for depth in ordered_depths:
        starts.append(len(flat_offsets))
        nodes = sorted(by_depth[depth], key=lambda offset: (-reuse_scores[offset], -offset))
        flat_offsets.extend(nodes)
        counts.append(len(nodes))
    cached = ProductDAGDepthTensorSchedule(
        node_offsets=torch.tensor(flat_offsets, dtype=torch.long, device=device),
        level_starts=tuple(starts),
        level_counts=tuple(counts),
        level_depths=ordered_depths,
    )
    dag._reverse_depth_schedule_cache[key] = cached
    return cached


if triton is not None:  # pragma: no cover - exercised only on CUDA/Triton systems
    @triton.jit
    def _reverse_product_dag_output_init_kernel(
        output_adjoint_ptr,
        omega_ptr,
        out_desc_ptr,
        out_node_ptr,
        out_coeff_ptr,
        term_start,
        n_atoms,
        n_descriptors,
        n_total,
        n_terms,
    ):
        atoms = tl.program_id(0) * 64 + tl.arange(0, 64)
        terms = term_start + tl.program_id(1) * 64 + tl.arange(0, 64)
        atom_mask = atoms < n_atoms
        term_mask = terms < n_terms
        desc = tl.load(out_desc_ptr + terms, mask=term_mask, other=0)
        node = tl.load(out_node_ptr + terms, mask=term_mask, other=0)
        coeff = tl.load(out_coeff_ptr + terms, mask=term_mask, other=0.0)
        adj = tl.load(
            output_adjoint_ptr + atoms[:, None] * n_descriptors + desc[None, :],
            mask=atom_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        tl.atomic_add(
            omega_ptr + atoms[:, None] * n_total + node[None, :],
            adj * coeff[None, :],
            mask=atom_mask[:, None] & term_mask[None, :],
        )

    @triton.jit
    def _reverse_product_dag_kernel(
        output_adjoint_ptr,
        all_values_ptr,
        root_adjoint_ptr,
        out_desc_ptr,
        out_node_ptr,
        out_coeff_ptr,
        node_left_ptr,
        node_right_ptr,
        n_atoms,
        n_descriptors,
        n_roots,
        n_nodes,
        n_total,
        n_terms,
        BLOCK_ATOMS,
    ):
        atoms = tl.program_id(0) * BLOCK_ATOMS + tl.arange(0, BLOCK_ATOMS)
        atom_mask = atoms < n_atoms

        for term_idx in tl.static_range(0, n_terms):
            desc = tl.load(out_desc_ptr + term_idx)
            node = tl.load(out_node_ptr + term_idx)
            coeff = tl.load(out_coeff_ptr + term_idx)
            out_adj = tl.load(
                output_adjoint_ptr + atoms * n_descriptors + desc,
                mask=atom_mask,
                other=0.0,
            )
            tl.atomic_add(
                root_adjoint_ptr + atoms * n_total + node,
                out_adj * coeff,
                mask=atom_mask,
            )

        for reverse_offset in tl.static_range(0, n_nodes):
            node_offset = n_nodes - 1 - reverse_offset
            node_index = n_roots + node_offset
            left = tl.load(node_left_ptr + node_offset)
            right = tl.load(node_right_ptr + node_offset)
            node_omega = tl.load(
                root_adjoint_ptr + atoms * n_total + node_index,
                mask=atom_mask,
                other=0.0,
            )
            left_val = tl.load(all_values_ptr + atoms * n_total + left, mask=atom_mask, other=0.0)
            right_val = tl.load(all_values_ptr + atoms * n_total + right, mask=atom_mask, other=0.0)
            tl.atomic_add(root_adjoint_ptr + atoms * n_total + left, node_omega * right_val, mask=atom_mask)
            tl.atomic_add(root_adjoint_ptr + atoms * n_total + right, node_omega * left_val, mask=atom_mask)

    @triton.jit
    def _reverse_product_dag_complex_split_kernel(
        output_adjoint_real_ptr,
        output_adjoint_imag_ptr,
        values_real_ptr,
        values_imag_ptr,
        omega_real_ptr,
        omega_imag_ptr,
        out_desc_ptr,
        out_node_ptr,
        out_coeff_real_ptr,
        out_coeff_imag_ptr,
        node_left_ptr,
        node_right_ptr,
        n_atoms,
        n_descriptors,
        n_roots,
        n_nodes,
        n_total,
        n_terms,
        BLOCK_ATOMS,
    ):
        atoms = tl.program_id(0) * BLOCK_ATOMS + tl.arange(0, BLOCK_ATOMS)
        atom_mask = atoms < n_atoms

        for term_idx in tl.static_range(0, n_terms):
            desc = tl.load(out_desc_ptr + term_idx)
            node = tl.load(out_node_ptr + term_idx)
            coeff_re = tl.load(out_coeff_real_ptr + term_idx)
            coeff_im = tl.load(out_coeff_imag_ptr + term_idx)
            adj_re = tl.load(
                output_adjoint_real_ptr + atoms * n_descriptors + desc,
                mask=atom_mask,
                other=0.0,
            )
            adj_im = tl.load(
                output_adjoint_imag_ptr + atoms * n_descriptors + desc,
                mask=atom_mask,
                other=0.0,
            )
            term_re = adj_re * coeff_re - adj_im * coeff_im
            term_im = adj_re * coeff_im + adj_im * coeff_re
            tl.atomic_add(omega_real_ptr + atoms * n_total + node, term_re, mask=atom_mask)
            tl.atomic_add(omega_imag_ptr + atoms * n_total + node, term_im, mask=atom_mask)

        for reverse_offset in tl.static_range(0, n_nodes):
            node_offset = n_nodes - 1 - reverse_offset
            node_index = n_roots + node_offset
            left = tl.load(node_left_ptr + node_offset)
            right = tl.load(node_right_ptr + node_offset)
            omega_re = tl.load(omega_real_ptr + atoms * n_total + node_index, mask=atom_mask, other=0.0)
            omega_im = tl.load(omega_imag_ptr + atoms * n_total + node_index, mask=atom_mask, other=0.0)

            left_re = tl.load(values_real_ptr + atoms * n_total + left, mask=atom_mask, other=0.0)
            left_im = tl.load(values_imag_ptr + atoms * n_total + left, mask=atom_mask, other=0.0)
            right_re = tl.load(values_real_ptr + atoms * n_total + right, mask=atom_mask, other=0.0)
            right_im = tl.load(values_imag_ptr + atoms * n_total + right, mask=atom_mask, other=0.0)

            left_contrib_re = omega_re * right_re - omega_im * right_im
            left_contrib_im = omega_re * right_im + omega_im * right_re
            right_contrib_re = omega_re * left_re - omega_im * left_im
            right_contrib_im = omega_re * left_im + omega_im * left_re

            tl.atomic_add(omega_real_ptr + atoms * n_total + left, left_contrib_re, mask=atom_mask)
            tl.atomic_add(omega_imag_ptr + atoms * n_total + left, left_contrib_im, mask=atom_mask)
            tl.atomic_add(omega_real_ptr + atoms * n_total + right, right_contrib_re, mask=atom_mask)
            tl.atomic_add(omega_imag_ptr + atoms * n_total + right, right_contrib_im, mask=atom_mask)

    @triton.jit
    def _reverse_product_dag_node_chunk_kernel(
        all_values_ptr,
        omega_ptr,
        node_left_ptr,
        node_right_ptr,
        chunk_start,
        n_atoms,
        n_roots,
        n_nodes,
        n_total,
        BLOCK_ATOMS,
        CHUNK_SIZE,
    ):
        atoms = tl.program_id(0) * BLOCK_ATOMS + tl.arange(0, BLOCK_ATOMS)
        atom_mask = atoms < n_atoms

        for reverse_offset in tl.static_range(0, CHUNK_SIZE):
            node_offset = chunk_start + CHUNK_SIZE - 1 - reverse_offset
            active = node_offset < n_nodes
            node_index = n_roots + node_offset
            left = tl.load(node_left_ptr + node_offset, mask=active, other=0)
            right = tl.load(node_right_ptr + node_offset, mask=active, other=0)
            node_omega = tl.load(
                omega_ptr + atoms * n_total + node_index,
                mask=atom_mask & active,
                other=0.0,
            )
            left_val = tl.load(all_values_ptr + atoms * n_total + left, mask=atom_mask & active, other=0.0)
            right_val = tl.load(all_values_ptr + atoms * n_total + right, mask=atom_mask & active, other=0.0)
            tl.atomic_add(omega_ptr + atoms * n_total + left, node_omega * right_val, mask=atom_mask & active)
            tl.atomic_add(omega_ptr + atoms * n_total + right, node_omega * left_val, mask=atom_mask & active)

    @triton.jit
    def _reverse_product_dag_scheduled_node_chunk_kernel(
        all_values_ptr,
        omega_ptr,
        scheduled_offsets_ptr,
        node_left_ptr,
        node_right_ptr,
        chunk_index,
        n_atoms,
        n_roots,
        n_total,
        BLOCK_ATOMS,
        CHUNK_SIZE,
    ):
        atoms = tl.program_id(0) * BLOCK_ATOMS + tl.arange(0, BLOCK_ATOMS)
        atom_mask = atoms < n_atoms

        for local_offset in tl.static_range(0, CHUNK_SIZE):
            node_offset = tl.load(scheduled_offsets_ptr + chunk_index * CHUNK_SIZE + local_offset)
            active = node_offset >= 0
            node_index = n_roots + node_offset
            left = tl.load(node_left_ptr + node_offset, mask=active, other=0)
            right = tl.load(node_right_ptr + node_offset, mask=active, other=0)
            node_omega = tl.load(
                omega_ptr + atoms * n_total + node_index,
                mask=atom_mask & active,
                other=0.0,
            )
            left_val = tl.load(all_values_ptr + atoms * n_total + left, mask=atom_mask & active, other=0.0)
            right_val = tl.load(all_values_ptr + atoms * n_total + right, mask=atom_mask & active, other=0.0)
            tl.atomic_add(omega_ptr + atoms * n_total + left, node_omega * right_val, mask=atom_mask & active)
            tl.atomic_add(omega_ptr + atoms * n_total + right, node_omega * left_val, mask=atom_mask & active)

    @triton.jit
    def _reverse_product_dag_depth_level_kernel(
        all_values_ptr,
        omega_ptr,
        level_offsets_ptr,
        node_left_ptr,
        node_right_ptr,
        level_start,
        level_count,
        n_atoms,
        n_roots,
        n_total,
    ):
        atom_offsets = tl.program_id(0) * 32 + tl.arange(0, 32)
        node_local_offsets = tl.program_id(1) * 32 + tl.arange(0, 32)
        atom_mask = atom_offsets < n_atoms
        node_mask = node_local_offsets < level_count
        node_offsets = tl.load(level_offsets_ptr + level_start + node_local_offsets, mask=node_mask, other=0)
        node_indices = n_roots + node_offsets
        left = tl.load(node_left_ptr + node_offsets, mask=node_mask, other=0)
        right = tl.load(node_right_ptr + node_offsets, mask=node_mask, other=0)
        node_omega = tl.load(
            omega_ptr + atom_offsets[:, None] * n_total + node_indices[None, :],
            mask=atom_mask[:, None] & node_mask[None, :],
            other=0.0,
        )
        left_val = tl.load(
            all_values_ptr + atom_offsets[:, None] * n_total + left[None, :],
            mask=atom_mask[:, None] & node_mask[None, :],
            other=0.0,
        )
        right_val = tl.load(
            all_values_ptr + atom_offsets[:, None] * n_total + right[None, :],
            mask=atom_mask[:, None] & node_mask[None, :],
            other=0.0,
        )
        tl.atomic_add(
            omega_ptr + atom_offsets[:, None] * n_total + left[None, :],
            node_omega * right_val,
            mask=atom_mask[:, None] & node_mask[None, :],
        )
        tl.atomic_add(
            omega_ptr + atom_offsets[:, None] * n_total + right[None, :],
            node_omega * left_val,
            mask=atom_mask[:, None] & node_mask[None, :],
        )


def _triton_reverse_available(output_adjoint, dag_values, dag):
    if _triton_disabled():
        return False
    min_work = int(_ace_env("TRITON_PRODUCT_DAG_MIN_WORK", "4096"))
    max_nodes = int(_ace_env("TRITON_PRODUCT_DAG_MAX_NODES", "128"))
    max_terms = int(_ace_env("TRITON_PRODUCT_DAG_MAX_TERMS", "512"))
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and output_adjoint.is_cuda
        and dag_values.all_node_values.is_cuda
        and output_adjoint.is_contiguous()
        and dag_values.all_node_values.is_contiguous()
        and output_adjoint.dtype in (torch.float32, torch.float64)
        and dag_values.all_node_values.dtype == output_adjoint.dtype
        and not output_adjoint.is_complex()
        and not dag_values.all_node_values.is_complex()
        and len(dag.output_terms) > 0
        and len(dag.nodes) > 0
        and len(dag.nodes) <= max_nodes
        and len(dag.output_terms) <= max_terms
        and int(output_adjoint.shape[0]) * int(dag.n_nodes_total + len(dag.output_terms)) >= min_work
    )


def _triton_reverse_complex_split_available(output_adjoint, dag_values, dag):
    if _triton_disabled():
        return False
    min_work = int(_ace_env("TRITON_PRODUCT_DAG_MIN_WORK", "4096"))
    max_nodes = int(_ace_env("TRITON_PRODUCT_DAG_MAX_NODES", "128"))
    max_terms = int(_ace_env("TRITON_PRODUCT_DAG_MAX_TERMS", "512"))
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and output_adjoint.is_cuda
        and dag_values.all_node_values.is_cuda
        and output_adjoint.is_contiguous()
        and dag_values.all_node_values.is_contiguous()
        and output_adjoint.is_complex()
        and dag_values.all_node_values.is_complex()
        and output_adjoint.dtype in (torch.complex64, torch.complex128)
        and dag_values.all_node_values.dtype == output_adjoint.dtype
        and len(dag.output_terms) > 0
        and len(dag.nodes) > 0
        and len(dag.nodes) <= max_nodes
        and len(dag.output_terms) <= max_terms
        and int(output_adjoint.shape[0]) * int(dag.n_nodes_total + len(dag.output_terms)) >= min_work
    )


def _complex_real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _real_reverse_base_available(output_adjoint, dag_values, dag):
    if _triton_disabled():
        return False
    min_work = int(_ace_env("TRITON_PRODUCT_DAG_MIN_WORK", "4096"))
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and output_adjoint.is_cuda
        and dag_values.all_node_values.is_cuda
        and output_adjoint.is_contiguous()
        and dag_values.all_node_values.is_contiguous()
        and output_adjoint.dtype in (torch.float32, torch.float64)
        and dag_values.all_node_values.dtype == output_adjoint.dtype
        and not output_adjoint.is_complex()
        and not dag_values.all_node_values.is_complex()
        and len(dag.output_terms) > 0
        and len(dag.nodes) > 0
        and int(output_adjoint.shape[0]) * int(dag.n_nodes_total + len(dag.output_terms)) >= min_work
    )


def _initialize_reverse_omega_reference(output_adjoint, dag_values, dag):
    omega = torch.zeros_like(dag_values.all_node_values)
    if dag.output_terms:
        schedule = _tensor_schedule(dag, device=output_adjoint.device, dtype=output_adjoint.dtype)
        term_omega = (
            output_adjoint.index_select(1, schedule.output_descriptor_indices)
            * schedule.output_coeffs.unsqueeze(0)
        )
        omega.index_add_(1, schedule.output_node_indices, term_omega)
    return omega


def _initialize_reverse_omega_maybe_triton(
    output_adjoint,
    dag_values,
    dag,
    schedule,
):
    omega = torch.zeros_like(dag_values.all_node_values)
    min_terms = int(_ace_env("TRITON_PRODUCT_DAG_OUTPUT_INIT_MIN_TERMS", "256"))
    max_term_block = int(_ace_env("TRITON_PRODUCT_DAG_OUTPUT_INIT_BLOCK_TERMS", "64"))
    block_terms = max(1, min(max_term_block, int(_ace_env("TRITON_PRODUCT_DAG_MAX_TERMS", "512"))))
    can_triton = bool(
        triton is not None
        and not torch.is_grad_enabled()
        and output_adjoint.is_cuda
        and omega.is_cuda
        and output_adjoint.is_contiguous()
        and output_adjoint.dtype in (torch.float32, torch.float64)
        and not output_adjoint.is_complex()
        and len(dag.output_terms) >= min_terms
        and block_terms == 64
    )
    if not can_triton:
        if dag.output_terms:
            term_omega = (
                output_adjoint.index_select(1, schedule.output_descriptor_indices)
                * schedule.output_coeffs.unsqueeze(0)
            )
            omega.index_add_(1, schedule.output_node_indices, term_omega)
        return omega, "torch_output_init"
    block_atoms = 64
    grid = (
        triton.cdiv(int(output_adjoint.shape[0]), block_atoms),
        triton.cdiv(int(len(dag.output_terms)), block_terms),
    )
    try:
        _reverse_product_dag_output_init_kernel[grid](
            output_adjoint,
            omega,
            schedule.output_descriptor_indices,
            schedule.output_node_indices,
            schedule.output_coeffs,
            term_start=0,
            n_atoms=int(output_adjoint.shape[0]),
            n_descriptors=int(output_adjoint.shape[1]),
            n_total=int(dag.n_nodes_total),
            n_terms=int(len(dag.output_terms)),
        )
        return omega, "triton_output_init"
    except Exception:
        if os.environ.get("GNE3_DEBUG_TRITON") == "1":
            raise
        if dag.output_terms:
            term_omega = (
                output_adjoint.index_select(1, schedule.output_descriptor_indices)
                * schedule.output_coeffs.unsqueeze(0)
            )
            omega.index_add_(1, schedule.output_node_indices, term_omega)
        return omega, "torch_output_init"


def reverse_product_dag_adjoint_chunked_triton(
    output_adjoint,
    dag_values,
    dag,
):
    if not _real_reverse_base_available(output_adjoint, dag_values, dag):
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"
    chunk_size = int(_ace_env("TRITON_PRODUCT_DAG_CHUNK_NODES", "64"))
    chunk_size = max(1, min(chunk_size, int(_ace_env("TRITON_PRODUCT_DAG_MAX_NODES", "128"))))
    chunk_policy = _ace_env("TRITON_PRODUCT_DAG_CHUNK_POLICY", "subtree")
    schedule = _tensor_schedule(dag, device=output_adjoint.device, dtype=output_adjoint.dtype)
    all_values = dag_values.all_node_values.contiguous()
    output_adjoint = output_adjoint.contiguous()
    chunk_schedule = _reverse_chunk_schedule(
        dag,
        device=output_adjoint.device,
        chunk_size=chunk_size,
        policy=chunk_policy,
    )
    max_chunks = int(_ace_env("TRITON_PRODUCT_DAG_MAX_CHUNKS", "2048"))
    if chunk_schedule.chunk_count > max_chunks:
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"
    omega, init_backend = _initialize_reverse_omega_maybe_triton(output_adjoint, dag_values, dag, schedule)
    block_atoms = 128
    grid = (triton.cdiv(int(output_adjoint.shape[0]), block_atoms),)
    try:
        if chunk_schedule.policy == "flat":
            last_start = ((len(dag.nodes) - 1) // chunk_size) * chunk_size
            for chunk_start in range(last_start, -1, -chunk_size):
                _reverse_product_dag_node_chunk_kernel[grid](
                    all_values,
                    omega,
                    schedule.node_left_indices,
                    schedule.node_right_indices,
                    chunk_start=int(chunk_start),
                    n_atoms=int(output_adjoint.shape[0]),
                    n_roots=int(dag.n_roots),
                    n_nodes=int(len(dag.nodes)),
                    n_total=int(dag.n_nodes_total),
                    BLOCK_ATOMS=block_atoms,
                    CHUNK_SIZE=int(chunk_size),
                )
            return omega[:, : dag.n_roots], f"triton_product_dag_reverse_chunked_flat+{init_backend}"
        for chunk_index in range(chunk_schedule.chunk_count):
            _reverse_product_dag_scheduled_node_chunk_kernel[grid](
                all_values,
                omega,
                chunk_schedule.node_offsets,
                schedule.node_left_indices,
                schedule.node_right_indices,
                chunk_index=int(chunk_index),
                n_atoms=int(output_adjoint.shape[0]),
                n_roots=int(dag.n_roots),
                n_total=int(dag.n_nodes_total),
                BLOCK_ATOMS=block_atoms,
                CHUNK_SIZE=int(chunk_size),
            )
        return omega[:, : dag.n_roots], f"triton_product_dag_reverse_chunked_{chunk_schedule.policy}+{init_backend}"
    except Exception:
        if os.environ.get("GNE3_DEBUG_TRITON") == "1":
            raise
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"


def reverse_product_dag_adjoint_depth_triton(
    output_adjoint,
    dag_values,
    dag,
):
    if not _real_reverse_base_available(output_adjoint, dag_values, dag):
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"
    schedule = _tensor_schedule(dag, device=output_adjoint.device, dtype=output_adjoint.dtype)
    depth_schedule = _reverse_depth_schedule(dag, device=output_adjoint.device)
    max_levels = int(_ace_env("TRITON_PRODUCT_DAG_MAX_DEPTH_LEVELS", "64"))
    if depth_schedule.level_count == 0 or depth_schedule.level_count > max_levels:
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"
    all_values = dag_values.all_node_values.contiguous()
    output_adjoint = output_adjoint.contiguous()
    omega, init_backend = _initialize_reverse_omega_maybe_triton(output_adjoint, dag_values, dag, schedule)
    block_atoms = int(_ace_env("TRITON_PRODUCT_DAG_DEPTH_BLOCK_ATOMS", "32"))
    block_nodes = int(_ace_env("TRITON_PRODUCT_DAG_DEPTH_BLOCK_NODES", "32"))
    block_atoms = max(1, min(block_atoms, 128))
    block_nodes = max(1, min(block_nodes, 128))
    if block_atoms != 32 or block_nodes != 32:
        return (
            reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag),
            "torch_reference_unsupported_triton_depth_tile_override",
        )
    try:
        for level_start, level_count in zip(depth_schedule.level_starts, depth_schedule.level_counts):
            grid = (
                triton.cdiv(int(output_adjoint.shape[0]), block_atoms),
                triton.cdiv(int(level_count), block_nodes),
            )
            _reverse_product_dag_depth_level_kernel[grid](
                all_values,
                omega,
                depth_schedule.node_offsets,
                schedule.node_left_indices,
                schedule.node_right_indices,
                level_start=int(level_start),
                level_count=int(level_count),
                n_atoms=int(output_adjoint.shape[0]),
                n_roots=int(dag.n_roots),
                n_total=int(dag.n_nodes_total),
            )
        return omega[:, : dag.n_roots], f"triton_product_dag_reverse_depth+{init_backend}"
    except Exception:
        if os.environ.get("GNE3_DEBUG_TRITON") == "1":
            raise
        return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"


def reverse_product_dag_adjoint_triton(
    output_adjoint,
    dag_values,
    dag,
):
    if _triton_reverse_complex_split_available(output_adjoint, dag_values, dag):
        return (
            reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag),
            "torch_reference_complex_triton_specialization_pending",
        )
    if _real_reverse_base_available(output_adjoint, dag_values, dag):
        reverse_mode = _ace_env("TRITON_PRODUCT_DAG_REVERSE_MODE", "depth").strip().lower().replace("-", "_")
        if reverse_mode == "depth":
            return reverse_product_dag_adjoint_depth_triton(output_adjoint, dag_values, dag)
        return (
            reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag),
            "torch_reference_unsupported_triton_reverse_mode",
        )
    return reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag), "torch_reference"


def evaluate_product_dag(atomic_base, dag):
    if atomic_base.ndim != 2:
        raise ValueError("atomic_base must have shape [n_atoms, n_channels]")
    if atomic_base.shape[1] < dag.n_roots:
        raise ValueError("atomic_base does not contain all ProductDAG root channels")
    n_atoms = int(atomic_base.shape[0])
    node_values = [atomic_base[:, idx] for idx in range(dag.n_roots)]
    for node in dag.nodes:
        node_values.append(node_values[node.left] * node_values[node.right])
    values = torch.stack(node_values, dim=1)

    descriptor_values = torch.zeros(
        (n_atoms, dag.descriptor_count),
        dtype=atomic_base.dtype,
        device=atomic_base.device,
    )
    if dag.output_terms:
        schedule = _tensor_schedule(dag, device=atomic_base.device, dtype=atomic_base.dtype)
        term_values = (
            values.index_select(1, schedule.output_node_indices)
            * schedule.output_coeffs.unsqueeze(0)
        )
        descriptor_values.index_add_(1, schedule.output_descriptor_indices, term_values)
    return ProductDAGValues(all_node_values=values, descriptor_values=descriptor_values)


def evaluate_product_dag_linear_form(atomic_base, dag, descriptor_weight):
    """Evaluate a ProductDAG linear form and its root-channel adjoint."""

    if atomic_base.ndim != 2:
        raise ValueError("atomic_base must have shape [n_atoms, n_channels]")
    if atomic_base.shape[1] < dag.n_roots:
        raise ValueError("atomic_base does not contain all ProductDAG root channels")
    n_atoms = int(atomic_base.shape[0])
    weights = torch.as_tensor(descriptor_weight, dtype=atomic_base.dtype, device=atomic_base.device)
    if weights.ndim == 1:
        weights = weights.reshape(1, dag.descriptor_count).expand(n_atoms, dag.descriptor_count)
    if tuple(weights.shape) != (n_atoms, dag.descriptor_count):
        raise ValueError(f"descriptor_weight must have shape {(dag.descriptor_count,)} or {(n_atoms, dag.descriptor_count)}")
    node_values = [atomic_base[:, idx] for idx in range(dag.n_roots)]
    for node in dag.nodes:
        node_values.append(node_values[node.left] * node_values[node.right])
    values = torch.stack(node_values, dim=1)
    site_linear = torch.zeros((n_atoms,), dtype=atomic_base.dtype, device=atomic_base.device)
    omega = torch.zeros_like(values)
    if dag.output_terms:
        schedule = _tensor_schedule(dag, device=atomic_base.device, dtype=atomic_base.dtype)
        term_weight = weights.index_select(1, schedule.output_descriptor_indices)
        term_coeff = schedule.output_coeffs.unsqueeze(0)
        term_linear = term_weight * term_coeff
        term_values = values.index_select(1, schedule.output_node_indices) * term_linear
        site_linear = site_linear + term_values.sum(dim=1)
        omega.index_add_(1, schedule.output_node_indices, term_linear)

    for offset in range(len(dag.nodes) - 1, -1, -1):
        node_index = dag.n_roots + offset
        node = dag.nodes[offset]
        node_omega = omega[:, node_index]
        omega[:, node.left] += node_omega * values[:, node.right]
        omega[:, node.right] += node_omega * values[:, node.left]
    return site_linear, omega[:, :dag.n_roots]


def reverse_product_dag_adjoint_reference(output_adjoint, dag_values, dag):
    if output_adjoint.ndim != 2:
        raise ValueError("output_adjoint must have shape [n_atoms, n_descriptors]")
    if output_adjoint.shape != dag_values.descriptor_values.shape:
        raise ValueError("output_adjoint shape must match descriptor_values shape")
    omega = torch.zeros_like(dag_values.all_node_values)
    if dag.output_terms:
        schedule = _tensor_schedule(dag, device=output_adjoint.device, dtype=output_adjoint.dtype)
        term_omega = (
            output_adjoint.index_select(1, schedule.output_descriptor_indices)
            * schedule.output_coeffs.unsqueeze(0)
        )
        omega.index_add_(1, schedule.output_node_indices, term_omega)

    for offset in range(len(dag.nodes) - 1, -1, -1):
        node_index = dag.n_roots + offset
        node = dag.nodes[offset]
        node_omega = omega[:, node_index]
        omega[:, node.left] += node_omega * dag_values.all_node_values[:, node.right]
        omega[:, node.right] += node_omega * dag_values.all_node_values[:, node.left]
    return omega[:, :dag.n_roots]


def reverse_product_dag_adjoint(output_adjoint, dag_values, dag):
    root_adjoint, _ = reverse_product_dag_adjoint_triton(output_adjoint, dag_values, dag)
    return root_adjoint
