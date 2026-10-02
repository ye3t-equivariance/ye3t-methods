import os

import pytest
import torch

from ye3t.runtime import native_execution_plan_capabilities
from ye3t_ace.equivariant_calc.labeling import SingleChannelLabel
from ye3t_ace.equivariant_calc.site_basis_v2 import (
    SiteBasisConfig,
    SiteBasisV2,
)


pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(
        not (
            os.environ.get("YE3T_TEST_CPP_EXTENSION")
            or native_execution_plan_capabilities()["prebuilt_extension"]
        ),
        reason="requires the prebuilt extension or YE3T_TEST_CPP_EXTENSION=1",
    ),
]


def _channels():
    channels = []
    for radial_index, angular_momentum in ((1, 0), (2, 1), (3, 2), (4, 3)):
        for magnetic in range(-angular_momentum, angular_momentum + 1):
            channels.append(
                SingleChannelLabel(
                    mu0=0,
                    mu=0,
                    kappa0=0,
                    kappa=0,
                    n=radial_index,
                    l=angular_momentum,
                    m=magnetic,
                )
            )
    return tuple(channels)


def _basis(
    source_backend,
    spherical_backend,
    atomic_base_normalization="none",
):
    return SiteBasisV2(
        SiteBasisConfig(
            rc=[3.5],
            lmbda=[0.35],
            nradmax=5,
            lmax=3,
            possible_types=(0,),
            charge_mode="none",
            atomic_base_normalization=atomic_base_normalization,
            factor_normalization="none",
            spherical_backend=spherical_backend,
            source_backend=source_backend,
            native_source_min_edges=0,
            dtype=torch.float64,
            complex_dtype=torch.complex128,
        )
    )


@pytest.mark.parametrize("spherical_backend", ["complex", "real"])
@pytest.mark.parametrize("source_backend", ["native_cpu", "native"])
def test_native_source_basis_matches_torch_values_and_edge_derivatives(
    monkeypatch,
    spherical_backend,
    source_backend,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 2, 0, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    atom_types = torch.zeros(4, dtype=torch.long)
    channels = _channels()

    _, expected_values, expected_derivatives = _basis(
        "torch",
        spherical_backend,
    ).compute_channel_edges_with_dx(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    native_basis = _basis(
        source_backend,
        spherical_backend,
    )
    _, actual_values, actual_derivatives = native_basis.compute_channel_edges_with_dx(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )

    torch.testing.assert_close(
        actual_values,
        expected_values,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        actual_derivatives,
        expected_derivatives,
        rtol=2e-11,
        atol=2e-11,
    )
    report = native_basis.source_runtime_report()
    assert report["requested_backend"] == source_backend
    assert report["radial_backend"] == "native"
    assert report["angular_backend"] == "native"
    assert report["radial_table_backend"] == "native"
    assert report["angular_table_backend"] == "native"
    assert report["plain_product_backend"] == "native_cpu"
    assert report["density_accumulation_backend"] == "index_add"


@pytest.mark.parametrize("spherical_backend", ["real", "complex"])
@pytest.mark.parametrize("return_strain_derivative", [False, True])
@pytest.mark.parametrize(
    "atomic_base_normalization",
    ["none", "soft_neighbor"],
)
def test_native_plain_source_streaming_adjoint_matches_torch(
    monkeypatch,
    spherical_backend,
    return_strain_derivative,
    atomic_base_normalization,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 2, 0, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    atom_types = torch.zeros(4, dtype=torch.long)
    channels = _channels()
    reference_basis = _basis(
        "torch",
        spherical_backend,
        atomic_base_normalization,
    )
    _, reference_raw, reference_final = (
        reference_basis.compute_channels_raw_and_final(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
        )
    )
    channel_adjoint = torch.randn_like(reference_final)
    expected = (
        reference_basis.position_vjp_from_raw_channel_adjoint_streaming(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            reference_raw,
            channel_adjoint,
            return_strain_derivative=return_strain_derivative,
        )
    )

    native_basis = _basis(
        "native_cpu",
        spherical_backend,
        atomic_base_normalization,
    )
    _, native_raw, _ = native_basis.compute_channels_raw_and_final(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    actual = (
        native_basis.position_vjp_from_raw_channel_adjoint_streaming(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            native_raw,
            channel_adjoint,
            return_strain_derivative=return_strain_derivative,
        )
    )
    if return_strain_derivative:
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=2e-11,
                atol=2e-11,
            )
    else:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-11,
            atol=2e-11,
        )
    report = native_basis.source_runtime_report()
    assert report["plain_adjoint_backend"] == "native_cpu"


def test_native_plain_source_product_matches_scalar_charge_derivatives(
    monkeypatch,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1], [1, 2, 0, 2]],
        dtype=torch.long,
    )
    atom_types = torch.zeros(3, dtype=torch.long)
    charges = torch.tensor([0.2, -0.35, 0.15], dtype=torch.float64)
    channels = tuple(
        SingleChannelLabel(
            mu0=0,
            mu=0,
            kappa0=kappa0,
            kappa=kappa,
            n=2,
            l=1,
            m=magnetic,
        )
        for kappa0, kappa in ((0, 1), (1, 0), (1, 1))
        for magnetic in range(-1, 2)
    )

    def charged_basis(source_backend):
        return SiteBasisV2(
            SiteBasisConfig(
                rc=[3.5],
                lmbda=[0.35],
                nradmax=2,
                lmax=1,
                kmax=1,
                possible_types=(0,),
                charge_mode="scalar",
                q_min=(-1.0,),
                q_max=(1.0,),
                atomic_base_normalization="none",
                factor_normalization="none",
                spherical_backend="complex",
                source_backend=source_backend,
                native_source_min_edges=0,
                dtype=torch.float64,
                complex_dtype=torch.complex128,
            )
        )

    reference_basis = charged_basis("torch")
    _, expected_values, expected_dx = (
        reference_basis.compute_channel_edges_with_dx(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            charges=charges,
        )
    )
    expected_charge = reference_basis._last_edge_charge_derivatives
    native_basis = charged_basis("native_cpu")
    _, actual_values, actual_dx = (
        native_basis.compute_channel_edges_with_dx(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            charges=charges,
        )
    )
    actual_charge = native_basis._last_edge_charge_derivatives

    torch.testing.assert_close(
        actual_values,
        expected_values,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        actual_dx,
        expected_dx,
        rtol=2e-11,
        atol=2e-11,
    )
    for actual, expected in zip(actual_charge, expected_charge):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-12,
            atol=2e-12,
        )
    assert (
        native_basis.source_runtime_report()["plain_product_backend"]
        == "native_cpu"
    )


@pytest.mark.parametrize("spherical_backend", ["complex", "real"])
@pytest.mark.parametrize("source_backend", ["native_cpu", "native"])
def test_native_atomic_density_accumulation_matches_torch(
    monkeypatch,
    spherical_backend,
    source_backend,
):
    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 2, 0, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    atom_types = torch.zeros(4, dtype=torch.long)
    channels = _channels()
    _, expected = _basis("torch", spherical_backend).compute_atomic_base(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    channel_adjoint = torch.randn(
        4,
        len(channels),
        dtype=torch.complex128,
    )
    torch_basis = _basis("torch", spherical_backend)
    _, _, expected_position_gradient = (
        torch_basis.position_vjp_from_channel_adjoint(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            channel_adjoint,
        )
    )
    native_basis = _basis(source_backend, spherical_backend)
    _, actual = native_basis.compute_atomic_base(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    _, _, actual_position_gradient = (
        native_basis.position_vjp_from_channel_adjoint(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            channel_adjoint,
        )
    )
    torch.testing.assert_close(
        actual_position_gradient,
        expected_position_gradient,
        rtol=2e-11,
        atol=2e-11,
    )
    assert (
        native_basis.source_runtime_report()[
            "density_accumulation_backend"
        ]
        == "native_cpu"
    )
    assert (
        native_basis.source_runtime_report()["density_adjoint_backend"]
        == "native_cpu"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "cheb_exp_cos_radial_with_derivative"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "spherical_harmonics_with_derivative"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "density_accumulate"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "density_accumulate_adjoint"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "plain_site_basis_product_with_derivative"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "plain_site_basis_product_adjoint"
    not in native_execution_plan_capabilities()["cuda_operations"],
    reason="requires the YE3T native CUDA radial and spherical operators",
)
@pytest.mark.parametrize("spherical_backend", ["complex", "real"])
@pytest.mark.parametrize("source_backend", ["auto", "native_cuda"])
def test_cuda_source_uses_native_radial_angular_and_density_with_position_vjp(
    spherical_backend,
    source_backend,
):
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
        device="cuda",
    )
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 2, 0, 1, 0, 1],
        ],
        dtype=torch.long,
        device="cuda",
    )
    atom_types = torch.zeros(4, dtype=torch.long, device="cuda")
    channels = _channels()

    torch_basis = _basis(
        "torch",
        spherical_backend,
    )
    _, expected_values, expected_derivatives = (
        torch_basis.compute_channel_edges_with_dx(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
        )
    )
    native_basis = _basis(source_backend, spherical_backend)
    _, actual_values, actual_derivatives = (
        native_basis.compute_channel_edges_with_dx(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
        )
    )
    torch.testing.assert_close(
        actual_values,
        expected_values,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        actual_derivatives,
        expected_derivatives,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    if spherical_backend == "real":
        _, expected_real_values, expected_real_derivatives = (
            torch_basis.compute_channel_edges_with_dx(
                edge_vectors,
                edge_index,
                atom_types,
                channels,
                real_output=True,
            )
        )
        _, actual_real_values, actual_real_derivatives = (
            native_basis.compute_channel_edges_with_dx(
                edge_vectors,
                edge_index,
                atom_types,
                channels,
                real_output=True,
            )
        )
        assert actual_real_values.dtype == torch.float64
        assert actual_real_derivatives.dtype == torch.float64
        torch.testing.assert_close(
            expected_real_values,
            expected_values.real,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        torch.testing.assert_close(
            expected_real_derivatives,
            expected_derivatives.real,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        torch.testing.assert_close(
            actual_real_values,
            expected_real_values,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        torch.testing.assert_close(
            actual_real_derivatives,
            expected_real_derivatives,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    _, expected_atomic = torch_basis.compute_atomic_base(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    _, actual_atomic = native_basis.compute_atomic_base(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    torch.testing.assert_close(
        actual_atomic,
        expected_atomic,
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    channel_adjoint = torch.randn(
        4,
        len(channels),
        dtype=torch.complex128,
        device="cuda",
    )
    _, _, expected_position_gradient = (
        torch_basis.position_vjp_from_channel_adjoint(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            channel_adjoint,
        )
    )
    _, _, actual_position_gradient = (
        native_basis.position_vjp_from_channel_adjoint(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            channel_adjoint,
        )
    )
    torch.testing.assert_close(
        actual_position_gradient,
        expected_position_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    _, reference_raw, reference_final = (
        torch_basis.compute_channels_raw_and_final(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
        )
    )
    streaming_adjoint = torch.randn_like(reference_final)
    expected_streaming = (
        torch_basis.position_vjp_from_raw_channel_adjoint_streaming(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            reference_raw,
            streaming_adjoint,
        )
    )
    _, native_raw, _ = native_basis.compute_channels_raw_and_final(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
    )
    actual_streaming = (
        native_basis.position_vjp_from_raw_channel_adjoint_streaming(
            edge_vectors,
            edge_index,
            atom_types,
            channels,
            native_raw,
            streaming_adjoint,
        )
    )
    torch.testing.assert_close(
        actual_streaming,
        expected_streaming,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    report = native_basis.source_runtime_report()
    assert report["radial_backend"] == "native_cuda"
    assert report["angular_backend"] == "native_cuda"
    assert report["radial_table_backend"] == "native_cuda"
    assert report["angular_table_backend"] == "native_cuda"
    assert report["plain_product_backend"] == "native_cuda"
    assert report["plain_adjoint_backend"] == "native_cuda"
    if source_backend == "native_cuda":
        assert report["density_accumulation_backend"] == "native_cuda"
        assert report["density_adjoint_backend"] == "native_cuda"
    else:
        assert report["density_accumulation_backend"] == "index_add"
        assert report["density_adjoint_backend"] == "index_select"


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "cheb_exp_cos_radial_table_with_derivative"
    not in native_execution_plan_capabilities()["cuda_operations"]
    or "spherical_harmonics_table_with_derivative"
    not in native_execution_plan_capabilities()["cuda_operations"],
    reason="requires native CUDA radial and spherical source tables",
)
@pytest.mark.parametrize("source_backend", ["auto", "native_cuda"])
def test_cuda_compute_atomic_base_native_tables_match_torch_vjp_hvp(
    source_backend,
):
    edge_vectors = torch.tensor(
        [
            [0.31, 0.47, 0.83],
            [-0.42, 0.58, 0.27],
            [0.73, -0.24, 0.51],
            [-0.67, -0.19, 0.37],
            [0.28, 0.63, -0.44],
            [-0.35, 0.22, -0.91],
            [0.59, -0.71, -0.18],
            [-0.76, -0.41, -0.29],
        ],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    reference_edges = edge_vectors.detach().clone().requires_grad_(True)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 2, 0, 1, 0, 1],
        ],
        dtype=torch.long,
        device="cuda",
    )
    atom_types = torch.zeros(4, dtype=torch.long, device="cuda")
    channels = _channels()
    native_basis = _basis(source_backend, "real")
    torch_basis = _basis("torch", "real")
    _, actual = native_basis.compute_atomic_base(
        edge_vectors,
        edge_index,
        atom_types,
        channels,
        real_output=True,
    )
    _, expected = torch_basis.compute_atomic_base(
        reference_edges,
        edge_index,
        atom_types,
        channels,
        real_output=True,
    )
    weights = torch.linspace(
        -0.8,
        0.9,
        int(actual.numel()),
        dtype=actual.dtype,
        device=actual.device,
    ).reshape_as(actual)
    actual_loss = (actual * weights).sum() + 0.125 * actual.square().sum()
    expected_loss = (expected * weights).sum() + 0.125 * expected.square().sum()
    actual_grad = torch.autograd.grad(
        actual_loss,
        edge_vectors,
        create_graph=True,
    )[0]
    expected_grad = torch.autograd.grad(
        expected_loss,
        reference_edges,
        create_graph=True,
    )[0]
    direction = torch.linspace(
        0.7,
        -0.3,
        int(edge_vectors.numel()),
        dtype=edge_vectors.dtype,
        device=edge_vectors.device,
    ).reshape_as(edge_vectors)
    actual_hvp = torch.autograd.grad(
        (actual_grad * direction).sum(),
        edge_vectors,
    )[0]
    expected_hvp = torch.autograd.grad(
        (expected_grad * direction).sum(),
        reference_edges,
    )[0]

    torch.testing.assert_close(actual, expected, atol=2.0e-11, rtol=2.0e-11)
    torch.testing.assert_close(
        actual_grad,
        expected_grad,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        actual_hvp,
        expected_hvp,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    report = native_basis.source_runtime_report()
    assert report["radial_table_backend"] == "native_cuda"
    assert report["angular_table_backend"] == "native_cuda"
