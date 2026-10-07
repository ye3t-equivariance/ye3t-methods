from itertools import permutations
from math import sqrt

import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t.couplings.lifted_cauchy_scalar import _exact_scalar_from_payload
from ye3t.couplings import shifted_jacobi_ladder_with_derivative, shifted_jacobi_normalization_squared
from ye3t.couplings.tagged_cauchy_carriers import _pooled_physical_row
from ye3t_methods.atomistic import YE3TDescriptors
from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource
from ye3t_methods.atomistic.tagged_cauchy_carriers import tagged_support_chunks, tagged_support_layout


def test_tagged_carrier_periodic_occurrences_and_cache(tmp_path):
    config = {
        "basis": {
            "type": "tagged_cauchy_carriers", "species": ["Ni"],
            "cutoff_A": 2.0,
            "catalogue": {
                "ranks": (1, 2), "tag_counts": (0, 1, 2),
                "nmax_per_rank": {1: 1, 2: 1},
                "lmax_per_rank": {1: 1, 2: 1}, "input_Lmax": 2,
                "max_source_blocks": 2, "max_features_per_rank": 16,
            },
        },
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    }
    atoms = Atoms("Ni", positions=[[0, 0, 0]], cell=[1.5, 8, 8],
                  pbc=[True, False, False])
    descriptor = YE3TDescriptors.ye3t_basis(config)
    first = descriptor.create(atoms)
    evaluator = descriptor.metadata["_tagged_carrier_evaluators"][("cpu", "float64")]
    second = descriptor.create(atoms)
    physical = descriptor.create(atoms, descriptor_evaluation="pooled_carriers")
    assert physical["normalization"] == "raw"
    assert physical["image_coordinate_policy"] == "unreduced_compiler_carrier_coordinates"
    assert physical["pooling_convention"] == "ordered_distinct_occurrence_sum_v1"
    assert physical["schedule_hashes"] == {
        int(schedule["tag_count"]): schedule["self_hash"]
        for schedule in descriptor.metadata["tagged_cauchy_carrier_source_plan"]["schedules"]
    }
    assert set(physical["pooled_carriers"]) == set(first["carriers"]) == {0, 1, 2}
    for tag_count, carrier in first["carriers"].items():
        image = physical["pooled_carriers"][tag_count]
        assert image["labels"] == carrier["labels"]
        expected = np.zeros((len(atoms), carrier["values"].shape[1]))
        np.add.at(expected, carrier["centers"], carrier["values"])
        np.testing.assert_allclose(image["values"], expected, rtol=0, atol=1e-12)
    assert {label["target_L"] for image in physical["pooled_carriers"].values()
            for label in image["labels"]} >= {0, 1, 2}
    edges = torch.as_tensor(first["edge_index"], dtype=torch.long)
    displacement = (torch.as_tensor(first["shifts"], dtype=torch.float64)
                    @ torch.as_tensor(atoms.cell.array, dtype=torch.float64))
    primitive, density = evaluator.geometry_values(
        displacement, torch.zeros(len(atoms), dtype=torch.long), edges, len(atoms),
    )
    primitive, density = primitive.numpy(), density.numpy()
    channel_offsets = {}
    offset = 0
    for channel in evaluator.channels:
        key = (channel["neighbor_species"], channel["radial_channel"],
               channel["l"], channel["source_family_id"])
        channel_offsets[key] = offset
        offset += 2 * channel["l"] + 1
    source_rows = {
        record["label"]["coordinate_id"]: (source, record)
        for source in descriptor.metadata["tagged_cauchy_carriers_compiled"]["sources"]
        for record in source["descriptors"]
    }
    for tag_count, image in physical["pooled_carriers"].items():
        for label in image["labels"]:
            source, record = source_rows[label["coordinate_id"]]
            assert source["request"]["tag_count"] == tag_count
            local_offsets = []
            for channel in source["request"]["channels"]:
                key = (channel["neighbor_species"], channel["radial_channel"],
                       channel["l"], channel["source_family_id"])
                local_offsets.append(channel_offsets[key])
            expected = np.zeros((len(atoms), len(record["real_terms_by_component"])))
            for center in range(len(atoms)):
                neighbors = [index for index in range(edges.shape[1])
                             if int(edges[0, index]) == center]
                for tags in permutations(neighbors, tag_count):
                    for component, row in enumerate(record["real_terms_by_component"]):
                        for term in row["terms"]:
                            value = float(_exact_scalar_from_payload(term["coefficient"]))
                            for channel, role, magnetic in term["coordinates"]:
                                component_index = local_offsets[channel] + magnetic
                                value *= (primitive[tags[role], component_index]
                                          if role < tag_count else
                                          density[center, component_index])
                            expected[center, component] += value
            np.testing.assert_allclose(
                image["values"][:, slice(*label["component_slice"])], expected,
                rtol=1e-11, atol=1e-12,
            )
    assert descriptor.metadata["_tagged_carrier_evaluators"][("cpu", "float64")] is evaluator
    np.testing.assert_array_equal(first["carriers"][2]["values"],
                                  second["carriers"][2]["values"])
    assert {tuple(shift) for shift in first["shifts"]} == {(-1, 0, 0), (1, 0, 0)}
    assert first["carriers"][2]["tag_edges"].shape == (2, 2)
    assert np.all(first["carriers"][2]["tag_edges"][:, 0]
                  != first["carriers"][2]["tag_edges"][:, 1])
    assert any(label["tag_character"] == -1
               for label in first["carriers"][2]["labels"])
    two_tag = first["carriers"][2]
    assert {tuple(row) for row in two_tag["tag_edges"]} == {(0, 1), (1, 0)}
    for label in two_tag["labels"]:
        start, end = label["component_slice"]
        np.testing.assert_allclose(
            two_tag["values"][0, start:end],
            label["tag_character"] * two_tag["values"][1, start:end],
            rtol=0, atol=1e-12,
        )
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    cluster = Atoms("Ni4", positions=[
        [0.0, 0.0, 0.0], [1.1, 0.2, 0.1],
        [0.1, 1.3, 0.2], [0.2, 0.1, 1.4],
    ], cell=[8.0, 8.0, 8.0], pbc=False)
    angle = 0.37
    spin_y = np.array([
        [np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    spin_z = np.array([
        [np.cos(0.23), -np.sin(0.23), 0.0],
        [np.sin(0.23), np.cos(0.23), 0.0], [0.0, 0.0, 1.0],
    ])
    spin = spin_z @ spin_y
    rotated_cluster = cluster.copy()
    rotated_cluster.positions = cluster.positions @ spin.T
    rotated_cluster.cell = np.asarray(cluster.cell) @ spin.T
    inverted_cluster = cluster.copy()
    inverted_cluster.positions = -cluster.positions
    base_images = descriptor.create(cluster, descriptor_evaluation="pooled_carriers")["pooled_carriers"]
    raw_cluster = descriptor.create(cluster)["carriers"]
    rotated_images = descriptor.create(rotated_cluster, descriptor_evaluation="pooled_carriers")["pooled_carriers"]
    inverted_images = descriptor.create(inverted_cluster, descriptor_evaluation="pooled_carriers")["pooled_carriers"]
    exercised = set()
    odd_raw_nonzero = False
    for tag_count, block in base_images.items():
        raw = raw_cluster[tag_count]
        expected = np.zeros_like(block["values"])
        np.add.at(expected, raw["centers"], raw["values"])
        np.testing.assert_allclose(block["values"], expected, rtol=0, atol=1e-12)
        for label in block["labels"]:
            begin, end = label["component_slice"]
            value = block["values"][:, begin:end]
            turned = rotated_images[tag_count]["values"][:, begin:end]
            inverted = inverted_images[tag_count]["values"][:, begin:end]
            L = label["target_L"]
            np.testing.assert_allclose(
                inverted, label["target_parity"] * value,
                rtol=1e-10, atol=1e-11,
            )
            if label["tag_character"] == -1 and tag_count == 2:
                raw = raw_cluster[tag_count]["values"][:, begin:end]
                odd_raw_nonzero |= np.linalg.norm(raw) > 1e-10
                np.testing.assert_allclose(value, 0.0, atol=1e-11)
            if L in (1, 2) and np.linalg.norm(value) > 1e-10:
                exercised.add(L)
                original_complex = real_tesseral_to_complex_multiplet(
                    torch.as_tensor(value).reshape(len(cluster), -1, 2 * L + 1), L,
                ).numpy()
                rotated_complex = real_tesseral_to_complex_multiplet(
                    torch.as_tensor(turned).reshape(len(cluster), -1, 2 * L + 1), L,
                ).numpy()
                np.testing.assert_allclose(
                    rotated_complex, original_complex @ wigner_D_numeric(L, spin).T,
                    rtol=1e-8, atol=1e-10,
                )
    assert exercised == {1, 2}
    assert odd_raw_nonzero
    cluster_edges = torch.as_tensor(descriptor.create(cluster)["edge_index"], dtype=torch.long)
    cluster_disp = torch.as_tensor(cluster.positions, dtype=torch.float64).index_select(
        0, cluster_edges[1]) - torch.as_tensor(cluster.positions, dtype=torch.float64).index_select(
        0, cluster_edges[0])
    radius = torch.linalg.vector_norm(cluster_disp, dim=1)
    x = radius / descriptor.cutoff
    geometry = (cluster_disp, torch.zeros(len(radius), dtype=torch.long), x,
                cluster_disp / radius[:, None], torch.ones(len(radius), dtype=torch.bool),
                descriptor.cutoff)
    moment_cache = {}
    for source in descriptor.metadata["tagged_cauchy_carriers_compiled"]["sources"]:
        tag_count = int(source["request"]["tag_count"])
        image_labels = {label["coordinate_id"]: label for label in base_images[tag_count]["labels"]}
        for record in source["descriptors"]:
            label = image_labels[record["label"]["coordinate_id"]]
            for component_index, component in enumerate(record["real_terms_by_component"]):
                lowered = _pooled_physical_row(component, source["request"]["channels"], tag_count)
                for monomial in lowered:
                    for key in monomial:
                        if key in moment_cache:
                            continue
                        species, family, support, degree, angular_l, magnetic = key
                        assert species == "Ni" and support == "pair_normalized_cutoff_v1"
                        solid = LiftedCauchyPolynomialSource._compiler_ordered_regular_solid(
                            angular_l, geometry)[0]
                        jacobi = shifted_jacobi_ladder_with_derivative(
                            degree, 4, 2 * angular_l + 2, x)[0][degree]
                        primitive = (solid[:, magnetic] *
                                     sqrt(float(shifted_jacobi_normalization_squared(degree, angular_l))) *
                                     (1 - x).square() * jacobi)
                        moment_cache[key] = torch.zeros(len(cluster), dtype=torch.float64).index_add(
                            0, cluster_edges[0], primitive).numpy()
                expected = sum(
                    float(coefficient) * np.prod([moment_cache[key] for key in monomial], axis=0)
                    for monomial, coefficient in lowered.items()
                ) if lowered else np.zeros(len(cluster))
                begin = label["component_slice"][0]
                np.testing.assert_allclose(
                    base_images[tag_count]["values"][:, begin + component_index], expected,
                    rtol=2e-10, atol=2e-11,
                )
    selected = descriptor.create(cluster, descriptor_evaluation="physical_image")
    selected_rotated = descriptor.create(rotated_cluster, descriptor_evaluation="physical_image")
    selected_inverted = descriptor.create(inverted_cluster, descriptor_evaluation="physical_image")
    assert selected["image_coordinate_policy"] == "exact_original_compiler_pivots_v1"
    assert selected["physical_image_plan"]["complete_multiplet_reconstruction"]
    assert selected["physical_image_plan_hash"] == selected["physical_image_plan"]["self_hash"]
    assert selected["physical_image_output_coordinate_ids"] == tuple(
        label["coordinate_id"] for block in selected["physical_image"].values()
        for label in block["labels"])
    assert set(selected["physical_image_output_coordinate_ids"]) == set(
        selected["physical_image_plan"]["selected_coordinate_ids"])
    assert sum(len(block["labels"]) for block in selected["physical_image"].values()) < (
        sum(len(block["labels"]) for block in base_images.values()))
    for tag_count, selected_block in selected["physical_image"].items():
        raw_block = base_images[tag_count]
        raw_labels = {label["coordinate_id"]: label for label in raw_block["labels"]}
        for label in selected_block["labels"]:
            raw_label = raw_labels[label["coordinate_id"]]
            np.testing.assert_allclose(
                selected_block["values"][:, slice(*label["component_slice"])],
                raw_block["values"][:, slice(*raw_label["component_slice"])],
                rtol=0, atol=1e-12,
            )
            begin, end = label["component_slice"]
            value = selected_block["values"][:, begin:end]
            turned = selected_rotated["physical_image"][tag_count]["values"][:, begin:end]
            inverted = selected_inverted["physical_image"][tag_count]["values"][:, begin:end]
            np.testing.assert_allclose(inverted, label["target_parity"] * value,
                                       rtol=1e-10, atol=1e-11)
            L = label["target_L"]
            if L in (1, 2):
                complex_value = real_tesseral_to_complex_multiplet(
                    torch.as_tensor(value).reshape(len(cluster), -1, 2 * L + 1), L).numpy()
                complex_turned = real_tesseral_to_complex_multiplet(
                    torch.as_tensor(turned).reshape(len(cluster), -1, 2 * L + 1), L).numpy()
                np.testing.assert_allclose(complex_turned,
                                           complex_value @ wigner_D_numeric(L, spin).T,
                                           rtol=1e-8, atol=1e-10)
    rotated = atoms.copy()
    rotated.rotate(90, "z", center=(0, 0, 0), rotate_cell=True)
    rotated_carriers = descriptor.create(rotated)["carriers"]
    for tag_count, block in first["carriers"].items():
        transformed = rotated_carriers[tag_count]
        for label, transformed_label in zip(block["labels"], transformed["labels"], strict=True):
            assert label == transformed_label
            start, end = label["component_slice"]
            np.testing.assert_allclose(
                np.sort(np.linalg.norm(block["values"][:, start:end], axis=1)),
                np.sort(np.linalg.norm(transformed["values"][:, start:end], axis=1)),
                rtol=1e-10, atol=1e-12,
            )
    cache_files = {path.name: path.stat().st_mtime_ns for path in tmp_path.glob("*.json")}
    assert cache_files
    YE3TDescriptors.ye3t_basis(config)
    assert cache_files == {
        path.name: path.stat().st_mtime_ns for path in tmp_path.glob("*.json")
    }
    bad_config = {**config, "basis": {**config["basis"], "radial_lambda": 0.3}}
    with pytest.raises(ValueError, match="Unsupported tagged carrier basis settings"):
        YE3TDescriptors.ye3t_basis(bad_config)


def test_multirank_joint_image_reconstructs_cross_rank_source_on_ase_atoms(tmp_path):
    descriptor = YE3TDescriptors.ye3t_basis({
        "basis": {"type": "tagged_cauchy_carriers", "species": ["Ni"],
                  "cutoff_A": 2.4, "catalogue": {
                      "ranks": [1, 2], "nmax_per_rank": {1: 3, 2: 1},
                      "lmax_per_rank": {1: 1, 2: 1},
                      "source_block_partitions_by_rank": {
                          1: [[1]], 2: [[1, 1]]},
                      "tag_counts_by_rank": {1: [0], 2: [0, 2]},
                      "input_Lmax": 1}},
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path}, "model": {},
    })
    atoms = Atoms("Ni4", positions=[[0, 0, 0], [1.0, .2, .1],
                                   [.1, 1.2, .2], [.2, .1, 1.3]],
                  cell=[8, 8, 8], pbc=False)
    raw = descriptor.create(atoms, descriptor_evaluation="pooled_carriers")
    selected = descriptor.create(atoms, descriptor_evaluation="physical_image")
    plan = selected["physical_image_plan"]
    assert plan["rank_policy"] == "joint_physical_image_after_rankwise_compilation"
    assert plan["permutation_policy"] == "rank_specific_formal_parents_not_a_common_S_N_action"
    assert set(plan["candidate_tensor_orders"]) == {1, 2}
    raw_by_id = {
        label["coordinate_id"]: (label, block["values"][:, slice(*label["component_slice"])])
        for block in raw["pooled_carriers"].values() for label in block["labels"]}
    selected_by_id = {
        label["coordinate_id"]: (label, block["values"][:, slice(*label["component_slice"])])
        for block in selected["physical_image"].values() for label in block["labels"]}
    order_by_id = dict(zip(plan["candidate_coordinate_ids"],
                           plan["candidate_tensor_orders"], strict=True))
    cross_rank = []
    for coordinate_id, old_order, reconstruction in zip(
            plan["candidate_coordinate_ids"], plan["candidate_tensor_orders"],
            plan["reconstruction"], strict=True):
        label, actual = raw_by_id[coordinate_id]
        rebuilt = np.zeros_like(actual)
        for term in reconstruction:
            retained_id = term["coordinate_id"]
            selected_label, value = selected_by_id[retained_id]
            assert selected_label["target_L"] == label["target_L"]
            assert selected_label["target_parity"] == label["target_parity"]
            rebuilt += float(term["coefficient"]["binary64"][0]) * value
            if order_by_id[retained_id] != old_order:
                cross_rank.append((old_order, order_by_id[retained_id]))
        np.testing.assert_allclose(actual, rebuilt, rtol=2e-10, atol=2e-11)
    assert (2, 1) in cross_rank


def test_tagged_pooled_carriers_retain_a_mixed_internal_young_path(tmp_path):
    descriptor = YE3TDescriptors.ye3t_basis({
        "basis": {
            "type": "tagged_cauchy_carriers", "species": ["Ni"],
            "cutoff_A": 2.0,
            "catalogue": {
                "ranks": (3,), "tag_counts": (2,),
                "nmax_per_rank": {3: 1}, "lmax_per_rank": {3: 1},
                "input_Lmax": 2,
                "source_block_partitions_by_rank": {3: [[3]]},
                "max_features_per_rank": {3: 16},
            },
        },
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    })
    atoms = Atoms("Ni4", positions=[
        [0.0, 0.0, 0.0], [1.1, 0.2, 0.1],
        [0.1, 1.3, 0.2], [0.2, 0.1, 1.4],
    ], cell=[8.0, 8.0, 8.0], pbc=False)
    image = descriptor.create(atoms, descriptor_evaluation="pooled_carriers")[
        "pooled_carriers"
    ][2]
    mixed = [label for label in image["labels"]
             if label["block_kappas"] == ((2, 1),)
             and label["tag_character"] == 1
             and label["target_L"] in (1, 2)]
    assert mixed
    assert any(np.linalg.norm(image["values"][:, slice(*label["component_slice"])]) > 1e-10
               for label in mixed)
    selected = descriptor.create(atoms, descriptor_evaluation="physical_image")[
        "physical_image"
    ][2]
    selected_mixed = [label for label in selected["labels"]
                      if label["block_kappas"] == ((2, 1),)
                      and label["tag_character"] == 1
                      and label["target_L"] in (1, 2)]
    assert selected_mixed
    assert any(np.linalg.norm(selected["values"][:, slice(*label["component_slice"])]) > 1e-10
               for label in selected_mixed)
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    ry = np.array([[np.cos(.41), 0.0, np.sin(.41)], [0.0, 1.0, 0.0],
                   [-np.sin(.41), 0.0, np.cos(.41)]])
    rz = np.array([[np.cos(.29), -np.sin(.29), 0.0],
                   [np.sin(.29), np.cos(.29), 0.0], [0.0, 0.0, 1.0]])
    rotation = rz @ ry
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    turned.cell = atoms.cell.array @ rotation.T
    turned_values = descriptor.create(turned, descriptor_evaluation="physical_image")[
        "physical_image"][2]["values"]
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    inverted_values = descriptor.create(inverted, descriptor_evaluation="physical_image")[
        "physical_image"][2]["values"]
    order = np.array([2, 0, 3, 1])
    reordered = atoms[order]
    reordered_values = descriptor.create(reordered, descriptor_evaluation="physical_image")[
        "physical_image"][2]["values"]
    for label in selected_mixed:
        begin, end = label["component_slice"]
        original = selected["values"][:, begin:end]
        L = label["target_L"]
        assert np.linalg.norm(original) > 1e-10
        np.testing.assert_allclose(inverted_values[:, begin:end],
                                   label["target_parity"] * original,
                                   rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(reordered_values[:, begin:end], original[order],
                                   rtol=1e-10, atol=1e-11)
        complex_original = real_tesseral_to_complex_multiplet(
            torch.as_tensor(original).reshape(len(atoms), -1, 2 * L + 1), L).numpy()
        complex_turned = real_tesseral_to_complex_multiplet(
            torch.as_tensor(turned_values[:, begin:end]).reshape(len(atoms), -1, 2 * L + 1), L).numpy()
        np.testing.assert_allclose(complex_turned,
                                   complex_original @ wigner_D_numeric(L, rotation).T,
                                   rtol=1e-8, atol=1e-10)


def test_selected_tagged_image_two_species_radial_channels_and_pair_cutoffs(tmp_path):
    pair_cutoffs = {"Ni-Ni": 2.4, "Ni-Cu": 1.5,
                    "Cu-Ni": 1.4, "Cu-Cu": 1.0}
    descriptor = YE3TDescriptors.ye3t_basis({
        "basis": {"type": "tagged_cauchy_carriers", "species": ["Ni", "Cu"],
                  "cutoff_A": 2.4, "pair_cutoffs_A": pair_cutoffs,
                  "catalogue": {"ranks": (2,), "tag_counts": (0, 1, 2),
                                "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 0},
                                "input_Lmax": 0, "max_source_blocks": 2,
                                "max_features_per_rank": {2: 64}}},
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    })
    atoms = Atoms("NiCuCuNi", positions=[
        [0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.8, 0.1, 0.0], [0.0, 1.2, 0.1],
    ], cell=[8.0, 8.0, 8.0], pbc=False)
    result = descriptor.create(atoms, descriptor_evaluation="physical_image")
    selected = result["physical_image"]
    selected_ids = set(result["physical_image_output_coordinate_ids"])
    sources = descriptor.metadata["tagged_cauchy_carriers_compiled"]["sources"]
    records = [(source, record) for source in sources for record in source["descriptors"]
               if record["label"]["coordinate_id"] in selected_ids]
    assert records
    assert any(any(channel["neighbor_species"] == "Cu" and
                   channel["radial_channel"] == 1
                   for channel in source["request"]["channels"])
               for source, _ in records)
    edges = result["edge_index"]
    distances = np.linalg.norm(atoms.positions[edges[1]] - atoms.positions[edges[0]], axis=1)
    species = atoms.get_chemical_symbols()
    cutoff = np.asarray([pair_cutoffs[species[center] + "-" + species[neighbor]]
                         for center, neighbor in edges.T])
    assert np.any((distances > cutoff) & (distances < descriptor.cutoff))
    moments = {}
    for source, record in records:
        tag_count = source["request"]["tag_count"]
        label = next(label for label in selected[tag_count]["labels"]
                     if label["coordinate_id"] == record["label"]["coordinate_id"])
        for component_index, component in enumerate(record["real_terms_by_component"]):
            lowered = _pooled_physical_row(component, source["request"]["channels"], tag_count)
            for monomial in lowered:
                for key in monomial:
                    if key in moments:
                        continue
                    neighbor_species, family, support, degree, angular_l, magnetic = key
                    assert family == "orthogonal_shifted_jacobi_origin_regular_v1"
                    assert support == "pair_normalized_cutoff_v1"
                    assert angular_l == magnetic == 0
                    active = (np.asarray([species[neighbor] == neighbor_species
                                          for neighbor in edges[1]]) & (distances < cutoff))
                    x = np.where(active, distances / cutoff, 0.5)
                    x_tensor = torch.as_tensor(x, dtype=torch.float64)
                    radial = (sqrt(float(shifted_jacobi_normalization_squared(degree, 0)))
                              * (1 - x_tensor).square()
                              * shifted_jacobi_ladder_with_derivative(
                                  degree, 4, 2, x_tensor)[0][degree]).numpy()
                    row = np.zeros(len(atoms))
                    np.add.at(row, edges[0], active * radial)
                    moments[key] = row
            expected = sum((float(coefficient) * np.prod(
                [moments[key] for key in monomial], axis=0)
                for monomial, coefficient in lowered.items()), np.zeros(len(atoms)))
            begin = label["component_slice"][0]
            np.testing.assert_allclose(selected[tag_count]["values"][:, begin + component_index],
                                       expected, rtol=2e-10, atol=2e-11)


def test_tagged_selected_image_handles_compiler_empty_image(tmp_path):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import tagged_cauchy_carriers_request, tagged_cauchy_carrier_schedule

    descriptor = YE3TDescriptors.ye3t_basis({
        "basis": {"type": "tagged_cauchy_carriers", "species": ["Ni"],
                  "cutoff_A": 2.0, "catalogue": {
                      "ranks": (2,), "tag_counts": (2,),
                      "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
                      "input_Lmax": 2, "max_source_blocks": 1,
                      "max_features_per_rank": {2: 8}}},
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    })
    odd = compile_coupling(tagged_cauchy_carriers_request(
        [{"neighbor_species": "Ni", "radial_channel": 0, "l": 1,
          "source_family_id": "orthogonal_shifted_jacobi_origin_regular_v1"}],
        [2], tag_count=2, target_Ls=[1]))
    assert len(odd["descriptors"]) == 1
    assert odd["descriptors"][0]["label"]["tag_character"] == -1
    descriptor.metadata["tagged_cauchy_carriers_compiled"] = {
        "sources": (odd,), "self_hash": odd["self_hash"]}
    descriptor.metadata["tagged_cauchy_carrier_source_plan"] = {
        "schedules": (tagged_cauchy_carrier_schedule((odd,)),)}
    atoms = Atoms("Ni2", positions=[[0, 0, 0], [1.1, .2, .1]],
                  cell=[8, 8, 8], pbc=False)
    result = descriptor.create(atoms, descriptor_evaluation="physical_image")
    assert result["physical_image"] == {}
    assert result["physical_image_output_coordinate_ids"] == ()
    assert result["physical_image_plan"]["selected_coordinate_ids"] == ()
    assert result["schedule_hashes"] == {}
    assert result["edge_index"].shape == (2, 2)
    assert descriptor.metadata.get("_tagged_carrier_evaluators", {}) == {}


def test_exact_physical_image_position_and_strain_derivatives(tmp_path):
    descriptor = YE3TDescriptors.ye3t_basis({
        "basis": {"type": "tagged_cauchy_carriers", "species": ["Ni"],
                  "cutoff_A": 2.0, "catalogue": {
                      "ranks": (1, 2), "tag_counts": (0, 1, 2),
                      "nmax_per_rank": {1: 1, 2: 1},
                      "lmax_per_rank": {1: 1, 2: 1},
                      "input_Lmax": 2, "max_source_blocks": 2,
                      "max_features_per_rank": 16}},
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    })

    def weights_for(result, scalar_only):
        weights = {}
        for tag_count, block in result["physical_image"].items():
            matrix = np.zeros_like(block["values"])
            for label in block["labels"]:
                if (label["target_L"] == 0) == scalar_only:
                    begin, end = label["component_slice"]
                    matrix[:, begin:end] = (np.arange(begin + 1, end + 1)
                                            if scalar_only else np.arange(begin + 1, end + 1)[None, :])
            if not scalar_only:
                matrix[1:] = 0
            weights[tag_count] = matrix
        assert any(np.linalg.norm(row) > 0 for row in weights.values())
        return weights

    def public_value(atoms, weights):
        image = descriptor.create(atoms, descriptor_evaluation="physical_image")
        return sum(float(np.sum(block["values"] * weights[tag_count]))
                   for tag_count, block in image["physical_image"].items())

    def differentiable_value(atoms, positions, cell, weights, reference):
        edges = torch.as_tensor(reference["edge_index"], dtype=torch.long)
        shifts = torch.as_tensor(reference["shifts"], dtype=torch.float64)
        displacement = (positions.index_select(0, edges[1])
                        - positions.index_select(0, edges[0]) + shifts @ cell)
        evaluator = descriptor.metadata["_tagged_carrier_evaluators"][
            ("cpu", "float64", "physical_image")]
        edge_values, density = evaluator.geometry_values(
            displacement, torch.zeros(len(atoms), dtype=torch.long), edges, len(atoms))
        layout = tagged_support_layout(edges, len(atoms), shifts.to(torch.long))
        value = positions.new_zeros(())
        for schedule in descriptor.metadata["_tagged_carrier_physical_image_schedules"]:
            tag_count = int(schedule["tag_count"])
            block = positions.new_zeros((len(atoms), int(schedule["output_dimension"])))
            for support in tagged_support_chunks(layout, tag_count, 4096):
                chunk = evaluator(tag_count, edge_values, density, support, normalized=False)
                block = block.index_add(0, support["centers"], chunk)
            value = value + (block * torch.as_tensor(weights[tag_count])).sum()
        return value

    cluster = Atoms("Ni4", positions=[
        [0.0, 0.0, 0.0], [1.1, 0.2, 0.1],
        [0.1, 1.3, 0.2], [0.2, 0.1, 1.4],
    ], cell=[8.0, 8.0, 8.0], pbc=False)
    cluster_image = descriptor.create(cluster, descriptor_evaluation="physical_image")
    covariant_weights = weights_for(cluster_image, False)
    assert {label["target_L"] for block in cluster_image["physical_image"].values()
            for label in block["labels"] if label["target_L"] in (1, 2)} == {1, 2}
    positions = torch.tensor(cluster.positions, dtype=torch.float64, requires_grad=True)
    cell = torch.tensor(cluster.cell.array, dtype=torch.float64)
    energy = differentiable_value(cluster, positions, cell, covariant_weights, cluster_image)
    gradient = torch.autograd.grad(energy, positions)[0].numpy()
    assert np.linalg.norm(gradient) > 1e-8
    step = 1e-5
    for atom, axis in ((1, 0), (2, 1), (3, 2)):
        plus, minus = cluster.copy(), cluster.copy()
        plus.positions[atom, axis] += step
        minus.positions[atom, axis] -= step
        finite = (public_value(plus, covariant_weights)
                  - public_value(minus, covariant_weights)) / (2 * step)
        np.testing.assert_allclose(gradient[atom, axis], finite, rtol=2e-5, atol=2e-7)

    periodic = Atoms("Ni2", positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.3]],
                     cell=[2.5, 6.0, 6.0], pbc=[True, False, False])
    periodic_image = descriptor.create(periodic, descriptor_evaluation="physical_image")
    assert np.any(periodic_image["shifts"])
    scalar_weights = weights_for(periodic_image, True)
    strain = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    deformation = torch.eye(3, dtype=torch.float64) + torch.stack((
        torch.stack((strain[0], strain[5] / 2, strain[4] / 2)),
        torch.stack((strain[5] / 2, strain[1], strain[3] / 2)),
        torch.stack((strain[4] / 2, strain[3] / 2, strain[2])),
    ))
    strain_value = differentiable_value(
        periodic, torch.as_tensor(periodic.positions) @ deformation.T,
        torch.as_tensor(periodic.cell.array) @ deformation.T,
        scalar_weights, periodic_image)
    stress = torch.autograd.grad(strain_value, strain)[0].numpy() / periodic.get_volume()
    assert np.linalg.norm(stress) > 1e-9
    for component, (row, column) in enumerate(((0, 0), (1, 1), (2, 2),
                                               (1, 2), (0, 2), (0, 1))):
        plus, minus = periodic.copy(), periodic.copy()
        plus_deformation = np.eye(3)
        minus_deformation = np.eye(3)
        factor = 1 if row == column else 0.5
        plus_deformation[row, column] += step * factor
        minus_deformation[row, column] -= step * factor
        if row != column:
            plus_deformation[column, row] += step * factor
            minus_deformation[column, row] -= step * factor
        plus.positions = periodic.positions @ plus_deformation.T
        minus.positions = periodic.positions @ minus_deformation.T
        plus.cell = periodic.cell.array @ plus_deformation.T
        minus.cell = periodic.cell.array @ minus_deformation.T
        finite = (public_value(plus, scalar_weights)
                  - public_value(minus, scalar_weights)) / (2 * step * periodic.get_volume())
        np.testing.assert_allclose(stress[component], finite, rtol=3e-5, atol=3e-7)
