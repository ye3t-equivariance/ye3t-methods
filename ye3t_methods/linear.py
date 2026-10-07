"""Compact entry points for the existing fixed-descriptor linear models."""

from copy import deepcopy
from collections.abc import Mapping
from contextlib import ExitStack
import hashlib
import json
import math
import os
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ye3t_methods.atomistic import YE3TDescriptors, YE3TModel, YE3TRepresentation
from ye3t_methods.atomistic.cache import default_linear_cache_directory
from ye3t_methods.atomistic.ace.linear_ace import (
    LinearACEScalarCalculator,
    LinearACEScalarModelBundle,
    _linear_ace_geometry_row,
    _make_sklearn_estimator,
    load_linear_ace_ase_bundle,
    save_linear_ace_ase_bundle,
)
from ye3t_methods.atomistic.tagged_cauchy_image import (
    TaggedCauchyImageLinearModel,
    load_tagged_cauchy_image_model,
)
from ye3t_methods.atomistic.cluster_phi import (
    HybridACEPhiCalculator,
    PhiBranchConfig,
    load_hybrid_ace_phi_ase_bundle,
    phi_motif_coupling_report,
    save_hybrid_ace_phi_ase_bundle,
)
from ye3t_methods.atomistic.linear_statistics import solve_ridge_statistics
from ye3t_methods.atomistic.tagged_cauchy_image_fit import tagged_cauchy_image_geometry_row
from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import serialize_site_basis_config
from .config import CataloguePreview, resolve_basis_config, resolve_linear_fit_config


_FIT_UNSET = object()


def _full_m_json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _full_m_json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_full_m_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _full_m_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Full-M artifact contains a value that cannot be saved as JSON.")


def _full_m_canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_json_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate JSON model artifact key: {key}")
        record[key] = value
    return record


def _density_full_m_convention_hash(L, parity):
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    matrix = real_tesseral_to_complex_multiplet(
        torch.eye(2 * L + 1, dtype=torch.float64), L).numpy()
    record = {"L": L, "parity": parity,
              "M_values": list(range(-L, L + 1)),
              "real_to_complex": [[[float(value.real), float(value.imag)]
                                   for value in row] for row in matrix]}
    return hashlib.sha256(_full_m_canonical_bytes(record)).hexdigest()


def _density_full_m_validation(L, parity):
    natural = parity == ("odd" if L % 2 else "even")
    return {"m_alignment": "passed",
            "natural_parity" if natural else "declared_parity":
                "passed" if natural else "compiler_selected",
            "rotation": "not_performed_at_write"}


def _density_full_m_coefficient_hash(ordered):
    digest = hashlib.sha256()
    for rows in ordered:
        digest.update(struct.pack("<q", len(rows)))
        for spec in rows:
            coefficients = np.asarray(spec.coeffs, dtype=np.complex128)
            if not np.isfinite(coefficients).all():
                raise ValueError("Density full-M compiler has nonfinite coefficients.")
            digest.update(struct.pack("<q", len(coefficients)))
            digest.update(np.asarray(coefficients.real, dtype="<f8").tobytes())
            digest.update(np.asarray(coefficients.imag, dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_density_full_m_intertwiner(ordered, L):
    """Check the stored density polynomials under an independent O(3) generator."""
    from ye3t.core.rotation import wigner_D_numeric

    angle = 2e-5
    forward = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
    backward = forward.T
    angular_orders = {channel.l for rows in ordered for spec in rows
                      for channel in spec.channels} | {L}
    generators = {}
    for degree in angular_orders:
        matrix = (wigner_D_numeric(degree, forward) -
                  wigner_D_numeric(degree, backward)) / (2 * angle)
        generators[degree] = matrix
    target = generators[L]
    for column in range(len(ordered[0])):
        specs = [rows[column] for rows in ordered]
        channel_ids = {}
        for channel in specs[L].channels:
            channel_ids.setdefault(channel, len(channel_ids))
        degrees_by_id = {channel_id: channel.l for channel, channel_id
                         in channel_ids.items()}
        polynomials = []
        for spec in specs:
            polynomial = {}
            for ms, coefficient in zip(spec.ms_combinations, spec.coeffs):
                monomial = tuple(sorted((channel_ids[channel], m)
                                        for channel, m in zip(spec.channels, ms)))
                polynomial[monomial] = polynomial.get(monomial, 0j) + coefficient
            polynomials.append(polynomial)
        scale = max((abs(value) for polynomial in polynomials
                     for value in polynomial.values()), default=0.0)
        if scale < 1e-14:
            raise ValueError("Density full-M coefficient polynomial is zero.")
        tolerance = 5e-7 * scale
        for output, polynomial in enumerate(polynomials):
            M = output - L
            conjugate = polynomials[L - M]
            flipped = {}
            for monomial, coefficient in polynomial.items():
                opposite = tuple(sorted((channel, -m) for channel, m in monomial))
                flipped[opposite] = flipped.get(opposite, 0j) + coefficient.conjugate()
            if any(abs(flipped.get(key, 0j) - conjugate.get(key, 0j)) > tolerance
                   for key in flipped.keys() | conjugate.keys()):
                raise ValueError("Density full-M coefficients violate the real-form relation.")
            derivative = {}
            for monomial, coefficient in polynomial.items():
                for slot, (channel_id, m) in enumerate(monomial):
                    degree = degrees_by_id[channel_id]
                    generator = generators[degree]
                    for shifted in (m - 1, m + 1):
                        if abs(shifted) > degree:
                            continue
                        changed = list(monomial)
                        changed[slot] = (channel_id, shifted)
                        changed = tuple(sorted(changed))
                        derivative[changed] = (derivative.get(changed, 0j) +
                                               coefficient * generator[shifted + degree,
                                                                       m + degree])
            expected = {}
            for source, source_polynomial in enumerate(polynomials):
                factor = target[output, source]
                if abs(source - output) != 1:
                    continue
                for monomial, coefficient in source_polynomial.items():
                    expected[monomial] = (expected.get(monomial, 0j) +
                                          factor * coefficient)
            if any(abs(derivative.get(key, 0j) - expected.get(key, 0j)) > tolerance
                   for key in derivative.keys() | expected.keys()):
                raise ValueError("Density full-M coefficients violate rotation covariance.")


def _density_full_m_compiler_record(calculator, ordered):
    blocks = []
    for rows in ordered:
        block = []
        for spec in rows:
            coefficients = np.asarray(spec.coeffs, dtype=np.complex128)
            if not np.isfinite(coefficients).all():
                raise ValueError("Density full-M compiler produced nonfinite coefficients.")
            block.append({
                "key": spec.key, "label": spec.label.to_dict(),
                "channels": [{name: _full_m_json_ready(getattr(channel, name))
                              for name in channel.__record_fields__}
                             for channel in spec.channels],
                "ms_combinations": _full_m_json_ready(spec.ms_combinations),
                "coeffs": [[float(value.real), float(value.imag)]
                           for value in coefficients],
                "L_R": int(spec.L_R), "M_R": int(spec.M_R),
            })
        blocks.append(block)
    return {
        "schema": "ye3t_density_full_m_compiler_v1",
        "site_basis": serialize_site_basis_config(calculator.site_basis_config),
        "compact_labels": [label.to_dict() for label in calculator.labels],
        "specs_by_M": blocks,
        "coefficient_sha256": _density_full_m_coefficient_hash(ordered),
    }


def _read_legacy_composite(target):
    """Pin bounded, hash-verified paper bytes and their ZBL manifest."""
    from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator

    manifest_path = target.parent / "model_manifest.json"
    with manifest_path.open("rb") as handle:
        manifest_bytes = handle.read(2 * 1024 * 1024 + 1)
    if len(manifest_bytes) > 2 * 1024 * 1024:
        raise ValueError("Legacy composite manifest exceeds the 2 MiB limit.")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "ye3t_cost_comparison_promoted_model_v4":
        raise ValueError("The legacy composite requires a promoted v4 model manifest.")
    with target.open("rb") as handle:
        composite_bytes = handle.read(64 * 1024 * 1024 + 1)
    if len(composite_bytes) > 64 * 1024 * 1024:
        raise ValueError("Legacy composite model exceeds the 64 MiB limit.")
    composite = json.loads(composite_bytes)
    if composite.get("schema") != "ye3t_tagged_cauchy_composite_v1":
        raise ValueError("The pinned model is not a legacy tagged composite.")
    components = (composite["ordinary_component"], composite["tagged_component"])
    names = {target.name, *(item["path"] for item in components)}
    artifacts = manifest["artifacts"]
    artifact_names = {item["path"] for item in artifacts}
    if (len(names) != 3 or len(artifacts) != len(artifact_names) or
            artifact_names not in (names, names | {"portfolio_upgrade.json"})):
        raise ValueError("Composite components do not match the promoted manifest artifacts.")
    pinned = {}
    for item in artifacts:
        name = str(item["path"])
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Composite artifact paths must be colocated basenames.")
        path = target.parent / name
        size = int(item["bytes"])
        if size <= 0 or size > 64 * 1024 * 1024:
            raise ValueError("Legacy composite artifact exceeds the 64 MiB limit.")
        if name == target.name:
            data = composite_bytes
        else:
            with path.open("rb") as handle:
                data = handle.read(size + 1)
        if len(data) != size:
            raise ValueError(f"Composite artifact byte count mismatch: {name}.")
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Composite artifact SHA-256 mismatch: {name}.")
        pinned[name] = data
    by_name = {item["path"]: item["sha256"] for item in artifacts}
    if any(by_name[item["path"]] != item["sha256"] for item in components):
        raise ValueError("Composite component hashes differ from the promoted manifest.")
    tagged = json.loads(pinned[composite["tagged_component"]["path"]])
    for label, record in (("composite", composite), ("tagged component", tagged)):
        supplied = record.get("self_hash")
        body = {key: value for key, value in record.items() if key != "self_hash"}
        actual = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if supplied != actual:
            raise ValueError(f"Legacy {label} self hash mismatch.")
    if "portfolio_upgrade.json" in pinned:
        upgrade = json.loads(pinned["portfolio_upgrade.json"])
        portfolio = tagged.get("tagged_execution_portfolio")
        if (upgrade.get("schema") != "ye3t_tagged_portfolio_upgrade_v1" or
                upgrade.get("composite_self_hash") != composite.get("self_hash") or
                upgrade.get("tagged_self_hash") != tagged.get("self_hash") or
                not isinstance(portfolio, dict) or
                upgrade.get("portfolio_hash") != portfolio.get("portfolio_hash")):
            raise ValueError("Promoted portfolio sidecar differs from the composite or tagged plan.")
        from ye3t.execution_plan import compile_tagged_moment_execution_portfolio

        beta = tagged["beta"]
        if not isinstance(beta, dict):
            if len(tagged["species_order"]) != 1:
                raise ValueError("Shared tagged beta requires one species.")
            beta = {tagged["species_order"][0]: beta}
        expected = compile_tagged_moment_execution_portfolio(
            tagged["real_moment_program"], beta,
        )
        if json.dumps(portfolio, sort_keys=True, separators=(",", ":")) != json.dumps(
                expected, sort_keys=True, separators=(",", ":")):
            raise ValueError("Promoted portfolio differs from its compiler-owned program/readout plan.")
    metadata = composite["metadata"]
    model = manifest["model"]
    count = int(metadata["feature_count"])
    if (count <= 0 or count != int(model["feature_count"]) or
            count != int(metadata["ordinary_feature_count"]) + int(metadata["tagged_feature_count"]) or
            int(model["ordinary_feature_count"]) != int(metadata["ordinary_feature_count"]) or
            int(model["tagged_feature_count"]) != int(metadata["tagged_feature_count"])):
        raise ValueError("Composite feature counts differ from the promoted manifest.")
    species = tuple(composite["species_order"])
    if not species or len(set(species)) != len(species) or species != tuple(manifest["reference_potential"]["type_order"]):
        raise ValueError("Composite species order differs from the ZBL reference manifest.")
    YE3TZBLCalculator.from_model_manifest(manifest)
    return {
        "artifact_name": target.name, "artifact_bytes": pinned,
        "manifest_json": manifest_bytes,
        "feature_count": count, "species": species,
        "cutoff_A": float(manifest["radial"]["cutoff_A"]),
        "artifact_sha256": by_name[target.name],
    }


def _read_legacy_compat_archive(target):
    """Check internal consistency of a trusted legacy compatibility ZIP."""
    members = {
        "compat/model.ye3t.json", "compat/ordinary_backbone.yace",
        "compat/tagged_correction.ye3t.json", "compat/portfolio_upgrade.json",
        "compat/model_manifest.json",
    }
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Compatibility archive JSON has a duplicate key.")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Compatibility archive JSON has a nonfinite constant: " + value)

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Compatibility archive JSON has a nonfinite number.")
        return number

    def parse_json(payload):
        try:
            source = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Compatibility archive JSON requires UTF-8.") from error
        return json.loads(source, object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant, parse_float=finite_float)

    with ExitStack() as resources:
        stream = resources.enter_context(target.open("rb"))
        archive_size = stream.seek(0, 2)
        if archive_size > 256 * 1024 * 1024:
            raise ValueError("Compatibility archive exceeds the 256 MiB limit.")
        stream.seek(max(0, archive_size - (65535 + 22)))
        trailer = stream.read(65535 + 22)
        end_offset = trailer.rfind(b"PK\x05\x06")
        if end_offset < 0 or len(trailer) - end_offset < 22:
            raise ValueError("Compatibility archive has no bounded ZIP central directory.")
        _, disk, central_disk, disk_entries, total_entries, central_size, central_offset, comment_size = (
            struct.unpack_from("<4sHHHHIIH", trailer, end_offset))
        end_absolute = archive_size - len(trailer) + end_offset
        if end_absolute >= 20:
            stream.seek(end_absolute - 20)
            if stream.read(4) == b"PK\x06\x07":
                raise ValueError("Compatibility archive does not accept ZIP64.")
        if (disk != 0 or central_disk != 0 or disk_entries != len(members) + 1 or
                total_entries != len(members) + 1 or central_size > 16 * 1024 or
                central_offset + central_size != end_absolute or
                end_offset + 22 + comment_size != len(trailer)):
            raise ValueError("Compatibility archive exceeds its ZIP central directory limit.")
        archive = resources.enter_context(zipfile.ZipFile(stream))
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if (len(names) != len(set(names)) or
                set(names) != members | {"manifest.json"}):
            raise ValueError("Compatibility archive member inventory differs from its schema.")
        by_name = {item.filename: item for item in infos}
        if any(item.is_dir() or item.flag_bits & 1 or
               item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) or
               ((item.external_attr >> 16) & 0o170000) == 0o120000 or
               item.file_size <= 0 or item.file_size > 64 * 1024 * 1024 or
               item.file_size > max(1024, 1000 * item.compress_size)
               for item in infos):
            raise ValueError("Compatibility archive contains an unsafe or oversized member.")
        if (by_name["manifest.json"].file_size > 2 * 1024 * 1024 or
                sum(item.file_size for item in infos) > 256 * 1024 * 1024):
            raise ValueError("Compatibility archive exceeds its manifest or total size limit.")
        with archive.open("manifest.json") as handle:
            manifest_bytes = handle.read(2 * 1024 * 1024 + 1)
            if len(manifest_bytes) > 2 * 1024 * 1024 or handle.read(1):
                raise ValueError("Compatibility archive manifest exceeds the 2 MiB limit.")
        manifest = parse_json(manifest_bytes)
        if (not isinstance(manifest, dict) or
                not isinstance(manifest.get("members"), dict) or
                manifest.get("schema") != "ye3t_legacy_compat_archive_v1" or
                manifest.get("maturity") != "compatibility_only" or
                manifest.get("output_type") != "energy_forces_stress" or
                set(manifest.get("members", {})) != members):
            raise ValueError("Compatibility archive has an unsupported manifest schema.")
        pinned = {}
        for name in sorted(members):
            item = by_name[name]
            record = manifest["members"][name]
            if (not isinstance(record, dict) or
                    type(record.get("bytes")) is not int or
                    record["bytes"] != item.file_size or
                    not isinstance(record.get("sha256"), str) or
                    len(record["sha256"]) != 64 or
                    any(char not in "0123456789abcdef" for char in record["sha256"])):
                raise ValueError("Compatibility archive member metadata differs: " + name)
            digest = hashlib.sha256()
            payload = bytearray()
            with archive.open(name) as handle:
                while True:
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    payload.extend(chunk)
                    if len(payload) > item.file_size:
                        raise ValueError("Compatibility archive member exceeds declared size: " + name)
                    digest.update(chunk)
            if len(payload) != item.file_size or digest.hexdigest() != record["sha256"]:
                raise ValueError("Compatibility archive member SHA-256 mismatch: " + name)
            if name.endswith(".json"):
                parse_json(payload)
            pinned[Path(name).name] = bytes(payload)
    with tempfile.TemporaryDirectory(prefix="ye3t_legacy_archive_") as directory:
        for name, payload in pinned.items():
            (Path(directory) / name).write_bytes(payload)
        fitted = _read_legacy_composite(Path(directory) / "model.ye3t.json")
    species_order = manifest.get("species_order")
    if (not isinstance(species_order, list) or not species_order or
            any(not isinstance(species, str) or not species for species in species_order) or
            len(set(species_order)) != len(species_order) or
            tuple(species_order) != fitted["species"]):
        raise ValueError("Compatibility archive species differ from its verified model.")
    legacy_manifest = json.loads(fitted["manifest_json"])
    if (len(fitted["species"]) != 1 or
            manifest.get("model") != fitted["species"][0] + "/" +
            legacy_manifest["model_id"]):
        raise ValueError("Compatibility archive model identity differs from its verified model.")
    fitted["archive_schema"] = manifest["schema"]
    return fitted


class FeatureLabel:
    """One public descriptor coordinate; its index is not a multiplicity index."""

    def __init__(self, feature_index, source, identity, details):
        self.feature_index = int(feature_index)
        self.source = str(source)
        self.identity = str(identity)
        self._details = deepcopy(details)

    def as_dict(self):
        return {
            "feature_index": self.feature_index,
            "source": self.source,
            "identity": self.identity,
            **deepcopy(self._details),
        }

    def __str__(self):
        fields = self._details
        if self.source == "portable_linear":
            return (f"[{self.feature_index}] B N={fields['N']} L=0 "
                    f"{fields['branch']} {self.identity}")
        if self.source == "density":
            radial = tuple(fields["radial_indices"])
            angular = tuple(fields["input_angular_momenta"])
            return (
                f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
                f"n={radial} l={angular}"
            )
        if self.source == "bar_phi":
            return (
                f"[{self.feature_index}] barPhi N={fields['N']} L=0 "
                f"motif={fields['motif_name']}"
            )
        if self.source == "explicit_phi":
            return (
                f"[{self.feature_index}] Phi N={fields['N']} "
                f"lambda={tuple(fields['parent_partition'])} L={fields['L']} "
                f"alpha={fields['selected_full_alpha']}"
            )
        if self.source == "tagged_carriers":
            return (f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
                    f"tags={fields['tag_count']} {self.identity}")
        raw = fields["compiler_raw_opportunities"]
        if raw and "tag_kappa" not in raw[0]:
            tag_counts = sorted({int(item["tag_count"]) for item in raw})
            block_kappas = sorted({tuple(tuple(part) for part in
                item["label"]["block_kappas"]) for item in raw})
            return (
                f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
                f"tag_counts={tag_counts} block_kappas={block_kappas[:2]} "
                f"({len(raw)} compiler opportunities)"
            )
        tags = sorted({tuple(item["tag_kappa"]) for item in raw})
        roles = sorted({tuple(item["role_kappa"]) for item in raw})
        tag_view = f"{tags[:2]}" + (f" +{len(tags) - 2} more" if len(tags) > 2 else "")
        role_view = f"{roles[:2]}" + (f" +{len(roles) - 2} more" if len(roles) > 2 else "")
        return (
            f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
            f"tag_kappa={tag_view} role_kappa={role_view} "
            f"({len(raw)} compiler opportunities)"
        )

    def latex(self):
        fields = self._details
        head = f"B_{{{self.feature_index}}}^{{N={fields['N']},L={fields['L']}}}"
        if self.source == "bar_phi":
            return rf"\overline{{\Phi}}_{{{self.feature_index}}}^{{N={fields['N']}}}"
        if self.source == "explicit_phi":
            return rf"\Phi_{{{self.feature_index}}}^{{N={fields['N']},L={fields['L']}}}"
        if self.source != "density":
            return head
        radial = ",".join(str(value) for value in fields["radial_indices"])
        angular = ",".join(str(value) for value in fields["input_angular_momenta"])
        return head + rf"\left[\mathbf{{n}}=({radial}),\boldsymbol{{\ell}}=({angular})\right]"


def _density_labels(specs, elements, pace_public_index=False,
                    chemical_kind="explicit", physical_eta_bound=False,
                    multiplet=False):
    labels = []
    for index, spec in enumerate(specs):
        label = spec.label
        channels = []
        for channel in spec.channels:
            native_n = int(channel.n)
            chemical = ({"chemical_kind": "fixed_embedding",
                         "chemical_column": int(channel.mu)}
                        if chemical_kind == "fixed_embedding" else
                        {"neighbor_species": str(elements[int(channel.mu)])})
            channels.append({
                "eta": {"central_species": str(elements[int(channel.mu0)]),
                        **chemical,
                        "radial_index": native_n - 1 if pace_public_index else native_n,
                        **({"compiler_content_id": int(label.n_tuple[len(channels)])}
                           if physical_eta_bound else {}),
                        "central_charge_index": int(channel.kappa0),
                        "neighbor_charge_index": int(channel.kappa),
                        **({"native_pace_n": native_n} if pace_public_index else {})},
                "l": int(channel.l),
            })
        identity = str(spec.key)
        if multiplet:
            if "|M=0" not in identity or int(spec.L_R) <= 0:
                raise ValueError("Density multiplet labels require an M=0 compiler coordinate.")
            identity = identity.replace("|M=0", "|M=*")
        labels.append(FeatureLabel(index, "density", identity, {
            "N": int(label.rank),
            "L": int(spec.L_R),
            **({"M_values": tuple(range(-int(spec.L_R), int(spec.L_R) + 1))}
               if multiplet else {"M": int(spec.M_R)}),
            "radial_indices": tuple(int(channel.n) - 1 if pace_public_index else int(channel.n)
                                    for channel in spec.channels)
                              if physical_eta_bound else tuple(
                                  int(value) - 1 if pace_public_index else int(value)
                                  for value in label.n_tuple),
            **({"compiler_content_ids": tuple(int(value) for value in label.n_tuple)}
               if physical_eta_bound else {}),
            "input_angular_momenta": tuple(int(value) for value in label.l_tuple),
            "angular_intermediates": tuple(int(value) for value in label.internal_Ls),
            "one_factor_channels": tuple(channels),
            "compiler_basis_key": deepcopy(label.basis_key),
        }))
    return tuple(labels)


def _ordered_public_labels_sha256(labels):
    encoded = json.dumps([label.as_dict() for label in labels],
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _physical_eta_binding_sha256(specs, radial_caps):
    binding = {}
    inverse = {}
    for spec in specs:
        if len(spec.label.n_tuple) != len(spec.channels):
            raise ValueError("Physical eta binding has a different number of compiler and source leaves.")
        for content_id, channel in zip(spec.label.n_tuple, spec.channels):
            content_id = int(content_id)
            source = (int(channel.mu), int(channel.n))
            if content_id in binding and binding[content_id] != source:
                raise ValueError("Physical eta content ID has inconsistent source channels.")
            if source in inverse and inverse[source] != content_id:
                raise ValueError("Physical eta source channel has inconsistent content IDs.")
            binding[content_id] = source
            inverse[source] = content_id
    if not binding:
        raise ValueError("Physical eta binding has no source channels.")
    encoded = json.dumps({
        "binding": [[content_id, *binding[content_id]] for content_id in sorted(binding)],
        "physical_radial_nmax_per_rank": radial_caps,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _density_physical_source_sha256(site_basis_config, elements, cutoff, type_map):
    source = {
        "elements": list(elements), "cutoff_A": float(cutoff),
        "type_map": {str(name): int(value) for name, value in type_map.items()},
        "site_basis_config": serialize_site_basis_config(site_basis_config),
    }
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tagged_labels(records, tensor_order, raw_labels, raw_from_image=None):
    if raw_from_image is None:
        by_id = {str(item["raw_opportunity_id"]): item for item in raw_labels}
    else:
        by_feature = [[] for _ in records]
        for raw_label, terms in zip(raw_labels, raw_from_image, strict=True):
            for term in terms:
                opportunity = deepcopy(raw_label)
                opportunity["image_coefficient"] = deepcopy(term["coefficient"])
                by_feature[int(term["feature_index"])].append(opportunity)
    labels = []
    for index, record in enumerate(records):
        opportunities = (tuple(by_feature[index]) if raw_from_image is not None else tuple(
            deepcopy(by_id[str(item["raw_opportunity_id"])]) for item in record["contributors"]))
        labels.append(FeatureLabel(index, "tagged_cauchy_image", f"tagged-image:{index}", {
            "N": int(record.get("tensor_order", tensor_order)),
            "L": 0,
            "M": 0,
            "compiler_coordinate_provenance": deepcopy(record),
            "compiler_raw_opportunities": opportunities,
        }))
    return tuple(labels)


def _bar_phi_labels(config):
    labels = []
    for index, spec in enumerate(config.motif_specs):
        report = phi_motif_coupling_report(spec, target_L=0)
        if not report["validation_report"]["passed"]:
            raise ValueError(f"YE3T Phi coupling plan rejected motif {spec.name!r}.")
        labels.append(FeatureLabel(index, "bar_phi", spec.name, {
            "N": int(spec.template.vertex_count), "L": 0, "M": 0,
            "motif_name": spec.name,
            "motif_template": spec.template.to_dict(),
            "one_factor_channels": tuple(ch.to_dict() for ch in spec.channels),
            "slot_orbit_partition": report["slot_orbit_partition"],
            "compiler_coupling_plan": report["plan"],
            "normalization": (
                "weighted_embedding_average" if config.normalize_motif_features
                else "weighted_embedding_sum"
            ),
        }))
    return tuple(labels)


def _bar_phi_feature_rows(model, atoms, type_map, *, forces, stress):
    """Evaluate the existing explicit-motif feature map and its derivatives."""
    dtype = model.config.torch_dtype
    cell0 = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=dtype)
    if stress:
        if float(atoms.get_volume()) <= 0.0:
            raise ValueError("Phi stress rows require a positive cell volume.")
        cell = cell0.detach().clone().requires_grad_(True)
        scaled = torch.as_tensor(np.asarray(atoms.positions, float), dtype=dtype) @ torch.linalg.inv(cell0)
        positions = scaled @ cell
    else:
        cell = cell0
        positions = torch.tensor(np.asarray(atoms.positions, float), dtype=dtype, requires_grad=bool(forces))
    types = torch.as_tensor(
        [type_map[name] for name in atoms.get_chemical_symbols()], dtype=torch.long,
    )
    pbc = np.asarray(atoms.pbc, dtype=bool)
    model._validate_periodic_cutoff_margin(cell, pbc)
    src, dst, disp, _dist, values = model._edge_data(positions, types, cell=cell, pbc=pbc)
    sites, _payloads, _count, _weights = model._motif_values(positions, src, dst, disp, values, cell=cell, pbc=pbc)
    sums = sites.sum(dim=0)
    energy = np.concatenate(([float(len(atoms))], sums.detach().cpu().numpy()))
    force_rows = np.zeros((len(atoms) * 3, sums.numel() + 1), dtype=float)
    stress_rows = np.zeros((6, sums.numel() + 1), dtype=float)
    for index, feature in enumerate(sums):
        if not feature.requires_grad or not (forces or stress):
            continue
        inputs = (positions, cell) if stress else (positions,)
        gradients = torch.autograd.grad(feature, inputs, retain_graph=True, allow_unused=True)
        if forces and gradients[0] is not None:
            force_rows[:, index + 1] = -gradients[0].detach().cpu().numpy().reshape(-1)
        if stress and gradients[1] is not None:
            cell_gradient = gradients[1].detach().cpu().numpy()
            strain_gradient = np.asarray(atoms.cell.array, float).T @ cell_gradient
            tensor = 0.5 * (strain_gradient + strain_gradient.T) / float(atoms.get_volume())
            stress_rows[:, index + 1] = tensor[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    return energy, force_rows, stress_rows, sites.detach().cpu().numpy()


def _rank_values(value, ranks, name):
    if isinstance(value, int):
        return tuple(int(value) for _ in ranks)
    values = tuple(int(item) for item in value)
    if len(values) != len(ranks):
        raise ValueError(f"{name} must be an integer or have one value per rank.")
    return values


class Basis:
    """Resolved density, scalar tagged, or explicit motif basis."""

    def __init__(self, *, elements, source="density", cutoff, max_rank=None,
                 nmax=4, lmax=2, radial_decay=None, tag_counts=None,
                 rank=None, nmax_per_rank=None, lmax_per_rank=None,
                 source_block_partitions_by_rank=None, angular_patterns_by_rank=None,
                 angular_basis_backend=None,
                 max_records_per_rank=None, max_features_per_rank=None,
                 radial_degrees=None, tensor_order=None, angular_degree=None,
                 backend=None, motif_family="full", motif_specs=None,
                 channels=None, edge_cutoff=None, edge_basis_backend="site_basis",
                 periodic_image_mode="unique", normalize_motif_features=True,
                 compiled_cache_dir=None, descriptor_cache_dir=None,
                 compiler_validation="certificate", pair_cutoffs_A=None):
        self.elements = tuple(str(value) for value in elements)
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be a nonempty unique sequence.")
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("cutoff must be finite and positive in Angstrom.")
        self.source = str(source)
        catalogue_options = (rank, nmax_per_rank, lmax_per_rank,
            source_block_partitions_by_rank, angular_patterns_by_rank,
            max_records_per_rank, max_features_per_rank)
        if self.source == "density":
            if pair_cutoffs_A is not None:
                raise ValueError("pair_cutoffs_A is available for the tagged source only.")
            radial_decay = 0.25 if radial_decay is None else float(radial_decay)
            if compiled_cache_dir is not None:
                raise ValueError("Density uses descriptor_cache_dir, not compiled_cache_dir.")
            if tag_counts is not None or radial_degrees is not None or tensor_order is not None or angular_basis_backend is not None or any(
                    value is not None for value in catalogue_options):
                raise ValueError("Tagged source options require source='tagged_cauchy_image'.")
            rank_count = 3 if max_rank is None else int(max_rank)
            if rank_count < 1:
                raise ValueError("max_rank must be positive.")
            ranks = tuple(range(1, rank_count + 1))
            radial = _rank_values(nmax, ranks, "nmax")
            angular = _rank_values(lmax, ranks, "lmax")
            self.backend = "pytorch" if backend is None else str(backend)
            cache_dir = (default_linear_cache_directory() / "compiler" / "ordinary_density"
                         if descriptor_cache_dir is None else Path(descriptor_cache_dir))
            config = {
                "elements": self.elements,
                "type_map": {name: index for index, name in enumerate(self.elements)},
                "cutoff": self.cutoff,
                "ranks": ranks,
                "nmax": radial,
                "lmax": angular,
                "lmin": (0,) * rank_count,
                "L_R": 0,
                "M_R_values": (0,),
                "basis_type": "no_charge",
                "k_o_max": 0,
                "k_max": (0,) * rank_count,
                "max_labels_per_rank": None,
                "site_basis": {
                    "mode": "explicit",
                    "rc": self.cutoff,
                    "lmbda": float(radial_decay),
                },
                "backend": self.backend,
                "descriptor_cache_dir": cache_dir,
            }
            self._descriptor = YE3TDescriptors.ace(config)
            self._resolved = {
                "source": self.source, "elements": self.elements,
                "cutoff_A": self.cutoff, "ranks": ranks,
                "nmax": radial, "lmax": angular,
                "radial_decay": float(radial_decay), "backend": self.backend,
                "descriptor_cache_dir": str(cache_dir),
            }
            self._labels = _density_labels(self._descriptor.descriptor_specs, self.elements)
        elif self.source == "tagged_cauchy_image":
            if radial_decay is not None:
                raise ValueError("The tagged shifted-Jacobi source has no radial_decay setting.")
            if descriptor_cache_dir is not None:
                raise ValueError("Tagged Cauchy image uses compiled_cache_dir, not descriptor_cache_dir.")
            if tag_counts is None:
                raise ValueError("Tagged basis requires explicit tag_counts.")
            if max_rank is not None:
                raise ValueError("Tagged rank is specified by rank, not max_rank.")
            self.backend = "auto" if backend is None else str(backend)
            cache_dir = (default_linear_cache_directory() / "compiler" / "tagged_cauchy_image"
                         if compiled_cache_dir is None else Path(compiled_cache_dir))
            if compiler_validation not in {"full", "certificate"}:
                raise ValueError("compiler_validation must be full or certificate.")
            uses_catalogue = any(value is not None for value in catalogue_options)
            if uses_catalogue:
                if any(value is not None for value in (radial_degrees, tensor_order, angular_degree)):
                    raise ValueError("Use rank and per-rank caps without tensor_order, radial_degrees, or angular_degree.")
                if any(value is None for value in (rank, nmax_per_rank, lmax_per_rank,
                                                   source_block_partitions_by_rank)):
                    raise ValueError("Tagged catalogue basis requires rank, nmax_per_rank, "
                                     "lmax_per_rank, and source_block_partitions_by_rank.")
                order = int(rank)
                if order < 1:
                    raise ValueError("rank must be positive.")
                for name, setting in (("nmax_per_rank", nmax_per_rank),
                                      ("lmax_per_rank", lmax_per_rank),
                                      ("source_block_partitions_by_rank", source_block_partitions_by_rank)):
                    if (not isinstance(setting, dict) or len(setting) != 1
                            or {int(key) for key in setting} != {order}):
                        raise ValueError(f"{name} must contain exactly rank {order}.")
                radial_cap = int(next(iter(nmax_per_rank.values())))
                angular_cap = int(next(iter(lmax_per_rank.values())))
                if radial_cap < 1 or angular_cap < 0:
                    raise ValueError("nmax_per_rank must be positive and lmax_per_rank nonnegative.")
                angular_compiler = ("exact_weight_space_v1" if angular_basis_backend is None
                                    else str(angular_basis_backend))
                if angular_compiler not in {"legacy_exact", "exact_weight_space_v1"}:
                    raise ValueError("angular_basis_backend must be legacy_exact or exact_weight_space_v1.")
                partitions = tuple(tuple(int(part) for part in parts)
                                   for parts in next(iter(source_block_partitions_by_rank.values())))
                if not partitions or any(not parts or min(parts) < 1 or sum(parts) != order
                                         for parts in partitions):
                    raise ValueError("Each source block partition must contain positive sizes summing to rank.")
                catalogue = {
                    "nmax_per_rank": {order: radial_cap},
                    "lmax_per_rank": {order: angular_cap},
                    "source_block_partitions_by_rank": {order: partitions},
                    "tag_counts_by_rank": {order: tuple(int(value) for value in tag_counts)},
                    "angular_basis_backend": angular_compiler,
                }
                if angular_patterns_by_rank is not None:
                    if (not isinstance(angular_patterns_by_rank, dict)
                            or len(angular_patterns_by_rank) != 1
                            or {int(key) for key in angular_patterns_by_rank} != {order}):
                        raise ValueError("angular_patterns_by_rank must contain exactly the selected rank.")
                    patterns = tuple(tuple(int(value) for value in pattern)
                                     for pattern in next(iter(angular_patterns_by_rank.values())))
                    if not patterns or any(len(pattern) != order or min(pattern) < 0
                                           or max(pattern) > angular_cap for pattern in patterns):
                        raise ValueError("Each angular pattern must have rank entries in 0..lmax_per_rank.")
                    catalogue["angular_patterns_by_rank"] = {order: patterns}
                for name, value in (("max_records_per_rank", max_records_per_rank),
                                    ("max_features_per_rank", max_features_per_rank)):
                    if value is not None:
                        catalogue[name] = {order: int(value)}
                tagged_request = {"catalogue": catalogue, "cutoff_A": self.cutoff,
                                  "coefficient_materialization": "compile",
                                  "compiled_cache_dir": cache_dir,
                                  "compiler_validation": compiler_validation}
            else:
                if angular_basis_backend is not None:
                    raise ValueError("angular_basis_backend requires rank and per-rank tagged catalogue settings.")
                if radial_degrees is None:
                    raise ValueError("Tagged basis requires rank and per-rank caps; legacy requests need radial_degrees.")
                order = 4 if tensor_order is None else int(tensor_order)
                angular = 1 if angular_degree is None else int(angular_degree)
                tagged_request = {
                    "tensor_order": order,
                    "selected_raw_tag_counts": tuple(int(value) for value in tag_counts),
                    "radial_degrees": tuple(int(value) for value in radial_degrees),
                    "angular_degree": angular,
                    "cutoff_A": self.cutoff,
                    "coefficient_materialization": "compile",
                    "compiled_cache_dir": cache_dir,
                    "compiler_validation": compiler_validation,
                }
            if pair_cutoffs_A is not None:
                tagged_request["pair_cutoffs_A"] = dict(pair_cutoffs_A)
            config = {
                "elements": self.elements,
                "representation": YE3TRepresentation.tagged_cauchy_image(),
                "tagged_cauchy_image": tagged_request,
                "backend": self.backend,
            }
            self._descriptor = YE3TDescriptors.ye3t_basis(config)
            self._resolved = {
                "source": self.source, "elements": tuple(self._descriptor.elements),
                "cutoff_A": self.cutoff, "N": order,
                "tag_counts": tuple(int(v) for v in tag_counts),
                "backend": self.backend,
                "compiled_cache_dir": str(cache_dir),
                "compiler_validation": compiler_validation,
            }
            if pair_cutoffs_A is not None:
                self._resolved["pair_cutoffs_A"] = dict(pair_cutoffs_A)
            if uses_catalogue:
                self._resolved["catalogue"] = catalogue
                self._resolved["angular_basis_backend"] = angular_compiler
            else:
                self._resolved["radial_degrees"] = tuple(int(v) for v in radial_degrees)
                self._resolved["angular_degree"] = angular
            self.elements = tuple(self._descriptor.elements)
            compiled = self._descriptor.metadata["tagged_cauchy_image_compiled"]
            self._resolved["polynomial_backend"] = self._descriptor.metadata[
                "tagged_cauchy_image_evaluator"].backend
            self._labels = _tagged_labels(
                self._descriptor.feature_labels, order,
                compiled.payload.get("raw_coordinate_labels", compiled.payload.get("raw_labels")),
                compiled.payload.get("raw_from_image") if uses_catalogue else None,
            )
        elif self.source == "bar_phi":
            if pair_cutoffs_A is not None:
                raise ValueError("pair_cutoffs_A is available for the tagged source only.")
            if compiled_cache_dir is not None or descriptor_cache_dir is not None:
                raise ValueError("bar_phi does not use density or tagged compiler cache settings.")
            if max_rank is not None or tag_counts is not None or radial_degrees is not None or tensor_order is not None or angular_basis_backend is not None or any(
                    value is not None for value in catalogue_options):
                raise ValueError("bar_phi uses explicit motif slots; density and tagged truncations do not apply.")
            self.backend = "pytorch" if backend is None else str(backend)
            if self.backend != "pytorch":
                raise ValueError("bar_phi currently uses the PyTorch reference evaluator.")
            phi = PhiBranchConfig(
                cutoff=self.cutoff,
                edge_cutoff=edge_cutoff,
                channels=() if channels is None else tuple(channels),
                motif_specs=() if motif_specs is None else tuple(motif_specs),
                motif_family=motif_family,
                edge_basis_backend=edge_basis_backend,
                periodic_image_mode=periodic_image_mode,
                normalize_motif_features=normalize_motif_features,
            )
            config = {
                "elements": self.elements,
                "type_map": {name: index for index, name in enumerate(self.elements)},
                "branches": ("bar_phi",),
                "phi": phi.to_dict(),
                "backend": self.backend,
            }
            self._descriptor = YE3TDescriptors.phi(config)
            self._resolved = {
                "source": self.source, "elements": self.elements,
                "cutoff_A": self.cutoff, "backend": self.backend,
                "phi_config": self._descriptor.metadata["phi_config"],
            }
            self._labels = _bar_phi_labels(phi)
        else:
            raise ValueError("source must be 'density', 'tagged_cauchy_image', or 'bar_phi'.")

    @classmethod
    def combine(cls, components):
        """Concatenate independently compiled scalar components in named order."""
        if not isinstance(components, Mapping) or len(components) < 2:
            raise ValueError("Basis.combine requires at least two named components.")
        names = tuple(components)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Component names must be nonempty strings.")
        bases = tuple(components.values())
        if any(not isinstance(item, cls) for item in bases):
            raise TypeError("Every component must be a Basis.")
        for item in bases:
            if item.source == "configured":
                item._materialize_configured()
        if any(item._descriptor is None and not hasattr(item, "_loaded_tagged_model")
               for item in bases):
            raise ValueError("Basis.combine requires constructed component bases with evaluators.")
        if any(item.source not in {"density", "tagged_cauchy_image"} for item in bases):
            raise ValueError("Basis.combine currently supports scalar density and tagged components.")
        species = bases[0].elements
        if any(item.elements != species for item in bases):
            raise ValueError("Combined components must have the same ordered species map.")
        if not any(item.source == "tagged_cauchy_image" for item in bases):
            raise ValueError("Combined scalar fitting currently requires a tagged component.")
        if not any(item.source == "density" for item in bases):
            raise ValueError("Combined scalar fitting currently requires a density component.")
        if any(any(label.as_dict().get("L") != 0 for label in item.labels) for item in bases):
            raise ValueError("Basis.combine currently requires scalar output in every component.")
        contracts = [item._resolved["representation"] for item in bases
                     if "representation" in item._resolved]
        if contracts:
            parent_contracts = [
                json.dumps({"group": record["group"], "parent": record["parent"]},
                           sort_keys=True, separators=(",", ":"))
                for record in contracts]
            if len(set(parent_contracts)) != 1:
                raise ValueError("Combined components must share global parent and parity.")
        labels = []
        manifest = []
        for name, item in zip(names, bases):
            single_factors = deepcopy(item._resolved.get("single_factors"))
            source = (single_factors or {}).get("radial")
            if source is None:
                source = {key: deepcopy(item._resolved[key]) for key in
                          ("radial_decay", "radial_degrees", "site_basis_config")
                          if key in item._resolved}
            elif source.get("units") != "Angstrom":
                raise ValueError("Combined scalar radial sources must use Angstrom units.")
            manifest.append({"name": name, "kind": item.source,
                             "cutoff_A": float(item.cutoff), "radial_source": source,
                             "single_factors": single_factors,
                             "resolution_sha256": (item._resolution.sha256
                                                   if hasattr(item, "_resolution") else None),
                             "feature_count": len(item.labels)})
            for label in item.labels:
                details = label.as_dict()
                for key in ("feature_index", "source", "identity"):
                    details.pop(key)
                labels.append(FeatureLabel(len(labels), label.source,
                                           name + ":" + label.identity,
                                           {"component": name, **details}))
        basis = object.__new__(cls)
        basis.source = "combined_scalar"
        basis.elements = species
        basis.cutoff = max(item.cutoff for item in bases)
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._components = dict(zip(names, bases))
        basis._labels = tuple(labels)
        basis._resolved = {"source": basis.source, "elements": species,
                           "cutoff_A": basis.cutoff, "components": manifest,
                           "output": {"L": 0, "scope": "per_atom"}}
        return basis

    @classmethod
    def from_config(cls, config, *, representation, runtime,
                    _check_optional_dependencies=True):
        """Purpose: Construct a public Basis from a validated config section.

        Mathematical contract: Source content is resolved before compiler
        counts; no application-local coupling labels are invented.
        Inputs: Basis config, core representation, and runtime config.
        Outputs: A Basis with catalogue preview and frozen resolution report.
        Does not: Materialize coefficients until create, labels, or fit.
        """
        resolution = resolve_basis_config(
            config, representation, runtime,
            check_optional_dependencies=_check_optional_dependencies)
        payload = resolution.to_dict()
        basis = object.__new__(cls)
        ordered_phi = (len(payload["components"]) == 1
                       and payload["components"][0]["kind"] == "explicit_phi")
        basis.source = "explicit_phi" if ordered_phi else "configured"
        basis.elements = tuple(payload["species"])
        basis.cutoff = max(float(row["single_factors"]["radial"]["cutoff_A"])
                           for row in payload["components"])
        basis.backend = payload["runtime"]["evaluator"]
        basis._descriptor = None
        basis._labels = ()
        basis._resolution = resolution
        basis._catalogue = CataloguePreview(resolution)
        basis._resolved = payload
        basis._construction = {"basis": deepcopy(config),
                               "representation": representation.to_dict(),
                               "runtime": deepcopy(runtime)}
        if ordered_phi:
            from ye3t.couplings import plan as coupling_plan

            component = payload["components"][0]
            basis._phi_plan = coupling_plan(
                **component["compiler_request"],
                metadata={"selected_typed_alpha": component["selected_typed_alpha"]},
            )
            labels = basis._phi_plan.report.labels_for_target(2)
            alpha = component["selected_typed_alpha"]
            compiler_label = labels[alpha]
            if (compiler_label.multiplicity_index != alpha
                    or compiler_label.partition != (4, 4)
                    or compiler_label.L_R != 2):
                raise ArithmeticError("Ordered Phi compiler label disagrees with selected route.")
            binding = basis._phi_plan.validation_report["alpha_bindings"][alpha]
            basis._labels = (FeatureLabel(0, "explicit_phi", f"compiler_alpha:{alpha}", {
                "N": 8, "L": 2, "parent_partition": [4, 4],
                "selected_full_alpha": alpha,
                "compiler_label": compiler_label.to_dict(),
                "alpha_binding": binding,
                "compiler_plan_hash": basis._phi_plan.convention_hash,
                "output_convention": component["output_convention"],
            }),)
            basis._phi_compiled = None
            basis._phi_site_basis = None
        return basis

    def _density_multiplet_compact_labels(self, component):
        from ye3t.couplings import count as count_couplings, normalize_compact_label
        from ye3t.core.basis.validation import iter_canonical_leaf_labelings

        target = self._resolved["representation"]["parent"]
        L = int(target["L"])
        parity = target["parity"]
        ranks = tuple(int(rank) for rank in component["ranks"])
        expected = self.catalogue.counts()["by_component"][component["name"]]
        labels = []
        by_rank = {}
        for rank in ranks:
            start = len(labels)
            for content, angular in iter_canonical_leaf_labelings(
                    rank, component["active_compiler_content_ids_by_rank"][str(rank)],
                    range(component["lmax_per_rank"][str(rank)] + 1),
                    multiplicity_partitions=component["source_block_partitions_by_rank"][str(rank)]):
                if ("odd" if sum(angular) % 2 else "even") != parity:
                    continue
                report = count_couplings({
                    "content": content,
                    "target_rotation": {"L_R": L, "parity": parity, "group": "O3"},
                    "target_permutation": "young:" + str(rank),
                    "carrier": "ACE_density", "metadata": {"input_Ls": angular},
                }, input_Ls=angular)
                labels.extend(normalize_compact_label(row)
                              for row in report.labels_for_target(L))
            by_rank[rank] = len(labels) - start
        if (not labels or len(set(labels)) != len(labels) or
                by_rank != expected["by_rank_per_center"]):
            raise RuntimeError("Density full-M compiler labels disagree with exact preview.")
        return tuple(labels)

    def _materialize_density_multiplets(self, component):
        from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import build_descriptor_specs_from_settings

        target = self._resolved["representation"]["parent"]
        L = int(target["L"])
        ranks = tuple(int(rank) for rank in component["ranks"])
        expected = self.catalogue.counts()["by_component"][component["name"]]
        labels = self._density_multiplet_compact_labels(component)
        chemical = component["single_factors"]["chemical"]
        bound = chemical["kind"] == "fixed_embedding" or len(self.elements) > 1
        physical_channels = None
        if bound:
            physical_channels = tuple(
                (int(row["compiler_content_id"]),
                 int(row["chemical"]["chemical_index"]),
                 int(row["native_pace_n"]))
                for row in component["physical_eta_by_center"][self.elements[0]])
            if any(tuple((int(row["compiler_content_id"]),
                          int(row["chemical"]["chemical_index"]),
                          int(row["native_pace_n"]))
                         for row in component["physical_eta_by_center"][center]) !=
                   physical_channels for center in self.elements[1:]):
                raise RuntimeError("Density full-M physical channels differ by center species.")
        nmax = tuple(max(component["active_compiler_content_ids_by_rank"][str(rank)])
                     if bound else component["nmax_per_rank"][str(rank)]
                     for rank in ranks)
        lmax = tuple(component["lmax_per_rank"][str(rank)] for rank in ranks)
        representation = YE3TRepresentation.ace(
            basis_mode=None, fast_path_policy="disable",
            metadata={"global_young_sector": "(N)",
                      "basis_convention": "pace_complex_magnetic_y00_1"})
        descriptor = YE3TDescriptors.ace({
            "elements": self.elements,
            "type_map": {name: index for index, name in enumerate(self.elements)},
            "cutoff": self.cutoff, "ranks": ranks, "basis_type": "no_charge",
            "k_o_max": 0, "k_max": [0] * len(ranks), "nmax": nmax,
            "lmax": lmax, "lmin": [0] * len(ranks), "L_R": L,
            "M_R_values": list(range(-L, L + 1)), "compact_labels": tuple(labels),
            "parity_filter": ("natural" if target["parity"] ==
                              ("odd" if L % 2 else "even") else "none"),
            **({"_physical_content_channels": physical_channels} if bound else {}),
            "factorized_descriptor_runtime_policy": "disable",
            "site_basis_config": self._density_full_m_site_config(component),
            "representation": representation, "backend": "pytorch",
            "strict_backend": True, "validate_backend": True, "device": "cpu",
        })
        calculator = descriptor.ace_descriptor.calculator
        collection = build_descriptor_specs_from_settings(
            calculator.labels, calculator.settings, calculator.coupling_library,
            center_mu_values=tuple(range(len(self.elements))),
            physical_content_channels=physical_channels)
        ordered = tuple(tuple(collection.specs_by_M[M]) for M in range(-L, L + 1))
        baseline = tuple(spec.key.replace("|M=0", "|M=*") for spec in ordered[L])
        reference_columns = tuple((spec.label, spec.channels) for spec in ordered[L])
        if (len(baseline) != expected["all_centers"] or
                tuple(spec.key for spec in descriptor.descriptor_specs) !=
                tuple(spec.key for spec in ordered[L]) or
                any(tuple(spec.key.replace(f"|M={M}", "|M=*") for spec in rows)
                    != baseline or
                    tuple((spec.label, spec.channels) for spec in rows) !=
                    reference_columns
                    for M, rows in zip(range(-L, L + 1), ordered))):
            raise RuntimeError("Density full-M magnetic blocks differ in compiler label order.")
        _validate_density_full_m_intertwiner(ordered, L)
        self.source = "density"
        self.backend = "pytorch"
        self._descriptor = descriptor
        self._density_full_m_evaluator = calculator.evaluator
        self._density_full_m_type_map = descriptor.type_map
        self._density_full_m_specs = ordered
        self._density_full_m = True
        self._density_full_m_compiled_record = _density_full_m_compiler_record(
            calculator, ordered)
        self._density_full_m_plan_hash = hashlib.sha256(_full_m_canonical_bytes(
            self._density_full_m_compiled_record)).hexdigest()
        self._density_full_m_convention_hash = _density_full_m_convention_hash(
            L, target["parity"])
        self._labels = _density_labels(
            ordered[L], self.elements, pace_public_index=True,
            chemical_kind=chemical["kind"], physical_eta_bound=bound,
            multiplet=True)
        self._resolved["output_layout"] = "atoms_multiplets_real_tesseral_M"

    def _density_full_m_site_config(self, component):
        radial = component["single_factors"]["radial"]
        chemical = component["single_factors"]["chemical"]
        return {
            "rc": [self.cutoff], "lmbda": [radial["lambda"]],
            "nradmax": max(component["nmax_per_rank"].values()),
            "lmax": max(component["lmax_per_rank"].values()), "kmax": 0,
            "possible_types": list(range(len(self.elements))),
            "radial_basis": "PACE_ChebExpCos",
            "chemical_basis": "fixed_embedding" if chemical["kind"] == "fixed_embedding" else "delta",
            **({"chemical_embedding": chemical["matrix"]}
               if chemical["kind"] == "fixed_embedding" else {}),
            "charge_mode": "none", "atomic_base_normalization": "none",
            "factor_normalization": "none", "spherical_backend": "complex",
            "spherical_normalization": "pace_y00_one", "source_backend": "torch",
            "dtype": "float64", "pace_cutoff_width": [radial["cutoff_width_A"]],
            "pace_spline_spacing": [0.001], "pace_inner_cutoff": [0.0],
            "pace_inner_cutoff_width": [0.0], "pace_crad_policy": "identity",
        }

    def _materialize_configured(self, *, ordinary_catalogue=None):
        if self.source != "configured":
            return
        capability = self._resolution.capability_report
        if not capability["basis_create_available"]:
            raise RuntimeError(capability["basis_create_reason"])
        if len(self._resolved["components"]) > 1:
            if ordinary_catalogue is not None:
                raise ValueError("A component catalogue cannot replay an entire combined basis.")
            from ye3t import YE3TRepresentation as CoreRepresentation
            source = self._construction
            rep = CoreRepresentation.from_config(source["representation"])
            components = {}
            for name, raw in source["basis"]["components"].items():
                local = {
                    "single_factors": deepcopy(raw.get(
                        "single_factors", source["basis"]["single_factors"])),
                    "tensor_product": deepcopy(raw["tensor_product"]),
                    "catalogue": deepcopy(raw["catalogue"]),
                }
                local["single_factors"]["species"] = list(self.elements)
                components[name] = Basis.from_config(
                    local, representation=rep, runtime=source["runtime"])
            combined = Basis.combine(components)
            if combined.elements != self.elements or combined.cutoff != self.cutoff:
                raise RuntimeError("Configured combined source identity changed during materialization.")
            self.__dict__.update(combined.__dict__)
            self._resolved["construction_resolution"] = self._resolution.to_dict()
            self._resolved["construction_resolution_sha256"] = self._resolution.sha256
            return
        component = self._resolved["components"][0]
        if component["kind"] == "density":
            if self._resolved["representation"]["parent"]["L"] > 0:
                if ordinary_catalogue is not None:
                    raise ValueError("Scalar ordinary catalogue cannot bind a full-M density basis.")
                self._materialize_density_multiplets(component)
                return
            from ye3t_methods.atomistic.ace.catalogue_selection import resolve_ordinary_scalar_catalogues
            from ye3t_methods.atomistic.ace.lammps_export import compile_ordinary_scalar_catalogue

            ranks = tuple(int(rank) for rank in component["ranks"])
            counts = self.catalogue.counts()["by_component"][component["name"]]
            expected = int(counts["per_center"])
            chemical = component["single_factors"]["chemical"]
            embedded = chemical["kind"] == "fixed_embedding"
            expanded_chemistry = embedded or len(self.elements) > 1
            chemical_width = component["chemical_channel_count"]
            physical_eta_bound = expanded_chemistry
            if expanded_chemistry and not physical_eta_bound and (expected % chemical_width or any(
                    count % chemical_width for count in
                    counts["by_rank_per_center"].values())):
                raise RuntimeError("Configured chemistry count cannot bind to radial compiler rows.")
            compiled_expected = (expected if physical_eta_bound else
                                 expected // chemical_width if expanded_chemistry else expected)
            eta_records = component["physical_eta_by_center"][self.elements[0]]
            physical_channels = tuple(
                (int(row["compiler_content_id"]),
                 int(row["chemical"]["chemical_index"]),
                 int(row["native_pace_n"]))
                for row in eta_records)
            if physical_eta_bound and any(tuple(
                    (int(row["compiler_content_id"]),
                     int(row["chemical"]["chemical_index"]),
                     int(row["native_pace_n"]))
                    for row in component["physical_eta_by_center"][center]) !=
                    physical_channels for center in self.elements[1:]):
                raise RuntimeError("Physical eta bindings differ between central species.")
            catalogue_id = "configured_ordinary_" + self._resolution.sha256[:16]
            request = {
                "tensor_orders": ranks,
                "nmax_by_tensor_order": (
                    {rank: max(component["active_compiler_content_ids_by_rank"][str(rank)])
                     for rank in ranks} if physical_eta_bound else component["nmax_per_rank"]),
                "lmax_by_tensor_order": component["lmax_per_rank"],
                "channel_multiplicity_partitions_by_order":
                    component["source_block_partitions_by_rank"],
                "target_descriptor_counts": [compiled_expected],
                **({"content_ids_by_tensor_order":
                    component["active_compiler_content_ids_by_rank"]}
                   if physical_eta_bound else {}),
            }
            if ordinary_catalogue is None:
                source, _manifest = resolve_ordinary_scalar_catalogues(
                    request, catalogue_id=catalogue_id,
                    target_parity=self._resolved["representation"]["parent"]["parity"])
                application = compile_ordinary_scalar_catalogue(
                    source, profile_id=catalogue_id + "_" + str(compiled_expected))
            else:
                application = deepcopy(ordinary_catalogue)
            application_counts = {rank: 0 for rank in ranks}
            for row in application["rows"]:
                application_counts[len(row["compact_label"]["n_tuple"])] += 1
            expected_by_rank = ({rank: count // chemical_width
                                 for rank, count in counts["by_rank_per_center"].items()}
                                if expanded_chemistry and not physical_eta_bound else
                                counts["by_rank_per_center"])
            if (len(application["rows"]) != compiled_expected or
                    application_counts != expected_by_rank):
                raise RuntimeError("Configured ordinary compiler catalogue differs from exact preview.")
            radial = component["single_factors"]["radial"]
            nmax = tuple((max(component["active_compiler_content_ids_by_rank"][str(rank)])
                          if physical_eta_bound else component["nmax_per_rank"][str(rank)])
                         for rank in ranks)
            lmax = tuple(component["lmax_per_rank"][str(rank)] for rank in ranks)
            representation = YE3TRepresentation.ace(
                basis_mode=None, fast_path_policy="disable",
                metadata={"global_young_sector": "(N)",
                          "basis_convention": "pace_complex_magnetic_y00_1"},
            )
            descriptor = YE3TDescriptors.ace({
                "elements": self.elements,
                "type_map": {name: index for index, name in enumerate(self.elements)},
                "cutoff": self.cutoff,
                "ranks": ranks,
                "basis_type": "no_charge",
                "k_o_max": 0,
                "k_max": [0] * len(ranks),
                "nmax": nmax,
                "lmax": lmax,
                "lmin": [0] * len(ranks),
                "L_R": 0,
                "M_R_values": [0],
                "ordinary_scalar_catalogue": application,
                **({"restrict_neighbor_mu": tuple(range(chemical_width))}
                   if embedded and not physical_eta_bound else {}),
                **({"_physical_content_channels": physical_channels}
                   if physical_eta_bound else {}),
                "factorized_descriptor_runtime_policy": "disable",
                "site_basis_config": {
                    "rc": [self.cutoff],
                    "lmbda": [radial["lambda"]],
                    "nradmax": max(component["nmax_per_rank"].values()),
                    "lmax": max(lmax),
                    "kmax": 0,
                    "possible_types": list(range(len(self.elements))),
                    "radial_basis": "PACE_ChebExpCos",
                    "chemical_basis": "fixed_embedding" if embedded else "delta",
                    **({"chemical_embedding": chemical["matrix"]} if embedded else {}),
                    "charge_mode": "none",
                    "atomic_base_normalization": "none",
                    "factor_normalization": "none",
                    "spherical_backend": "complex",
                    "spherical_normalization": "pace_y00_one",
                    "source_backend": "torch",
                    "dtype": "float64",
                    "pace_cutoff_width": [radial["cutoff_width_A"]],
                    "pace_spline_spacing": [0.001],
                    "pace_inner_cutoff": [0.0],
                    "pace_inner_cutoff_width": [0.0],
                    "pace_crad_policy": "identity",
                },
                "representation": representation,
                "backend": "pytorch",
                "strict_backend": True,
                "validate_backend": True,
                "device": "cpu",
            })
            expected_total = expected * len(self.elements)
            variants_per_label = (len(self.elements) if physical_eta_bound else
                                  len(self.elements) * chemical_width
                                  if expanded_chemistry else 1)
            if len(descriptor.descriptor_specs) != expected_total or [
                    spec.label.to_dict() for spec in descriptor.descriptor_specs] != [
                    row["compact_label"] for row in application["rows"]
                    for _ in range(variants_per_label)]:
                raise RuntimeError("Configured ordinary descriptor order differs from compiler catalogue.")
            self.source = "density"
            self.backend = "pytorch"
            self._descriptor = descriptor
            self._labels = _density_labels(
                descriptor.descriptor_specs, self.elements,
                pace_public_index=True, chemical_kind=chemical["kind"],
                physical_eta_bound=physical_eta_bound)
            return
        if self._resolved["representation"]["parent"]["L"] > 0:
            from ye3t.couplings import (
                tagged_cauchy_carrier_physical_image_plan, tagged_cauchy_carrier_schedule,
            )

            ranks = tuple(int(rank) for rank in component["ranks"])
            preflight = self.catalogue.repeated_content_summary()["by_component"][component["name"]]
            record_cap = sum(row["candidate_fixed_contents_before_parity"] for row in preflight)
            cache_mode = self._resolved["runtime"]["cache"]["mode"]
            cache_dir = (None if cache_mode == "off" else
                         default_linear_cache_directory() / "compiler" / "tagged_cauchy_carriers")
            descriptor = YE3TDescriptors.ye3t_basis({
                "basis": {"type": "tagged_cauchy_carriers", "species": self.elements,
                          "cutoff_A": self.cutoff,
                          "catalogue": {
                              "ranks": ranks,
                              **({"tag_counts": component["tag_counts_per_rank"][str(ranks[0])]}
                                 if len(ranks) == 1 else
                                 {"tag_counts_by_rank": component["tag_counts_per_rank"]}),
                              "nmax_per_rank": component["nmax_per_rank"],
                              "lmax_per_rank": component["lmax_per_rank"],
                              "source_block_partitions_by_rank":
                                  component["source_block_partitions_by_rank"],
                              "input_Lmax": self._resolved["representation"]["parent"]["L"],
                              "max_records_per_rank": record_cap,
                          }},
                "representation": {"mode": "tagged_cauchy_carriers",
                                   "sector_policy": "tagged_mixed"},
                "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                            "compiled_cache_dir": cache_dir},
                "model": {},
            }, _compiled_tagged_carriers=getattr(self, "_embedded_tagged_carriers", None))
            compiled = descriptor.metadata["tagged_cauchy_carriers_compiled"]
            plan = tagged_cauchy_carrier_physical_image_plan(compiled["sources"])
            selected = set(plan["selected_coordinate_ids"])
            grouped = {}
            for source in compiled["sources"]:
                grouped.setdefault(int(source["request"]["tag_count"]), []).append(source)
            target_L = int(self._resolved["representation"]["parent"]["L"])
            parity = (1 if self._resolved["representation"]["parent"]["parity"] == "even"
                      else -1)
            labels, selection, selected_schedules = [], [], []
            for tag_count in sorted(grouped):
                if not any(record["label"]["coordinate_id"] in selected
                           for source in grouped[tag_count] for record in source["descriptors"]):
                    continue
                schedule = tagged_cauchy_carrier_schedule(
                    grouped[tag_count], coordinate_ids=selected)
                selected_schedules.append(schedule)
                for record in schedule["inventory"]:
                    label = record["label"]
                    if label["target_L"] != target_L or label["target_parity"] != parity:
                        continue
                    labels.append(FeatureLabel(len(labels), "tagged_carriers",
                        label["coordinate_id"], {
                            "N": sum(label["formal_parent"]), "L": target_L,
                            "M_values": tuple(range(-target_L, target_L + 1)),
                            "tag_count": tag_count, "target_parity": parity,
                            "compiler_label": label,
                            "physical_image_plan_hash": plan["self_hash"],
                        }))
                    selection.append((tag_count, tuple(record["component_slice"])))
            self.source = "tagged_carriers"
            self.backend = "reference"
            self._descriptor = descriptor
            descriptor.metadata["_tagged_carrier_physical_image_plan"] = plan
            descriptor.metadata["_tagged_carrier_physical_image_schedules"] = tuple(
                selected_schedules)
            self._labels = tuple(labels)
            self._tagged_carrier_selection = tuple(selection)
            self._tagged_carrier_plan_hash = plan["self_hash"]
            self._resolved["physical_image_plan_hash"] = plan["self_hash"]
            self._resolved["physical_image_rank_policy"] = plan["rank_policy"]
            self._resolved["compiler_hash"] = compiled["self_hash"]
            self._resolved["output_layout"] = "atoms_multiplets_real_tesseral_M"
            return
        if ordinary_catalogue is not None:
            raise ValueError("An ordinary scalar catalogue cannot materialize a tagged component.")
        ranks = tuple(int(rank) for rank in component["ranks"])
        summary = self.catalogue.repeated_content_summary()["by_component"][component["name"]]
        record_counts = {
            rank: sum(row["candidate_fixed_contents_before_parity"]
                      for row in summary if row["rank"] == rank)
            for rank in ranks
        }
        preview = self.catalogue.counts()["by_component"][component["name"]]
        raw_count = int(preview["raw_opportunity_count"])
        raw_by_rank = preview["raw_opportunities_by_rank"]
        if any(record_counts[rank] < 1 or raw_by_rank.get(rank, 0) < 1
               for rank in ranks) or raw_count < 1:
            raise ValueError("Configured tagged basis has no compiler source records or raw opportunities.")
        if len(ranks) == 1:
            rank = ranks[0]
            materialized = Basis(
                elements=self.elements, source="tagged_cauchy_image", cutoff=self.cutoff,
                rank=rank, tag_counts=component["tag_counts_per_rank"][str(rank)],
                nmax_per_rank={rank: component["nmax_per_rank"][str(rank)]},
                lmax_per_rank={rank: component["lmax_per_rank"][str(rank)]},
                source_block_partitions_by_rank={
                    rank: component["source_block_partitions_by_rank"][str(rank)]},
                **({"angular_patterns_by_rank": {
                    rank: component["angular_patterns_by_rank"][str(rank)]}}
                   if "angular_patterns_by_rank" in component else {}),
                max_records_per_rank=record_counts[rank],
                max_features_per_rank=raw_count,
                backend="reference", compiler_validation="full",
            )
            descriptor = materialized._descriptor
            labels = materialized.labels
        else:
            catalogue = {
                "nmax_per_rank": component["nmax_per_rank"],
                "lmax_per_rank": component["lmax_per_rank"],
                "source_block_partitions_by_rank": component["source_block_partitions_by_rank"],
                "tag_counts_by_rank": component["tag_counts_per_rank"],
                **({"angular_patterns_by_rank": component["angular_patterns_by_rank"]}
                   if "angular_patterns_by_rank" in component else {}),
                "max_records_per_rank": record_counts,
                "max_features_per_rank": raw_by_rank,
            }
            descriptor = YE3TDescriptors.ye3t_basis({
                "elements": self.elements,
                "representation": YE3TRepresentation.tagged_cauchy_image(),
                "tagged_cauchy_image": {
                    "catalogue": catalogue,
                    "cutoff_A": self.cutoff,
                    "compiled_cache_dir": default_linear_cache_directory()
                        / "compiler" / "tagged_cauchy_image",
                    "compiler_validation": "full",
                },
                "backend": "reference",
            })
            compiled = descriptor.metadata["tagged_cauchy_image_compiled"]
            labels = _tagged_labels(
                descriptor.feature_labels, None,
                compiled.payload.get("raw_coordinate_labels", compiled.payload.get("raw_labels")),
                compiled.payload.get("raw_from_image"),
            )
        preflight = descriptor.metadata["tagged_cauchy_image_preflight"]
        compiled = descriptor.metadata["tagged_cauchy_image_compiled"]
        if (int(preflight.raw_label_count) != raw_count or
                len(labels) != int(descriptor.metadata["feature_count"]) or
                len(labels) != len(compiled.payload["image_coordinate_provenance"])):
            raise RuntimeError("Configured tagged physical image differs from compiler preview or labels.")
        self.source = "tagged_cauchy_image"
        self.backend = "reference"
        self._descriptor = descriptor
        self._labels = labels

    @classmethod
    def _from_density_bundle(cls, bundle, cutoff, type_map):
        label_convention = (bundle.fit_metadata or {}).get(
            "ye3t_methods_public_label_convention")
        pace_public_index = False
        physical_eta_bound = any(
            str(spec.key).endswith("|physical_eta_bound")
            for spec in bundle.descriptor_specs)
        if physical_eta_bound and (not isinstance(label_convention, dict) or
                                   label_convention.get("schema") !=
                                   "ye3t_methods_density_pace_labels_v3"):
            raise ValueError("Bound physical eta density requires a v3 public source record.")
        if label_convention is None and (
                getattr(bundle.site_basis_config, "chemical_basis", "delta") != "delta" or
                getattr(bundle.site_basis_config, "chemical_embedding", None) is not None):
            raise ValueError("Embedded density chemistry requires a bound public source record.")
        if label_convention is not None:
            schema = label_convention.get("schema") if isinstance(label_convention, dict) else None
            keys = {"schema", "radial_family", "radial_index_base",
                    "native_radial_index_base", "resolution_sha256",
                    "ordered_public_labels_sha256"}
            if (not isinstance(label_convention, dict) or
                    schema not in {"ye3t_methods_density_pace_labels_v1",
                                   "ye3t_methods_density_pace_labels_v2",
                                   "ye3t_methods_density_pace_labels_v3"} or
                    set(label_convention) !=
                    (keys | {"physical_source_sha256", "physical_eta_binding_sha256",
                             "physical_radial_nmax_per_rank"}
                     if schema.endswith("_v3") else
                     keys | {"physical_source_sha256"}
                     if schema.endswith("_v2") else keys) or
                    label_convention.get("radial_family") != "pace_chebexp_cos" or
                    label_convention.get("radial_index_base") != 0 or
                    label_convention.get("native_radial_index_base") != 1 or
                    not isinstance(label_convention.get("resolution_sha256"), str) or
                    len(label_convention["resolution_sha256"]) != 64 or
                    any(char not in "0123456789abcdef" for char in
                        label_convention["resolution_sha256"]) or
                    getattr(bundle.site_basis_config, "radial_basis", None) !=
                    "PACE_ChebExpCos"):
                raise ValueError("Saved density public label convention is invalid.")
            if schema.endswith("_v1"):
                if getattr(bundle.site_basis_config, "chemical_basis", "delta") != "delta":
                    raise ValueError("Legacy density label convention cannot bind embedded chemistry.")
            elif label_convention["physical_source_sha256"] != (
                    _density_physical_source_sha256(
                        bundle.site_basis_config, bundle.settings.elems, cutoff, type_map)):
                raise ValueError("Saved density physical source differs from its recorded identity.")
            if schema.endswith("_v3"):
                radial_caps = label_convention["physical_radial_nmax_per_rank"]
                if (not isinstance(radial_caps, dict) or
                        set(radial_caps) != {str(rank) for rank in bundle.settings.ranks} or
                        any(type(value) is not int or value < 1 or
                            value > bundle.site_basis_config.nradmax
                            for value in radial_caps.values()) or
                        max(radial_caps.values()) != bundle.site_basis_config.nradmax or
                        any(int(channel.n) > radial_caps[str(spec.label.rank)]
                            for spec in bundle.descriptor_specs for channel in spec.channels)):
                    raise ValueError("Saved density physical radial caps are invalid.")
                if (not physical_eta_bound or
                        label_convention["physical_eta_binding_sha256"] !=
                        _physical_eta_binding_sha256(bundle.descriptor_specs, radial_caps)):
                    raise ValueError("Saved density physical eta binding differs from its recorded identity.")
            pace_public_index = True
        basis = object.__new__(cls)
        basis.source = "density"
        basis.elements = tuple(bundle.settings.elems)
        basis.cutoff = float(cutoff)
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._resolved = {
            "source": "density", "elements": basis.elements,
            "cutoff_A": basis.cutoff,
            "ranks": tuple(bundle.settings.ranks),
            "nmax": (tuple(label_convention["physical_radial_nmax_per_rank"][str(rank)]
                           for rank in bundle.settings.ranks)
                     if physical_eta_bound else tuple(bundle.settings.nmax)),
            **({"compiler_content_capacity_nmax": tuple(bundle.settings.nmax)}
               if physical_eta_bound else {}),
            "lmax": tuple(bundle.settings.lmax),
            "type_map": dict(type_map),
            "site_basis_config": bundle.site_basis_config,
        }
        basis._labels = _density_labels(
            bundle.descriptor_specs, basis.elements,
            pace_public_index=pace_public_index,
            chemical_kind=getattr(bundle.site_basis_config, "chemical_basis", "delta"),
            physical_eta_bound=physical_eta_bound)
        if pace_public_index and _ordered_public_labels_sha256(basis._labels) != (
                label_convention["ordered_public_labels_sha256"]):
            raise ValueError("Saved density public labels differ from their recorded order.")
        return basis

    @classmethod
    def _from_tagged_model(cls, model):
        basis = object.__new__(cls)
        basis.source = "tagged_cauchy_image"
        basis.elements = tuple(model.species_order)
        basis.cutoff = float(model.evaluator.cutoff)
        basis.backend = str(model.evaluator.backend)
        basis._descriptor = None
        request = model.evaluator.compiled.plan.report.request
        payload = model.evaluator.compiled.payload
        records = payload["image_coordinate_provenance"]
        if "tensor_order" in request:
            order = int(request["tensor_order"])
            ranks = (order,)
        else:
            ranks = tuple(sorted({int(record["tensor_order"]) for record in records}))
            order = ranks[0] if len(ranks) == 1 else None
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "compiler_request": request,
            "compiler_hash": model.evaluator.compiled.self_hash,
            "N": order, "ranks": ranks,
        }
        basis._labels = _tagged_labels(
            records, order,
            payload.get("raw_coordinate_labels", payload.get("raw_labels")),
            payload.get("raw_from_image") if "catalogue" in request else None,
        )
        return basis

    @classmethod
    def _from_legacy_composite(cls, record):
        basis = object.__new__(cls)
        basis.source = "legacy_composite"
        basis.elements = record["species"]
        basis.cutoff = record["cutoff_A"]
        basis.backend = "native_cpu"
        basis._descriptor = None
        basis._labels = None
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "feature_count": record["feature_count"],
            "artifact_sha256": record["artifact_sha256"],
            "label_status": "unavailable_in_legacy_composite",
        }
        return basis

    @classmethod
    def _from_portable_linear(cls, record):
        basis = object.__new__(cls)
        basis.source = "portable_linear"
        basis.elements = ("Ni",)
        basis.cutoff = float(record["sources"]["radial_fit"]["cutoff_A"])
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._portable = record
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "backend": basis.backend,
            "feature_count": len(record["labels"]),
            "archive_schema": (record["refit"]["schema"] if "refit" in record else
                               record["manifest"]["schema"]),
            "native_plan_status": (record["refit"]["native_plan_status"]
                                   if "refit" in record else "saved_base_plan"),
        }
        labels = []
        for row in record["labels"]:
            if row["branch"] == "ordinary":
                rank = len(row["label"]["n_tuple"])
                identity = row["feature_id"]
            else:
                ranks = {int(item["rank"]) for item in row["descriptor_labels"]}
                if len(ranks) != 1:
                    raise ValueError("Portable tagged feature mixes tensor ranks.")
                rank = ranks.pop()
                identity = row["compiler_request_hash"] + ":" + str(
                    row["tagged_program_column"])
            labels.append(FeatureLabel(row["fit_column"], basis.source, identity,
                                       {"N": rank, "L": 0, **row}))
        basis._labels = tuple(labels)
        return basis

    @classmethod
    def _from_bar_phi_model(cls, model, type_map):
        basis = object.__new__(cls)
        basis.source = "bar_phi"
        basis.elements = tuple(type_map)
        basis.cutoff = float(model.config.phi.cutoff)
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "backend": basis.backend,
            "type_map": dict(type_map), "phi_config": model.config.to_dict(),
        }
        basis._labels = _bar_phi_labels(model.config.phi)
        return basis

    @property
    def labels(self):
        if self.source == "configured":
            self._materialize_configured()
        if self.source == "legacy_composite":
            raise RuntimeError(
                "The legacy composite has no serialized compiler label order; "
                "ordered labels require the portable YE3T bundle."
            )
        return self._labels

    @property
    def catalogue(self):
        if not hasattr(self, "_catalogue"):
            raise AttributeError("Legacy Basis construction has no public config catalogue preview.")
        return self._catalogue

    @property
    def resolution(self):
        if not hasattr(self, "_resolution"):
            raise AttributeError("Legacy Basis construction has no public config resolution report.")
        return self._resolution

    @property
    def resolved(self):
        return deepcopy(self._resolved)

    def create(self, atoms):
        if self.source == "explicit_phi":
            raise ValueError("Ordered explicit Phi evaluates one selected cluster; call create_cluster.")
        if self.source == "configured":
            self._materialize_configured()
        if self.source == "density" and getattr(self, "_density_full_m", False):
            from ye3t.core.tesseral import complex_multiplet_to_real_tesseral
            from ye3t_methods.atomistic.equivariant_calc import neighbor_data_from_ase_atoms

            L = int(self._resolved["representation"]["parent"]["L"])
            neighbor = neighbor_data_from_ase_atoms(
                atoms, self.cutoff, self._density_full_m_type_map)
            edge_index = torch.as_tensor(neighbor.edge_index, dtype=torch.long)
            displacements = torch.as_tensor(neighbor.x_ij, dtype=torch.float64)
            atom_types = torch.as_tensor(neighbor.atom_types, dtype=torch.long)
            specs = tuple(spec for block in self._density_full_m_specs for spec in block)
            with torch.no_grad():
                complex_rows = self._density_full_m_evaluator(
                    x_ij=displacements, edge_index=edge_index,
                    atom_types=atom_types, descriptors=specs,
                    real_if_scalar=False)
                complex_rows = complex_rows.reshape(
                    len(atoms), 2 * L + 1, len(self._labels)).transpose(1, 2)
                real_rows = complex_multiplet_to_real_tesseral(
                    complex_rows, L, range(-L, L + 1))
            rows = real_rows.detach().cpu().numpy()
            if (rows.shape != (len(atoms), len(self._labels), 2 * L + 1) or
                    not np.isfinite(rows).all()):
                raise RuntimeError("Density full-M evaluation returned invalid multiplets.")
            return rows
        if self.source == "tagged_carriers":
            result = self._descriptor.create(atoms, descriptor_evaluation="physical_image")
            if result["physical_image_plan_hash"] != self._tagged_carrier_plan_hash:
                raise RuntimeError("Tagged physical-image plan changed after basis materialization.")
            rows = np.stack(tuple(
                result["physical_image"][tag_count]["values"][:, slice(*span)]
                for tag_count, span in self._tagged_carrier_selection), axis=1
            ) if self._tagged_carrier_selection else np.empty(
                (len(atoms), 0, 2 * self._resolved["representation"]["parent"]["L"] + 1))
            if (rows.shape != (len(atoms), len(self._labels),
                               2 * self._resolved["representation"]["parent"]["L"] + 1)
                    or not np.isrealobj(rows) or not np.isfinite(rows).all()):
                raise RuntimeError("Tagged Basis.create requires complete finite real multiplets.")
            return np.asarray(rows, dtype=np.float64)
        if self.source == "combined_scalar":
            return np.concatenate(tuple(item.create(atoms)
                                        for item in self._components.values()), axis=1)
        if self.source == "portable_linear":
            from .portable_archive import portable_feature_rows

            return portable_feature_rows(self._portable, atoms)
        if self.source == "legacy_composite":
            raise RuntimeError("The legacy composite has no standalone ordered descriptor rows.")
        if self.source == "tagged_cauchy_image" and hasattr(self, "_loaded_tagged_model"):
            evaluator = self._loaded_tagged_model.evaluator
            symbols = atoms.get_chemical_symbols()
            if set(symbols) - set(evaluator.type_map):
                raise ValueError("Structure contains a species outside the loaded tagged model.")
            with torch.no_grad():
                values = evaluator.materialize(
                    torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64),
                    torch.as_tensor([evaluator.type_map[name] for name in symbols],
                                    dtype=torch.long),
                    cell=np.asarray(atoms.cell.array), pbc=np.asarray(atoms.pbc),
                )[2].detach().cpu().numpy()
            if (values.shape != (len(atoms), len(self.labels)) or
                    not np.isrealobj(values) or not np.isfinite(values).all()):
                raise RuntimeError("Loaded tagged component produced invalid scalar rows.")
            return np.asarray(values, dtype=np.float64)
        if self._descriptor is None:
            raise RuntimeError("A loaded model retains label identity; construct a Basis to evaluate standalone descriptors.")
        if self.source == "bar_phi":
            model = YE3TModel.phi(self._descriptor, {"branches": ("bar_phi",)})
            type_map = {name: index for index, name in enumerate(self.elements)}
            values = _bar_phi_feature_rows(model, atoms, type_map, forces=False, stress=False)[3]
        else:
            values = self._descriptor.create(atoms)
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        rows = np.asarray(values)
        if (rows.ndim != 2 or rows.shape != (len(atoms), len(self.labels)) or
                not np.isrealobj(rows) or not np.isfinite(rows).all()):
            raise RuntimeError("Scalar Basis.create requires finite real per-atom descriptor rows.")
        return np.asarray(rows, dtype=np.float64)

    def create_cluster(self, atoms, center, ordered_occurrences):
        """Evaluate one ordered rank-eight star with explicit periodic images."""

        if self.source != "explicit_phi":
            raise ValueError("create_cluster requires an explicit_phi Basis.")
        from ye3t.couplings import compile as compile_coupling
        from ye3t.core.tesseral import complex_multiplet_to_real_tesseral
        from ye3t_methods.atomistic.equivariant_calc.labeling import SingleChannelLabel
        from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import SiteBasisConfig, SiteBasisV2

        if type(center) is not int or not 0 <= center < len(atoms):
            raise ValueError("center must be a valid atom index.")
        occurrences = tuple(ordered_occurrences)
        if len(occurrences) != 8:
            raise ValueError("The first explicit Phi star requires eight ordered occurrences.")
        parsed = []
        pbc = np.asarray(atoms.pbc, dtype=bool)
        for occurrence in occurrences:
            if (not isinstance(occurrence, (tuple, list)) or len(occurrence) != 2
                    or type(occurrence[0]) is not int
                    or not 0 <= occurrence[0] < len(atoms)
                    or not isinstance(occurrence[1], (tuple, list))
                    or len(occurrence[1]) != 3
                    or any(type(value) is not int for value in occurrence[1])):
                raise ValueError("Each occurrence must be (atom_index, integer_cell_shift[3]).")
            atom_index, shift = int(occurrence[0]), tuple(occurrence[1])
            if any(value and not pbc[axis] for axis, value in enumerate(shift)):
                raise ValueError("An occurrence cannot shift a nonperiodic cell axis.")
            parsed.append((atom_index, shift))
        if len(set(parsed)) != 8:
            raise ValueError("Ordered Phi occurrences must be distinct atom-image pairs.")
        if set(atoms.get_chemical_symbols()) - set(self.elements):
            raise ValueError("Structure contains a species outside this explicit Phi source.")
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        shifts = np.asarray([shift for _atom, shift in parsed], dtype=np.float64)
        neighbors = np.asarray([atom for atom, _shift in parsed], dtype=np.int64)
        displacement = positions[neighbors] + shifts @ cell - positions[center]
        distance = np.linalg.norm(displacement, axis=1)
        if (not np.isfinite(displacement).all() or np.any(distance == 0.0)
                or np.any(distance >= self.cutoff)):
            raise ValueError("Ordered Phi occurrences must have finite nonzero distance below cutoff.")
        component = self._resolved["components"][0]
        radial = component["single_factors"]["radial"]
        if self._phi_compiled is None:
            cache_mode = self._resolved["runtime"].get("cache", {}).get("mode", "auto")
            self._phi_compiled = compile_coupling(
                self._phi_plan,
                subduction_materialization_backend="numeric_cached",
                subduction_cache_dir=False if cache_mode == "off" else None,
            )
            table = self._phi_compiled.coupler.factorized_coefficient_tables[0]
            if (not self._phi_compiled.certificate.passed
                    or table["selected_full_alpha"] != component["selected_typed_alpha"]
                    or table["shape"][1] != 70):
                raise ArithmeticError("Ordered Phi selected-route compiler axes changed.")
        if self._phi_site_basis is None:
            self._phi_site_basis = SiteBasisV2(SiteBasisConfig(
                rc=[self.cutoff], lmbda=[radial["lambda"]],
                nradmax=2, lmax=1, possible_types=(0,),
                radial_basis="PACE_ChebExpCos", chemical_basis="delta",
                charge_mode="none", atomic_base_normalization="none",
                factor_normalization="none", spherical_backend="complex",
                spherical_normalization="pace_y00_one", source_backend="torch",
                dtype=torch.float64, complex_dtype=torch.complex128,
                pace_cutoff_width=[radial["cutoff_width_A"]],
                pace_spline_spacing=[0.001], pace_inner_cutoff=[0.0],
                pace_inner_cutoff_width=[0.0], pace_crad_policy="identity",
            ))
        channels = tuple(
            SingleChannelLabel(mu0=0, mu=0, kappa0=0, kappa=0,
                               n=n, l=1, m=m)
            for n in (1, 2) for m in (-1, 0, 1)
        )
        edge_index = torch.tensor(
            [[center] * 8, neighbors.tolist()], dtype=torch.long
        )
        _labels, edge_values, _edge_dx = self._phi_site_basis.compute_channel_edges_with_dx(
            torch.as_tensor(displacement, dtype=torch.float64), edge_index,
            torch.zeros(len(atoms), dtype=torch.long), channels,
        )
        if tuple(_labels) != channels or tuple(edge_values.shape) != (8, 6):
            raise ArithmeticError("Ordered Phi source channel ordering changed.")
        slots = torch.stack(tuple(
            edge_values[index, (0 if index < 4 else 3):(3 if index < 4 else 6)]
            for index in range(8)
        ))
        coupled = self._phi_compiled.coupler.evaluate_selected_typed_slots_torch(slots)
        real = complex_multiplet_to_real_tesseral(coupled, 2, range(-2, 3))
        if tuple(real.shape) != (1, 14, 5) or not torch.isfinite(real).all():
            raise ArithmeticError("Ordered Phi physical evaluator returned invalid axes or values.")
        return real.detach().cpu().numpy()

    def create_many(self, structures):
        """Evaluate one ordered per-atom descriptor array per structure."""
        return tuple(self.create(atoms) for atoms in structures)

    def describe(self, index, format="text"):
        label = self.labels[int(index)]
        if format == "latex":
            return label.latex()
        if format not in {"text", "ascii"}:
            raise ValueError("format must be 'text', 'ascii', or 'latex'.")
        fields = label.as_dict()
        if self.source == "bar_phi":
            plan = fields.pop("compiler_coupling_plan")
            fields["compiler_coupling_summary"] = {
                "api": plan["multiplicity_report"]["provenance"]["api"],
                "convention_hash": plan["convention_hash"],
                "counts_by_target": plan["multiplicity_report"]["counts_by_target"],
            }
        details = json.dumps(fields, sort_keys=True, ensure_ascii=True, default=str)
        if len(details) > 1200:
            details = details[:1200] + f"... ({len(details) - 1200} characters omitted)"
        return str(label) + "\n" + details

    def __str__(self):
        if self.source == "explicit_phi":
            return (f"Basis(source=explicit_phi, elements={self.elements}, cutoff_A={self.cutoff:g}, "
                    "output_axes=(selected_path, 14 tableaux, 5 magnetic components))")
        if self.source == "combined_scalar":
            return (f"Basis(source=combined_scalar, elements={self.elements}, "
                    f"cutoff_A={self.cutoff:g}, components={tuple(self._components)}, "
                    f"features={len(self.labels)})")
        if hasattr(self, "_resolution"):
            return (f"Basis(source={self.source}, elements={self.elements}, cutoff_A={self.cutoff:g}, "
                    f"resolution_sha256={self._resolution.sha256}, "
                    f"create_available={self._resolution.capability_report['basis_create_available']})")
        if self.source == "legacy_composite":
            return (f"Basis(source=legacy_composite, elements={self.elements}, "
                    f"cutoff_A={self.cutoff:g}, features={self._resolved['feature_count']}, "
                    "labels=unavailable)")
        if self.source == "portable_linear":
            return (f"Basis(source=portable_linear, elements={self.elements}, "
                    f"cutoff_A={self.cutoff:g}, features={len(self.labels)}, "
                    "labels=ordered)")
        tagged_rank = self._resolved.get("N")
        tagged_scope = (f"N={tagged_rank}" if tagged_rank is not None
                        else f"ranks={self._resolved.get('ranks', 'unknown')}")
        truncation = (
            f"ranks={self._resolved['ranks']}, nmax={self._resolved['nmax']}, "
            f"lmax={self._resolved['lmax']}"
            if self.source == "density" else
            f"motifs={len(self.labels)}, family={self._resolved['phi_config']['phi']['motif_family']}"
            if self.source == "bar_phi" else
            f"{tagged_scope}, "
            f"tag_counts={self._resolved.get('tag_counts', 'saved')}"
        )
        head = (
            f"Basis(source={self.source}, elements={self.elements}, "
            f"cutoff_A={self.cutoff:g}, {truncation}, features={len(self.labels)})"
        )
        rows = [str(label) for label in self.labels[:5]]
        if len(self.labels) > 5:
            rows.append(f"... {len(self.labels) - 5} rows omitted")
        return "\n".join((head, *rows))

    __repr__ = __str__


def _read_density_full_m_artifact(payload):
    """Restore saved compiler coordinates without choosing a new coupling gauge."""
    from ye3t import YE3TRepresentation as CoreRepresentation
    from ye3t.core.labels import CompactLabel, DescriptorSpec
    from ye3t_methods.atomistic.equivariant_calc.ace_eval_v2 import ACECovariantEvaluator
    from ye3t_methods.atomistic.equivariant_calc.labeling import SingleChannelLabel
    from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import (
        deserialize_site_basis_config,
    )
    from .tesseral_targets import cartesian_tesseral_convention_hash

    if payload.get("schema") not in {"ye3t_methods_density_full_m_per_atom_v1",
                                     "ye3t_methods_density_full_m_per_atom_v2"}:
        raise ValueError("Unknown density full-M artifact schema.")
    allowed = {"schema", "construction", "resolution_sha256", "compiled_density",
               "compiler_hash", "selected_coordinate_ids", "coordinate_convention",
               "real_form_sha256", "validation", "fit", "self_hash"}
    if payload["schema"] == "ye3t_methods_density_full_m_per_atom_v2":
        allowed.add("native_property_plan")
    if (set(payload) != allowed or not isinstance(payload["self_hash"], str) or
            hashlib.sha256(_full_m_canonical_bytes({
                key: value for key, value in payload.items() if key != "self_hash"
            })).hexdigest() != payload["self_hash"]):
        raise ValueError("Density full-M artifact schema or self-hash mismatch.")
    construction = payload["construction"]
    if not isinstance(construction, dict) or set(construction) != {
            "basis", "representation", "runtime"}:
        raise ValueError("Density full-M construction record is invalid.")
    representation = CoreRepresentation.from_config(construction["representation"])
    basis = Basis.from_config(construction["basis"], representation=representation,
                              runtime=construction["runtime"])
    parent = basis._resolved["representation"]["parent"]
    L = int(parent["L"])
    if (basis._resolution.sha256 != payload["resolution_sha256"] or
            len(basis._resolved["components"]) != 1 or
            basis._resolved["components"][0]["kind"] != "density" or
            L <= 0 or
            not basis._resolution.capability_report["basis_create_available"]):
        raise ValueError("Density full-M construction or resolution changed.")
    convention = {"group": "O3", "basis": "real_tesseral_tensor_components",
                  "axis_order": "cos_L_to_cos_1_zero_sin_1_to_sin_L",
                  "M_values": list(range(-L, L + 1)), "L": L,
                  "parity": parent["parity"]}
    if (payload["coordinate_convention"] != convention or
            payload["real_form_sha256"] != _density_full_m_convention_hash(
                L, parent["parity"]) or
            payload["validation"] != _density_full_m_validation(
                L, parent["parity"])):
        raise ValueError("Density full-M real-form convention or validation changed.")
    component = basis._resolved["components"][0]
    expected_labels = basis._density_multiplet_compact_labels(component)
    compiled = payload["compiled_density"]
    if (not isinstance(compiled, dict) or set(compiled) != {
            "schema", "site_basis", "compact_labels", "specs_by_M",
            "coefficient_sha256"} or
            compiled["schema"] != "ye3t_density_full_m_compiler_v1" or
            hashlib.sha256(_full_m_canonical_bytes(compiled)).hexdigest() !=
            payload["compiler_hash"] or
            [label.to_dict() for label in expected_labels] !=
            compiled["compact_labels"]):
        raise ValueError("Density full-M compiler labels or hash changed.")
    site = deserialize_site_basis_config(compiled["site_basis"])
    expected_site = deserialize_site_basis_config(
        basis._density_full_m_site_config(component))
    if serialize_site_basis_config(site) != serialize_site_basis_config(expected_site):
        raise ValueError("Density full-M site basis differs from construction.")
    raw_blocks = compiled["specs_by_M"]
    if not isinstance(raw_blocks, list) or len(raw_blocks) != 2 * L + 1:
        raise ValueError("Density full-M artifact lacks complete magnetic blocks.")
    ordered = []
    channel_fields = set(SingleChannelLabel.__record_fields__)
    for M, raw_rows in zip(range(-L, L + 1), raw_blocks):
        if not isinstance(raw_rows, list) or len(raw_rows) != (
                basis.catalogue.counts()["by_component"][component["name"]]["all_centers"]):
            raise ValueError("Density full-M magnetic block width differs from count.")
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, dict) or set(raw) != {
                    "key", "label", "channels", "ms_combinations", "coeffs",
                    "L_R", "M_R"}:
                raise ValueError("Density full-M descriptor specification is invalid.")
            label = CompactLabel.from_dict(raw["label"])
            if label not in expected_labels or raw["L_R"] != L or raw["M_R"] != M:
                raise ValueError("Density full-M descriptor label or angular target changed.")
            if not isinstance(raw["channels"], list) or len(raw["channels"]) != label.rank:
                raise ValueError("Density full-M source channels are incomplete.")
            if any(not isinstance(channel, dict) or set(channel) != channel_fields
                   for channel in raw["channels"]):
                raise ValueError("Density full-M source-channel schema changed.")
            if any(channel["kappa0"] != 0 or channel["kappa"] != 0 or
                   channel["m"] != 0 or channel["l_aux"] is not None or
                   channel["m_aux"] is not None or channel["eta"] is not None
                   for channel in raw["channels"]):
                raise ValueError("Density full-M source requires canonical no-charge channels.")
            channels = tuple(SingleChannelLabel(**channel)
                             for channel in raw["channels"])
            if (len(set(channel.mu0 for channel in channels)) != 1 or
                    channels[0].mu0 not in range(len(basis.elements)) or
                    any(channel.l != l or channel.n <= 0 or channel.mu < 0
                        for channel, l in zip(channels, label.l_tuple))):
                raise ValueError("Density full-M source channels differ from label or species.")
            if (not isinstance(raw["ms_combinations"], list) or
                    not isinstance(raw["coeffs"], list) or
                    not raw["coeffs"] or
                    len(raw["ms_combinations"]) != len(raw["coeffs"])):
                raise ValueError("Density full-M coefficient term rows are invalid.")
            ms_rows = tuple(tuple(row) for row in raw["ms_combinations"])
            if any(len(ms) != label.rank or sum(ms) != M or
                   any(not isinstance(m, int) or isinstance(m, bool) or
                       abs(m) > l for m, l in zip(ms, label.l_tuple))
                   for ms in ms_rows):
                raise ValueError("Density full-M magnetic tuples are invalid.")
            if any(not isinstance(value, list) or len(value) != 2 or
                   any(not isinstance(part, (int, float)) or
                       not np.isfinite(part) for part in value)
                   for value in raw["coeffs"]):
                raise ValueError("Density full-M coefficient pairs are invalid.")
            coefficients = tuple(complex(*value) for value in raw["coeffs"])
            bound = (component["single_factors"]["chemical"]["kind"] ==
                     "fixed_embedding" or len(basis.elements) > 1)
            key_suffix = f"|M={M}" + ("|physical_eta_bound" if bound else "")
            if (not isinstance(raw["key"], str) or
                    not raw["key"].startswith(label.full_key() + "|variant=") or
                    not raw["key"].endswith(key_suffix)):
                raise ValueError("Density full-M descriptor key changed.")
            rows.append(DescriptorSpec(raw["key"], label, channels, ms_rows,
                                       coefficients, L, M))
        ordered.append(tuple(rows))
    ordered = tuple(ordered)
    reference = tuple((spec.label, spec.channels,
                       spec.key.replace("|M=0", "|M=*"))
                      for spec in ordered[L])
    if any(tuple((spec.label, spec.channels,
                  spec.key.replace(f"|M={M}", "|M=*")) for spec in rows) != reference
           for M, rows in zip(range(-L, L + 1), ordered)):
        raise ValueError("Density full-M magnetic feature axes differ.")
    counts = basis.catalogue.counts()["by_component"][component["name"]]
    by_rank = {rank: sum(spec.rank == rank for spec in ordered[L])
               for rank in component["ranks"]}
    if (by_rank != {rank: count * len(basis.elements)
                   for rank, count in counts["by_rank_per_center"].items()} or
            _density_full_m_coefficient_hash(ordered) !=
            compiled["coefficient_sha256"]):
        raise ValueError("Density full-M rank counts or coefficient bytes changed.")
    _validate_density_full_m_intertwiner(ordered, L)
    if (component["single_factors"]["chemical"]["kind"] == "fixed_embedding" or
            len(basis.elements) > 1):
        for spec in ordered[L]:
            center = basis.elements[spec.channels[0].mu0]
            channels_by_content = {
                int(row["compiler_content_id"]): row
                for row in component["physical_eta_by_center"][center]}
            for content_id, channel in zip(spec.label.n_tuple, spec.channels):
                row = channels_by_content[content_id]
                if (channel.mu != int(row["chemical"]["chemical_index"]) or
                        channel.n != int(row["native_pace_n"])):
                    raise ValueError("Density full-M physical content mapping changed.")
    labels = _density_labels(ordered[L], basis.elements, pace_public_index=True,
                             chemical_kind=component["single_factors"]["chemical"]["kind"],
                             physical_eta_bound=(
                                 component["single_factors"]["chemical"]["kind"] ==
                                 "fixed_embedding" or len(basis.elements) > 1),
                             multiplet=True)
    if [label.identity for label in labels] != payload["selected_coordinate_ids"]:
        raise ValueError("Density full-M ordered public coordinates changed.")
    basis.source = "density"
    basis.backend = "pytorch"
    basis._labels = labels
    basis._density_full_m_specs = ordered
    basis._density_full_m_evaluator = ACECovariantEvaluator(
        site, backend="pytorch", strict_backend=True, validate_backend=True,
        factorized_descriptor_runtime_policy="disable")
    basis._density_full_m_type_map = {
        name: index for index, name in enumerate(basis.elements)}
    basis._density_full_m = True
    basis._density_full_m_compiled_record = compiled
    basis._density_full_m_plan_hash = payload["compiler_hash"]
    basis._density_full_m_convention_hash = payload["real_form_sha256"]
    basis._resolved["output_layout"] = "atoms_multiplets_real_tesseral_M"
    fitted = payload["fit"]
    if not isinstance(fitted, dict) or set(fitted) != {"kind", "beta", "fit_metadata"}:
        raise ValueError("Density full-M fit record is invalid.")
    beta = np.asarray(fitted["beta"], dtype=np.float64)
    metadata = fitted["fit_metadata"]
    expected_columns = [{"coordinate_id": label.identity} for label in labels]
    metric = metadata.get("coordinate_penalty_metric") if isinstance(metadata, dict) else None
    if (fitted["kind"] != "density_full_m_per_atom" or
            beta.shape != (len(labels),) or not np.isfinite(beta).all() or
            not isinstance(metadata, dict) or
            metadata.get("design_column_order") !=
            "compiled_center_and_channel_multiplet_shared_over_M" or
            metadata.get("fit_method") not in {"ridge", "lasso", "ardregression"} or
            not isinstance(metric, dict) or
            metric.get("schema") != "selected_compiler_coordinate_euclidean_v1" or
            metric.get("columns") != expected_columns or
            metric.get("diagonal") != [1.0] * len(labels) or
            metric.get("physical_image_plan_hash") != payload["compiler_hash"] or
            metric.get("interpretation") !=
            "coefficient_norm_in_saved_selected_coordinates" or
            metadata.get("n_cols") != len(labels) or
            not isinstance(metadata.get("n_rows"), int) or metadata["n_rows"] < 1 or
            metadata.get("target_input") not in {"real_tesseral", "cartesian"} or
            not isinstance(metadata.get("target_units"), str) or
            not metadata["target_units"]):
        raise ValueError("Density full-M fitted columns or coefficients changed.")
    fit_request = metadata.get("resolved_fit_config")
    if (not isinstance(fit_request, dict) or
            hashlib.sha256(_full_m_canonical_bytes(fit_request)).hexdigest() !=
            metadata.get("resolved_fit_config_sha256") or
            fit_request.get("construction_resolution_sha256") != basis._resolution.sha256 or
            fit_request.get("targets", {}).get("per_atom", {}).get("input") !=
            metadata["target_input"]):
        raise ValueError("Density full-M saved fit request changed.")
    posterior = metadata.get("predictive_uncertainty")
    if metadata["fit_method"] == "ardregression":
        if (not isinstance(posterior, dict) or
                posterior.get("schema") != "ye3t_linear_ard_posterior_v1" or
                posterior.get("design_column_order") !=
                "compiled_center_and_channel_multiplet_shared_over_M"):
            raise ValueError("Density full-M ARD posterior contract changed.")
        active = np.asarray(posterior.get("active_column_indices"), dtype=int)
        covariance = np.asarray(posterior.get("coefficient_covariance_active"),
                                dtype=np.float64)
        if (active.ndim != 1 or len(set(active.tolist())) != len(active) or
                np.any(active < 0) or np.any(active >= len(labels)) or
                covariance.shape != (len(active), len(active)) or
                not np.isfinite(covariance).all() or
                not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10) or
                covariance.size and
                np.min(np.linalg.eigvalsh(covariance)) < -1e-10):
            raise ValueError("Density full-M ARD covariance is invalid.")
    elif posterior is not None:
        raise ValueError("Density full-M non-ARD fit has an unexpected posterior.")
    if (metadata["target_input"] == "cartesian" and
            metadata.get("cartesian_tesseral_convention_sha256") !=
            cartesian_tesseral_convention_hash(L, parent["parity"])):
        raise ValueError("Density full-M Cartesian convention changed.")
    fitted["beta"] = beta
    if (payload["schema"] == "ye3t_methods_density_full_m_per_atom_v2" and
            payload["native_property_plan"] !=
            _density_full_m_property_plan(basis, fitted, convention)):
        raise ValueError("Density native property plan differs from compiler and fit.")
    model = LinearModel(basis)
    model._fitted = fitted
    return model


def _density_full_m_property_plan(basis, fitted, convention):
    """Bind the saved complex compiler rows to one native source and readout."""
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    compiled = basis._density_full_m_compiled_record
    blocks = compiled["specs_by_M"]
    identifiers = [label.identity for label in basis.labels]
    if (len(blocks) != 2 * convention["L"] + 1 or
            any(len(rows) != len(identifiers) for rows in blocks)):
        raise ValueError("Density native plan feature axes differ from compiler blocks.")
    features = []
    for index, coordinate_id in enumerate(identifiers):
        rows = [block[index] for block in blocks]
        centers = {row["channels"][0]["mu0"] for row in rows}
        if len(centers) != 1:
            raise ValueError("Density native plan center species changes across M.")
        features.append({
            "coordinate_id": coordinate_id,
            "compiler_feature_index": index,
            "center_type": int(next(iter(centers))),
            "spec_keys_by_M": [row["key"] for row in rows],
        })
    matrix = real_tesseral_to_complex_multiplet(
        torch.eye(2 * convention["L"] + 1, dtype=torch.float64),
        convention["L"]).numpy()
    component = basis._resolved["components"][0]
    plan = {
        "schema": "ye3t_density_full_m_native_property_plan_v1",
        "compiler_hash": basis._density_full_m_plan_hash,
        "coefficient_sha256": compiled["coefficient_sha256"],
        "real_form_sha256": basis._density_full_m_convention_hash,
        "source": {
            "site_basis": compiled["site_basis"],
            "species_order": list(basis.elements),
            "physical_eta_by_center": component["physical_eta_by_center"],
            "pair_cutoffs_A": {left + "-" + right: basis.cutoff
                               for left in basis.elements for right in basis.elements},
        },
        "target": {
            **convention,
            "real_to_complex_matrix": [
                [[float(value.real), float(value.imag)] for value in row]
                for row in matrix],
        },
        "selected_coordinate_ids": identifiers,
        "features": features,
        "readout": {
            "binding": "compiled_feature_shared_over_M",
            "coefficients": np.asarray(fitted["beta"], dtype=np.float64).tolist(),
            "fit_sha256": hashlib.sha256(_full_m_canonical_bytes(
                _full_m_json_ready(fitted))).hexdigest(),
        },
        "provenance": "ye3t.couplings.compile",
    }
    plan = _full_m_json_ready(plan)
    plan["self_hash"] = hashlib.sha256(_full_m_canonical_bytes(plan)).hexdigest()
    return plan


def _tagged_full_m_property_plan(basis, fitted, convention):
    """Bind a fitted readout to compiler-owned linear-support schedules."""
    from ye3t.couplings import (shifted_jacobi_normalization_squared,
                                shifted_jacobi_power_coefficients,
                                tagged_cauchy_carrier_schedule)

    compiled = basis._descriptor.metadata["tagged_cauchy_carriers_compiled"]
    grouped = {}
    for source in compiled["sources"]:
        grouped.setdefault(int(source["request"]["tag_count"]), []).append(source)
    selected = {label.identity for label in basis.labels}
    schedules = []
    feature_rows = []
    one = {"real": [{"radicand_numerator": 1, "radicand_denominator": 1,
                     "coefficient_numerator": 1, "coefficient_denominator": 1}],
           "imag": [], "binary64": [1.0, 0.0]}
    for tag_count in sorted(grouped):
        if not any(record["label"]["coordinate_id"] in selected
                   for source in grouped[tag_count] for record in source["descriptors"]):
            continue
        schedule = tagged_cauchy_carrier_schedule(
            grouped[tag_count], coordinate_ids=selected,
            support_realization="edge_marginal" if tag_count == 2 else "ordered_tags")
        schedule = _full_m_json_ready(schedule)
        if tag_count == 2:
            certificate = schedule["marginal_image_certificate"]
            candidates = certificate["available_coordinate_ids"]
            retained = certificate["selected_coordinate_ids"]
            if (not certificate["exact_all_M_reconstruction"] or
                    candidates != retained or
                    len(candidates) != len(certificate["reconstruction"]) or
                    any(row != [{"coordinate_id": coordinate, "coefficient": one}]
                        for coordinate, row in zip(candidates,
                                                   certificate["reconstruction"]))):
                raise ValueError("Tagged native property plan changes the fitted feature basis.")
        schedules.append(schedule)
        for record in schedule["inventory"]:
            feature_rows.append({
                "coordinate_id": record["label"]["coordinate_id"],
                "tag_count": tag_count, "schedule_index": len(schedules) - 1,
                "component_slice": list(record["component_slice"]),
            })
    ids = [row["coordinate_id"] for row in feature_rows]
    if (ids != [label.identity for label in basis.labels] or
            any(stop - start != 2 * convention["L"] + 1
                for start, stop in (row["component_slice"] for row in feature_rows))):
        raise ValueError("Tagged native property plan differs from fitted feature order.")
    component = basis._resolved["components"][0]
    channel_rows = []
    seen_channels = set()
    for schedule in schedules:
        for channel in schedule["channels"]:
            identity = (channel["neighbor_species"], int(channel["radial_channel"]),
                        int(channel["l"]), channel["source_family_id"])
            if identity in seen_channels:
                continue
            seen_channels.add(identity)
            q, l = identity[1:3]
            norm_squared = shifted_jacobi_normalization_squared(q, l)
            channel_rows.append({
                "channel": channel,
                "jacobi_alpha": 4, "jacobi_beta": 2 * l + 2,
                "shifted_jacobi_power_coefficients": list(
                    shifted_jacobi_power_coefficients(q, 4, 2 * l + 2)),
                "normalization_squared": {
                    "numerator": norm_squared.numerator,
                    "denominator": norm_squared.denominator},
                "binary64_normalization": math.sqrt(float(norm_squared)),
                "angular_convention": "compiler_ordered_regular_solid_v1",
                "real_axis_order": "cos_l_to_cos_1_zero_sin_1_to_sin_l",
            })
    pair_cutoffs = basis._descriptor.metadata[
        "tagged_cauchy_carriers_config"]["pair_cutoffs_A"]
    if pair_cutoffs is None:
        pair_cutoffs = {left + "-" + right: basis.cutoff
                        for left in basis.elements for right in basis.elements}
    plan = {
        "schema": "ye3t_tagged_full_m_native_property_plan_v1",
        "compiler_hash": compiled["self_hash"],
        "physical_image_plan_hash": basis._tagged_carrier_plan_hash,
        "source": {"species_order": list(basis.elements),
                   "cutoff_A": basis.cutoff,
                   "pair_cutoffs_A": pair_cutoffs,
                   "radial": component["single_factors"]["radial"],
                   "chemical": component["single_factors"]["chemical"],
                   "channels": channel_rows},
        "target": convention,
        "selected_coordinate_ids": ids,
        "feature_slices": feature_rows,
        "schedules": schedules,
        "readout": {"binding": "central_species_then_selected_multiplet_shared_over_M",
                    "coefficients_by_species": np.asarray(fitted["beta"], dtype=np.float64).reshape(
                        len(basis.elements), len(ids)).tolist(),
                    "fit_sha256": hashlib.sha256(_full_m_canonical_bytes(
                        _full_m_json_ready(fitted))).hexdigest()},
        "provenance": "ye3t.couplings.tagged_cauchy_carrier_schedule",
    }
    plan = _full_m_json_ready(plan)
    plan["self_hash"] = hashlib.sha256(_full_m_canonical_bytes(plan)).hexdigest()
    return plan


def _prepared_training_structures(structures, energy_key, force_key, stress_key,
                                  force_weight, stress_weight):
    prepared = []
    for index, atoms in enumerate(structures):
        if not hasattr(atoms, "get_chemical_symbols"):
            raise TypeError("fit expects a sequence of ASE Atoms objects.")
        results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
        energy = getattr(atoms, "info", {}).get(energy_key, results.get(energy_key))
        if energy is None:
            raise ValueError(f"Structure {index} lacks precomputed {energy_key!r} energy.")
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError(f"Structure {index} has a nonfinite {energy_key!r} energy.")
        force = getattr(atoms, "arrays", {}).get(force_key, results.get(force_key))
        if force_weight and force is None:
            raise ValueError(f"Structure {index} lacks precomputed {force_key!r} forces.")
        stress = getattr(atoms, "info", {}).get(stress_key, results.get(stress_key))
        if stress_weight and stress is None:
            raise ValueError(f"Structure {index} lacks precomputed {stress_key!r} stress.")
        clean = atoms.copy()
        clean.calc = None
        clean.info[energy_key] = energy
        if force is not None:
            values = np.asarray(force, dtype=float)
            if values.shape != (len(clean), 3) or not np.isfinite(values).all():
                raise ValueError(f"Structure {index} has invalid force shape or values.")
            clean.arrays[force_key] = values.copy()
        if stress is not None:
            values = np.asarray(stress, dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"Structure {index} has invalid ASE Voigt stress.")
            clean.info[stress_key] = values.copy()
        prepared.append(clean)
    if not prepared:
        raise ValueError("fit requires at least one structure.")
    return prepared


def _fit_sklearn_design(blocks, method, params, column_order):
    """Fit fixed compiler columns and retain the optional ARD covariance."""
    X = np.concatenate([block[0] for block in blocks], axis=0)
    y = np.concatenate([block[1] for block in blocks], axis=0)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("The scikit-learn design contains nonfinite values.")
    estimator = _make_sklearn_estimator(method, sklearn_params=params)
    estimator.fit(X, y)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    if coefficients.shape != (X.shape[1],) or not np.isfinite(coefficients).all():
        raise FloatingPointError("The scikit-learn fit returned invalid coefficients.")
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError("sklearn_params contains a value that cannot be saved portably.")

    metadata = {"fit_method": method, "sklearn_params": plain(dict(params or {})),
                "design_column_order": column_order, "n_rows": int(X.shape[0]),
                "n_cols": int(X.shape[1])}
    if method == "ardregression":
        precision = np.asarray(estimator.lambda_, dtype=np.float64)
        threshold = float(estimator.threshold_lambda)
        active = np.flatnonzero(precision < threshold)
        covariance = np.asarray(estimator.sigma_, dtype=np.float64)
        if covariance.shape != (active.size, active.size):
            raise RuntimeError("ARD posterior covariance does not match active columns.")
        metadata["predictive_uncertainty"] = {
            "schema": "ye3t_linear_ard_posterior_v1",
            "status": "python_offline_only",
            "design_column_order": column_order,
            "active_column_indices": active.tolist(),
            "coefficient_precision": precision.tolist(),
            "coefficient_covariance_active": covariance.tolist(),
            "noise_precision": float(estimator.alpha_),
            "threshold_lambda": threshold,
            "variance_formula": "x_active @ sigma @ x_active.T (epistemic readout only)",
        }
    return coefficients, metadata


def _combined_design_columns(basis, fit_offsets):
    columns = []
    for name, item in basis._components.items():
        for element in (basis.elements if item.source == "tagged_cauchy_image" else (None,)):
            for label in item.labels:
                columns.append({"component": name, "central_species": element,
                                "feature_index": label.feature_index,
                                "label_identity": label.identity})
    if fit_offsets:
        columns.extend({"component": "reference_energy",
                        "central_species": name, "feature_index": None,
                        "label_identity": "E0:" + name}
                       for name in basis.elements)
    return columns


def _combined_scalar_geometry_row(basis, atoms, forces, stress):
    """Assemble compiler-ordered component columns and one species-offset block."""
    if stress and (np.linalg.matrix_rank(np.asarray(atoms.cell.array, float)) < 3 or
                   not np.isfinite(atoms.get_volume()) or atoms.get_volume() <= 0.0):
        raise ValueError("Combined stress fitting requires a positive full-rank cell.")
    energy_parts = []
    force_parts = []
    stress_parts = []
    species = basis.elements
    symbols = tuple(atoms.get_chemical_symbols())
    if set(symbols) - set(species):
        raise ValueError("Structure contains a species outside the combined basis.")
    for item in basis._components.values():
        if item.source == "density":
            descriptor = item._descriptor
            row = _linear_ace_geometry_row(
                atoms, evaluator=descriptor.ace_descriptor.calculator.evaluator,
                descriptors=descriptor.descriptor_specs, cutoff=item.cutoff,
                type_map=descriptor.type_map, device="cpu", forces=forces,
                stress=stress, chunk_size=None,
            )
            energy_parts.append(row["energy"].detach().cpu().numpy())
            if forces:
                force_parts.append(row["forces"].detach().cpu().numpy())
            if stress:
                stress_parts.append(row["stress"].detach().cpu().numpy())
        else:
            row = tagged_cauchy_image_geometry_row(item._descriptor, atoms)
            energy_parts.append(row["feature_sums"].reshape(-1))
            if forces:
                force_parts.append(row["force_design"])
            if stress:
                stress_parts.append(row["stress_design"])
    offsets = np.asarray([symbols.count(name) for name in species], dtype=np.float64)
    energy = np.concatenate((*energy_parts, offsets))
    force = None if not forces else np.column_stack((
        *force_parts, np.zeros((3 * len(atoms), len(species)), dtype=np.float64)))
    strain = None if not stress else np.column_stack((
        *stress_parts, np.zeros((6, len(species)), dtype=np.float64)))
    if not np.isfinite(energy).all() or (forces and not np.isfinite(force).all()) or (
            stress and not np.isfinite(strain).all()):
        raise FloatingPointError("Combined geometry row contains nonfinite values.")
    return energy, force, strain


def _tagged_carrier_species_design(basis, atoms, rows):
    """Keep central-species columns distinct while sharing weights over M."""
    symbols = atoms.get_chemical_symbols()
    if set(symbols) - set(basis.elements):
        raise ValueError("Full-M structure contains a species outside the basis.")
    indicators = np.asarray([[name == species for species in basis.elements]
                             for name in symbols], dtype=np.float64)
    return (indicators[:, :, None, None] * rows[:, None, :, :]).reshape(
        len(atoms), len(basis.elements) * rows.shape[1], rows.shape[2])


def _bind_density_public_labels(basis, fitted):
    actual = tuple(str(spec.key) for spec in fitted.descriptor_specs)
    expected = tuple(label.identity for label in basis.labels)
    if actual != expected:
        raise RuntimeError("Fitted descriptor columns differ from the Basis label order.")
    if hasattr(basis, "_resolution"):
        physical_eta_bound = any(str(spec.key).endswith("|physical_eta_bound")
                                 for spec in fitted.descriptor_specs)
        fitted.fit_metadata["ye3t_methods_public_label_convention"] = {
            "schema": ("ye3t_methods_density_pace_labels_v3" if physical_eta_bound
                       else "ye3t_methods_density_pace_labels_v2"),
            "radial_family": "pace_chebexp_cos",
            "radial_index_base": 0,
            "native_radial_index_base": 1,
            "resolution_sha256": basis._resolution.sha256,
            "physical_source_sha256": _density_physical_source_sha256(
                fitted.site_basis_config, basis.elements,
                basis.cutoff, basis._descriptor.type_map),
            "ordered_public_labels_sha256": _ordered_public_labels_sha256(basis.labels),
            **({"physical_eta_binding_sha256":
                _physical_eta_binding_sha256(
                    fitted.descriptor_specs,
                    basis._resolution.to_dict()["components"][0]["nmax_per_rank"]),
                "physical_radial_nmax_per_rank":
                basis._resolution.to_dict()["components"][0]["nmax_per_rank"]}
               if physical_eta_bound else {}),
        }


class _TaggedCarrierPropertyCalculator(Calculator):
    """ASE property adapter for a fitted selected full-M tagged readout."""

    implemented_properties = ["per_atom_real_tesseral_mean"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.implemented_properties = ["per_atom_real_tesseral_mean"]
        if "predictive_uncertainty" in model._fitted["fit_metadata"]:
            self.implemented_properties.append("per_atom_real_tesseral_covariance")

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, all_changes if system_changes is None
                          else system_changes)
        prediction = self.model.predict(
            atoms, uncertainty="predictive_uncertainty" in
            self.model._fitted["fit_metadata"])
        self.results["per_atom_real_tesseral_mean"] = prediction["mean_real_tesseral"]
        if "covariance_real_tesseral" in prediction:
            self.results["per_atom_real_tesseral_covariance"] = prediction[
                "covariance_real_tesseral"]


class LinearModel:
    """Fit, persist, inspect, and evaluate a fixed YE3T descriptor readout."""

    def __init__(self, basis, *, reference_energies=None):
        if not isinstance(basis, Basis):
            raise TypeError("LinearModel requires a Basis.")
        if basis.source == "bar_phi" and reference_energies:
            raise ValueError("The existing bar_phi model has no per-species reference-energy term.")
        self.basis = basis
        self.reference_energies = dict(reference_energies or {})
        self._fitted = None

    @property
    def labels(self):
        return self.basis.labels

    def fit(self, structures, *, regularization=_FIT_UNSET, fit_method=_FIT_UNSET,
            sklearn_params=_FIT_UNSET, energy_weight=_FIT_UNSET,
            force_weight=_FIT_UNSET, stress_weight=_FIT_UNSET, energy_key=_FIT_UNSET,
            force_key=_FIT_UNSET, stress_key=_FIT_UNSET, fit_E0=_FIT_UNSET,
            config=None):
        if self.basis.source == "portable_linear":
            if config is None or any(value is not _FIT_UNSET for value in (
                    regularization, fit_method, sklearn_params, energy_weight,
                    force_weight, stress_weight, energy_key, force_key,
                    stress_key, fit_E0)):
                raise ValueError("Selected portable Ni refit requires a seven-section config.")
            from .portable_refit import fit_portable_linear

            frames = tuple(structures)
            record = fit_portable_linear(self.basis._portable, frames, config)
            previous_basis, previous_fitted = self.basis, self._fitted
            try:
                self.basis = Basis._from_portable_linear(record)
                self._fitted = record
                checks = record["refit"]["fit_metadata"]["validation_checks"]
                report = {"checks": checks, "results": {}}
                if checks:
                    first = frames[0].copy()
                    calculator = self.ase_calculator(evaluator="torch")
                    first.calc = calculator
                    energy = float(first.get_potential_energy())
                    force = first.get_forces()
                    if "force_fd" in checks:
                        atom_index, axis = np.unravel_index(
                            np.argmax(np.abs(force)), force.shape)
                        step = 1e-5
                        energies = []
                        for sign in (-1, 1):
                            displaced = first.copy()
                            displaced.positions[atom_index, axis] += sign * step
                            displaced.calc = calculator
                            energies.append(float(displaced.get_potential_energy()))
                        finite = -(energies[1] - energies[0]) / (2 * step)
                        error = abs(float(force[atom_index, axis]) - finite)
                        if error > 1e-4 * max(1.0, abs(float(force[atom_index, axis]))):
                            raise ValueError("Portable Ni refit force_fd validation failed.")
                        report["results"]["force_fd"] = {
                            "atom_index": int(atom_index), "cartesian_axis": int(axis),
                            "force_eV_per_A": float(force[atom_index, axis]),
                            "finite_difference_eV_per_A": finite,
                            "absolute_error": error,
                        }
                    if "round_trip" in checks:
                        with tempfile.TemporaryDirectory(prefix="ye3t_ni_refit_validate_") as directory:
                            restored = LinearModel.read(self.write(Path(directory) / "model.ye3t"))
                            replay = first.copy()
                            replay.calc = restored.ase_calculator(evaluator="torch")
                            if (abs(float(replay.get_potential_energy()) - energy) > 1e-8 or
                                    not np.allclose(replay.get_forces(), force,
                                                    rtol=0, atol=1e-7)):
                                raise ValueError("Portable Ni refit round_trip validation failed.")
                        report["results"]["round_trip"] = {"passed": True}
                record["refit"]["fit_metadata"]["configured_validation"] = report
            except Exception:
                self.basis, self._fitted = previous_basis, previous_fitted
                raise
            return self
        if config is not None:
            if any(value is not _FIT_UNSET for value in (
                    regularization, fit_method, sklearn_params, energy_weight,
                    force_weight, stress_weight, energy_key, force_key,
                    stress_key, fit_E0)):
                raise ValueError("Use config or direct fit arguments, not both.")
            request = resolve_linear_fit_config(config, self.basis)
            if request.get("output_scope") == "per_atom":
                from .tesseral_targets import (
                    cartesian_to_real_tesseral, cartesian_tesseral_convention_hash,
                )

                if self.reference_energies:
                    raise ValueError("Per-atom covariant models do not use scalar reference energies.")
                frames = tuple(structures)
                if not frames or not self.basis.labels:
                    raise ValueError("Per-atom fit needs structures and a nonempty selected physical image.")
                L = self.basis._resolved["representation"]["parent"]["L"]
                blocks = []
                for index, atoms in enumerate(frames):
                    if not hasattr(atoms, "get_chemical_symbols"):
                        raise TypeError("Per-atom fit expects ASE Atoms structures.")
                    if len(atoms) == 0:
                        raise ValueError("Per-atom fit does not accept an empty ASE structure.")
                    values = atoms.arrays.get(request["target_key"])
                    if values is None:
                        raise ValueError(f"Structure {index} lacks per-atom target {request['target_key']!r}.")
                    values = np.asarray(values, dtype=np.float64)
                    if request["target_input"] == "cartesian":
                        values = cartesian_to_real_tesseral(
                            values, L, self.basis._resolved["representation"]["parent"]["parity"])
                    if values.shape != (len(atoms), 2 * L + 1) or not np.isfinite(values).all():
                        raise ValueError(f"Structure {index} has invalid real-tesseral target shape or values.")
                    rows = self.basis.create(atoms)
                    species_rows = (_tagged_carrier_species_design(self.basis, atoms, rows)
                                    if self.basis.source == "tagged_carriers" else rows)
                    design = species_rows.transpose(0, 2, 1).reshape(
                        -1, species_rows.shape[1])
                    blocks.append((design, values.reshape(-1)))
                width = (len(self.basis.elements) * len(self.basis.labels)
                         if self.basis.source == "tagged_carriers" else len(self.basis.labels))
                column_order = ("central_species_then_selected_multiplet_shared_over_M"
                                if self.basis.source == "tagged_carriers" else
                                "compiled_center_and_channel_multiplet_shared_over_M")
                if request["method"] == "ridge":
                    gram = sum((design.T @ design for design, _ in blocks),
                               np.zeros((width, width), dtype=np.float64))
                    rhs = sum((design.T @ target for design, target in blocks),
                              np.zeros(width, dtype=np.float64))
                    coefficients = np.linalg.lstsq(
                        gram + request["regularization"] * np.eye(width), rhs, rcond=None)[0]
                    metadata = {"fit_method": "ridge", "alpha": request["regularization"],
                                "n_rows": sum(len(target) for _, target in blocks),
                                "n_cols": width,
                                "design_column_order": column_order}
                else:
                    coefficients, metadata = _fit_sklearn_design(
                        blocks, request["method"], request["sklearn_params"],
                        column_order)
                if coefficients.shape != (width,) or not np.isfinite(coefficients).all():
                    raise FloatingPointError("Per-atom fit returned invalid shared-M coefficients.")
                metadata["coordinate_penalty_metric"] = {
                    "schema": "selected_compiler_coordinate_euclidean_v1",
                    "physical_image_plan_hash": (
                        self.basis._tagged_carrier_plan_hash
                        if self.basis.source == "tagged_carriers" else
                        self.basis._density_full_m_plan_hash),
                    "columns": ([{"central_species": species,
                                  "coordinate_id": label.identity}
                                 for species in self.basis.elements for label in self.basis.labels]
                                if self.basis.source == "tagged_carriers" else
                                [{"coordinate_id": label.identity}
                                 for label in self.basis.labels]),
                    "diagonal": [1.0] * width,
                    "interpretation": "coefficient_norm_in_saved_selected_coordinates",
                }
                metadata["resolved_fit_config"] = request["resolved_fit_config"]
                metadata["resolved_fit_config_sha256"] = request["resolved_fit_config_sha256"]
                metadata["target_units"] = request["target_units"]
                metadata["target_input"] = request["target_input"]
                metadata["cartesian_tesseral_convention_sha256"] = (
                    cartesian_tesseral_convention_hash(
                        L, self.basis._resolved["representation"]["parent"]["parity"])
                    if request["target_input"] == "cartesian" else None)
                previous = self._fitted
                self._fitted = {"kind": ("tagged_full_m_per_atom"
                                          if self.basis.source == "tagged_carriers" else
                                          "density_full_m_per_atom"), "beta": coefficients,
                                "fit_metadata": metadata}
                try:
                    report = {"checks": list(request["validation_checks"]), "results": {}}
                    if "round_trip" in request["validation_checks"]:
                        with tempfile.TemporaryDirectory(prefix="ye3t_full_m_fit_validate_") as directory:
                            saved = self.write(Path(directory) / "model.ye3t.json")
                            restored = LinearModel.read(saved)
                            if not np.allclose(restored.predict(frames[0])["mean_real_tesseral"],
                                               self.predict(frames[0])["mean_real_tesseral"],
                                               rtol=0, atol=1e-12):
                                raise ValueError("Per-atom fit round-trip validation failed.")
                        report["results"]["round_trip"] = {"passed": True}
                    metadata["configured_validation"] = report
                except Exception:
                    self._fitted = previous
                    raise
                return self
            offsets = request["reference_energies"]
            if self.reference_energies and self.reference_energies != offsets:
                raise ValueError("Constructor reference energies differ from fit config.")
            frames = tuple(structures)
            old_offsets = self.reference_energies
            old_fitted = self._fitted
            try:
                self.reference_energies = offsets
                self.fit(
                    frames, regularization=request["regularization"],
                    fit_method=request["method"],
                    sklearn_params=request["sklearn_params"],
                    energy_weight=request["energy_weight"],
                    force_weight=request["force_weight"],
                    stress_weight=request["stress_weight"],
                    energy_key=request["energy_key"],
                    force_key=request["force_key"],
                    stress_key=request["stress_key"],
                    fit_E0=request["fit_E0"] if self.basis.source == "combined_scalar"
                    or len(self.basis._resolution.to_dict()["components"]) > 1
                    or self.basis._resolution.to_dict()["components"][0]["kind"] == "density"
                    else None,
                )
                checks = request["validation_checks"]
                report = {"checks": list(checks), "results": {}}
                if checks:
                    first = frames[0].copy()
                    calculator = self.ase_calculator(evaluator="auto")
                    first.calc = calculator
                    if "force_fd" in checks:
                        force = float(first.get_forces()[0, 0])
                        step = 1e-5
                        energies = []
                        for sign in (-1, 1):
                            displaced = first.copy()
                            displaced.positions[0, 0] += sign * step
                            displaced.calc = calculator
                            energies.append(float(displaced.get_potential_energy()))
                        finite = -(energies[1] - energies[0]) / (2 * step)
                        error = abs(force - finite)
                        if error > 1e-4 * max(1.0, abs(force)):
                            raise ValueError("Configured force_fd validation failed.")
                        report["results"]["force_fd"] = {"force_eV_per_A": force,
                                                          "finite_difference_eV_per_A": finite,
                                                          "absolute_error": error}
                    if "round_trip" in checks:
                        suffix = {"combined_scalar": ".ye3t", "density": ".pt",
                                  "tagged_cauchy_image": ".ye3t.json"}.get(self.basis.source)
                        if suffix is None:
                            raise ValueError("Configured round_trip has no supported scalar artifact.")
                        with tempfile.TemporaryDirectory(prefix="ye3t_fit_validate_") as directory:
                            saved = self.write(Path(directory) / ("model" + suffix))
                            restored = LinearModel.read(saved)
                            replay = first.copy()
                            replay.calc = restored.ase_calculator(evaluator="torch")
                            if (abs(replay.get_potential_energy() - first.get_potential_energy()) > 1e-8 or
                                    not np.allclose(replay.get_forces(), first.get_forces(),
                                                    rtol=0, atol=1e-7) or
                                    request["stress_weight"] and
                                    not np.allclose(replay.get_stress(), first.get_stress(),
                                                    rtol=0, atol=1e-8)):
                                raise ValueError("Configured round_trip validation failed.")
                        report["results"]["round_trip"] = {"passed": True}
                if self.basis.source == "combined_scalar":
                    self._fitted["fit_metadata"]["configured_validation"] = report
                    self._fitted["fit_metadata"]["resolved_fit_config"] = request["resolved_fit_config"]
                    self._fitted["fit_metadata"]["resolved_fit_config_sha256"] = request["resolved_fit_config_sha256"]
                elif isinstance(getattr(self._fitted, "fit_metadata", None), dict):
                    self._fitted.fit_metadata["configured_validation"] = report
                    self._fitted.fit_metadata["resolved_fit_config"] = request["resolved_fit_config"]
                    self._fitted.fit_metadata["resolved_fit_config_sha256"] = request["resolved_fit_config_sha256"]
                return self
            except Exception:
                self.reference_energies = old_offsets
                self._fitted = old_fitted
                raise
        regularization = 1e-8 if regularization is _FIT_UNSET else regularization
        fit_method = "ridge" if fit_method is _FIT_UNSET else fit_method
        sklearn_params = None if sklearn_params is _FIT_UNSET else sklearn_params
        energy_weight = 1.0 if energy_weight is _FIT_UNSET else energy_weight
        force_weight = 1.0 if force_weight is _FIT_UNSET else force_weight
        stress_weight = 0.0 if stress_weight is _FIT_UNSET else stress_weight
        energy_key = "energy" if energy_key is _FIT_UNSET else energy_key
        force_key = "forces" if force_key is _FIT_UNSET else force_key
        stress_key = "stress" if stress_key is _FIT_UNSET else stress_key
        fit_E0 = None if fit_E0 is _FIT_UNSET else fit_E0
        if self.basis.source == "configured":
            self.basis._materialize_configured()
        if self.basis.source == "tagged_carriers" or (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            raise ValueError("Full-M per-atom fitting requires a seven-section config with per_atom targets.")
        if self.basis._descriptor is None and self.basis.source != "combined_scalar":
            raise RuntimeError("Fit requires a constructed Basis, not one reloaded from an artifact.")
        prepared = _prepared_training_structures(
            structures, energy_key, force_key, stress_key, force_weight, stress_weight,
        )
        method = str(fit_method).strip().lower()
        if method == "ard":
            method = "ardregression"
        if method not in {"ridge", "linear_regression", "ols", "lasso", "ridgecv", "ardregression"}:
            raise ValueError("fit_method must be ridge, linear_regression, lasso, ridgecv, or ardregression.")
        if method == "ridge" and sklearn_params:
            raise ValueError("sklearn_params requires a scikit-learn fit_method.")
        if method != "ridge" and regularization != 1e-8:
            raise ValueError("For scikit-learn methods set regularization through sklearn_params.")
        weights = (float(energy_weight), float(force_weight), float(stress_weight))
        if any(not np.isfinite(value) or value < 0.0 for value in weights) or not any(weights):
            raise ValueError("Fit weights must be finite, nonnegative, and not all zero.")
        if fit_E0 is not None and self.basis.source not in {"combined_scalar", "density"}:
            raise ValueError("fit_E0 requires a density or combined scalar basis.")
        if self.basis.source == "density" and fit_E0 is not None:
            if not isinstance(fit_E0, (bool, np.bool_)):
                raise TypeError("fit_E0 must be a boolean.")
            if not np.isfinite(regularization) or regularization < 0.0:
                raise ValueError("regularization must be finite and nonnegative.")
            species = self.basis.elements
            if self.reference_energies and set(self.reference_energies) != set(species):
                raise ValueError("Density reference energies must cover every model species.")
            if fit_E0 and not energy_weight:
                raise ValueError("Fitting per-species E0 requires positive energy_weight.")
            if fit_E0:
                composition = np.asarray([
                    [atoms.get_chemical_symbols().count(name) for name in species]
                    for atoms in prepared], dtype=np.float64)
                if np.linalg.matrix_rank(composition) != len(species):
                    raise ValueError("Fitting per-species E0 requires full column rank in training compositions.")
            descriptor = self.basis._descriptor
            feature_width = len(descriptor.descriptor_specs)
            width = feature_width + (len(species) if fit_E0 else 0)
            gram = np.zeros((width, width), dtype=np.float64)
            rhs = np.zeros(width, dtype=np.float64)
            blocks = []
            n_rows = 0
            for atoms in prepared:
                row = _linear_ace_geometry_row(
                    atoms, evaluator=descriptor.ace_descriptor.calculator.evaluator,
                    descriptors=descriptor.descriptor_specs, cutoff=self.basis.cutoff,
                    type_map=descriptor.type_map, device="cpu", forces=bool(force_weight),
                    stress=bool(stress_weight), chunk_size=None)
                zeros_force = np.zeros((3 * len(atoms), len(species)), dtype=np.float64)
                zeros_stress = np.zeros((6, len(species)), dtype=np.float64)
                counts = np.asarray([atoms.get_chemical_symbols().count(name)
                                     for name in species], dtype=np.float64)
                energy_row = row["energy"].detach().cpu().numpy()
                force_rows = (None if not force_weight else
                              row["forces"].detach().cpu().numpy())
                stress_rows = (None if not stress_weight else
                               row["stress"].detach().cpu().numpy())
                if fit_E0:
                    energy_row = np.r_[energy_row, counts]
                    if force_weight:
                        force_rows = np.column_stack((force_rows, zeros_force))
                    if stress_weight:
                        stress_rows = np.column_stack((stress_rows, zeros_stress))
                reference = sum(self.reference_energies.get(name, 0.0)
                                for name in atoms.get_chemical_symbols())
                targets = ((energy_row[None, :],
                            np.asarray([float(atoms.info[energy_key]) - reference]),
                            energy_weight),
                           (force_rows, np.asarray(atoms.arrays[force_key]).reshape(-1)
                            if force_weight else None, force_weight),
                           (stress_rows, np.asarray(atoms.info[stress_key])
                            if stress_weight else None, stress_weight))
                for design, target, weight in targets:
                    if not weight:
                        continue
                    scaled = np.sqrt(weight) * design
                    scaled_target = np.sqrt(weight) * target
                    gram += scaled.T @ scaled
                    rhs += scaled.T @ scaled_target
                    n_rows += len(scaled_target)
                    if method != "ridge":
                        blocks.append((scaled, scaled_target))
            if method == "ridge":
                penalty = np.zeros(width, dtype=np.float64)
                penalty[:feature_width] = float(regularization)
                coefficients = np.linalg.lstsq(
                    gram + np.diag(penalty), rhs, rcond=None)[0]
                metadata = {"fit_method": "ridge", "alpha": float(regularization),
                            "n_cols": width, "n_rows": n_rows,
                            "design_column_order": ("descriptor_features_then_species_offsets"
                                                    if fit_E0 else "descriptor_features_only")}
            else:
                coefficients, metadata = _fit_sklearn_design(
                    blocks, method, sklearn_params,
                    "descriptor_features_then_species_offsets" if fit_E0
                    else "descriptor_features_only")
            if coefficients.shape != (width,) or not np.isfinite(coefficients).all():
                raise FloatingPointError("Density fit returned invalid coefficients.")
            offsets = {name: float(self.reference_energies.get(name, 0.0) +
                                   (coefficients[feature_width + index] if fit_E0 else 0.0))
                       for index, name in enumerate(species)}
            metadata["fit_E0"] = bool(fit_E0)
            metadata["per_species_E0_eV"] = offsets
            metadata["reference_energy_targets"] = {
                "enabled": True,
                "reference_energies": dict(offsets),
                "target_convention": "E_target = E_raw - sum_type(n_type * E_ref[type])",
            }
            if fit_E0 and method != "ridge":
                metadata["e0_correction_prior"] = {
                    "origin_eV": {name: float(self.reference_energies.get(name, 0.0))
                                  for name in species},
                    "policy": "sklearn prior or penalty acts on descriptor and E0 correction columns",
                    "baseline_dependent": True,
                }
            metadata["design_columns"] = ([{"feature_index": index,
                                             "label_identity": label.identity}
                                            for index, label in enumerate(self.labels)] +
                                          ([{"central_species": name,
                                             "label_identity": "E0:" + name}
                                            for name in species] if fit_E0 else []))
            fitted = LinearACEScalarModelBundle(
                settings=descriptor.settings,
                site_basis_config=descriptor.site_basis_config,
                descriptor_specs=descriptor.descriptor_specs,
                weight=coefficients[:feature_width], bias=0.0,
                basis_mode=None, fit_method=method, fit_metadata=metadata)
            _bind_density_public_labels(self.basis, fitted)
            self.reference_energies = offsets
            self._fitted = fitted
            return self
        if self.basis.source == "combined_scalar":
            if any(item._descriptor is None
                   for item in self.basis._components.values()):
                raise RuntimeError("Refitting a loaded combined bundle requires newly constructed component bases.")
            if not np.isfinite(regularization) or regularization < 0.0:
                raise ValueError("regularization must be finite and nonnegative.")
            if fit_E0 is not None and not isinstance(fit_E0, (bool, np.bool_)):
                raise TypeError("fit_E0 must be a boolean.")
            species = self.basis.elements
            if self.reference_energies and set(self.reference_energies) != set(species):
                raise ValueError("Combined reference energies must cover every model species.")
            fit_offsets = (not self.reference_energies if fit_E0 is None else
                           bool(fit_E0))
            if fit_offsets and not energy_weight:
                raise ValueError("Fitting per-species E0 requires positive energy_weight.")
            if set(species) - {name for atoms in prepared for name in atoms.get_chemical_symbols()}:
                raise ValueError("Every combined-model species needs a training structure for its offset.")
            if fit_offsets:
                composition = np.asarray([
                    [atoms.get_chemical_symbols().count(name) for name in species]
                    for atoms in prepared], dtype=np.float64)
                if np.linalg.matrix_rank(composition) != len(species):
                    raise ValueError("Fitting per-species E0 requires full column rank in training compositions.")
            feature_width = sum(len(item.labels) * (
                len(species) if item.source == "tagged_cauchy_image" else 1)
                for item in self.basis._components.values())
            design_columns = _combined_design_columns(self.basis, fit_offsets)
            width = feature_width + (len(species) if fit_offsets else 0)
            if len(design_columns) != width:
                raise RuntimeError("Combined design-column manifest disagrees with feature width.")
            gram = np.zeros((width, width), dtype=np.float64)
            rhs = np.zeros(width, dtype=np.float64)
            blocks = []
            for atoms in prepared:
                energy_row, force_rows, stress_rows = _combined_scalar_geometry_row(
                    self.basis, atoms, bool(force_weight), bool(stress_weight))
                energy_row = energy_row[:width]
                if force_rows is not None:
                    force_rows = force_rows[:, :width]
                if stress_rows is not None:
                    stress_rows = stress_rows[:, :width]
                reference = sum(self.reference_energies.get(name, 0.0)
                                for name in atoms.get_chemical_symbols())
                targets = ((energy_row[None, :],
                            np.asarray([float(atoms.info[energy_key]) - reference]),
                            energy_weight),
                           (force_rows, np.asarray(atoms.arrays[force_key]).reshape(-1)
                            if force_weight else None, force_weight),
                           (stress_rows, np.asarray(atoms.info[stress_key])
                            if stress_weight else None, stress_weight))
                for design, target, weight in targets:
                    if not weight:
                        continue
                    scaled = np.sqrt(weight) * design
                    scaled_target = np.sqrt(weight) * target
                    gram += scaled.T @ scaled
                    rhs += scaled.T @ scaled_target
                    if method != "ridge":
                        blocks.append((scaled, scaled_target))
            if method == "ridge":
                penalty = np.zeros(width, dtype=np.float64)
                penalty[:feature_width] = float(regularization)
                coefficients = np.linalg.lstsq(
                    gram + np.diag(penalty), rhs, rcond=None)[0]
                metadata = {"fit_method": "ridge", "alpha": float(regularization),
                            "design_column_order": "component_features_then_species_offsets",
                            "n_cols": width, "fit_E0": fit_offsets}
            else:
                coefficients, metadata = _fit_sklearn_design(
                    blocks, method, sklearn_params,
                    "component_features_then_species_offsets")
            if coefficients.shape != (width,) or not np.isfinite(coefficients).all():
                raise FloatingPointError("Combined scalar fit returned invalid coefficients.")
            final_offsets = {name: float((coefficients[feature_width + index]
                                          if fit_offsets else 0.0) +
                                         self.reference_energies.get(name, 0.0))
                             for index, name in enumerate(species)}
            metadata["fit_E0"] = fit_offsets
            metadata["per_species_E0_eV"] = dict(final_offsets)
            if fit_offsets and method != "ridge":
                metadata["e0_correction_prior"] = {
                    "origin_eV": {name: float(self.reference_energies.get(name, 0.0))
                                  for name in species},
                    "policy": "sklearn prior or penalty acts on descriptor and E0 correction columns",
                    "baseline_dependent": True,
                }
            metadata["design_columns"] = design_columns
            fitted = {}
            position = 0
            tagged_offset_owner = next(name for name, item in
                                       self.basis._components.items()
                                       if item.source == "tagged_cauchy_image")
            for name, item in self.basis._components.items():
                count = len(item.labels)
                if item.source == "density":
                    descriptor = item._descriptor
                    fitted[name] = LinearACEScalarModelBundle(
                        settings=descriptor.settings,
                        site_basis_config=descriptor.site_basis_config,
                        descriptor_specs=descriptor.descriptor_specs,
                        weight=coefficients[position:position + count].copy(),
                        bias=0.0, basis_mode=None, fit_method=method,
                        fit_metadata={"combined_component": name},
                    )
                    position += count
                else:
                    evaluator = item._descriptor.metadata["tagged_cauchy_image_evaluator"]
                    beta = coefficients[position:position + len(species) * count].reshape(
                        len(species), count)
                    fitted[name] = TaggedCauchyImageLinearModel(
                        evaluator, {element: beta[index].copy()
                                    for index, element in enumerate(species)},
                        final_offsets if name == tagged_offset_owner else
                        {element: 0.0 for element in species},
                    )
                    position += len(species) * count
            self._fitted = {"components": fitted, "offsets_eV": final_offsets,
                            "fit_metadata": metadata}
            return self
        if self.basis.source == "bar_phi":
            if not np.isfinite(regularization) or regularization < 0.0:
                raise ValueError("regularization must be finite and nonnegative.")
            fitted = YE3TModel.phi(self.basis._descriptor, {"branches": ("bar_phi",)})
            width = len(self.labels) + 1
            gram = np.zeros((width, width), dtype=float)
            rhs = np.zeros(width, dtype=float)
            blocks = []
            type_map = {name: index for index, name in enumerate(self.basis.elements)}
            for atoms in prepared:
                energy_row, force_rows, stress_rows, _sites = _bar_phi_feature_rows(
                    fitted, atoms, type_map, forces=bool(force_weight), stress=bool(stress_weight),
                )
                if energy_weight:
                    gram += energy_weight * np.outer(energy_row, energy_row)
                    rhs += energy_weight * energy_row * float(atoms.info[energy_key])
                    if method != "ridge":
                        blocks.append((np.sqrt(energy_weight) * energy_row[None, :],
                                       np.sqrt(energy_weight) * np.asarray([atoms.info[energy_key]])))
                if force_weight:
                    gram += force_weight * (force_rows.T @ force_rows)
                    rhs += force_weight * (force_rows.T @ np.asarray(atoms.arrays[force_key]).reshape(-1))
                    if method != "ridge":
                        blocks.append((np.sqrt(force_weight) * force_rows,
                                       np.sqrt(force_weight) * np.asarray(atoms.arrays[force_key]).reshape(-1)))
                if stress_weight:
                    gram += stress_weight * (stress_rows.T @ stress_rows)
                    rhs += stress_weight * (stress_rows.T @ np.asarray(atoms.info[stress_key]))
                    if method != "ridge":
                        blocks.append((np.sqrt(stress_weight) * stress_rows,
                                       np.sqrt(stress_weight) * np.asarray(atoms.info[stress_key])))
            if method == "ridge":
                coefficients = solve_ridge_statistics(
                    {"gram": gram, "rhs": rhs, "runtime_from_fit_coordinates": np.eye(width)},
                    float(regularization),
                )["runtime_coefficients"]
            else:
                coefficients, fitted.fit_metadata = _fit_sklearn_design(
                    blocks, method, sklearn_params, "atom_count_bias_then_descriptor_features",
                )
            if not np.isfinite(coefficients).all():
                raise FloatingPointError("bar_phi fit returned nonfinite coefficients.")
            with torch.no_grad():
                fitted.bar_phi_bias.copy_(torch.as_tensor([coefficients[0]], dtype=fitted.config.torch_dtype))
                fitted.bar_phi_weight.copy_(torch.as_tensor(coefficients[1:], dtype=fitted.config.torch_dtype))
            self._fitted = fitted
            return self
        if self.basis.source == "tagged_cauchy_image" and method != "ridge":
            evaluator = self.basis._descriptor.metadata["tagged_cauchy_image_evaluator"]
            species = tuple(evaluator.species_order)
            if self.reference_energies and set(self.reference_energies) != set(species):
                raise ValueError("Tagged reference energies must cover every model species.")
            feature_count = int(evaluator.feature_count)
            if feature_count != len(self.labels):
                raise RuntimeError("Tagged evaluator width differs from the Basis labels.")
            beta_width = len(species) * feature_count
            blocks = []
            for atoms in prepared:
                row = tagged_cauchy_image_geometry_row(self.basis._descriptor, atoms)
                energy_row = np.concatenate((row["feature_sums"].reshape(-1), row["species_counts"]))
                zero_offsets = np.zeros((len(atoms) * 3, len(species)))
                if energy_weight:
                    reference = sum(self.reference_energies.get(name, 0.0)
                                    for name in atoms.get_chemical_symbols())
                    blocks.append((np.sqrt(energy_weight) * energy_row[None, :],
                                   np.asarray([np.sqrt(energy_weight) *
                                               (float(atoms.info[energy_key]) - reference)])))
                if force_weight:
                    blocks.append((np.sqrt(force_weight) * np.column_stack((row["force_design"], zero_offsets)),
                                   np.sqrt(force_weight) * np.asarray(atoms.arrays[force_key]).reshape(-1)))
                if stress_weight:
                    blocks.append((np.sqrt(stress_weight) * np.column_stack((
                        row["stress_design"], np.zeros((6, len(species))))),
                        np.sqrt(stress_weight) * np.asarray(atoms.info[stress_key])))
            coefficients, metadata = _fit_sklearn_design(
                blocks, method, sklearn_params, "species_major_descriptor_features_then_species_offsets",
            )
            beta = coefficients[:beta_width].reshape(len(species), feature_count)
            fitted = TaggedCauchyImageLinearModel(
                evaluator, {name: beta[index] for index, name in enumerate(species)},
                {name: coefficients[beta_width + index] for index, name in enumerate(species)},
            )
            if self.reference_energies:
                fitted.reference_terms = {"atomic_energies": dict(self.reference_energies)}
            fitted.fit_metadata = metadata
            fitted._ye3t_linear_fit_metadata = dict(metadata)
            self._fitted = fitted
            return self
        if self.basis.source == "density" and stress_weight and method != "ridge":
            descriptor = self.basis._descriptor
            blocks = []
            for atoms in prepared:
                row = _linear_ace_geometry_row(
                    atoms, evaluator=descriptor.ace_descriptor.calculator.evaluator,
                    descriptors=descriptor.descriptor_specs, cutoff=self.basis.cutoff,
                    type_map=descriptor.type_map, device="cpu",
                    forces=bool(force_weight), stress=True, chunk_size=None)
                columns = len(descriptor.descriptor_specs)
                reference = sum(self.reference_energies.get(name, 0.0)
                                for name in atoms.get_chemical_symbols())
                if energy_weight:
                    design = np.r_[row["energy"].detach().cpu().numpy(), len(atoms)][None, :]
                    target = np.asarray([float(atoms.info[energy_key]) - reference])
                    blocks.append((np.sqrt(energy_weight) * design,
                                   np.sqrt(energy_weight) * target))
                if force_weight:
                    design = np.column_stack((
                        row["forces"].detach().cpu().numpy(),
                        np.zeros((3 * len(atoms), 1), dtype=np.float64)))
                    blocks.append((np.sqrt(force_weight) * design,
                                   np.sqrt(force_weight) *
                                   np.asarray(atoms.arrays[force_key]).reshape(-1)))
                design = np.column_stack((
                    row["stress"].detach().cpu().numpy(), np.zeros((6, 1))))
                blocks.append((np.sqrt(stress_weight) * design,
                               np.sqrt(stress_weight) * np.asarray(atoms.info[stress_key])))
            coefficients, metadata = _fit_sklearn_design(
                blocks, method, sklearn_params,
                "descriptor_features_then_atom_count_bias")
            if len(coefficients) != columns + 1:
                raise RuntimeError("Density fit width differs from compiler descriptor columns.")
            metadata.update({"energy_weight": float(energy_weight),
                             "force_weight": float(force_weight),
                             "stress_weight": float(stress_weight),
                             "include_bias_column": True,
                             "intercept_policy": "atom_count_bias_column"})
            fitted = LinearACEScalarModelBundle(
                settings=descriptor.settings,
                site_basis_config=descriptor.site_basis_config,
                descriptor_specs=descriptor.descriptor_specs,
                weight=coefficients[:-1], bias=float(coefficients[-1]),
                basis_mode=None, fit_method=method, fit_metadata=metadata)
        else:
            config = {
                "ridge_alpha": float(regularization),
                "fit_method": ("ridge_normal_equations" if method == "ridge" else method)
                              if self.basis.source == "density" else "ridge_streaming_gram",
                "energy_weight": float(energy_weight),
                "force_weight": float(force_weight),
                "stress_weight": float(stress_weight),
                "energy_key": str(energy_key),
                "force_key": str(force_key),
                "stress_key": str(stress_key),
                "reference_energies": dict(self.reference_energies),
            }
            if method != "ridge":
                config["sklearn_params"] = dict(sklearn_params or {})
            if self.basis.source == "tagged_cauchy_image":
                config["stress_weight"] = float(stress_weight)
                config["stress_key"] = str(stress_key)
                config["restore_references"] = bool(self.reference_energies)
            fitted = YE3TModel.linear(self.basis._descriptor, config, structures=prepared)
        if self.basis.source == "density":
            _bind_density_public_labels(self.basis, fitted)
        elif int(fitted.evaluator.feature_count) != len(self.labels):
            raise RuntimeError("Fitted tagged image width differs from the Basis labels.")
        self._fitted = fitted
        return self

    def predict(self, atoms, *, uncertainty=False):
        """Predict real-tesseral per-atom multiplets with shared coefficients."""
        if self._fitted is None or self.basis.source != "tagged_carriers" and not (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            raise ValueError("predict requires a fitted full-M per-atom model.")
        rows = self.basis.create(atoms)
        species_rows = (_tagged_carrier_species_design(self.basis, atoms, rows)
                        if self.basis.source == "tagged_carriers" else rows)
        beta = np.asarray(self._fitted["beta"], dtype=np.float64)
        if beta.shape != (species_rows.shape[1],):
            raise RuntimeError("Per-atom coefficient width differs from selected compiler rows.")
        mean = np.einsum("nfm,f->nm", species_rows, beta)
        if not np.isfinite(mean).all():
            raise FloatingPointError("Per-atom prediction is nonfinite.")
        L = self.basis._resolved["representation"]["parent"]["L"]
        prediction = {"mean_real_tesseral": mean,
                      "M_values": tuple(range(-L, L + 1)),
                      "units": self._fitted["fit_metadata"]["target_units"]}
        if uncertainty:
            posterior = self._fitted["fit_metadata"].get("predictive_uncertainty")
            if not isinstance(posterior, dict) or posterior.get("schema") != "ye3t_linear_ard_posterior_v1":
                raise ValueError("Full-M uncertainty requires an ARD posterior.")
            column_order = ("central_species_then_selected_multiplet_shared_over_M"
                            if self.basis.source == "tagged_carriers" else
                            "compiled_center_and_channel_multiplet_shared_over_M")
            if posterior.get("design_column_order") != column_order:
                raise ValueError("ARD posterior column order differs from the full-M basis.")
            active = np.asarray(posterior["active_column_indices"], dtype=int)
            covariance = np.asarray(posterior["coefficient_covariance_active"], dtype=np.float64)
            if (active.ndim != 1 or len(set(active.tolist())) != len(active) or
                    np.any(active < 0) or np.any(active >= len(beta)) or
                    covariance.shape != (len(active), len(active)) or
                    not np.isfinite(covariance).all() or
                    not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10)):
                raise ValueError("Saved full-M ARD covariance is invalid.")
            selected = species_rows[:, active, :]
            component_covariance = np.einsum("nam,ab,nbk->nmk",
                                             selected, covariance, selected)
            component_covariance = .5 * (component_covariance +
                                         component_covariance.transpose(0, 2, 1))
            if (not np.isfinite(component_covariance).all() or
                    np.min(np.linalg.eigvalsh(component_covariance)) < -1e-10):
                raise ValueError("Full-M predictive covariance is invalid.")
            prediction["covariance_real_tesseral"] = component_covariance
        return prediction

    def ase_calculator(self, *, backend=None, evaluator=None, neighbors=None, **kwargs):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before constructing an ASE calculator.")
        if evaluator is not None and evaluator not in {
                "auto", "torch", "reference", "native_cpu"}:
            raise ValueError("evaluator must be auto, torch, reference, or native_cpu.")
        if backend == "torch":
            backend = ("pytorch" if self.basis.source in {
                "density", "bar_phi", "portable_linear"} else "reference")
        if neighbors is None:
            neighbors = (self.basis._resolution.capability_report["requested_neighbors"]
                         if hasattr(self.basis, "_resolution") else "auto")
            if self.basis.source == "tagged_cauchy_image" and (
                    evaluator in {"torch", "reference"} or backend == "reference"):
                neighbors = "auto"
            if self.basis.source == "density" and neighbors == "matscipy" and (
                    evaluator == "torch" or backend == "pytorch"):
                neighbors = "ase"
        if neighbors not in {"auto", "ase", "matscipy"}:
            raise ValueError("neighbors must be auto, ase, or matscipy.")
        if evaluator is not None:
            if backend is not None:
                raise ValueError("Specify either evaluator or backend, not both.")
            if evaluator == "auto":
                backend = None
            elif evaluator == "torch":
                backend = ("pytorch" if self.basis.source in {
                    "density", "bar_phi", "portable_linear"} else "reference")
            else:
                backend = evaluator
        if backend == "auto":
            backend = None
        if backend is None and hasattr(self.basis, "_resolution"):
            selected = self.basis._resolution.capability_report["selected_evaluator"]
            if selected == "native_cpu":
                backend = "native_cpu"
        if self.basis.source == "tagged_carriers" or (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            expected_backend = "reference" if self.basis.source == "tagged_carriers" else "pytorch"
            if backend not in (None, expected_backend) or neighbors not in ("auto", "ase") or kwargs:
                raise ValueError("Full-M ASE properties require the selected evaluator and ASE neighbors.")
            return _TaggedCarrierPropertyCalculator(self)
        if self.basis.source == "combined_scalar":
            if backend not in (None, "reference", "pytorch", "native_cpu") or (
                    backend != "native_cpu" and neighbors not in ("auto", "ase")):
                raise ValueError("Combined scalar ASE requires Torch/reference or native_cpu with supported neighbors.")
            if kwargs:
                raise ValueError("Combined scalar ASE does not accept component-specific runtime options.")
            from ase.calculators.mixing import SumCalculator

            calculators = []
            for name, item in self.basis._components.items():
                fitted = self._fitted["components"][name]
                if item.source == "density":
                    if backend == "native_cpu":
                        component_model = LinearModel(item)
                        component_model._fitted = fitted
                        calculators.append(component_model.ase_calculator(
                            backend="native_cpu", neighbors=neighbors))
                    else:
                        calculators.append(LinearACEScalarCalculator(
                            fitted, item.cutoff, item._descriptor.type_map,
                            reference_energies={}, backend="pytorch"))
                else:
                    calculators.append(fitted.ase_calculator(
                        backend="native_cpu" if backend == "native_cpu" else "reference",
                        neighbors=neighbors if backend == "native_cpu" else "auto"))
            return SumCalculator(calculators)
        configured_pace = self.basis.source == "density" and (
            hasattr(self.basis, "_resolution") or
            (self._fitted.fit_metadata or {}).get(
                "ye3t_methods_public_label_convention") is not None)
        if (neighbors != "auto" and backend is None and self.basis.source in {
                "density", "tagged_cauchy_image"} and
                not (configured_pace and neighbors == "ase")):
            backend = "native_cpu"
        if self.basis.source == "portable_linear":
            if backend not in (None, "pytorch"):
                raise ValueError("Portable Ni scalar ASE currently supports the Torch evaluator.")
            if "refit" in self._fitted and backend is None:
                raise ValueError("Portable Ni refits require evaluator='torch'; no native AUTO plan is saved.")
            from .portable_archive import portable_torch_calculator

            return portable_torch_calculator(self._fitted, neighbors=neighbors,
                                             **kwargs)
        if self.basis.source == "legacy_composite":
            if backend not in (None, "native_cpu"):
                raise ValueError("The legacy composite has only a validated native_cpu ASE evaluator.")
            from ase.calculators.mixing import SumCalculator
            from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
            from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator

            with tempfile.TemporaryDirectory(prefix="ye3t_legacy_composite_") as directory:
                for name, data in self._fitted["artifact_bytes"].items():
                    (Path(directory) / name).write_bytes(data)
                residual = YE3TTaggedCauchyCalculator.from_artifact(
                    Path(directory) / self._fitted["artifact_name"],
                    neighbors=neighbors, **kwargs,
                )
            reference = YE3TZBLCalculator.from_model_manifest(
                json.loads(self._fitted["manifest_json"]),
            )
            return SumCalculator((residual, reference))
        if neighbors != "auto" and self.basis.source == "bar_phi":
            raise ValueError("Explicit neighbors selection requires a native_cpu scalar ASE evaluator.")
        if self.basis.source == "density":
            if backend not in (None, "pytorch", "native_cpu"):
                raise ValueError("Density ASE evaluator must be torch or native_cpu.")
            if backend == "native_cpu":
                convention = (self._fitted.fit_metadata or {}).get(
                    "ye3t_methods_public_label_convention", {})
                if (any(str(spec.key).endswith("|physical_eta_bound")
                        for spec in self._fitted.descriptor_specs) or
                    (len(self.basis.elements) > 1 and
                     convention.get("schema") in {
                         "ye3t_methods_density_pace_labels_v2",
                         "ye3t_methods_density_pace_labels_v3"})):
                    raise ValueError("Configured multi-species density has no validated native_cpu evaluator.")
                from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator

                temporary = tempfile.TemporaryDirectory(prefix="ye3t_yace_native_")
                try:
                    path = Path(temporary.name) / "model.yace"
                    self.export_lammps(path)
                    if self.reference_energies:
                        from ye3t_methods.atomistic.ace.yace import read_yace

                        exported = read_yace(path, compatibility="lammps_pace_linear_v1")
                        expected = [float(self._fitted.bias) +
                                    float(self.reference_energies.get(name, 0.0))
                                    for name in self.basis.elements]
                        if not np.allclose(exported["E0"], expected, rtol=0.0, atol=1e-12):
                            raise ValueError(
                                "Native YACE export did not preserve the fitted reference energies."
                            )
                    calculator = YE3TYACENativeCalculator.from_artifact(
                        path, neighbors=neighbors, **kwargs,
                    )
                    calculator._temporary = temporary
                    return calculator
                except Exception:
                    temporary.cleanup()
                    raise
            if neighbors not in ("auto", "ase"):
                raise ValueError("Torch density ASE supports auto or ase neighbors.")
            return LinearACEScalarCalculator(
                self._fitted, self.basis.cutoff,
                self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
                reference_energies=self.reference_energies,
                backend="pytorch" if backend is None else backend,
                **kwargs,
            )
        if self.basis.source == "bar_phi":
            if backend not in (None, "pytorch", "reference"):
                raise ValueError("bar_phi ASE currently uses the PyTorch reference evaluator.")
            return HybridACEPhiCalculator(
                self._fitted, type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ), **kwargs,
            )
        return self._fitted.ase_calculator(
            backend="reference" if backend is None else backend,
            neighbors=neighbors, **kwargs,
        )

    def predict_uncertainty(self, atoms):
        """Return ARD readout standard deviations in eV for each site and total E.

        These are posterior coefficient uncertainties conditional on the fixed
        descriptor map. They exclude model error and observation noise.
        """
        if self.basis.source == "tagged_carriers" or (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            raise ValueError("Use predict(atoms, uncertainty=True) for full-M component covariance.")
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before predicting uncertainty.")
        if self.basis.source == "legacy_composite":
            raise ValueError("The legacy composite has no serialized ARD posterior.")
        if self.basis.source == "portable_linear":
            raise ValueError("The bounded Ni portable archive has no serialized ARD posterior.")
        metadata = dict((self._fitted["fit_metadata"] if self.basis.source == "combined_scalar"
                         else getattr(self._fitted, "fit_metadata", {})) or {})
        posterior = metadata.get("predictive_uncertainty")
        if not isinstance(posterior, dict) or posterior.get("schema") != "ye3t_linear_ard_posterior_v1":
            raise ValueError("predict_uncertainty requires a saved ARDRegression posterior.")
        if self.basis.source == "combined_scalar":
            symbols = np.asarray(atoms.get_chemical_symbols())
            if set(symbols) - set(self.basis.elements):
                raise ValueError("Uncertainty structure contains a species outside the combined basis.")
            blocks = []
            for item in self.basis._components.values():
                sites = item.create(atoms)
                if item.source == "density":
                    blocks.append(sites)
                else:
                    for species in self.basis.elements:
                        blocks.append(sites * (symbols == species)[:, None])
            if metadata.get("fit_E0"):
                blocks.append(np.column_stack(tuple(
                    (symbols == species).astype(np.float64)
                    for species in self.basis.elements)))
            design = np.column_stack(blocks)
            if design.shape[1] != len(metadata.get("design_columns", ())):
                raise ValueError("Combined ARD posterior design width differs from its saved columns.")
            order = "component_features_then_species_offsets"
        elif self.basis.source == "density":
            calculator = self.ase_calculator(backend="pytorch")
            pos, cell, edge_index, shifts, atom_types = calculator._geometry_from_atoms(
                atoms, requires_grad=False,
            )
            displacement = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
            with torch.no_grad():
                features = calculator.evaluator(
                    x_ij=displacement, edge_index=edge_index, atom_types=atom_types,
                    descriptors=self._fitted.descriptor_specs, real_if_scalar=True,
                ).detach().cpu().numpy()
            if metadata.get("fit_E0") is True:
                symbols = np.asarray(atoms.get_chemical_symbols())
                if set(symbols) - set(self.basis.elements):
                    raise ValueError("Uncertainty structure contains a species outside the density basis.")
                design = np.column_stack((features, *(
                    (symbols == species).astype(np.float64)
                    for species in self.basis.elements)))
                order = "descriptor_features_then_species_offsets"
            elif metadata.get("fit_E0") is False:
                design = features
                order = "descriptor_features_only"
            else:
                design = np.column_stack((features, np.ones(len(atoms))))
                order = "descriptor_features_then_atom_count_bias"
        elif self.basis.source == "bar_phi":
            type_map = self.basis._resolved.get(
                "type_map", {name: index for index, name in enumerate(self.basis.elements)},
            )
            sites = _bar_phi_feature_rows(
                self._fitted, atoms, type_map, forces=False, stress=False,
            )[3]
            design = np.column_stack((np.ones(len(atoms)), sites))
            order = "atom_count_bias_then_descriptor_features"
        else:
            evaluator = self._fitted.evaluator
            symbols = atoms.get_chemical_symbols()
            atom_types = np.asarray([evaluator.type_map[name] for name in symbols], dtype=int)
            with torch.no_grad():
                features = evaluator.materialize(
                    torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64),
                    torch.as_tensor(atom_types, dtype=torch.long),
                    cell=np.asarray(atoms.cell.array), pbc=np.asarray(atoms.pbc),
                )[2].detach().cpu().numpy()
            width = len(evaluator.species_order) * (evaluator.feature_count + 1)
            design = np.zeros((len(atoms), width), dtype=np.float64)
            for index, species_index in enumerate(atom_types):
                start = species_index * evaluator.feature_count
                design[index, start:start + evaluator.feature_count] = features[index]
                design[index, len(evaluator.species_order) * evaluator.feature_count + species_index] = 1.0
            order = "species_major_descriptor_features_then_species_offsets"
        if posterior.get("design_column_order") != order:
            raise ValueError("ARD posterior column order does not match this model.")
        raw_active = posterior["active_column_indices"]
        if (not isinstance(raw_active, list) or
                any(type(index) is not int for index in raw_active)):
            raise ValueError("Saved ARD posterior has noninteger column indices.")
        active = np.asarray(raw_active, dtype=int)
        covariance = np.asarray(
            posterior["coefficient_covariance_active"], dtype=np.float64,
        ).reshape((len(active), len(active)))
        if (covariance.shape != (len(active), len(active)) or
                np.any(active < 0) or np.any(active >= design.shape[1]) or
                len(np.unique(active)) != len(active) or not np.isfinite(covariance).all()):
            raise ValueError("Saved ARD posterior has invalid column indices or covariance.")
        selected = design[:, active]
        atomic_variance = np.einsum("if,fg,ig->i", selected, covariance, selected)
        total_row = selected.sum(axis=0)
        total_variance = float(total_row @ covariance @ total_row)
        if min(float(np.min(atomic_variance)), total_variance) < -1e-10:
            raise ValueError("Saved ARD posterior has a negative predictive variance.")
        return {
            "atomic_energy_std_eV": np.sqrt(np.maximum(atomic_variance, 0.0)),
            "total_energy_std_eV": float(np.sqrt(max(total_variance, 0.0))),
            "kind": "conditional_linear_readout_epistemic",
        }

    def _validate_density_reference_energies(self):
        if self.basis.source != "density" or getattr(self.basis, "_density_full_m", False):
            return
        metadata = self._fitted.fit_metadata or {}
        target = metadata.get("reference_energy_targets")
        if target is None:
            return  # Older bundles have only the top-level reference map.
        if not isinstance(target, dict):
            raise ValueError("Density reference-energy metadata is invalid.")
        if not target.get("enabled") and not self.reference_energies:
            return
        embedded = target.get("reference_energies")
        expected = (set(self.basis.elements) if "per_species_E0_eV" in metadata
                    else set(self.reference_energies))
        if (not isinstance(embedded, dict) or
                not target.get("enabled") or
                set(embedded) != expected or
                set(self.reference_energies) != expected or
                any(not np.isfinite(float(value)) for value in embedded.values()) or
                any(float(embedded[name]) != float(self.reference_energies[name])
                    for name in expected)):
            raise ValueError("Density reference energies disagree between model and bundle.")
        final = metadata.get("per_species_E0_eV")
        if final is not None and (not isinstance(final, dict) or
                                  set(final) != expected or
                                  any(not np.isfinite(float(value)) for value in final.values()) or
                                  any(float(final[name]) != float(embedded[name])
                                      for name in expected)):
            raise ValueError("Density fitted E0 metadata disagrees with the reference energies.")

    def write(self, path):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before writing it.")
        if self.basis.source == "combined_scalar":
            from .combined_archive import write_combined_scalar_archive

            target = Path(path)
            if not target.suffix:
                target = target.with_suffix(".ye3t")
            if target.suffix != ".ye3t":
                raise ValueError("Combined scalar bundles use a .ye3t path.")
            return write_combined_scalar_archive(self, target)
        if self.basis.source == "legacy_composite":
            raise ValueError("The legacy composite cannot be rewritten as a portable YE3T bundle.")
        target = Path(path)
        if self.basis.source == "tagged_carriers" or (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            tagged = self.basis.source == "tagged_carriers"
            kind = "tagged_full_m_per_atom" if tagged else "density_full_m_per_atom"
            if self._fitted.get("kind") != kind:
                raise ValueError("Full-M artifact requires a matching per-atom fit.")
            if not target.suffix:
                target = target.with_suffix(".ye3t.json")
            if not target.name.endswith(".ye3t.json"):
                raise ValueError("Full-M bundles use a .ye3t.json path.")
            parent = self.basis._resolved["representation"]["parent"]
            convention = {
                "group": "O3", "basis": "real_tesseral_tensor_components",
                "axis_order": "cos_L_to_cos_1_zero_sin_1_to_sin_L",
                "M_values": list(range(-parent["L"], parent["L"] + 1)),
                "L": parent["L"], "parity": parent["parity"],
            }
            if tagged:
                payload = {
                    "schema": "ye3t_methods_tagged_full_m_per_atom_v2",
                    "construction": self.basis._construction,
                    "resolution_sha256": self.basis._resolution.sha256,
                    "compiled_catalogue": self.basis._descriptor.metadata[
                        "tagged_cauchy_carriers_compiled"],
                    "compiler_hash": self.basis._resolved["compiler_hash"],
                    "physical_image_plan_hash": self.basis._tagged_carrier_plan_hash,
                    "selected_coordinate_ids": [label.identity for label in self.basis.labels],
                    "coordinate_convention": convention,
                    "native_property_plan": _tagged_full_m_property_plan(
                        self.basis, self._fitted, convention),
                    "fit": self._fitted,
                }
            else:
                payload = {
                    "schema": "ye3t_methods_density_full_m_per_atom_v2",
                    "construction": self.basis._construction,
                    "resolution_sha256": self.basis._resolution.sha256,
                    "compiled_density": self.basis._density_full_m_compiled_record,
                    "compiler_hash": self.basis._density_full_m_plan_hash,
                    "selected_coordinate_ids": [label.identity for label in self.basis.labels],
                    "coordinate_convention": convention,
                    "real_form_sha256": self.basis._density_full_m_convention_hash,
                    "native_property_plan": _density_full_m_property_plan(
                        self.basis, self._fitted, convention),
                    "validation": _density_full_m_validation(
                        parent["L"], parent["parity"]),
                    "fit": self._fitted,
                }
            payload = _full_m_json_ready(payload)
            payload["self_hash"] = hashlib.sha256(_full_m_canonical_bytes(payload)).hexdigest()
            encoded = _full_m_canonical_bytes(payload)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                             prefix=target.name + ".", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            return target
        if self.basis.source == "portable_linear":
            if not target.suffix:
                target = target.with_suffix(".ye3t")
            if target.suffix != ".ye3t":
                raise ValueError("Portable scalar bundles use a .ye3t path.")
            if "refit" in self._fitted:
                from .portable_refit import write_portable_refit_archive

                return write_portable_refit_archive(self._fitted, target)
            target.write_bytes(self._fitted["archive_bytes"])
            return target
        if not target.suffix:
            suffix = {"density": ".pt", "tagged_cauchy_image": ".ye3t.json", "bar_phi": ".phi.pt"}[self.basis.source]
            target = target.with_suffix(suffix)
        if self.basis.source == "density":
            self._validate_density_reference_energies()
            if target.suffix != ".pt":
                raise ValueError("Density bundles use a .pt path.")
            save_linear_ace_ase_bundle(
                self._fitted, target, cutoff=self.basis.cutoff,
                type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
                reference_energies=self.reference_energies,
            )
        elif self.basis.source == "bar_phi":
            if not target.name.endswith(".phi.pt"):
                raise ValueError("Explicit Phi bundles use a .phi.pt path.")
            save_hybrid_ace_phi_ase_bundle(
                target, self._fitted,
                type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
            )
        else:
            if target.suffix != ".json":
                raise ValueError("Tagged bundles use a .ye3t.json path.")
            self._fitted.export_lammps(target)
        return target

    @classmethod
    def read(cls, path):
        target = Path(path)
        if not target.exists() and not target.suffix:
            candidates = (target.with_suffix(".pt"), target.with_suffix(".ye3t"),
                          target.with_suffix(".ye3t.json"), target.with_suffix(".phi.pt"))
            existing = [candidate for candidate in candidates if candidate.exists()]
            if len(existing) != 1:
                raise FileNotFoundError(f"Expected exactly one model artifact for {target!s}.")
            target = existing[0]
        if target.suffix == ".ye3t":
            from .combined_archive import read_combined_scalar_archive
            from .portable_archive import read_portable_linear_archive
            from .portable_refit import read_portable_refit_archive

            combined = read_combined_scalar_archive(target)
            if combined is not None:
                model = cls(combined["basis"])
                fitted = combined["fitted"]
            else:
                fitted = read_portable_refit_archive(target)
                if fitted is None:
                    fitted = read_portable_linear_archive(target)
            if combined is None and fitted is None:
                fitted = _read_legacy_compat_archive(target)
                model = cls(Basis._from_legacy_composite(fitted))
            elif combined is None:
                model = cls(Basis._from_portable_linear(fitted))
        elif target.name.endswith(".phi.pt"):
            fitted, type_map = load_hybrid_ace_phi_ase_bundle(target)
            if tuple(fitted.branches) != ("bar_phi",):
                raise ValueError("The compact linear Phi reader accepts only the bar_phi branch.")
            model = cls(Basis._from_bar_phi_model(fitted, type_map))
        elif target.suffix == ".pt":
            fitted, cutoff, type_map, refs = load_linear_ace_ase_bundle(target)
            model = cls(Basis._from_density_bundle(fitted, cutoff, type_map), reference_energies=refs)
        elif target.suffix == ".json":
            with target.open("rb") as handle:
                payload_bytes = handle.read(256 * 1024 * 1024 + 1)
            if len(payload_bytes) > 256 * 1024 * 1024:
                raise ValueError("JSON model artifact exceeds the 256 MiB reader limit.")
            payload = json.loads(payload_bytes, object_pairs_hook=_unique_json_object)
            if payload.get("schema") in {"ye3t_methods_density_full_m_per_atom_v1",
                                          "ye3t_methods_density_full_m_per_atom_v2"}:
                return _read_density_full_m_artifact(payload)
            if payload.get("schema") in {"ye3t_methods_tagged_full_m_per_atom_v1",
                                         "ye3t_methods_tagged_full_m_per_atom_v2"}:
                from ye3t import YE3TRepresentation as CoreRepresentation
                from .tesseral_targets import cartesian_tesseral_convention_hash

                allowed = {"schema", "construction", "resolution_sha256",
                           "compiled_catalogue", "compiler_hash",
                           "physical_image_plan_hash", "selected_coordinate_ids",
                           "coordinate_convention", "fit", "self_hash"}
                if payload["schema"] == "ye3t_methods_tagged_full_m_per_atom_v2":
                    allowed.add("native_property_plan")
                if (set(payload) != allowed or not isinstance(payload["self_hash"], str) or
                        hashlib.sha256(_full_m_canonical_bytes({
                            key: value for key, value in payload.items() if key != "self_hash"
                        })).hexdigest() != payload["self_hash"]):
                    raise ValueError("Full-M artifact schema or self-hash mismatch.")
                construction = payload["construction"]
                if not isinstance(construction, dict) or set(construction) != {
                        "basis", "representation", "runtime"}:
                    raise ValueError("Full-M construction record is invalid.")
                representation = CoreRepresentation.from_config(construction["representation"])
                basis = Basis.from_config(construction["basis"],
                                          representation=representation,
                                          runtime=construction["runtime"])
                if basis._resolution.sha256 != payload["resolution_sha256"]:
                    raise ValueError("Full-M basis resolution hash mismatch.")
                basis._embedded_tagged_carriers = payload["compiled_catalogue"]
                basis._materialize_configured()
                if (basis.source != "tagged_carriers" or
                        basis._resolved["compiler_hash"] != payload["compiler_hash"] or
                        basis._tagged_carrier_plan_hash != payload["physical_image_plan_hash"] or
                        [label.identity for label in basis.labels] !=
                        payload["selected_coordinate_ids"]):
                    raise ValueError("Full-M compiler plan or selected coordinates changed.")
                parent = basis._resolved["representation"]["parent"]
                expected_convention = {
                    "group": "O3", "basis": "real_tesseral_tensor_components",
                    "axis_order": "cos_L_to_cos_1_zero_sin_1_to_sin_L",
                    "M_values": list(range(-parent["L"], parent["L"] + 1)),
                    "L": parent["L"], "parity": parent["parity"],
                }
                if payload["coordinate_convention"] != expected_convention:
                    raise ValueError("Full-M Cartesian/tesseral coordinate convention changed.")
                fitted = payload["fit"]
                width = len(basis.elements) * len(basis.labels)
                beta = np.asarray(fitted.get("beta"), dtype=np.float64)
                metadata = fitted.get("fit_metadata")
                expected_columns = [{"central_species": species,
                                     "coordinate_id": label.identity}
                                    for species in basis.elements for label in basis.labels]
                metric = (metadata.get("coordinate_penalty_metric")
                          if isinstance(metadata, dict) else None)
                if (fitted.get("kind") != "tagged_full_m_per_atom" or
                        beta.shape != (width,) or not np.isfinite(beta).all() or
                        not isinstance(metadata, dict) or
                        metadata.get("design_column_order") !=
                        "central_species_then_selected_multiplet_shared_over_M" or
                        metadata.get("fit_method") not in {
                            "ridge", "lasso", "ardregression"} or
                        not isinstance(metric, dict) or
                        metric.get("schema") != "selected_compiler_coordinate_euclidean_v1" or
                        metric.get("columns") != expected_columns or
                        metric.get("diagonal") != [1.0] * width or
                        metric.get("physical_image_plan_hash") !=
                        basis._tagged_carrier_plan_hash or
                        metric.get("interpretation") !=
                        "coefficient_norm_in_saved_selected_coordinates" or
                        metadata.get("n_cols") != width or
                        not isinstance(metadata.get("n_rows"), int) or
                        metadata["n_rows"] < 1 or
                        metadata.get("target_input") not in {
                            "real_tesseral", "cartesian"} or
                        not isinstance(metadata.get("target_units"), str) or
                        not metadata["target_units"]):
                    raise ValueError("Full-M fitted columns or coefficients changed.")
                if (payload["schema"] == "ye3t_methods_tagged_full_m_per_atom_v2" and
                        payload["native_property_plan"] !=
                        _tagged_full_m_property_plan(basis, fitted, expected_convention)):
                    raise ValueError("Full-M native property plan differs from compiler and fit.")
                fit_request = metadata.get("resolved_fit_config")
                if (not isinstance(fit_request, dict) or
                        hashlib.sha256(json.dumps(
                            fit_request, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")).hexdigest() !=
                        metadata.get("resolved_fit_config_sha256") or
                        fit_request.get("construction_resolution_sha256") !=
                        basis._resolution.sha256 or
                        fit_request.get("targets", {}).get("per_atom", {}).get(
                            "input") != metadata["target_input"]):
                    raise ValueError("Full-M saved fit request hash or target changed.")
                posterior = metadata.get("predictive_uncertainty")
                if metadata["fit_method"] == "ardregression":
                    if not isinstance(posterior, dict) or posterior.get("schema") != (
                            "ye3t_linear_ard_posterior_v1") or posterior.get(
                            "design_column_order") != (
                            "central_species_then_selected_multiplet_shared_over_M"):
                        raise ValueError("Full-M ARD posterior contract changed.")
                    active = np.asarray(posterior.get("active_column_indices"), dtype=int)
                    covariance = np.asarray(posterior.get(
                        "coefficient_covariance_active"), dtype=np.float64)
                    if (active.ndim != 1 or len(set(active.tolist())) != len(active) or
                            np.any(active < 0) or np.any(active >= width) or
                            covariance.shape != (len(active), len(active)) or
                            not np.isfinite(covariance).all() or
                            not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10) or
                            (covariance.size and
                             np.min(np.linalg.eigvalsh(covariance)) < -1e-10)):
                        raise ValueError("Full-M ARD coefficient covariance is invalid.")
                elif posterior is not None:
                    raise ValueError("Full-M non-ARD fit has an unexpected posterior.")
                if metadata.get("target_input") == "cartesian" and (
                        metadata.get("cartesian_tesseral_convention_sha256") !=
                        cartesian_tesseral_convention_hash(parent["L"], parent["parity"])):
                    raise ValueError("Full-M Cartesian conversion hash mismatch.")
                fitted["beta"] = beta
                model = cls(basis)
            elif payload.get("schema") == "ye3t_tagged_cauchy_composite_v1":
                fitted = _read_legacy_composite(target)
                model = cls(Basis._from_legacy_composite(fitted))
            else:
                fitted = load_tagged_cauchy_image_model(target)
                model = cls(Basis._from_tagged_model(fitted))
        else:
            raise ValueError("Expected a .pt, .phi.pt, .ye3t, or .ye3t.json model.")
        model._fitted = fitted
        model._validate_density_reference_energies()
        return model

    def export_lammps(self, path):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before export.")
        if self.basis.source == "tagged_carriers" or (
                self.basis.source == "density" and
                getattr(self.basis, "_density_full_m", False)):
            raise ValueError("Full-M per-atom models have no validated LAMMPS exporter.")
        if self.basis.source == "legacy_composite":
            raise ValueError("Use the original verified composite artifact for LAMMPS.")
        if self.basis.source == "portable_linear":
            raise ValueError("Portable Ni scalar bundles have no validated LAMMPS export.")
        if self.basis.source == "combined_scalar":
            raise ValueError("Combined scalar bundles have no validated joint LAMMPS export.")
        if self.basis.source == "bar_phi":
            raise ValueError("The explicit Phi reference model has no LAMMPS export schema.")
        if self.basis.source == "density":
            metadata = self._fitted.fit_metadata or {}
            if self.reference_energies and "reference_energy_targets" not in metadata:
                self._fitted.fit_metadata = {
                    **metadata,
                    "reference_energy_targets": {
                        "enabled": True,
                        "reference_energies": dict(self.reference_energies),
                        "target_convention": "E_target = E_raw - sum_type(n_type * E_ref[type])",
                    },
                }
            self._validate_density_reference_energies()
            convention = (self._fitted.fit_metadata or {}).get(
                "ye3t_methods_public_label_convention", {})
            if (any(str(spec.key).endswith("|physical_eta_bound")
                    for spec in self._fitted.descriptor_specs) or
                (len(self.basis.elements) > 1 and
                 convention.get("schema") in {
                     "ye3t_methods_density_pace_labels_v2",
                     "ye3t_methods_density_pace_labels_v3"})):
                raise ValueError("Configured multi-species density has no validated LAMMPS/YACE export.")
            if getattr(self._fitted.site_basis_config, "chemical_basis", "delta") == (
                    "fixed_embedding"):
                raise ValueError("Fixed chemical embeddings have no validated LAMMPS/YACE export.")
            return self._fitted.export_lammps(path, elements=self.basis.elements)
        target = Path(path)
        self._fitted.export_lammps(target)
        return target

    def describe(self, index, format="text"):
        label = self.labels[int(index)]
        if format == "latex":
            return label.latex()
        detail = self.basis.describe(index, format=format)
        if self._fitted is None:
            return detail + "\ncoefficient: unfitted"
        if self.basis.source == "portable_linear":
            weights = self._fitted["weights"]
            ordinary_count = len(weights["ordinary"])
            coefficient = float(weights["ordinary"][int(index)] if int(index) < ordinary_count
                                else weights["tagged_selected"][int(index) - ordinary_count])
        elif self.basis.source == "density":
            coefficient = float(np.asarray(
                self._fitted["beta"] if getattr(self.basis, "_density_full_m", False)
                else self._fitted.weight)[int(index)])
        elif self.basis.source == "bar_phi":
            coefficient = float(self._fitted.bar_phi_weight.detach().cpu()[int(index)])
        else:
            coefficient = {
                name: float(self._fitted.beta_by_species[name][int(index)])
                for name in self.basis.elements
            }
        return detail + f"\ncoefficient: {coefficient}"

    def __str__(self):
        state = "fitted" if self._fitted is not None else "unfitted"
        if self.basis.source == "legacy_composite":
            return (f"LinearModel(source=legacy_composite, state={state}, "
                    f"features={self.basis._resolved['feature_count']}, "
                    f"elements={self.basis.elements}, labels=unavailable)")
        head = (
            f"LinearModel(source={self.basis.source}, state={state}, "
            f"features={len(self.labels)}, elements={self.basis.elements})"
        )
        rows = [str(label) for label in self.labels[:5]]
        if len(self.labels) > 5:
            rows.append(f"... {len(self.labels) - 5} rows omitted")
        return "\n".join((head, *rows))

    __repr__ = __str__
