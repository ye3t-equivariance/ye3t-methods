/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "ye3t_tagged_cauchy_model.h"

#include "ye3t_canonical_json_hash.h"
#include "ye3t_sha256.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace YE3T_LAMMPS {
namespace {

  constexpr std::uintmax_t MAX_MODEL_BYTES = 512ULL * 1024ULL * 1024ULL;

  [[noreturn]] void fail(const std::string &path, const std::string &message)
  {
    throw std::runtime_error(path + ": " + message);
  }

  void require_mapping(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsMap()) fail(path, "expected a mapping");
  }

  void require_sequence(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsSequence()) fail(path, "expected a sequence");
  }

  std::string scalar_string(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsScalar()) fail(path, "expected a string");
    try {
      return node.as<std::string>();
    } catch (const YAML::Exception &) {
      fail(path, "expected a string");
    }
  }

  std::int64_t integer64(const YAML::Node &node, const std::string &path, std::int64_t minimum,
                         std::int64_t maximum = std::numeric_limits<std::int64_t>::max())
  {
    if (!node || !node.IsScalar()) fail(path, "expected an integer");
    try {
      const long long value = node.as<long long>();
      if (value < minimum || value > maximum) fail(path, "integer is outside the supported range");
      return static_cast<std::int64_t>(value);
    } catch (const YAML::Exception &) {
      fail(path, "expected an integer");
    }
  }

  int integer(const YAML::Node &node, const std::string &path, int minimum,
              int maximum = std::numeric_limits<int>::max())
  {
    return static_cast<int>(integer64(node, path, minimum, maximum));
  }

  double finite_number(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsScalar()) fail(path, "expected a finite number");
    try {
      const double value = node.as<double>();
      if (!std::isfinite(value)) fail(path, "expected a finite number");
      return value;
    } catch (const YAML::Exception &) {
      fail(path, "expected a finite number");
    }
  }

  bool boolean_value(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsScalar()) fail(path, "expected a boolean");
    try {
      return node.as<bool>();
    } catch (const YAML::Exception &) {
      fail(path, "expected a boolean");
    }
  }

  bool lower_sha256(const std::string &value)
  {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
             return (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
           });
  }

  std::string sha256_field(const YAML::Node &node, const std::string &path)
  {
    const std::string value = scalar_string(node, path);
    if (!lower_sha256(value)) fail(path, "expected a lowercase SHA-256 digest");
    return value;
  }

  std::complex<double> binary64_complex(const YAML::Node &node, const std::string &path)
  {
    require_mapping(node, path);
    const YAML::Node values = node["binary64"];
    require_sequence(values, path + ".binary64");
    if (values.size() != 2)
      fail(path + ".binary64", "complex binary64 value must have two entries");
    const double real = finite_number(values[0], path + ".binary64[0]");
    const double imag = finite_number(values[1], path + ".binary64[1]");
    return {real, imag};
  }

  YAML::Node verified_binding(const YAML::Node &model, const std::string &model_path,
                              const std::string &member)
  {
    const YAML::Node binding = model[member];
    const std::string path = "model." + member;
    require_mapping(binding, path);
    if (binding.size() != 2 || !binding["payload"] || !binding["hash"])
      fail(path, "binding must contain exactly payload and hash");
    const std::string expected = sha256_field(binding["hash"], path + ".hash");
    const std::string payload_json =
        canonical_json_nested_member_value(model_path, member, "payload");
    if (sha256_string(payload_json) != expected) fail(path, "payload hash mismatch");
    require_mapping(binding["payload"], path + ".payload");
    return binding["payload"];
  }

  std::string json_string_value(const std::string &json, const std::string &path)
  {
    try {
      return scalar_string(YAML::Load(json), path);
    } catch (const YAML::Exception &) {
      fail(path, "expected a JSON string");
    }
  }

  // Solve `matrix * inverse = identity` for a small dense complex matrix by
  // Gauss-Jordan elimination with partial pivoting (row-major, width x width).
  // The artifact's real-form matrices are unitary (small l), so this is both
  // well-conditioned and cheap; the caller verifies the result independently.
  std::vector<std::complex<double>>
  invert_complex_matrix(const std::vector<std::complex<double>> &matrix, int width,
                        const std::string &path)
  {
    std::vector<std::complex<double>> work = matrix;
    std::vector<std::complex<double>> inverse(static_cast<std::size_t>(width) *
                                                  static_cast<std::size_t>(width),
                                              std::complex<double>(0.0, 0.0));
    for (int i = 0; i < width; ++i) inverse[static_cast<std::size_t>(i) * width + i] = 1.0;

    for (int column = 0; column < width; ++column) {
      int pivot_row = column;
      double pivot_magnitude = std::abs(work[static_cast<std::size_t>(column) * width + column]);
      for (int row = column + 1; row < width; ++row) {
        const double magnitude = std::abs(work[static_cast<std::size_t>(row) * width + column]);
        if (magnitude > pivot_magnitude) {
          pivot_magnitude = magnitude;
          pivot_row = row;
        }
      }
      if (pivot_magnitude < 1.0e-300)
        fail(path, "real_to_complex_matrix is singular to working precision");
      if (pivot_row != column) {
        for (int col = 0; col < width; ++col) {
          std::swap(work[static_cast<std::size_t>(column) * width + col],
                    work[static_cast<std::size_t>(pivot_row) * width + col]);
          std::swap(inverse[static_cast<std::size_t>(column) * width + col],
                    inverse[static_cast<std::size_t>(pivot_row) * width + col]);
        }
      }
      const std::complex<double> pivot = work[static_cast<std::size_t>(column) * width + column];
      for (int col = 0; col < width; ++col) {
        work[static_cast<std::size_t>(column) * width + col] /= pivot;
        inverse[static_cast<std::size_t>(column) * width + col] /= pivot;
      }
      for (int row = 0; row < width; ++row) {
        if (row == column) continue;
        const std::complex<double> factor = work[static_cast<std::size_t>(row) * width + column];
        if (factor == std::complex<double>(0.0, 0.0)) continue;
        for (int col = 0; col < width; ++col) {
          work[static_cast<std::size_t>(row) * width + col] -=
              factor * work[static_cast<std::size_t>(column) * width + col];
          inverse[static_cast<std::size_t>(row) * width + col] -=
              factor * inverse[static_cast<std::size_t>(column) * width + col];
        }
      }
    }
    return inverse;
  }

  enum class BinaryProductFactorization {
    CommutativeSymmetricPower,
    CanonicalPrefix,
  };

  const char *binary_product_factorization_name(BinaryProductFactorization factorization)
  {
    if (factorization == BinaryProductFactorization::CanonicalPrefix) return "canonical_prefix";
    return "commutative_symmetric_power";
  }

  TaggedCauchyBinaryProductPlan
  parse_binary_product_plan(const YAML::Node &node, const std::string &path,
                            int expected_base_count, std::size_t expected_root_count,
                            BinaryProductFactorization expected_factorization,
                            bool legacy_implicit_factorization)
  {
    require_mapping(node, path);
    if (!legacy_implicit_factorization &&
        scalar_string(node["factorization"], path + ".factorization") !=
            binary_product_factorization_name(expected_factorization))
      fail(path + ".factorization", "binary-plan factorization changed");
    TaggedCauchyBinaryProductPlan result;
    result.base_count = integer(node["base_count"], path + ".base_count", 0);
    if (result.base_count != expected_base_count)
      fail(path + ".base_count", "binary-plan base count changed");
    require_sequence(node["nodes"], path + ".nodes");
    for (std::size_t index = 0; index < node["nodes"].size(); ++index) {
      const std::string node_path = path + ".nodes[" + std::to_string(index) + "]";
      require_mapping(node["nodes"][index], node_path);
      const int maximum_value = result.base_count + static_cast<int>(index) - 1;
      TaggedCauchyBinaryNode entry;
      entry.left_value =
          integer(node["nodes"][index]["left_value"], node_path + ".left_value", 0, maximum_value);
      entry.right_value = integer(node["nodes"][index]["right_value"], node_path + ".right_value",
                                  0, maximum_value);
      result.nodes.push_back(entry);
    }
    require_sequence(node["roots"], path + ".roots");
    if (node["roots"].size() != expected_root_count)
      fail(path + ".roots", "binary-plan root count changed");
    const int maximum_root = result.base_count + static_cast<int>(result.nodes.size()) - 1;
    for (std::size_t index = 0; index < node["roots"].size(); ++index)
      result.roots.push_back(integer(
          node["roots"][index], path + ".roots[" + std::to_string(index) + "]", -1, maximum_root));
    result.direct_multiplication_count =
        integer64(node["direct_multiplication_count"], path + ".direct_multiplication_count", 0);
    result.binary_node_count = integer64(node["binary_node_count"], path + ".binary_node_count", 0);
    if (result.binary_node_count != static_cast<std::int64_t>(result.nodes.size()))
      fail(path + ".binary_node_count", "binary-plan node count changed");
    result.maximum_degree = integer(node["maximum_degree"], path + ".maximum_degree", 0);
    result.repeated_factor_product_count = integer64(node["repeated_factor_product_count"],
                                                     path + ".repeated_factor_product_count", 0);
    if (integer(node["division_operations"], path + ".division_operations", 0) != 0)
      fail(path + ".division_operations", "division-free plan uses division");
    const YAML::Node certificate = node["certificate"];
    require_mapping(certificate, path + ".certificate");
    for (const char *key : {"passed", "all_root_exponents_exact", "division_free_reverse"})
      if (!boolean_value(certificate[key], path + ".certificate." + key))
        fail(path + ".certificate." + key, "binary-plan certificate failed");
    if (legacy_implicit_factorization ||
        expected_factorization == BinaryProductFactorization::CommutativeSymmetricPower) {
      for (const char *key :
           {"hash_consed_commutative_nodes", "repeated_factors_use_symmetric_power_nodes"})
        if (!boolean_value(certificate[key], path + ".certificate." + key))
          fail(path + ".certificate." + key, "symmetric-power plan certificate failed");
    } else {
      for (const char *key :
           {"hash_consed_prefix_nodes", "preserves_canonical_left_to_right_order"})
        if (!boolean_value(certificate[key], path + ".certificate." + key))
          fail(path + ".certificate." + key, "canonical-prefix plan certificate failed");
      if (boolean_value(certificate["repeated_factors_use_symmetric_power_nodes"],
                        path + ".certificate.repeated_factors_use_symmetric_power_nodes"))
        fail(path + ".certificate.repeated_factors_use_symmetric_power_nodes",
             "canonical-prefix plan must not claim symmetric-power nodes");
    }
    return result;
  }

  std::vector<std::vector<int>>
  binary_plan_value_exponents(const TaggedCauchyBinaryProductPlan &plan, const std::string &path)
  {
    const std::size_t base_count = static_cast<std::size_t>(plan.base_count);
    std::vector<std::vector<int>> exponents(base_count + plan.nodes.size(),
                                            std::vector<int>(base_count, 0));
    for (std::size_t base = 0; base < base_count; ++base) exponents[base][base] = 1;
    for (std::size_t index = 0; index < plan.nodes.size(); ++index) {
      const TaggedCauchyBinaryNode &node = plan.nodes[index];
      std::vector<int> &output = exponents[base_count + index];
      const std::vector<int> &left = exponents[static_cast<std::size_t>(node.left_value)];
      const std::vector<int> &right = exponents[static_cast<std::size_t>(node.right_value)];
      for (std::size_t base = 0; base < base_count; ++base) {
        if (left[base] > std::numeric_limits<int>::max() - right[base])
          fail(path, "binary-plan exponent overflow");
        output[base] = left[base] + right[base];
      }
    }
    return exponents;
  }

  std::vector<int> binary_plan_root_exponent(const TaggedCauchyBinaryProductPlan &plan,
                                             const std::vector<std::vector<int>> &value_exponents,
                                             int root)
  {
    if (root < 0) return std::vector<int>(static_cast<std::size_t>(plan.base_count), 0);
    return value_exponents[static_cast<std::size_t>(root)];
  }

}    // namespace

TaggedCauchyModel TaggedCauchyModel::load(const std::string &supplied_path)
{
  namespace fs = std::filesystem;
  const fs::path model_path(supplied_path);
  std::error_code error;
  const std::uintmax_t size = fs::file_size(model_path, error);
  if (error) fail(supplied_path, "could not determine file size");
  if (size == 0 || size > MAX_MODEL_BYTES)
    fail(supplied_path, "file size is outside the supported bound");

  YAML::Node model;
  try {
    const std::string schema_hint = json_string_value(
        canonical_json_root_member_value(supplied_path, "schema"), "model.schema");
    if (schema_hint == "ye3t_tagged_cauchy_slice_v4") {
      // V4 carries a large offline proof (parents, formal labels and exact
      // reconstruction rows). Inference never traverses it. Materialize only
      // the fields consumed below; every checksum still uses the complete
      // original JSON, including that proof, through the canonical reader.
      model = YAML::Node(YAML::NodeType::Map);
      for (const char *member : {"schema", "self_hash", "compiler_artifact_hash",
                                "deployment_identity_hash", "conventions",
                                "source_binding", "schedule_binding", "readout_binding",
                                "tagged_execution_portfolio"})
        model[member] = YAML::Load(canonical_json_root_member_value(supplied_path, member));
      const std::string compiler_json =
          canonical_json_root_member_value(supplied_path, "compiler_artifact");
      const std::string compiler_payload_json =
          canonical_json_value_root_member(compiler_json, "payload");
      YAML::Node payload(YAML::NodeType::Map);
      for (const char *member : {"catalogue_hash", "real_schedule_core_hash",
                                "moment_schedules", "source_product_algebra"})
        payload[member] = YAML::Load(canonical_json_value_root_member(compiler_payload_json, member));
      model["compiler_artifact"]["payload"] = payload;
    } else {
      model = YAML::LoadFile(model_path.string());
    }
  } catch (const YAML::Exception &exception) {
    fail(supplied_path, std::string("invalid JSON: ") + exception.what());
  }
  require_mapping(model, "model");

  const std::string schema = scalar_string(model["schema"], "model.schema");
  if (schema == "ye3t_tagged_cauchy_composite_v1") {
    const std::string composite_self_hash = sha256_field(model["self_hash"], "model.self_hash");
    if (canonical_json_hash_without_root_member(model_path.string(), "self_hash") !=
        composite_self_hash)
      fail(model_path.string(), "composite model self-hash mismatch");

    const auto component = [&](const char *member) {
      const YAML::Node binding = model[member];
      const std::string binding_path = std::string("model.") + member;
      require_mapping(binding, binding_path);
      if (binding.size() != 2 || !binding["path"] || !binding["sha256"])
        fail(binding_path, "component must contain exactly path and sha256");
      const fs::path relative_path(scalar_string(binding["path"], binding_path + ".path"));
      if (relative_path.empty() || relative_path.is_absolute() || relative_path.has_parent_path())
        fail(binding_path + ".path", "component must be a filename beside the composite manifest");
      const fs::path resolved = model_path.parent_path() / relative_path;
      const std::string expected = sha256_field(binding["sha256"], binding_path + ".sha256");
      std::string actual;
      try {
        actual = sha256_file(resolved.string());
      } catch (const std::exception &exception) {
        fail(binding_path + ".path", exception.what());
      }
      if (actual != expected) fail(binding_path, "component byte hash mismatch");
      return std::make_pair(resolved, expected);
    };

    const auto ordinary = component("ordinary_component");
    const auto tagged = component("tagged_component");
    TaggedCauchyModel result = TaggedCauchyModel::load(tagged.first.string());

    require_sequence(model["species_order"], "model.species_order");
    if (model["species_order"].size() != result.species_order.size())
      fail("model.species_order", "composite and tagged component species counts differ");
    for (std::size_t index = 0; index < result.species_order.size(); ++index) {
      const std::string species = scalar_string(
          model["species_order"][index], "model.species_order[" + std::to_string(index) + "]");
      if (species != result.species_order[index])
        fail("model.species_order", "composite and tagged component species ordering differs");
    }

    result.composite_manifest_path = supplied_path;
    result.composite_self_hash = composite_self_hash;
    result.ordinary_model_path = ordinary.first.string();
    result.ordinary_model_hash = ordinary.second;
    result.tagged_component_hash = tagged.second;
    return result;
  }
  const bool physical_image_v4 = schema == "ye3t_tagged_cauchy_slice_v4";
  const bool physical_image_v3 = schema == "ye3t_tagged_cauchy_slice_v3" || physical_image_v4;
  if (!physical_image_v3 && schema != "ye3t_tagged_cauchy_slice_v2")
    fail("model.schema", "unsupported tagged-Cauchy schema");

  const std::string self_hash = sha256_field(model["self_hash"], "model.self_hash");
  if (canonical_json_hash_without_root_member(model_path.string(), "self_hash") != self_hash)
    fail(model_path.string(), "model self-hash mismatch");

  TaggedCauchyModel result;
  result.model_path = supplied_path;
  result.self_hash = self_hash;
  result.deployment_kind = physical_image_v3 ? TaggedCauchyDeploymentKind::PhysicalImageV3
                                             : TaggedCauchyDeploymentKind::LegacyMomentV2;
  if (physical_image_v4) result.deployment_kind = TaggedCauchyDeploymentKind::PhysicalImageV4;
  const YAML::Node algebra = physical_image_v4 && model["compiler_artifact"]["payload"]["source_product_algebra"]
      ? model["compiler_artifact"]["payload"]["source_product_algebra"]
      : (physical_image_v3 ? model["compiler_artifact"]["plan"]["report"]["request"]["source_product_algebra"]
                           : YAML::Node());

  YAML::Node source;
  YAML::Node program;
  YAML::Node readout;
  if (physical_image_v3) {
    source = verified_binding(model, supplied_path, "source_binding");
    program = verified_binding(model, supplied_path, "schedule_binding");
    readout = verified_binding(model, supplied_path, "readout_binding");

    result.compiler_artifact_hash =
        sha256_field(model["compiler_artifact_hash"], "model.compiler_artifact_hash");
    result.source_plan_hash =
        sha256_field(source["source_plan_hash"], "model.source_binding.payload.source_plan_hash");
    result.schedule_hash =
        sha256_field(program["program_hash"], "model.schedule_binding.payload.program_hash");
    result.readout_hash =
        sha256_field(model["readout_binding"]["hash"], "model.readout_binding.hash");
    result.deployment_identity_hash =
        sha256_field(model["deployment_identity_hash"], "model.deployment_identity_hash");

    const std::string compiler_json =
        canonical_json_root_member_value(supplied_path, "compiler_artifact");
    const std::string compiler_self =
        json_string_value(canonical_json_value_root_member(compiler_json, "self_hash"),
                          "model.compiler_artifact.self_hash");
    if (canonical_json_value_hash_without_root_member(compiler_json, "self_hash") != compiler_self)
      fail("model.compiler_artifact", "compiler artifact self-hash mismatch");
    if (compiler_self != result.compiler_artifact_hash)
      fail("model.compiler_artifact_hash", "compiler artifact hash does not match its payload");

    const std::string source_json =
        canonical_json_nested_member_value(supplied_path, "source_binding", "payload");
    if (canonical_json_value_hash_without_root_member(source_json, "source_plan_hash") !=
        result.source_plan_hash)
      fail("model.source_binding.payload", "source-plan self-hash mismatch");
    const std::string schedule_json =
        canonical_json_nested_member_value(supplied_path, "schedule_binding", "payload");
    if (canonical_json_value_hash_without_root_member(schedule_json, "program_hash") !=
        result.schedule_hash)
      fail("model.schedule_binding.payload", "schedule self-hash mismatch");

    const YAML::Node compiler_payload = model["compiler_artifact"]["payload"];
    require_mapping(compiler_payload, "model.compiler_artifact.payload");
    const std::string compiler_payload_json =
        canonical_json_value_root_member(compiler_json, "payload");
    const std::string committed_schedule_hash =
        sha256_field(compiler_payload["real_schedule_core_hash"],
                     "model.compiler_artifact.payload.real_schedule_core_hash");
    const std::string compiler_schedule_core_json =
        canonical_json_value_root_member(compiler_payload_json, "real_schedule_core");
    if (sha256_string(compiler_schedule_core_json) != committed_schedule_hash)
      fail("model.compiler_artifact.payload.real_schedule_core",
           "compiler real-schedule commitment hash mismatch");
    if (result.schedule_hash != committed_schedule_hash)
      fail("model.schedule_binding.payload.program_hash",
           "schedule differs from its compiler commitment");
    if (sha256_field(program["catalogue_hash"], "model.schedule_binding.payload.catalogue_hash") !=
        sha256_field(compiler_payload["catalogue_hash"],
                     "model.compiler_artifact.payload.catalogue_hash"))
      fail("model.schedule_binding.payload.catalogue_hash", "schedule catalogue binding changed");
    require_sequence(program["compiler_schedule_hashes"],
                     "model.schedule_binding.payload.compiler_schedule_hashes");
    require_sequence(compiler_payload["moment_schedules"],
                     "model.compiler_artifact.payload.moment_schedules");
    if (program["compiler_schedule_hashes"].size() != compiler_payload["moment_schedules"].size())
      fail("model.schedule_binding.payload.compiler_schedule_hashes",
           "compiler schedule count changed");
    for (std::size_t index = 0; index < program["compiler_schedule_hashes"].size(); ++index) {
      const std::string supplied = sha256_field(
          program["compiler_schedule_hashes"][index],
          "model.schedule_binding.payload.compiler_schedule_hashes[" + std::to_string(index) + "]");
      const std::string expected =
          sha256_field(compiler_payload["moment_schedules"][index]["schedule_hash"],
                       "model.compiler_artifact.payload.moment_schedules[" + std::to_string(index) +
                           "].schedule_hash");
      if (supplied != expected)
        fail("model.schedule_binding.payload.compiler_schedule_hashes",
             "compiler schedule ordering changed");
    }

    const YAML::Node source_algebra = algebra;
    require_mapping(source_algebra,
                    "model.compiler_artifact.plan.report.request.source_product_algebra");
    if (sha256_field(source["source_product_algebra_hash"],
                     "model.source_binding.payload.source_product_algebra_hash") !=
        sha256_field(source_algebra["record_hash"],
                     "model.compiler_artifact.plan.report.request."
                     "source_product_algebra.record_hash"))
      fail("model.source_binding.payload.source_product_algebra_hash",
           "source-product algebra binding changed");

    const std::string compiler_forms_json =
        canonical_json_value_root_member(compiler_payload_json, "real_forms");
    const std::string source_forms_json =
        canonical_json_value_root_member(source_json, "real_forms");
    if (compiler_forms_json != source_forms_json)
      fail("model.source_binding.payload.real_forms",
           "source real forms differ from the compiler artifact");
    if (sha256_field(source["compiler_artifact_hash"],
                     "model.source_binding.payload.compiler_artifact_hash") !=
        result.compiler_artifact_hash)
      fail("model.compiler_artifact_hash", "source compiler binding changed");

    const std::string conventions_json =
        canonical_json_root_member_value(supplied_path, "conventions");
    const std::string identity_json = "{\"compiler_artifact_hash\":\"" +
        result.compiler_artifact_hash + "\",\"conventions\":" + conventions_json +
        ",\"readout_hash\":\"" + result.readout_hash + "\",\"schedule_hash\":\"" +
        scalar_string(model["schedule_binding"]["hash"], "model.schedule_binding.hash") +
        "\",\"source_plan_hash\":\"" +
        scalar_string(model["source_binding"]["hash"], "model.source_binding.hash") + "\"}";
    if (sha256_string(identity_json) != result.deployment_identity_hash)
      fail("model.deployment_identity_hash", "deployment identity mismatch");

    const YAML::Node conventions = model["conventions"];
    require_mapping(conventions, "model.conventions");
    if (conventions.size() != 6)
      fail("model.conventions", "expected the complete V3 convention record");
    if (scalar_string(conventions["precision"], "model.conventions.precision") != "binary64" ||
        scalar_string(conventions["edge_displacement"], "model.conventions.edge_displacement") !=
            "R_neighbor_minus_R_center_plus_periodic_image" ||
        scalar_string(conventions["force_sign"], "model.conventions.force_sign") !=
            "F_equals_minus_dE_dR" ||
        scalar_string(conventions["lammps_virial"], "model.conventions.lammps_virial") !=
            "minus_strain_derivative" ||
        scalar_string(conventions["real_basis"], "model.conventions.real_basis") !=
            "compiler_derived_orthonormal_real_tesseral" ||
        scalar_string(conventions["exact_zero_distance_force_policy"],
                      "model.conventions.exact_zero_distance_force_policy") !=
            "reject_before_direction_evaluation_v1")
      fail("model.conventions", "unsupported native V3 convention");
  } else {
    source = model;
    program = model["real_moment_program"];
    readout = model;
  }

  // species_order
  {
    const std::string path = physical_image_v3 ? "model.source_binding.payload.species_order"
                                               : "model.species_order";
    require_sequence(source["species_order"], path);
    const YAML::Node node = source["species_order"];
    for (std::size_t index = 0; index < node.size(); ++index)
      result.species_order.push_back(
          scalar_string(node[index], path + "[" + std::to_string(index) + "]"));
    if (result.species_order.empty() ||
        std::set<std::string>(result.species_order.begin(), result.species_order.end()).size() !=
            result.species_order.size())
      fail(path, "species must be nonempty and unique");
  }
  const auto species_index = [&](const std::string &name) -> int {
    const auto found = std::find(result.species_order.begin(), result.species_order.end(), name);
    if (found == result.species_order.end()) return -1;
    return static_cast<int>(found - result.species_order.begin());
  };

  result.cutoff = finite_number(
      source["cutoff"], physical_image_v3 ? "model.source_binding.payload.cutoff" : "model.cutoff");
  if (result.cutoff <= 0.0) fail("model.cutoff", "cutoff must be positive");
  if (physical_image_v4) {
    const std::size_t species_count = result.species_order.size();
    result.pair_cutoffs.assign(species_count * species_count, result.cutoff);
    if (source["pair_cutoffs_A"]) {
      const YAML::Node pairs = source["pair_cutoffs_A"];
      require_mapping(pairs, "source.pair_cutoffs_A");
      if (pairs.size() != species_count * species_count)
        fail("source.pair_cutoffs_A", "expected every directed species pair");
      for (std::size_t left = 0; left < species_count; ++left)
        for (std::size_t right = 0; right < species_count; ++right) {
          const std::string key = result.species_order[left] + "-" + result.species_order[right];
          const double value = finite_number(pairs[key], "source.pair_cutoffs_A." + key);
          if (value <= 0.0 || value > result.cutoff)
            fail("source.pair_cutoffs_A", "pair cutoff must be positive and within host cutoff");
          result.pair_cutoffs[left * species_count + right] = value;
        }
    }
  } else if (source["pair_cutoffs_A"]) {
    fail("source.pair_cutoffs_A", "pair-specific sources require a V4 deployment");
  }

  result.tag_count = integer(
      physical_image_v3 ? program["tag_count"] : model["tag_count"],
      physical_image_v3 ? "model.schedule_binding.payload.tag_count" : "model.tag_count", 0);

  if (physical_image_v3) {
    const YAML::Node source_algebra = algebra;
    require_mapping(source_algebra,
                    "model.compiler_artifact.plan.report.request.source_product_algebra");
    require_sequence(readout["species_order"], "model.readout_binding.payload.species_order");
    if (readout["species_order"].size() != result.species_order.size())
      fail("model.readout_binding.payload.species_order",
           "readout/source species ordering differs");
    for (std::size_t index = 0; index < result.species_order.size(); ++index)
      if (scalar_string(readout["species_order"][index],
                        "model.readout_binding.payload.species_order[*]") !=
          result.species_order[index])
        fail("model.readout_binding.payload.species_order",
             "readout/source species ordering differs");
    const std::string source_schema =
        scalar_string(source["schema"], "model.source_binding.payload.schema");
    if (source_schema == "ye3t_tagged_cauchy_direct_source_v1") {
      result.source_realization = TaggedCauchySourceRealization::ExpandedPowerHornerV1;
      if (source["numerical_evaluation"])
        fail("model.source_binding.payload.numerical_evaluation",
             "V1 source plan must not contain V2 numerical metadata");
    } else if (source_schema == "ye3t_tagged_cauchy_direct_source_v2") {
      result.source_realization = TaggedCauchySourceRealization::ShiftedJacobiThreeTermV1;
      const YAML::Node evaluation = source["numerical_evaluation"];
      require_mapping(evaluation, "model.source_binding.payload.numerical_evaluation");
      if (evaluation.size() != 7 ||
          scalar_string(evaluation["schema"],
                        "model.source_binding.payload.numerical_evaluation.schema") !=
              "ye3t_shifted_jacobi_three_term_v1" ||
          integer(evaluation["alpha"], "model.source_binding.payload.numerical_evaluation.alpha",
                  0) != 4 ||
          scalar_string(evaluation["beta_rule"],
                        "model.source_binding.payload.numerical_evaluation.beta_rule") != "2*l+2" ||
          scalar_string(evaluation["argument"],
                        "model.source_binding.payload.numerical_evaluation.argument") != "2*x-1" ||
          scalar_string(evaluation["derivative"],
                        "model.source_binding.payload.numerical_evaluation.derivative") != "d/dx" ||
          scalar_string(evaluation["evaluation_order"],
                        "model.source_binding.payload.numerical_evaluation."
                        "evaluation_order") != "per_edge_per_l_ladder" ||
          scalar_string(evaluation["expanded_power_coefficients_use"],
                        "model.source_binding.payload.numerical_evaluation."
                        "expanded_power_coefficients_use") != "provenance_only")
        fail("model.source_binding.payload.numerical_evaluation",
             "unsupported V2 shifted-Jacobi numerical realization");
    } else {
      fail("model.source_binding.payload.schema", "unsupported V3 source-plan schema");
    }
    if (scalar_string(program["schema"], "model.schedule_binding.payload.schema") !=
            "ye3t_tagged_cauchy_real_schedule_v1" ||
        scalar_string(readout["schema"], "model.readout_binding.payload.schema") !=
            "ye3t_tagged_cauchy_readout_v1")
      fail("model", "unsupported V3 schedule or readout schema");
    if (scalar_string(program["core_schema"], "model.schedule_binding.payload.core_schema") !=
            "ye3t_tagged_cauchy_real_schedule_core_v1" ||
        scalar_string(program["lowering_convention"],
                      "model.schedule_binding.payload.lowering_convention") !=
            "exact_complex_to_real_tesseral_v1" ||
        scalar_string(program["coefficient_encoding"],
                      "model.schedule_binding.payload.coefficient_encoding") !=
            "exact_algebraic_plus_binary64_v1")
      fail("model.schedule_binding.payload", "unsupported compiler real-schedule convention");
    if (scalar_string(source["exact_zero_distance_force_policy"],
                      "model.source_binding.payload."
                      "exact_zero_distance_force_policy") !=
        "reject_before_direction_evaluation_v1")
      fail("model.source_binding.payload.exact_zero_distance_force_policy",
           "unsupported exact-overlap policy");
    if (scalar_string(source["source_family_id"],
                      "model.source_binding.payload.source_family_id") !=
            scalar_string(source_algebra["source_family_id"],
                          "model.compiler_artifact.plan.report.request."
                          "source_product_algebra.source_family_id") ||
        scalar_string(source["support_id"], "model.source_binding.payload.support_id") !=
            scalar_string(source_algebra["normalized_support"]["support_id"],
                          "model.compiler_artifact.plan.report.request."
                          "source_product_algebra.normalized_support.support_id"))
      fail("model.source_binding.payload",
           "source family or support differs from the compiler artifact");
    if (scalar_string(source["radial_coordinate"],
                      "model.source_binding.payload.radial_coordinate") != "x=r/r_c" ||
        scalar_string(source["radial_measure"], "model.source_binding.payload.radial_measure") !=
            "x^2_dx" ||
        scalar_string(source["envelope"], "model.source_binding.payload.envelope") != "(1-x)^2")
      fail("model.source_binding.payload", "unsupported radial coordinate, measure, or envelope");
    const YAML::Node source_certificate = source["certificate"];
    require_mapping(source_certificate, "model.source_binding.payload.certificate");
    if (source_certificate.size() != 4)
      fail("model.source_binding.payload.certificate",
           "expected the complete V3 source certificate");
    if (!boolean_value(source_certificate["passed"],
                       "model.source_binding.payload.certificate.passed") ||
        !boolean_value(source_certificate["cutoff_value_and_first_derivative_zero_exact"],
                       "model.source_binding.payload.certificate."
                       "cutoff_value_and_first_derivative_zero_exact") ||
        !boolean_value(source_certificate["compiler_source_inventory_consumed"],
                       "model.source_binding.payload.certificate."
                       "compiler_source_inventory_consumed") ||
        boolean_value(source_certificate["runtime_gram_solve"],
                      "model.source_binding.payload.certificate.runtime_gram_solve"))
      fail("model.source_binding.payload.certificate", "V3 source plan is uncertified");
    if (!boolean_value(program["schedule_includes_tag_combinatorics"],
                       "model.schedule_binding.payload."
                       "schedule_includes_tag_combinatorics"))
      fail("model.schedule_binding.payload.schedule_includes_tag_combinatorics",
           "V3 schedule must include tag combinatorics");
    if (boolean_value(program["falling_factorial_runtime_required"],
                      "model.schedule_binding.payload."
                      "falling_factorial_runtime_required"))
      fail("model.schedule_binding.payload.falling_factorial_runtime_required",
           "V3 physical-image schedule must not require runtime tag factors");
  } else {
    const std::string context_policy =
        scalar_string(model["context_policy"], "model.context_policy");
    if (context_policy != "inclusive")
      fail("model.context_policy", "unsupported context_policy (only 'inclusive' is native)");
    const std::string pooling = scalar_string(model["pooling"], "model.pooling");
    if (pooling != "ordered_sum")
      fail("model.pooling", "unsupported pooling (only 'ordered_sum' is native)");
  }

  // offsets
  {
    const std::string path = physical_image_v3 ? "model.readout_binding.payload.offsets"
                                               : "model.offsets";
    require_mapping(readout["offsets"], path);
    result.offsets.assign(result.species_order.size(), std::numeric_limits<double>::quiet_NaN());
    for (const auto &entry : readout["offsets"]) {
      const std::string name = scalar_string(entry.first, path + " key");
      const int index = species_index(name);
      if (index < 0) fail(path, "offset species '" + name + "' is absent from species_order");
      result.offsets[static_cast<std::size_t>(index)] =
          finite_number(entry.second, path + "." + name);
    }
    for (std::size_t index = 0; index < result.species_order.size(); ++index)
      if (!std::isfinite(result.offsets[index]))
        fail(path, "missing offset for species '" + result.species_order[index] + "'");
  }

  if (readout["reference_terms"]) {
    if (!physical_image_v4) fail("readout.reference_terms", "bound references require V4");
    const YAML::Node references = readout["reference_terms"];
    require_mapping(references, "readout.reference_terms");
    for (const auto &entry : references) {
      const std::string key = scalar_string(entry.first, "reference term key");
      if (key != "atomic_energies" && key != "zbl")
        fail("readout.reference_terms", "unsupported reference component");
    }
    const YAML::Node atomic = references["atomic_energies"];
    if (atomic) {
      require_mapping(atomic, "reference atomic_energies");
      if (atomic.size() != result.species_order.size())
        fail("reference atomic_energies", "expected every species");
      for (std::size_t species = 0; species < result.species_order.size(); ++species)
        result.offsets[species] += finite_number(atomic[result.species_order[species]],
                                                "reference atomic energy");
    }
    const YAML::Node zbl = references["zbl"];
    if (zbl && !zbl.IsNull()) {
      require_mapping(zbl, "reference zbl");
      if (scalar_string(zbl["schema"], "zbl.schema") != "ye3t_portable_zbl_reference_v1" ||
          scalar_string(zbl["engine"], "zbl.engine") != "numpy" ||
          scalar_string(zbl["units"], "zbl.units") != "metal" ||
          scalar_string(zbl["pair_style"], "zbl.pair_style") != "zbl" ||
          scalar_string(zbl["switch"], "zbl.switch") != "additive_gromacs_C2" ||
          finite_number(zbl["coulomb_constant_eV_A"], "zbl.coulomb_constant") != 14.399645)
        fail("reference zbl", "unsupported ZBL convention");
      const YAML::Node numbers = zbl["atomic_numbers"];
      require_mapping(numbers, "zbl.atomic_numbers");
      if (numbers.size() != result.species_order.size())
        fail("zbl.atomic_numbers", "expected every species");
      std::vector<int> charges;
      for (const auto &species : result.species_order)
        charges.push_back(integer(numbers[species], "zbl.atomic_numbers." + species, 1));
      const YAML::Node pairs = zbl["pair_cutoffs_A"];
      if (pairs) {
        require_mapping(pairs, "zbl.pair_cutoffs_A");
        if (pairs.size() != result.species_order.size() * (result.species_order.size() + 1) / 2)
          fail("zbl.pair_cutoffs_A", "expected one canonical unordered entry per pair");
        for (const auto &entry : pairs) {
          const std::string key = scalar_string(entry.first, "ZBL pair key");
          const auto dash = key.find('-');
          if (dash == std::string::npos || species_index(key.substr(0, dash)) < 0 ||
              species_index(key.substr(dash + 1)) < 0 || key.substr(0, dash) > key.substr(dash + 1))
            fail("zbl.pair_cutoffs_A", "unknown species pair");
        }
      }
      const std::string readout_json = canonical_json_nested_member_value(supplied_path, "readout_binding", "payload");
      const std::string references_json = canonical_json_value_root_member(readout_json, "reference_terms");
      const std::string zbl_json = canonical_json_value_root_member(references_json, "zbl");
      std::set<std::string> fields = {"schema", "engine", "pair_style", "units", "atomic_numbers",
                                     "coulomb_constant_eV_A", "switch"};
      if (pairs) fields.insert("pair_cutoffs_A");
      else { fields.insert("inner_cutoff_A"); fields.insert("outer_cutoff_A"); }
      for (const auto &entry : zbl) {
        const std::string key = scalar_string(entry.first, "ZBL metadata key");
        if (key != "structure_count" && key != "semantic_sha256" && !fields.count(key))
          fail("reference zbl", "unexpected numerical convention field");
      }
      std::string semantic_json = "{";
      for (const auto &key : fields) {
        if (semantic_json.size() > 1) semantic_json += ",";
        semantic_json += "\"" + key + "\":" + canonical_json_value_root_member(zbl_json, key);
      }
      semantic_json += "}";
      if (sha256_string(semantic_json) != sha256_field(zbl["semantic_sha256"], "zbl.semantic_sha256"))
        fail("reference zbl", "semantic hash mismatch");
      const std::size_t count = result.species_order.size();
      result.zbl_pairs.resize(count * count);
      // Independent evaluation of the published ZBL and additive GROMACS C2
      // equations: https://docs.lammps.org/pair_zbl.html and pair_gromacs.html.
      const double weights[] = {0.18175, 0.50986, 0.28022, 0.02817};
      const double rates[] = {3.19980, 0.94229, 0.40290, 0.20162};
      for (std::size_t left = 0; left < count; ++left)
        for (std::size_t right = 0; right < count; ++right) {
          auto &pair = result.zbl_pairs[left * count + right];
          if (pairs) {
            const std::string key = result.species_order[left] + "-" + result.species_order[right];
            const std::string reverse = result.species_order[right] + "-" + result.species_order[left];
            const YAML::Node values = pairs[key] ? pairs[key] : pairs[reverse];
            require_sequence(values, "zbl.pair_cutoffs_A." + key);
            if (values.size() != 2) fail("zbl.pair_cutoffs_A", "expected [inner, outer]");
            pair.inner = finite_number(values[0], "ZBL inner cutoff");
            pair.outer = finite_number(values[1], "ZBL outer cutoff");
            if (pairs[key] && pairs[reverse] &&
                (pair.inner != finite_number(pairs[reverse][0], "reverse ZBL inner") ||
                 pair.outer != finite_number(pairs[reverse][1], "reverse ZBL outer")))
              fail("zbl.pair_cutoffs_A", "ZBL pair switches must be symmetric");
          } else {
            pair.inner = finite_number(zbl["inner_cutoff_A"], "ZBL inner cutoff");
            pair.outer = finite_number(zbl["outer_cutoff_A"], "ZBL outer cutoff");
          }
          if (!(0 < pair.inner && pair.inner < pair.outer))
            fail("reference zbl", "cutoffs require 0 < inner < outer");
          result.cutoff = std::max(result.cutoff, pair.outer);
          pair.screening_length = 0.46850 / (std::pow(charges[left], 0.23) + std::pow(charges[right], 0.23));
          pair.amplitude = 14.399645 * charges[left] * charges[right];
          double phi = 0.0, first = 0.0, second = 0.0;
          for (int term = 0; term < 4; ++term) {
            const double rate = rates[term] / pair.screening_length;
            const double value = weights[term] * std::exp(-rate * pair.outer);
            phi += value; first -= rate * value; second += rate * rate * value;
          }
          const double r = pair.outer;
          const double endpoint = pair.amplitude * phi / r;
          const double derivative = pair.amplitude * (first / r - phi / (r*r));
          const double curvature = pair.amplitude * (second / r - 2*first/(r*r) + 2*phi/(r*r*r));
          const double width = pair.outer - pair.inner;
          pair.cubic = (-3*derivative + width*curvature)/(width*width);
          pair.quartic = (2*derivative - width*curvature)/(width*width*width);
          pair.constant = -endpoint + width*derivative/2 - width*width*curvature/12;
          if (!std::isfinite(pair.cubic) || !std::isfinite(pair.quartic) || !std::isfinite(pair.constant))
            fail("reference zbl", "non-finite switch coefficients");
        }
    }
  }

  // radial_definition
  if (!physical_image_v3) {
    const YAML::Node radial = model["radial_definition"];
    require_mapping(radial, "model.radial_definition");
    const std::string kind = scalar_string(radial["kind"], "model.radial_definition.kind");
    if (kind != "pace_cheb_exp_cos")
      fail("model.radial_definition.kind",
           "unsupported radial kind (only pace_cheb_exp_cos is native)");
    const YAML::Node parameters = radial["parameters"];
    require_mapping(parameters, "model.radial_definition.parameters");
    result.radial_rc = finite_number(parameters["rc"], "model.radial_definition.parameters.rc");
    if (result.radial_rc <= 0.0)
      fail("model.radial_definition.parameters.rc", "rc must be positive");
    if (std::abs(result.radial_rc - result.cutoff) > 1.0e-9 * std::max(1.0, result.cutoff))
      fail("model.radial_definition.parameters.rc", "radial rc must equal the top-level cutoff");
    result.radial_cutoff_width = finite_number(parameters["cutoff_width"],
                                               "model.radial_definition.parameters.cutoff_width");
    if (result.radial_cutoff_width < 0.0)
      fail("model.radial_definition.parameters.cutoff_width", "cutoff_width must be nonnegative");
    result.radial_lambda =
        finite_number(parameters["lmbda"], "model.radial_definition.parameters.lmbda");
    result.radial_count =
        integer(parameters["radial_count"], "model.radial_definition.parameters.radial_count", 1);
  } else {
    result.radial_rc = result.cutoff;
    result.radial_count = 1;
  }

  // Real forms come either from compiled_artifact.payload.real_forms
  // (single-content exports) or from the top-level real_form_records mapping
  // keyed by real_form_id (multi-content exports, where compiled_artifact is
  // null). Both carry the same per-form fields (real_form_id, angular_l,
  // magnetic_order, real_to_complex_matrix), so one parse_record body serves
  // both sources and both paths report identical error messages and paths.
  std::map<std::string, int> real_form_index_of;
  {
    const auto parse_record = [&](const YAML::Node &record, const std::string &path) {
      require_mapping(record, path);
      TaggedCauchyRealForm form;
      form.real_form_id = scalar_string(record["real_form_id"], path + ".real_form_id");
      form.l = integer(record["angular_l"], path + ".angular_l", 0);
      form.width = 2 * form.l + 1;

      require_sequence(record["magnetic_order"], path + ".magnetic_order");
      const YAML::Node magnetic = record["magnetic_order"];
      if (static_cast<int>(magnetic.size()) != form.width)
        fail(path + ".magnetic_order", "length must equal 2*l+1");
      std::set<int> seen_m;
      for (std::size_t row = 0; row < magnetic.size(); ++row) {
        const int m = integer(magnetic[row], path + ".magnetic_order[" + std::to_string(row) + "]",
                              -form.l, form.l);
        form.magnetic_order.push_back(m);
        if (!seen_m.insert(m).second)
          fail(path + ".magnetic_order", "magnetic_order must not repeat an m value");
      }

      require_sequence(record["real_to_complex_matrix"], path + ".real_to_complex_matrix");
      const YAML::Node matrix_rows = record["real_to_complex_matrix"];
      if (static_cast<int>(matrix_rows.size()) != form.width)
        fail(path + ".real_to_complex_matrix", "row count must equal 2*l+1");
      form.matrix.assign(static_cast<std::size_t>(form.width) *
                             static_cast<std::size_t>(form.width),
                         std::complex<double>(0.0, 0.0));
      for (int row = 0; row < form.width; ++row) {
        const YAML::Node row_node = matrix_rows[static_cast<std::size_t>(row)];
        require_sequence(row_node, path + ".real_to_complex_matrix[row]");
        if (static_cast<int>(row_node.size()) != form.width)
          fail(path + ".real_to_complex_matrix", "row width must equal 2*l+1");
        for (int col = 0; col < form.width; ++col)
          form.matrix[static_cast<std::size_t>(row) * form.width + col] = binary64_complex(
              row_node[static_cast<std::size_t>(col)], path + ".real_to_complex_matrix[row][col]");
      }

      if (physical_image_v3) {
        form.inverse.assign(static_cast<std::size_t>(form.width) *
                                static_cast<std::size_t>(form.width),
                            std::complex<double>(0.0, 0.0));
        for (int row = 0; row < form.width; ++row)
          for (int col = 0; col < form.width; ++col)
            form.inverse[static_cast<std::size_t>(col) * form.width + row] =
                std::conj(form.matrix[static_cast<std::size_t>(row) * form.width + col]);
      } else {
        form.inverse = invert_complex_matrix(form.matrix, form.width, path);
      }
      // Residual check mirrors the Python loader's 1e-14 identity check
      // (`_inverse_real_form_matrix`); Gauss-Jordan in double precision on
      // these small unitary matrices comfortably clears a tighter practical
      // bound.
      double residual = 0.0;
      for (int row = 0; row < form.width; ++row)
        for (int col = 0; col < form.width; ++col) {
          std::complex<double> value = 0.0;
          for (int k = 0; k < form.width; ++k)
            value += form.matrix[static_cast<std::size_t>(row) * form.width + k] *
                form.inverse[static_cast<std::size_t>(k) * form.width + col];
          if (row == col) value -= 1.0;
          residual = std::max(residual, std::abs(value));
        }
      if (residual > 1.0e-10)
        fail(path, "real_to_complex_matrix inverse failed the identity check");

      form.inverse_row_offsets.reserve(static_cast<std::size_t>(form.width) + 1);
      form.inverse_row_offsets.push_back(0);
      for (int row = 0; row < form.width; ++row) {
        for (int col = 0; col < form.width; ++col) {
          const std::complex<double> value =
              form.inverse[static_cast<std::size_t>(row) * form.width + col];
          if (value == std::complex<double>(0.0, 0.0)) continue;
          form.inverse_columns.push_back(col);
          form.inverse_values.push_back(value);
        }
        form.inverse_row_offsets.push_back(static_cast<int>(form.inverse_values.size()));
      }

      real_form_index_of[form.real_form_id] = static_cast<int>(result.real_forms.size());
      result.real_forms.push_back(std::move(form));
    };

    const YAML::Node compiled = model["compiled_artifact"];
    const YAML::Node multi_records = model["real_form_records"];
    if (physical_image_v3) {
      const YAML::Node forms = source["real_forms"];
      require_sequence(forms, "model.source_binding.payload.real_forms");
      for (std::size_t index = 0; index < forms.size(); ++index) {
        const std::string path =
            "model.source_binding.payload.real_forms[" + std::to_string(index) + "]";
        parse_record(forms[index], path);
      }
    } else if (compiled && compiled.IsMap()) {
      const YAML::Node payload = compiled["payload"];
      require_mapping(payload, "model.compiled_artifact.payload");
      const YAML::Node forms = payload["real_forms"];
      require_sequence(forms, "model.compiled_artifact.payload.real_forms");
      for (std::size_t index = 0; index < forms.size(); ++index) {
        const std::string path =
            "model.compiled_artifact.payload.real_forms[" + std::to_string(index) + "]";
        parse_record(forms[index], path);
      }
    } else if (multi_records && multi_records.IsMap()) {
      for (const auto &entry : multi_records) {
        const std::string key = scalar_string(entry.first, "model.real_form_records key");
        const std::string path = "model.real_form_records[" + key + "]";
        parse_record(entry.second, path);
        if (result.real_forms.back().real_form_id != key)
          fail(path, "real_form_records key does not match its record's real_form_id");
      }
    } else {
      fail("model.compiled_artifact",
           "neither compiled_artifact (single-content) nor real_form_records "
           "(multi-content) is present; the model has no source of real forms");
    }
  }

  // channel_real_forms
  {
    const YAML::Node node = physical_image_v3 ? source["channels"] : model["channel_real_forms"];
    const std::string channels_path = physical_image_v3 ? "model.source_binding.payload.channels"
                                                        : "model.channel_real_forms";
    require_sequence(node, channels_path);
    std::vector<std::tuple<std::string, std::string, std::string, int, int>> expected_v3_channels;
    YAML::Node program_channels;
    YAML::Node source_inventory;
    if (physical_image_v3) {
      const YAML::Node compiler_schedules =
          model["compiler_artifact"]["payload"]["moment_schedules"];
      std::set<std::tuple<std::string, std::string, std::string, int, int>> unique_channels;
      for (std::size_t schedule_index = 0; schedule_index < compiler_schedules.size();
           ++schedule_index) {
        const YAML::Node generators = compiler_schedules[schedule_index]["source_generators"];
        require_sequence(generators,
                         "model.compiler_artifact.payload.moment_schedules[*]."
                         "source_generators");
        for (std::size_t generator_index = 0; generator_index < generators.size();
             ++generator_index) {
          const YAML::Node generator = generators[generator_index];
          unique_channels.emplace(
              scalar_string(generator["neighbor_species"],
                            "compiler source generator neighbor_species"),
              scalar_string(generator["source_family_id"],
                            "compiler source generator source_family_id"),
              scalar_string(generator["support_id"], "compiler source generator support_id"),
              integer(generator["q"], "compiler source generator q", 0),
              integer(generator["l"], "compiler source generator l", 0));
        }
      }
      expected_v3_channels.assign(unique_channels.begin(), unique_channels.end());
      program_channels = program["channels"];
      require_sequence(program_channels, "model.schedule_binding.payload.channels");
      if (program_channels.size() != node.size() || expected_v3_channels.size() != node.size())
        fail(channels_path, "compiler/source/schedule channel counts differ");
      source_inventory = algebra["source_inventory"];
      require_sequence(source_inventory,
                       "model.compiler_artifact.plan.report.request."
                       "source_product_algebra.source_inventory");
    }
    std::vector<TaggedCauchyChannel> channels(node.size());
    std::vector<bool> seen(node.size(), false);
    for (std::size_t index = 0; index < node.size(); ++index) {
      const std::string path = channels_path + "[" + std::to_string(index) + "]";
      const YAML::Node entry = node[index];
      require_mapping(entry, path);
      const int channel_index = integer(entry["channel_index"], path + ".channel_index", 0,
                                        static_cast<int>(node.size()) - 1);
      if (seen[static_cast<std::size_t>(channel_index)])
        fail(path + ".channel_index", "duplicate channel_index");
      seen[static_cast<std::size_t>(channel_index)] = true;

      TaggedCauchyChannel channel;
      channel.channel_index = channel_index;
      channel.l = integer(entry["l"], path + ".l", 0);
      channel.radial_channel = physical_image_v3
          ? integer(entry["q"], path + ".q", 0)
          : integer(entry["radial_channel"], path + ".radial_channel", 0, result.radial_count - 1);
      const std::string species_name =
          scalar_string(entry["neighbor_species"], path + ".neighbor_species");
      channel.neighbor_species_index = species_index(species_name);
      if (channel.neighbor_species_index < 0)
        fail(path + ".neighbor_species",
             "channel neighbor_species '" + species_name + "' is absent from species_order");
      const std::string real_form_id = scalar_string(entry["real_form_id"], path + ".real_form_id");
      const auto found = real_form_index_of.find(real_form_id);
      if (found == real_form_index_of.end())
        fail(path + ".real_form_id",
             "real_form_id '" + real_form_id +
                 "' is absent from compiled_artifact.payload.real_forms");
      channel.real_form_index = found->second;
      if (result.real_forms[static_cast<std::size_t>(channel.real_form_index)].l != channel.l)
        fail(path, "channel l does not match its bound real_form angular_l");
      if (physical_image_v3) {
        const YAML::Node program_entry = program_channels[index];
        require_mapping(program_entry,
                        "model.schedule_binding.payload.channels[" + std::to_string(index) + "]");
        const auto identity = std::make_tuple(
            species_name, scalar_string(entry["source_family_id"], path + ".source_family_id"),
            scalar_string(entry["support_id"], path + ".support_id"), channel.radial_channel,
            channel.l);
        const auto program_identity = std::make_tuple(
            scalar_string(program_entry["neighbor_species"], "schedule channel neighbor_species"),
            scalar_string(program_entry["source_family_id"], "schedule channel source_family_id"),
            scalar_string(program_entry["support_id"], "schedule channel support_id"),
            integer(program_entry["q"], "schedule channel q", 0),
            integer(program_entry["l"], "schedule channel l", 0));
        if (identity != expected_v3_channels[index] || identity != program_identity ||
            integer(program_entry["channel_index"], "schedule channel channel_index", 0) !=
                channel_index ||
            scalar_string(program_entry["real_form_id"], "schedule channel real_form_id") !=
                real_form_id)
          fail(path, "compiler/source/schedule channel inventory changed");

        YAML::Node inventory_record;
        for (std::size_t inventory_index = 0; inventory_index < source_inventory.size();
             ++inventory_index) {
          const YAML::Node candidate = source_inventory[inventory_index]["source_key"];
          const auto candidate_identity = std::make_tuple(
              scalar_string(candidate["neighbor_species"], "source inventory neighbor_species"),
              scalar_string(candidate["source_family_id"], "source inventory source_family_id"),
              scalar_string(candidate["support_id"], "source inventory support_id"),
              integer(candidate["q"], "source inventory q", 0),
              integer(candidate["l"], "source inventory l", 0));
          if (candidate_identity == identity) {
            inventory_record = source_inventory[inventory_index];
            break;
          }
        }
        if (!inventory_record) fail(path, "channel is absent from the compiler source inventory");
        require_sequence(entry["binary64_power_coefficients"],
                         path + ".binary64_power_coefficients");
        require_sequence(entry["shifted_jacobi_power_coefficients"],
                         path + ".shifted_jacobi_power_coefficients");
        require_sequence(inventory_record["shifted_jacobi_power_coefficients"],
                         "compiler source inventory shifted_jacobi_power_coefficients");
        if (entry["binary64_power_coefficients"].size() !=
                entry["shifted_jacobi_power_coefficients"].size() ||
            entry["binary64_power_coefficients"].size() !=
                inventory_record["shifted_jacobi_power_coefficients"].size())
          fail(path + ".binary64_power_coefficients",
               "binary64/exact/compiler power-series widths differ");
        for (std::size_t coefficient = 0; coefficient < entry["binary64_power_coefficients"].size();
             ++coefficient) {
          const double value = finite_number(entry["binary64_power_coefficients"][coefficient],
                                             path + ".binary64_power_coefficients[" +
                                                 std::to_string(coefficient) + "]");
          const YAML::Node exact = entry["shifted_jacobi_power_coefficients"][coefficient];
          const YAML::Node compiler_exact =
              inventory_record["shifted_jacobi_power_coefficients"][coefficient];
          const std::int64_t numerator = integer64(exact["numerator"], path + ".radial numerator",
                                                   std::numeric_limits<std::int64_t>::min());
          const std::int64_t denominator =
              integer64(exact["denominator"], path + ".radial denominator", 1);
          if (numerator !=
                  integer64(compiler_exact["numerator"], "compiler radial numerator",
                            std::numeric_limits<std::int64_t>::min()) ||
              denominator !=
                  integer64(compiler_exact["denominator"], "compiler radial denominator", 1))
            fail(path, "exact radial coefficients differ from compiler inventory");
          const double expected = static_cast<double>(numerator) / static_cast<double>(denominator);
          const double tolerance = 4.0e-15 * std::max(1.0, std::abs(expected));
          if (std::abs(value - expected) > tolerance)
            fail(path, "binary64 radial coefficients differ from exact payload");
          channel.shifted_jacobi_power_coefficients.push_back(value);
        }
        channel.normalization =
            finite_number(entry["binary64_normalization"], path + ".binary64_normalization");
        channel.angular_scale =
            finite_number(entry["angular_racah_scale"], path + ".angular_racah_scale");
        if (channel.normalization <= 0.0 || channel.angular_scale <= 0.0)
          fail(path, "source normalization and angular scale must be positive");
        const YAML::Node exact_norm = entry["normalization_squared"];
        const YAML::Node compiler_norm = inventory_record["normalization_squared"];
        const std::int64_t norm_numerator =
            integer64(exact_norm["numerator"], path + ".normalization numerator", 1);
        const std::int64_t norm_denominator =
            integer64(exact_norm["denominator"], path + ".normalization denominator", 1);
        if (norm_numerator !=
                integer64(compiler_norm["numerator"], "compiler normalization numerator", 1) ||
            norm_denominator !=
                integer64(compiler_norm["denominator"], "compiler normalization denominator", 1))
          fail(path, "exact source normalization differs from compiler inventory");
        const double expected_normalization =
            std::sqrt(static_cast<double>(norm_numerator) / static_cast<double>(norm_denominator));
        const double expected_angular_scale =
            std::sqrt(4.0 * std::acos(-1.0) / (2 * channel.l + 1));
        if (std::abs(channel.normalization - expected_normalization) >
                4.0e-15 * std::max(1.0, expected_normalization) ||
            std::abs(channel.angular_scale - expected_angular_scale) >
                4.0e-15 * std::max(1.0, expected_angular_scale))
          fail(path, "binary64 source normalization differs from exact convention");
      }
      channels[static_cast<std::size_t>(channel_index)] = channel;
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool value) {
          return value;
        }))
      fail(channels_path, "channel_index must be dense (0..N-1)");

    int offset = 0;
    std::set<int> l_values;
    for (auto &channel : channels) {
      channel.component_offset = offset;
      offset += result.real_forms[static_cast<std::size_t>(channel.real_form_index)].width;
      l_values.insert(channel.l);
    }
    result.total_component_count = offset;
    result.distinct_angular_l.assign(l_values.begin(), l_values.end());
    result.channels = std::move(channels);
  }
  const auto channel_width = [&](int channel_index) {
    return result
        .real_forms[static_cast<std::size_t>(
            result.channels[static_cast<std::size_t>(channel_index)].real_form_index)]
        .width;
  };
  const auto flat_index_of = [&](int channel_index, int a, const std::string &path) {
    if (channel_index < 0 || channel_index >= static_cast<int>(result.channels.size()))
      fail(path, "channel index is out of range");
    if (a < 0 || a >= channel_width(channel_index))
      fail(path, "real component index is out of range for its channel");
    return result.channels[static_cast<std::size_t>(channel_index)].component_offset + a;
  };

  // real_moment_program
  if (!program || !program.IsMap())
    fail(physical_image_v3 ? "model.schedule_binding.payload" : "model.real_moment_program",
         "missing real program");

  const std::string program_path = physical_image_v3 ? "model.schedule_binding.payload"
                                                     : "model.real_moment_program";

  result.feature_count = integer(program["feature_count"], program_path + ".feature_count", 0);
  const int program_tag_count = integer(program["tag_count"], program_path + ".tag_count", 0);
  if (program_tag_count != result.tag_count)
    fail(program_path + ".tag_count", "tag_count differs from the top-level model.tag_count");
  if (!physical_image_v3) {
    const YAML::Node certificate = program["lowering_certificate"];
    require_mapping(certificate, program_path + ".lowering_certificate");
    const int certificate_tag_count =
        integer(certificate["tag_count"], program_path + ".lowering_certificate.tag_count", 0);
    if (certificate_tag_count != result.tag_count)
      fail(program_path + ".lowering_certificate.tag_count",
           "tag_count differs from the top-level model.tag_count");
  }
  if (physical_image_v3) {
    const int readout_features =
        integer(readout["feature_count"], "model.readout_binding.payload.feature_count", 0);
    if (readout_features != result.feature_count)
      fail("model.readout_binding.payload.feature_count", "readout/schedule feature width differs");
    const YAML::Node certificate = program["certificate"];
    require_mapping(certificate, program_path + ".certificate");
    if (!boolean_value(certificate["passed"], program_path + ".certificate.passed") ||
        !boolean_value(certificate["exact_realification"],
                       program_path + ".certificate.exact_realification") ||
        !boolean_value(certificate["compiler_adjoint_matches_real_forward_exact"],
                       program_path + ".certificate.compiler_adjoint_matches_real_forward_exact") ||
        !boolean_value(certificate["division_free_adjoint"],
                       program_path + ".certificate.division_free_adjoint"))
      fail(program_path + ".certificate", "V3 schedule is uncertified");
  }

  // offset_mode: optional (absent from schemas that predate it);
  // when present it must be one of the two known values. Both modes are a
  // single per-species constant added once per owned atom with no force or
  // virial contribution, so nothing else here or in the evaluator branches
  // on its value -- it is recorded purely for provenance/logging.
  if (!physical_image_v3 && model["offset_mode"]) {
    result.offset_mode = scalar_string(model["offset_mode"], "model.offset_mode");
    if (result.offset_mode != "fitted_species_offsets" && result.offset_mode != "composition_fixed")
      fail("model.offset_mode", "must be 'fitted_species_offsets' or 'composition_fixed'");
  }

  // beta: a {species: [feature_count doubles]} mapping, one vector
  // per species_order entry, every vector of length feature_count. The
  // plain-array (species-independent) form of the earlier schema is
  // still accepted, but ONLY when species_order has exactly one species
  // (that species implicitly owns the whole array) -- this is the recorded
  // backward-compatibility rule for the one-species case.
  {
    const YAML::Node node = readout["beta"];
    const std::string beta_path = physical_image_v3 ? "model.readout_binding.payload.beta"
                                                    : "model.beta";
    if (!node || (!node.IsMap() && !node.IsSequence()))
      fail(beta_path,
           "expected a mapping of species to feature vectors, "
           "or (single-species models only) a plain array");
    result.beta.assign(result.species_order.size(), std::vector<double>());
    const auto load_species_vector = [&](std::size_t species, const YAML::Node &values,
                                         const std::string &path) {
      require_sequence(values, path);
      if (static_cast<int>(values.size()) != result.feature_count)
        fail(path,
             "vector length (" + std::to_string(values.size()) +
                 ") does not equal schedule feature_count (" +
                 std::to_string(result.feature_count) + ")");
      std::vector<double> vector_values;
      vector_values.reserve(values.size());
      for (std::size_t index = 0; index < values.size(); ++index)
        vector_values.push_back(
            finite_number(values[index], path + "[" + std::to_string(index) + "]"));
      result.beta[species] = std::move(vector_values);
    };

    if (node.IsSequence()) {
      if (result.species_order.size() != 1)
        fail(beta_path,
             "a plain-array beta is only accepted when species_order has "
             "exactly one species (" +
                 std::to_string(result.species_order.size()) +
                 " species are present); export a {species: [...]} mapping "
                 "instead");
      load_species_vector(0, node, beta_path);
    } else {
      std::set<std::string> node_keys;
      for (const auto &entry : node) node_keys.insert(scalar_string(entry.first, "model.beta key"));
      const std::set<std::string> expected_keys(result.species_order.begin(),
                                                result.species_order.end());
      if (node_keys != expected_keys)
        fail(beta_path,
             "beta keys must be exactly species_order (a species present in "
             "beta but not species_order, or vice versa, is rejected)");
      for (std::size_t species = 0; species < result.species_order.size(); ++species)
        load_species_vector(species, node[result.species_order[species]],
                            beta_path + "." + result.species_order[species]);
    }
  }

  // real_density_keys
  {
    require_sequence(program["real_density_keys"], program_path + ".real_density_keys");
    const YAML::Node node = program["real_density_keys"];
    for (std::size_t index = 0; index < node.size(); ++index) {
      const std::string path = program_path + ".real_density_keys[" + std::to_string(index) + "]";
      require_sequence(node[index], path);
      if (node[index].size() != 2)
        fail(path, "a real density key must be a [channel_index, a] pair");
      const int channel = integer(node[index][0], path + "[0]", 0);
      const int a = integer(node[index][1], path + "[1]", 0);
      result.real_density_keys.emplace_back(channel, a);
      result.real_density_flat_index.push_back(flat_index_of(channel, a, path));
    }
  }

  // real_moment_keys
  {
    require_sequence(program["real_moment_keys"], program_path + ".real_moment_keys");
    const YAML::Node node = program["real_moment_keys"];
    for (std::size_t index = 0; index < node.size(); ++index) {
      const std::string path = program_path + ".real_moment_keys[" + std::to_string(index) + "]";
      require_sequence(node[index], path);
      std::vector<std::pair<int, int>> factors;
      std::vector<int> flat;
      const YAML::Node factor_list = node[index];
      for (std::size_t factor_index = 0; factor_index < factor_list.size(); ++factor_index) {
        const std::string factor_path = path + "[" + std::to_string(factor_index) + "]";
        require_sequence(factor_list[factor_index], factor_path);
        if (factor_list[factor_index].size() != 2)
          fail(factor_path, "a real moment factor must be a [channel_index, a] pair");
        const int channel = integer(factor_list[factor_index][0], factor_path + "[0]", 0);
        const int a = integer(factor_list[factor_index][1], factor_path + "[1]", 0);
        factors.emplace_back(channel, a);
        flat.push_back(flat_index_of(channel, a, factor_path));
      }
      result.real_moment_keys.push_back(std::move(factors));
      result.real_moment_flat_indices.push_back(std::move(flat));
    }
    if (physical_image_v3 && !result.real_moment_keys.empty())
      fail(program_path + ".real_moment_keys",
           "V3 physical-image schedule must not contain runtime moments");
  }

  // terms
  {
    require_sequence(program["terms"], program_path + ".terms");
    const YAML::Node node = program["terms"];
    for (std::size_t index = 0; index < node.size(); ++index) {
      const std::string path = program_path + ".terms[" + std::to_string(index) + "]";
      require_mapping(node[index], path);
      TaggedCauchyTerm term;
      term.feature_index = integer(node[index]["feature_index"], path + ".feature_index", 0,
                                   result.feature_count - 1);
      term.coefficient = finite_number(node[index]["coefficient"], path + ".coefficient");
      term.p = integer(node[index]["p"], path + ".p", 0, result.tag_count);
      if (physical_image_v3 && term.p != 0)
        fail(path + ".p", "V3 physical-image terms must use p=0");
      require_sequence(node[index]["density_factor_indices"], path + ".density_factor_indices");
      for (const auto &value : node[index]["density_factor_indices"])
        term.density_factor_indices.push_back(
            integer(value, path + ".density_factor_indices[*]", 0,
                    static_cast<int>(result.real_density_keys.size()) - 1));
      require_sequence(node[index]["moment_indices"], path + ".moment_indices");
      for (const auto &value : node[index]["moment_indices"])
        term.moment_indices.push_back(
            integer(value, path + ".moment_indices[*]", 0,
                    static_cast<int>(result.real_moment_keys.size()) - 1));
      if (physical_image_v3 && !term.moment_indices.empty())
        fail(path + ".moment_indices", "V3 physical-image terms must not contain runtime moments");
      result.terms.push_back(std::move(term));
    }
  }

  if (physical_image_v3) {
    require_sequence(program["adjoint_terms"], program_path + ".adjoint_terms");
    const YAML::Node node = program["adjoint_terms"];
    for (std::size_t index = 0; index < node.size(); ++index) {
      const std::string path = program_path + ".adjoint_terms[" + std::to_string(index) + "]";
      require_mapping(node[index], path);
      TaggedCauchyAdjointTerm term;
      term.feature_index = integer(node[index]["feature_index"], path + ".feature_index", 0,
                                   result.feature_count - 1);
      term.source_index = integer(node[index]["source_index"], path + ".source_index", 0,
                                  static_cast<int>(result.real_density_keys.size()) - 1);
      term.coefficient = finite_number(node[index]["coefficient"], path + ".coefficient");
      require_sequence(node[index]["remaining_source_indices"], path + ".remaining_source_indices");
      for (const auto &value : node[index]["remaining_source_indices"])
        term.remaining_source_indices.push_back(
            integer(value, path + ".remaining_source_indices[*]", 0,
                    static_cast<int>(result.real_density_keys.size()) - 1));
      result.adjoint_terms.push_back(std::move(term));
    }
    if (result.adjoint_terms.empty())
      fail(program_path + ".adjoint_terms", "V3 adjoint schedule is empty");

    using AdjointKey = std::tuple<int, int, std::vector<int>>;
    std::map<AdjointKey, double> derived;
    for (const auto &term : result.terms) {
      std::vector<int> factors = term.density_factor_indices;
      std::sort(factors.begin(), factors.end());
      for (std::size_t begin = 0; begin < factors.size();) {
        std::size_t end = begin + 1;
        while (end < factors.size() && factors[end] == factors[begin]) ++end;
        std::vector<int> remaining = factors;
        remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(begin));
        derived[{term.feature_index, factors[begin], remaining}] +=
            static_cast<double>(end - begin) * term.coefficient;
        begin = end;
      }
    }
    std::map<AdjointKey, double> supplied;
    for (const auto &term : result.adjoint_terms) {
      std::vector<int> remaining = term.remaining_source_indices;
      std::sort(remaining.begin(), remaining.end());
      supplied[{term.feature_index, term.source_index, remaining}] += term.coefficient;
    }
    if (derived.size() != supplied.size())
      fail(program_path + ".adjoint_terms", "binary64 forward/adjoint schedules differ");
    for (const auto &[key, expected] : derived) {
      const auto found = supplied.find(key);
      const double tolerance = 4.0e-14 * std::max(1.0, std::abs(expected));
      if (found == supplied.end() || std::abs(found->second - expected) > tolerance)
        fail(program_path + ".adjoint_terms", "binary64 forward/adjoint schedules differ");
    }
  }

  if (model["tagged_execution_portfolio"]) {
    const YAML::Node portfolio = model["tagged_execution_portfolio"];
    const std::string path = "model.tagged_execution_portfolio";
    require_mapping(portfolio, path);
    const std::string portfolio_schema = scalar_string(portfolio["schema"], path + ".schema");
    const bool legacy_portfolio = portfolio_schema == "ye3t_tagged_moment_execution_portfolio_v1";
    if (!legacy_portfolio && portfolio_schema != "ye3t_tagged_moment_execution_portfolio_v2")
      fail(path + ".schema", "unsupported tagged execution portfolio");
    const std::string portfolio_json =
        canonical_json_root_member_value(supplied_path, "tagged_execution_portfolio");
    result.execution_portfolio.portfolio_hash =
        sha256_field(portfolio["portfolio_hash"], path + ".portfolio_hash");
    if (canonical_json_value_hash_without_root_member(portfolio_json, "portfolio_hash") !=
        result.execution_portfolio.portfolio_hash)
      fail(path, "portfolio self-hash mismatch");
    result.execution_portfolio.program_hash =
        sha256_field(portfolio["program_hash"], path + ".program_hash");
    const std::string program_json = physical_image_v3
        ? canonical_json_nested_member_value(supplied_path, "schedule_binding", "payload")
        : canonical_json_root_member_value(supplied_path, "real_moment_program");
    if (sha256_string(program_json) != result.execution_portfolio.program_hash)
      fail(path + ".program_hash", "portfolio program binding changed");
    result.execution_portfolio.readout_hash =
        sha256_field(portfolio["readout_hash"], path + ".readout_hash");
    if (integer(portfolio["feature_count"], path + ".feature_count", 0) != result.feature_count ||
        integer(portfolio["density_count"], path + ".density_count", 0) !=
            static_cast<int>(result.real_density_keys.size()) ||
        integer(portfolio["moment_count"], path + ".moment_count", 0) !=
            static_cast<int>(result.real_moment_keys.size()))
      fail(path, "portfolio dimensions changed");

    result.execution_portfolio.moment_plan = parse_binary_product_plan(
        portfolio["moment_plan"], path + ".moment_plan",
        static_cast<int>(result.real_density_keys.size()), result.real_moment_keys.size(),
        BinaryProductFactorization::CommutativeSymmetricPower, legacy_portfolio);
    const YAML::Node outer_roots = portfolio["outer_plan"]["roots"];
    require_sequence(outer_roots, path + ".outer_plan.roots");
    result.execution_portfolio.outer_plan = parse_binary_product_plan(
        portfolio["outer_plan"], path + ".outer_plan",
        static_cast<int>(result.real_density_keys.size() + result.real_moment_keys.size()),
        outer_roots.size(),
        legacy_portfolio ? BinaryProductFactorization::CommutativeSymmetricPower
                         : BinaryProductFactorization::CanonicalPrefix,
        legacy_portfolio);

    require_sequence(portfolio["species_order"], path + ".species_order");
    if (portfolio["species_order"].size() != result.species_order.size())
      fail(path + ".species_order", "portfolio species count changed");
    require_mapping(portfolio["species_routes"], path + ".species_routes");
    std::set<std::string> declared_species;
    for (std::size_t species = 0; species < portfolio["species_order"].size(); ++species)
      declared_species.insert(
          scalar_string(portfolio["species_order"][species],
                        path + ".species_order[" + std::to_string(species) + "]"));
    const std::set<std::string> model_species(result.species_order.begin(),
                                              result.species_order.end());
    if (declared_species != model_species)
      fail(path + ".species_order", "portfolio species set changed");
    std::set<std::string> route_species;
    for (const auto &entry : portfolio["species_routes"])
      route_species.insert(scalar_string(entry.first, path + ".species_routes key"));
    if (route_species != model_species)
      fail(path + ".species_routes", "portfolio route species set changed");
    const int outer_maximum_value = result.execution_portfolio.outer_plan.base_count +
        static_cast<int>(result.execution_portfolio.outer_plan.nodes.size()) - 1;
    result.execution_portfolio.species_routes.resize(result.species_order.size());
    for (std::size_t species = 0; species < result.species_order.size(); ++species) {
      const std::string &name = result.species_order[species];
      const YAML::Node routes = portfolio["species_routes"][name];
      require_sequence(routes, path + ".species_routes." + name);
      for (std::size_t index = 0; index < routes.size(); ++index) {
        const std::string route_path =
            path + ".species_routes." + name + "[" + std::to_string(index) + "]";
        require_mapping(routes[index], route_path);
        TaggedCauchyExecutionRoute route;
        route.p = integer(routes[index]["p"], route_path + ".p", 0, result.tag_count);
        route.root_value = integer(routes[index]["root_value"], route_path + ".root_value", -1,
                                   outer_maximum_value);
        route.coefficient =
            finite_number(routes[index]["coefficient"], route_path + ".coefficient");
        result.execution_portfolio.species_routes[species].push_back(route);
      }
    }

    require_sequence(portfolio["candidates"], path + ".candidates");
    std::set<std::string> candidate_ids;
    for (std::size_t index = 0; index < portfolio["candidates"].size(); ++index) {
      const std::string candidate_path = path + ".candidates[" + std::to_string(index) + "]";
      const YAML::Node candidate = portfolio["candidates"][index];
      require_mapping(candidate, candidate_path);
      const std::string id =
          scalar_string(candidate["candidate_id"], candidate_path + ".candidate_id");
      if (!candidate_ids.insert(id).second)
        fail(candidate_path + ".candidate_id", "duplicate candidate");
      scalar_string(candidate["kernel"], candidate_path + ".kernel");
      const bool eligible = boolean_value(candidate["eligible"], candidate_path + ".eligible");
      if (id == "compiled_direct")
        result.execution_portfolio.direct_eligible = eligible;
      else if (id == "generic_dag")
        result.execution_portfolio.generic_dag_eligible = eligible;
      else if (id == "symmetric_power")
        result.execution_portfolio.symmetric_power_eligible = eligible;
      else if (id == "block")
        result.execution_portfolio.block_eligible = eligible;
      else
        fail(candidate_path + ".candidate_id", "unknown candidate");
    }
    if (candidate_ids !=
        std::set<std::string>{"block", "compiled_direct", "generic_dag", "symmetric_power"})
      fail(path + ".candidates", "portfolio candidate set changed");
    const YAML::Node certificate = portfolio["certificate"];
    require_mapping(certificate, path + ".certificate");
    for (const char *key :
         {"passed", "compiler_owned_factorization",
          "every_repeated_block_uses_symmetric_power_nodes", "division_free_forward_reverse"})
      if (!boolean_value(certificate[key], path + ".certificate." + key))
        fail(path + ".certificate." + key, "portfolio certificate failed");
    if (boolean_value(certificate["runtime_label_inference_required"],
                      path + ".certificate.runtime_label_inference_required"))
      fail(path + ".certificate.runtime_label_inference_required",
           "runtime label inference is forbidden");
    if (!legacy_portfolio &&
        !boolean_value(certificate["preserves_direct_route_accumulation_order"],
                       path + ".certificate.preserves_direct_route_accumulation_order"))
      fail(path + ".certificate.preserves_direct_route_accumulation_order",
           "portfolio route-order certificate failed");

    const auto &moment_plan = result.execution_portfolio.moment_plan;
    const auto moment_value_exponents =
        binary_plan_value_exponents(moment_plan, path + ".moment_plan");
    std::map<std::pair<int, int>, int> density_key_index;
    for (std::size_t index = 0; index < result.real_density_keys.size(); ++index)
      density_key_index.emplace(result.real_density_keys[index], static_cast<int>(index));
    if (density_key_index.size() != result.real_density_keys.size())
      fail(path + ".moment_plan", "real-density keys are not unique");

    std::int64_t expected_direct_moment_multiplications = 0;
    std::int64_t expected_repeated_moments = 0;
    int expected_maximum_moment_degree = 0;
    for (std::size_t moment = 0; moment < result.real_moment_keys.size(); ++moment) {
      std::vector<int> expected(result.real_density_keys.size(), 0);
      std::set<int> distinct;
      for (const auto &factor : result.real_moment_keys[moment]) {
        const auto found = density_key_index.find(factor);
        if (found == density_key_index.end())
          fail(path + ".moment_plan", "moment references an unknown real-density key");
        ++expected[static_cast<std::size_t>(found->second)];
        distinct.insert(found->second);
      }
      const std::size_t degree = result.real_moment_keys[moment].size();
      expected_direct_moment_multiplications +=
          static_cast<std::int64_t>(degree > 0 ? degree - 1 : 0);
      expected_maximum_moment_degree =
          std::max(expected_maximum_moment_degree, static_cast<int>(degree));
      if (distinct.size() < degree) ++expected_repeated_moments;
      if (binary_plan_root_exponent(moment_plan, moment_value_exponents,
                                    moment_plan.roots[moment]) != expected)
        fail(path + ".moment_plan.roots[" + std::to_string(moment) + "]",
             "binary moment root exponent differs from real_moment_program");
    }
    if (moment_plan.direct_multiplication_count != expected_direct_moment_multiplications ||
        moment_plan.maximum_degree != expected_maximum_moment_degree ||
        moment_plan.repeated_factor_product_count != expected_repeated_moments)
      fail(path + ".moment_plan",
           "binary moment operation metadata differs from the exact program");

    const auto &outer_plan = result.execution_portfolio.outer_plan;
    const auto outer_value_exponents =
        binary_plan_value_exponents(outer_plan, path + ".outer_plan");
    const auto factors_from_exponent = [](const std::vector<int> &exponent) {
      std::vector<int> factors;
      for (std::size_t base = 0; base < exponent.size(); ++base)
        for (int count = 0; count < exponent[base]; ++count)
          factors.push_back(static_cast<int>(base));
      return factors;
    };
    using RouteKey = std::pair<int, std::vector<int>>;
    std::vector<std::map<RouteKey, double>> expected_routes(result.species_order.size());
    std::vector<std::vector<RouteKey>> expected_route_order(result.species_order.size());
    std::set<std::vector<int>> expected_outer_products;
    for (std::size_t species = 0; species < result.species_order.size(); ++species) {
      auto &expected = expected_routes[species];
      auto &order = expected_route_order[species];
      for (const TaggedCauchyTerm &term : result.terms) {
        std::vector<int> factors;
        factors.reserve(term.density_factor_indices.size() + term.moment_indices.size());
        factors.insert(factors.end(), term.density_factor_indices.begin(),
                       term.density_factor_indices.end());
        for (const int moment : term.moment_indices)
          factors.push_back(static_cast<int>(result.real_density_keys.size()) + moment);
        std::sort(factors.begin(), factors.end());
        const RouteKey key{term.p, factors};
        const double coefficient =
            result.beta[species][static_cast<std::size_t>(term.feature_index)] * term.coefficient;
        const auto found = expected.find(key);
        if (found == expected.end()) {
          expected.emplace(key, coefficient);
          order.push_back(key);
        } else {
          found->second += coefficient;
        }
      }
      for (auto entry = expected.begin(); entry != expected.end();) {
        if (entry->second == 0.0)
          entry = expected.erase(entry);
        else {
          expected_outer_products.insert(entry->first.second);
          ++entry;
        }
      }
      order.erase(std::remove_if(order.begin(), order.end(),
                                 [&expected](const RouteKey &key) {
                                   return expected.find(key) == expected.end();
                                 }),
                  order.end());
    }

    if (outer_plan.roots.size() != expected_outer_products.size())
      fail(path + ".outer_plan.roots", "outer root count differs from folded readout products");
    auto expected_product = expected_outer_products.begin();
    std::int64_t expected_direct_outer_multiplications = 0;
    std::int64_t expected_repeated_outer_products = 0;
    int expected_maximum_outer_degree = 0;
    for (std::size_t index = 0; index < outer_plan.roots.size(); ++index, ++expected_product) {
      const std::vector<int> actual = factors_from_exponent(
          binary_plan_root_exponent(outer_plan, outer_value_exponents, outer_plan.roots[index]));
      if (actual != *expected_product)
        fail(path + ".outer_plan.roots[" + std::to_string(index) + "]",
             "outer root differs from a folded readout product");
      const std::size_t degree = actual.size();
      expected_direct_outer_multiplications +=
          static_cast<std::int64_t>(degree > 0 ? degree - 1 : 0);
      expected_maximum_outer_degree =
          std::max(expected_maximum_outer_degree, static_cast<int>(degree));
      if (std::set<int>(actual.begin(), actual.end()).size() < degree)
        ++expected_repeated_outer_products;
    }
    if (outer_plan.direct_multiplication_count != expected_direct_outer_multiplications ||
        outer_plan.maximum_degree != expected_maximum_outer_degree ||
        outer_plan.repeated_factor_product_count != expected_repeated_outer_products)
      fail(path + ".outer_plan", "outer operation metadata differs from the folded readout");

    for (std::size_t species = 0; species < result.species_order.size(); ++species) {
      const auto &actual = result.execution_portfolio.species_routes[species];
      const auto &expected = expected_routes[species];
      const auto &order = expected_route_order[species];
      if (actual.size() != expected.size())
        fail(path + ".species_routes." + result.species_order[species],
             "route count differs from the folded readout");
      for (std::size_t index = 0; index < actual.size(); ++index) {
        const std::vector<int> factors = factors_from_exponent(
            binary_plan_root_exponent(outer_plan, outer_value_exponents, actual[index].root_value));
        const RouteKey key{actual[index].p, factors};
        const auto expected_route = expected.find(key);
        if (index >= order.size() || key != order[index] || expected_route == expected.end() ||
            actual[index].coefficient != expected_route->second)
          fail(path + ".species_routes." + result.species_order[species] + "[" +
                   std::to_string(index) + "]",
               "route differs from the exact folded readout");
      }
    }

    require_mapping(portfolio["costs"], path + ".costs");
    if (integer64(portfolio["costs"]["direct_moment_multiplications"],
                  path + ".costs.direct_moment_multiplications",
                  0) != expected_direct_moment_multiplications ||
        integer64(portfolio["costs"]["symmetric_power_moment_nodes"],
                  path + ".costs.symmetric_power_moment_nodes",
                  0) != static_cast<std::int64_t>(moment_plan.nodes.size()) ||
        integer64(portfolio["costs"]["direct_outer_multiplications"],
                  path + ".costs.direct_outer_multiplications",
                  0) != expected_direct_outer_multiplications ||
        integer64(portfolio["costs"]["block_outer_nodes"], path + ".costs.block_outer_nodes", 0) !=
            static_cast<std::int64_t>(outer_plan.nodes.size()))
      fail(path + ".costs", "portfolio costs differ from exact schedules");
    if (!result.execution_portfolio.direct_eligible ||
        !result.execution_portfolio.generic_dag_eligible ||
        result.execution_portfolio.symmetric_power_eligible != (expected_repeated_moments > 0) ||
        result.execution_portfolio.block_eligible !=
            (!result.real_moment_keys.empty() && !expected_outer_products.empty()))
      fail(path + ".candidates", "candidate eligibility differs from exact schedules");
    if (integer(certificate["moment_root_count"], path + ".certificate.moment_root_count", 0) !=
            static_cast<int>(result.real_moment_keys.size()) ||
        integer(certificate["outer_product_count"], path + ".certificate.outer_product_count", 0) !=
            static_cast<int>(expected_outer_products.size()))
      fail(path + ".certificate", "portfolio certificate counts changed");
    result.execution_portfolio.present = true;
  }

  return result;
}

double TaggedCauchyModel::memory_usage() const
{
  double bytes = sizeof(TaggedCauchyModel);
  bytes += model_path.capacity() + 1;
  bytes += self_hash.capacity() + 1;
  bytes += composite_manifest_path.capacity() + 1;
  bytes += composite_self_hash.capacity() + 1;
  bytes += ordinary_model_path.capacity() + 1;
  bytes += ordinary_model_hash.capacity() + 1;
  bytes += tagged_component_hash.capacity() + 1;
  bytes += compiler_artifact_hash.capacity() + 1;
  bytes += source_plan_hash.capacity() + 1;
  bytes += schedule_hash.capacity() + 1;
  bytes += readout_hash.capacity() + 1;
  bytes += deployment_identity_hash.capacity() + 1;
  bytes += species_order.capacity() * sizeof(std::string);
  for (const auto &species : species_order) bytes += species.capacity() + 1;
  bytes += offsets.capacity() * sizeof(double);
  bytes += offset_mode.capacity() + 1;
  bytes += pair_cutoffs.capacity() * sizeof(double);
  bytes += zbl_pairs.capacity() * sizeof(TaggedCauchyZBLPair);
  bytes += beta.capacity() * sizeof(std::vector<double>);
  for (const auto &species_beta : beta) bytes += species_beta.capacity() * sizeof(double);
  bytes += real_forms.capacity() * sizeof(TaggedCauchyRealForm);
  for (const auto &form : real_forms) {
    bytes += form.real_form_id.capacity() + 1;
    bytes += form.magnetic_order.capacity() * sizeof(int);
    bytes += form.matrix.capacity() * sizeof(std::complex<double>);
    bytes += form.inverse.capacity() * sizeof(std::complex<double>);
    bytes += form.inverse_row_offsets.capacity() * sizeof(int);
    bytes += form.inverse_columns.capacity() * sizeof(int);
    bytes += form.inverse_values.capacity() * sizeof(std::complex<double>);
  }
  bytes += channels.capacity() * sizeof(TaggedCauchyChannel);
  for (const auto &channel : channels)
    bytes += channel.shifted_jacobi_power_coefficients.capacity() * sizeof(double);
  bytes += distinct_angular_l.capacity() * sizeof(int);
  bytes += real_density_keys.capacity() * sizeof(std::pair<int, int>);
  bytes += real_density_flat_index.capacity() * sizeof(int);
  bytes += real_moment_keys.capacity() * sizeof(std::vector<std::pair<int, int>>);
  for (const auto &key : real_moment_keys) bytes += key.capacity() * sizeof(std::pair<int, int>);
  bytes += real_moment_flat_indices.capacity() * sizeof(std::vector<int>);
  for (const auto &flat : real_moment_flat_indices) bytes += flat.capacity() * sizeof(int);
  bytes += terms.capacity() * sizeof(TaggedCauchyTerm);
  for (const auto &term : terms) {
    bytes += term.density_factor_indices.capacity() * sizeof(int);
    bytes += term.moment_indices.capacity() * sizeof(int);
  }
  bytes += adjoint_terms.capacity() * sizeof(TaggedCauchyAdjointTerm);
  for (const auto &term : adjoint_terms)
    bytes += term.remaining_source_indices.capacity() * sizeof(int);
  const auto &portfolio = execution_portfolio;
  bytes += portfolio.portfolio_hash.capacity() + 1;
  bytes += portfolio.program_hash.capacity() + 1;
  bytes += portfolio.readout_hash.capacity() + 1;
  for (const auto *plan : {&portfolio.moment_plan, &portfolio.outer_plan}) {
    bytes += plan->nodes.capacity() * sizeof(TaggedCauchyBinaryNode);
    bytes += plan->roots.capacity() * sizeof(int);
  }
  bytes += portfolio.species_routes.capacity() * sizeof(std::vector<TaggedCauchyExecutionRoute>);
  for (const auto &routes : portfolio.species_routes)
    bytes += routes.capacity() * sizeof(TaggedCauchyExecutionRoute);
  return bytes;
}

}    // namespace YE3T_LAMMPS
