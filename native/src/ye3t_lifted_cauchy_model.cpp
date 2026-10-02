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

#include "ye3t_lifted_cauchy_model.h"

#include "ye3t_canonical_json_hash.h"
#include "ye3t_sha256.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace YE3T_LAMMPS {
namespace {

  constexpr std::uintmax_t MAX_MODEL_BYTES = 16ULL * 1024ULL * 1024ULL;
  constexpr std::uintmax_t MAX_NATIVE_BYTES = 128ULL * 1024ULL * 1024ULL;
  constexpr std::uintmax_t MAX_COMPILER_BYTES = 512ULL * 1024ULL * 1024ULL;
  constexpr std::uintmax_t MAX_COMPILER_BINDING_DOM_BYTES = 64ULL * 1024ULL * 1024ULL;
  constexpr std::int64_t MAX_SOURCE_VARIABLES = 1LL << 20;
  constexpr std::int64_t MAX_TERMS = 5'000'000;
  constexpr std::int64_t MAX_FACTOR_REFERENCES = 100'000'000;
  constexpr std::int64_t MAX_REALIFICATION_VISITS = 100'000'000;
  constexpr std::int64_t MAX_STATIC_ARRAY_BYTES = 128LL * 1024LL * 1024LL;

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

  void require_exact_fields(const YAML::Node &node, const std::set<std::string> &expected,
                            const std::string &path)
  {
    require_mapping(node, path);
    std::set<std::string> actual;
    for (const auto &entry : node) {
      if (!entry.first.IsScalar()) fail(path, "field names must be strings");
      actual.insert(entry.first.as<std::string>());
    }
    for (const auto &field : expected)
      if (actual.count(field) == 0) fail(path, "missing required field '" + field + "'");
    for (const auto &field : actual)
      if (expected.count(field) == 0) fail(path, "unsupported field '" + field + "'");
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

  bool boolean(const YAML::Node &node, const std::string &path)
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

  bool nonzero_decimal_integer(const std::string &value)
  {
    if (value.empty()) return false;
    std::size_t index = value[0] == '-' ? 1 : 0;
    if (index == value.size()) return false;
    bool nonzero = false;
    for (; index < value.size(); ++index) {
      if (value[index] < '0' || value[index] > '9') return false;
      nonzero = nonzero || value[index] != '0';
    }
    return nonzero;
  }

  bool same_binary64(double left, double right)
  {
    return std::abs(left - right) <= 2.0e-14 * std::max({1.0, std::abs(left), std::abs(right)});
  }

  std::string sha256(const YAML::Node &node, const std::string &path)
  {
    const std::string value = scalar_string(node, path);
    if (!lower_sha256(value)) fail(path, "expected a lowercase SHA-256 digest");
    return value;
  }

  void require_string(const YAML::Node &node, const std::string &expected, const std::string &path)
  {
    if (scalar_string(node, path) != expected) fail(path, "unsupported value");
  }

  void require_bool(const YAML::Node &node, bool expected, const std::string &path)
  {
    if (boolean(node, path) != expected) fail(path, "unsupported boolean value");
  }

  std::vector<std::string> string_sequence(const YAML::Node &node, const std::string &path,
                                           std::size_t maximum = 4096)
  {
    require_sequence(node, path);
    if (node.size() > maximum) fail(path, "sequence exceeds the supported bound");
    std::vector<std::string> result;
    result.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      result.push_back(scalar_string(node[index], path + "[" + std::to_string(index) + "]"));
    return result;
  }

  std::vector<int> integer_sequence(const YAML::Node &node, const std::string &path, int minimum,
                                    int maximum, std::size_t size_bound = 1ULL << 20)
  {
    require_sequence(node, path);
    if (node.size() > size_bound) fail(path, "sequence exceeds the supported bound");
    std::vector<int> result;
    result.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      result.push_back(
          integer(node[index], path + "[" + std::to_string(index) + "]", minimum, maximum));
    return result;
  }

  std::vector<std::int64_t> integer64_sequence(const YAML::Node &node, const std::string &path,
                                               std::int64_t minimum, std::int64_t maximum,
                                               std::size_t size_bound)
  {
    require_sequence(node, path);
    if (node.size() > size_bound) fail(path, "sequence exceeds the supported bound");
    std::vector<std::int64_t> result;
    result.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      result.push_back(
          integer64(node[index], path + "[" + std::to_string(index) + "]", minimum, maximum));
    return result;
  }

  std::vector<double> finite_sequence(const YAML::Node &node, const std::string &path,
                                      std::size_t size_bound)
  {
    require_sequence(node, path);
    if (node.size() > size_bound) fail(path, "sequence exceeds the supported bound");
    std::vector<double> result;
    result.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      result.push_back(finite_number(node[index], path + "[" + std::to_string(index) + "]"));
    return result;
  }

  std::complex<double> binary_complex(const YAML::Node &node, const std::string &path)
  {
    require_mapping(node, path);
    const auto values = finite_sequence(node["binary64"], path + ".binary64", 2);
    if (values.size() != 2)
      fail(path + ".binary64", "complex binary64 value must have two entries");
    return {values[0], values[1]};
  }

  void require_file_bound(const std::filesystem::path &path, std::uintmax_t maximum)
  {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error) fail(path.string(), "could not determine file size");
    if (size == 0 || size > maximum)
      fail(path.string(), "file size is outside the supported bound");
  }

  YAML::Node load_json(const std::filesystem::path &path, std::uintmax_t maximum)
  {
    require_file_bound(path, maximum);
    try {
      return YAML::LoadFile(path.string());
    } catch (const YAML::Exception &error) {
      fail(path.string(), std::string("invalid JSON: ") + error.what());
    }
  }

  std::filesystem::path bundle_member(const std::filesystem::path &root, const std::string &name,
                                      const std::string &path)
  {
    const std::filesystem::path relative(name);
    if (name.empty() || name == "." || name == ".." || relative.is_absolute() ||
        relative.filename().string() != name)
      fail(path, "bundle reference must be a simple relative filename");
    return root / relative;
  }

  std::map<std::string, std::string> read_manifest(const std::filesystem::path &path)
  {
    require_file_bound(path, 1ULL << 20);
    std::ifstream stream(path);
    if (!stream) fail(path.string(), "could not open manifest");
    std::map<std::string, std::string> result;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(stream, line)) {
      ++line_number;
      if (line.empty()) continue;
      if (line.size() < 67 || line[64] != ' ' || line[65] != ' ')
        fail(path.string(), "invalid manifest line " + std::to_string(line_number));
      const std::string digest = line.substr(0, 64);
      const std::string name = line.substr(66);
      if (!lower_sha256(digest)) fail(path.string(), "invalid manifest digest");
      (void) bundle_member(path.parent_path(), name, path.string());
      if (!result.emplace(name, digest).second) fail(path.string(), "duplicate manifest filename");
    }
    if (!stream.eof()) fail(path.string(), "failed while reading manifest");
    return result;
  }

  void require_manifest_file(const std::map<std::string, std::string> &manifest,
                             const std::filesystem::path &root, const std::string &name,
                             const std::string &expected_hash, const std::string &path)
  {
    const auto found = manifest.find(name);
    if (found == manifest.end() || found->second != expected_hash)
      fail(path, "manifest identity mismatch for " + name);
    const std::filesystem::path member = bundle_member(root, name, path);
    if (sha256_file(member.string()) != expected_hash)
      fail(path, "file SHA-256 mismatch for " + name);
  }

  std::string quoted_json_string(const std::string &value)
  {
    return "\"" + value + "\"";
  }

  std::string compute_deployment_identity_hash(const std::string &model_path)
  {
    std::string payload = "{";
    payload += "\"central_species_order\":" +
        canonical_json_root_member_value(model_path, "central_species_order");
    payload += ",\"compiler_artifact_self_hash\":" +
        canonical_json_nested_member_value(model_path, "compiler_artifact", "artifact_self_hash");
    payload += ",\"conventions\":" + canonical_json_root_member_value(model_path, "conventions");
    payload += ",\"ordinary_reference\":" +
        canonical_json_root_member_value(model_path, "ordinary_reference");
    payload += ",\"readout_coefficients\":" +
        canonical_json_nested_member_value(model_path, "readout", "coefficients");
    payload += ",\"readout_offsets\":" +
        canonical_json_nested_member_value(model_path, "readout", "offsets");
    payload +=
        ",\"source_plan_hash\":" + canonical_json_root_member_value(model_path, "source_plan_hash");
    payload += ",\"type_map\":" + canonical_json_root_member_value(model_path, "type_map");
    payload += '}';
    return sha256_string(payload);
  }

  std::string compute_composite_deployment_identity_hash(const std::string &model_path)
  {
    std::string payload = "{";
    payload += "\"central_species_order\":" +
        canonical_json_root_member_value(model_path, "central_species_order");
    payload += ",\"compiler_binding_self_hash\":" +
        canonical_json_nested_member_value(model_path, "compiler_binding", "binding_self_hash");
    payload += ",\"composite_artifact_hash\":" +
        canonical_json_nested_member_value(model_path, "compiler_binding",
                                           "composite_artifact_hash");
    payload += ",\"conventions\":" + canonical_json_root_member_value(model_path, "conventions");
    payload += ",\"ordinary_reference\":" +
        canonical_json_root_member_value(model_path, "ordinary_reference");
    payload += ",\"readout_coefficients\":" +
        canonical_json_nested_member_value(model_path, "readout", "coefficients");
    payload += ",\"readout_offsets\":" +
        canonical_json_nested_member_value(model_path, "readout", "offsets");
    payload +=
        ",\"source_plan_hash\":" + canonical_json_root_member_value(model_path, "source_plan_hash");
    payload += ",\"type_map\":" + canonical_json_root_member_value(model_path, "type_map");
    payload += '}';
    return sha256_string(payload);
  }

  void validate_conventions(const YAML::Node &node, const std::string &path)
  {
    require_exact_fields(node,
                         {"cutoff_support", "edge_displacement", "force_sign", "l1_component_order",
                          "lammps_virial", "lammps_virial_component_order", "precision",
                          "real_basis_id", "real_pullback", "self_interaction",
                          "strain_derivative"},
                         path);
    require_string(node["cutoff_support"], "0_lt_r_lt_rc", path + ".cutoff_support");
    require_string(node["edge_displacement"], "R_neighbor_minus_R_center_plus_periodic_image",
                   path + ".edge_displacement");
    require_string(node["force_sign"], "F_equals_minus_dE_dR", path + ".force_sign");
    if (string_sequence(node["l1_component_order"], path + ".l1_component_order") !=
        std::vector<std::string>{"x", "z", "minus_y"})
      fail(path + ".l1_component_order", "unsupported component ordering");
    require_string(node["lammps_virial"], "minus_strain_derivative", path + ".lammps_virial");
    if (string_sequence(node["lammps_virial_component_order"],
                        path + ".lammps_virial_component_order") !=
        std::vector<std::string>{"xx", "yy", "zz", "xy", "xz", "yz"})
      fail(path + ".lammps_virial_component_order", "unsupported virial ordering");
    require_string(node["precision"], "binary64", path + ".precision");
    require_string(node["real_basis_id"], "ye3t_physical_real_tesseral_v1",
                   path + ".real_basis_id");
    require_string(node["real_pullback"], "algebraic_transpose", path + ".real_pullback");
    require_string(node["self_interaction"], "excluded", path + ".self_interaction");
    require_string(node["strain_derivative"], "dE_d_epsilon", path + ".strain_derivative");
  }

  std::vector<int> parse_type_map(const YAML::Node &node, const std::vector<std::string> &species,
                                  const std::string &path)
  {
    require_mapping(node, path);
    if (node.size() != species.size()) fail(path, "type map and central species order differ");
    std::vector<int> result;
    std::set<int> values;
    result.reserve(species.size());
    for (const auto &name : species) {
      const YAML::Node value = node[name];
      if (!value) fail(path, "type map omits species " + name);
      const int parsed = integer(value, path + "." + name, 0);
      if (!values.insert(parsed).second) fail(path, "type map values must be unique");
      result.push_back(parsed);
    }
    return result;
  }

  using MonomialKey = std::vector<std::int64_t>;
  using DescriptorRow = std::map<MonomialKey, double>;
  using RealFormMatrix = std::vector<std::vector<std::complex<double>>>;

  struct CompilerPolynomialBinding {
    std::vector<DescriptorRow> rows;
    std::int64_t physical_term_count = 0;
    double maximum_imaginary_residual = 0.0;
    double maximum_coefficient_scale = 1.0;
  };

  struct SourceChannelBinding {
    int position = -1;
    int radial_channel = -1;
    int angular = -1;
    int component_count = 0;
    std::int64_t source_offset = -1;
    std::string neighbor_species;
    std::string source_family_id;
  };

  struct NativeChannelBlock {
    int position = -1;
    int angular = -1;
    int component_count = 0;
    std::int64_t source_offset = -1;
  };

  struct CompactCompilerBinding {
    std::string self_hash;
    std::string composite_artifact_hash;
    std::string component_row_certificate_hash;
    std::int64_t descriptor_count = 0;
    std::int64_t source_variable_count = 0;
    std::int64_t physical_term_count = 0;
    double maximum_imaginary_residual = 0.0;
    double maximum_coefficient_scale = 1.0;
  };

  CompactCompilerBinding validate_compact_compiler_binding(
      const std::filesystem::path &binding_path, const std::filesystem::path &model_path,
      const std::filesystem::path &root, const std::map<std::string, std::string> &manifest,
      std::set<std::string> &expected_files, const std::string &expected_self_hash,
      const std::string &expected_composite_hash, const std::string &source_plan_hash,
      int head_count, int descriptor_count)
  {
    const YAML::Node binding = load_json(binding_path, MAX_MODEL_BYTES);
    require_exact_fields(binding,
                         {"components", "composite_artifact_hash", "descriptor_coordinate_ids",
                          "fit_coordinate_provenance", "fit_coordinate_provenance_sha256",
                          "physical_real_lowering", "readout", "schema", "self_hash",
                          "source_plan_hash", "source_registry", "source_registry_sha256",
                          "validation"},
                         "compiler_binding");
    require_string(binding["schema"], "ye3t_lifted_cauchy_composite_compiler_binding_v1",
                   "compiler_binding.schema");
    const std::string self_hash = sha256(binding["self_hash"], "compiler_binding.self_hash");
    if (self_hash != expected_self_hash ||
        canonical_json_hash_without_root_member(binding_path.string(), "self_hash") != self_hash)
      fail(binding_path.string(), "compiler-binding self-hash mismatch");
    const std::string composite_hash =
        sha256(binding["composite_artifact_hash"], "compiler_binding.composite_artifact_hash");
    if (composite_hash != expected_composite_hash)
      fail("compiler_binding.composite_artifact_hash",
           "model and compiler-binding composite identities differ");
    if (sha256(binding["source_plan_hash"], "compiler_binding.source_plan_hash") !=
        source_plan_hash)
      fail("compiler_binding.source_plan_hash", "model and compiler-binding sources differ");
    const std::string fit_provenance_hash =
        sha256(binding["fit_coordinate_provenance_sha256"],
               "compiler_binding.fit_coordinate_provenance_sha256");
    require_mapping(binding["fit_coordinate_provenance"],
                    "compiler_binding.fit_coordinate_provenance");
    const std::string fit_provenance_payload = "{\"fit_metadata\":" +
        canonical_json_root_member_value(binding_path.string(), "fit_coordinate_provenance") + '}';
    if (fit_provenance_hash != sha256_string(fit_provenance_payload))
      fail("compiler_binding.fit_coordinate_provenance_sha256",
           "fit-coordinate provenance hash mismatch");
    if (canonical_json_root_member_value(binding_path.string(), "fit_coordinate_provenance") !=
        canonical_json_root_member_value(model_path.string(), "fit_metadata"))
      fail("compiler_binding.fit_coordinate_provenance",
           "compiler binding and model fit provenance differ");

    const auto coordinate_ids =
        string_sequence(binding["descriptor_coordinate_ids"],
                        "compiler_binding.descriptor_coordinate_ids", MAX_TERMS);
    if (coordinate_ids.size() != static_cast<std::size_t>(descriptor_count) ||
        std::set<std::string>(coordinate_ids.begin(), coordinate_ids.end()).size() !=
            coordinate_ids.size())
      fail("compiler_binding.descriptor_coordinate_ids",
           "descriptor coordinates are incomplete or duplicated");

    require_exact_fields(binding["readout"],
                         {"coefficient_shape", "coefficients_json_sha256", "coefficients_sha256",
                          "offset_shape", "offsets_json_sha256", "offsets_sha256"},
                         "compiler_binding.readout");
    const auto coefficient_shape = integer_sequence(binding["readout"]["coefficient_shape"],
                                                    "compiler_binding.readout.coefficient_shape", 0,
                                                    std::numeric_limits<int>::max(), 2);
    const auto offset_shape = integer_sequence(binding["readout"]["offset_shape"],
                                               "compiler_binding.readout.offset_shape", 0,
                                               std::numeric_limits<int>::max(), 1);
    if (coefficient_shape != std::vector<int>{head_count, descriptor_count} ||
        offset_shape != std::vector<int>{head_count})
      fail("compiler_binding.readout", "readout shapes changed");
    const std::string coefficient_json_hash =
        sha256(binding["readout"]["coefficients_json_sha256"],
               "compiler_binding.readout.coefficients_json_sha256");
    const std::string offset_json_hash = sha256(binding["readout"]["offsets_json_sha256"],
                                                "compiler_binding.readout.offsets_json_sha256");
    if (coefficient_json_hash !=
            sha256_string(canonical_json_nested_member_value(model_path.string(), "readout",
                                                             "coefficients")) ||
        offset_json_hash !=
            sha256_string(
                canonical_json_nested_member_value(model_path.string(), "readout", "offsets")))
      fail("compiler_binding.readout", "model readout identity changed");
    (void) sha256(binding["readout"]["coefficients_sha256"],
                  "compiler_binding.readout.coefficients_sha256");
    (void) sha256(binding["readout"]["offsets_sha256"], "compiler_binding.readout.offsets_sha256");

    require_exact_fields(binding["validation"],
                         {"component_artifact_hashes_verified", "component_order_verified",
                          "passed", "runtime_gram_solve", "shared_source_registry_verified",
                          "strict_sector_disjoint_record_asserted",
                          "strict_sector_identity_provenance"},
                         "compiler_binding.validation");
    require_bool(binding["validation"]["passed"], true, "compiler_binding.validation.passed");
    require_bool(binding["validation"]["component_artifact_hashes_verified"], true,
                 "compiler_binding.validation.component_artifact_hashes_verified");
    require_bool(binding["validation"]["component_order_verified"], true,
                 "compiler_binding.validation.component_order_verified");
    require_bool(binding["validation"]["strict_sector_disjoint_record_asserted"], true,
                 "compiler_binding.validation."
                 "strict_sector_disjoint_record_asserted");
    require_string(binding["validation"]["strict_sector_identity_provenance"],
                   "hash_bound_component_records_and_fit_certificate",
                   "compiler_binding.validation."
                   "strict_sector_identity_provenance");
    require_bool(binding["validation"]["shared_source_registry_verified"], true,
                 "compiler_binding.validation.shared_source_registry_verified");
    require_bool(binding["validation"]["runtime_gram_solve"], false,
                 "compiler_binding.validation.runtime_gram_solve");

    require_sequence(binding["source_registry"], "compiler_binding.source_registry");
    std::set<std::string> source_keys;
    for (std::size_t index = 0; index < binding["source_registry"].size(); ++index) {
      const YAML::Node channel = binding["source_registry"][index];
      const std::string path = "compiler_binding.source_registry[" + std::to_string(index) + "]";
      require_exact_fields(
          channel, {"channel_index", "l", "neighbor_species", "radial_channel", "source_family_id"},
          path);
      if (integer(channel["channel_index"], path + ".channel_index", 0) != static_cast<int>(index))
        fail(path + ".channel_index", "source channels must be dense and ordered");
      const std::string key =
          scalar_string(channel["neighbor_species"], path + ".neighbor_species") + "\n" +
          scalar_string(channel["source_family_id"], path + ".source_family_id") + "\n" +
          std::to_string(integer(channel["l"], path + ".l", 0)) + "\n" +
          std::to_string(integer(channel["radial_channel"], path + ".radial_channel", 0));
      if (!source_keys.insert(key).second) fail(path, "duplicate complete source channel");
    }
    if (source_keys.empty())
      fail("compiler_binding.source_registry", "source registry must be nonempty");
    const std::string source_registry_hash =
        sha256(binding["source_registry_sha256"], "compiler_binding.source_registry_sha256");
    const std::string source_registry_payload = "{\"channels\":" +
        canonical_json_root_member_value(binding_path.string(), "source_registry") + '}';
    if (source_registry_hash != sha256_string(source_registry_payload))
      fail("compiler_binding.source_registry_sha256", "source-registry hash mismatch");

    require_sequence(binding["components"], "compiler_binding.components");
    if (binding["components"].size() == 0 || binding["components"].size() > 4096)
      fail("compiler_binding.components", "component count is outside bounds");
    std::set<std::string> strict_sectors;
    int next_descriptor = 0;
    std::int64_t component_term_count = 0;
    for (std::size_t position = 0; position < binding["components"].size(); ++position) {
      const YAML::Node component = binding["components"][position];
      const std::string path = "compiler_binding.components[" + std::to_string(position) + "]";
      require_exact_fields(
          component,
          {"compiler_artifact", "component_channel_map", "component_id", "coordinate_ids",
           "descriptor_count", "descriptor_offset", "opportunity_id", "orthogonal_output_plan_hash",
           "physical_real_rows_sha256", "physical_scalar_reality_report_hash",
           "physical_real_term_count", "position", "selection_hash", "strict_sector_ids"},
          path);
      if (integer(component["position"], path + ".position", 0) != static_cast<int>(position) ||
          integer(component["descriptor_offset"], path + ".descriptor_offset", 0) !=
              next_descriptor)
        fail(path, "component position or descriptor offset changed");
      const int count =
          integer(component["descriptor_count"], path + ".descriptor_count", 1, descriptor_count);
      const auto local_coordinates =
          string_sequence(component["coordinate_ids"], path + ".coordinate_ids",
                          static_cast<std::size_t>(descriptor_count));
      if (next_descriptor > descriptor_count - count ||
          local_coordinates.size() != static_cast<std::size_t>(count) ||
          !std::equal(local_coordinates.begin(), local_coordinates.end(),
                      coordinate_ids.begin() + next_descriptor))
        fail(path + ".coordinate_ids", "component coordinate order changed");
      next_descriptor += count;
      const auto component_channel_map =
          integer_sequence(component["component_channel_map"], path + ".component_channel_map", 0,
                           static_cast<int>(source_keys.size()) - 1);
      if (component_channel_map.empty() ||
          std::set<int>(component_channel_map.begin(), component_channel_map.end()).size() !=
              component_channel_map.size())
        fail(path + ".component_channel_map",
             "component source channels must be nonempty and injective");
      (void) sha256(component["physical_real_rows_sha256"], path + ".physical_real_rows_sha256");
      (void) sha256(component["physical_scalar_reality_report_hash"],
                    path + ".physical_scalar_reality_report_hash");
      if (!component["orthogonal_output_plan_hash"].IsNull())
        (void) sha256(component["orthogonal_output_plan_hash"],
                      path + ".orthogonal_output_plan_hash");
      component_term_count += integer64(component["physical_real_term_count"],
                                        path + ".physical_real_term_count", 0, MAX_TERMS);
      const auto sectors =
          string_sequence(component["strict_sector_ids"], path + ".strict_sector_ids");
      if (sectors.empty()) fail(path + ".strict_sector_ids", "component has no strict sector");
      for (const auto &sector : sectors)
        if (!strict_sectors.insert(sector).second)
          fail(path + ".strict_sector_ids", "strict sectors overlap");
      (void) scalar_string(component["component_id"], path + ".component_id");
      (void) scalar_string(component["opportunity_id"], path + ".opportunity_id");
      (void) sha256(component["selection_hash"], path + ".selection_hash");

      const YAML::Node artifact = component["compiler_artifact"];
      require_exact_fields(artifact, {"artifact_self_hash", "file", "file_sha256"},
                           path + ".compiler_artifact");
      const std::string name = scalar_string(artifact["file"], path + ".compiler_artifact.file");
      const std::string file_hash =
          sha256(artifact["file_sha256"], path + ".compiler_artifact.file_sha256");
      const std::string artifact_hash =
          sha256(artifact["artifact_self_hash"], path + ".compiler_artifact.artifact_self_hash");
      const std::filesystem::path artifact_path =
          bundle_member(root, name, path + ".compiler_artifact.file");
      require_file_bound(artifact_path, MAX_COMPILER_BYTES);
      require_manifest_file(manifest, root, name, file_hash, path);
      expected_files.insert(name);
      if (canonical_json_root_member_value(artifact_path.string(), "self_hash") !=
              quoted_json_string(artifact_hash) ||
          canonical_json_hash_without_root_member(artifact_path.string(), "self_hash") !=
              artifact_hash)
        fail(artifact_path.string(), "component artifact self-hash mismatch");
    }
    if (next_descriptor != descriptor_count)
      fail("compiler_binding.components", "components do not cover the readout");

    require_exact_fields(binding["physical_real_lowering"],
                         {"algorithm", "component_row_certificate_sha256", "descriptor_count",
                          "maximum_absolute_imaginary_residual", "maximum_coefficient_scale",
                          "physical_real_term_count_before_readout", "real_basis_id",
                          "source_variable_count"},
                         "compiler_binding.physical_real_lowering");
    require_string(binding["physical_real_lowering"]["algorithm"],
                   "component_canonical_rows_then_shared_source_remap_v1",
                   "compiler_binding.physical_real_lowering.algorithm");
    require_string(binding["physical_real_lowering"]["real_basis_id"],
                   "ye3t_physical_real_tesseral_v1",
                   "compiler_binding.physical_real_lowering.real_basis_id");
    CompactCompilerBinding result;
    result.self_hash = self_hash;
    result.composite_artifact_hash = composite_hash;
    result.component_row_certificate_hash =
        sha256(binding["physical_real_lowering"]["component_row_certificate_sha256"],
               "compiler_binding.physical_real_lowering.component_row_certificate_"
               "sha256");
    result.descriptor_count =
        integer64(binding["physical_real_lowering"]["descriptor_count"],
                  "compiler_binding.physical_real_lowering.descriptor_count", 0, MAX_TERMS);
    result.physical_term_count =
        integer64(binding["physical_real_lowering"]["physical_real_term_count_before_readout"],
                  "compiler_binding.physical_real_lowering."
                  "physical_real_term_count_before_readout",
                  0, MAX_TERMS);
    result.maximum_imaginary_residual =
        finite_number(binding["physical_real_lowering"]["maximum_absolute_imaginary_residual"],
                      "compiler_binding.physical_real_lowering."
                      "maximum_absolute_imaginary_residual");
    result.maximum_coefficient_scale =
        finite_number(binding["physical_real_lowering"]["maximum_coefficient_scale"],
                      "compiler_binding.physical_real_lowering.maximum_coefficient_scale");
    result.source_variable_count = integer64(
        binding["physical_real_lowering"]["source_variable_count"],
        "compiler_binding.physical_real_lowering.source_variable_count", 1, MAX_SOURCE_VARIABLES);
    if (result.descriptor_count != descriptor_count ||
        result.physical_term_count != component_term_count)
      fail("compiler_binding.physical_real_lowering",
           "component and aggregate realification counts differ");
    if (result.maximum_imaginary_residual < 0.0 || result.maximum_coefficient_scale < 0.0 ||
        result.maximum_imaginary_residual >
            5.0e-11 * std::max(1.0, result.maximum_coefficient_scale))
      fail("compiler_binding.physical_real_lowering",
           "physical-real residual certificate is outside tolerance");
    return result;
  }

  // Rebuild the compiler-owned canonical descriptor rows under
  // A_complex = R * A_real.  This load-time audit binds the fitted readout to
  // the executable native polynomial without adding work to a timestep.
  CompilerPolynomialBinding reconstruct_compiler_rows(
      const std::filesystem::path &compiler_path, int role_dimension, std::int64_t source_count,
      bool native_v2, const std::vector<LiftedCauchySourceGroup> &groups, int descriptor_count)
  {
    const YAML::Node compiler = load_json(compiler_path, MAX_COMPILER_BINDING_DOM_BYTES);
    require_mapping(compiler, "compiler");
    require_string(compiler["schema"], "ye3t_linear_lifted_cauchy_scalar_v1", "compiler.schema");
    require_mapping(compiler["validation_report"], "compiler.validation_report");
    require_bool(compiler["validation_report"]["passed"], true,
                 "compiler.validation_report.passed");
    const bool ordered_canonical_exact =
        boolean(compiler["validation_report"]["ordered_canonical_exact"],
                "compiler.validation_report.ordered_canonical_exact");
    require_string(compiler["validation_report"]["real_form_convention"],
                   "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1",
                   "compiler.validation_report.real_form_convention");

    const YAML::Node payload = compiler["payload"];
    require_mapping(payload, "compiler.payload");
    require_mapping(payload["capabilities"], "compiler.payload.capabilities");
    const bool ordered_reference = boolean(payload["capabilities"]["ordered_reference"],
                                           "compiler.payload.capabilities.ordered_reference");
    if (ordered_canonical_exact != ordered_reference)
      fail("compiler.validation_report.ordered_canonical_exact",
           "must match the ordered-reference capability");
    if (integer(payload["role_dimension"], "compiler.payload.role_dimension", 1) != role_dimension)
      fail("compiler.payload.role_dimension", "compiler and native role dimensions differ");

    std::map<int, SourceChannelBinding> source_bindings;
    for (const auto &group : groups) {
      for (std::size_t local = 0; local < group.channel_indices.size(); ++local) {
        SourceChannelBinding binding;
        binding.position = group.channel_positions[local];
        binding.radial_channel = static_cast<int>(local);
        binding.angular = group.angular;
        binding.component_count = group.real_component_count;
        binding.source_offset = group.q_source_variable_offsets[2 * local];
        binding.neighbor_species = group.neighbor_species;
        binding.source_family_id = group.source_family_id;
        if (!source_bindings.emplace(group.channel_indices[local], std::move(binding)).second)
          fail("native.source_groups", "compiler channel indices must have unique source bindings");
      }
    }

    require_sequence(payload["channels"], "compiler.payload.channels");
    if (payload["channels"].size() != source_bindings.size())
      fail("compiler.payload.channels", "compiler and native channel counts differ");
    std::map<int, int> channel_positions;
    for (std::size_t position = 0; position < payload["channels"].size(); ++position) {
      const YAML::Node channel = payload["channels"][position];
      const std::string path = "compiler.payload.channels[" + std::to_string(position) + "]";
      require_mapping(channel, path);
      const int channel_index = integer(channel["channel_index"], path + ".channel_index", 0);
      if (channel_index != static_cast<int>(position))
        fail(path, "native runtime requires dense compiler channel ordering");
      if (!channel["channel_id"] ||
          (!native_v2 && integer(channel["channel_id"], path + ".channel_id", 0) != channel_index))
        fail(path, "compiler channel identifier is invalid");
      const auto found = source_bindings.find(channel_index);
      if (found == source_bindings.end() || found->second.position != static_cast<int>(position) ||
          integer(channel["l"], path + ".l", 0) != found->second.angular ||
          integer(channel["radial_channel"], path + ".radial_channel", 0) !=
              found->second.radial_channel ||
          scalar_string(channel["neighbor_species"], path + ".neighbor_species") !=
              found->second.neighbor_species ||
          scalar_string(channel["source_family_id"], path + ".source_family_id") !=
              found->second.source_family_id)
        fail(path, "compiler channel disagrees with the native source binding");
      channel_positions.emplace(channel_index, static_cast<int>(position));
    }

    require_sequence(payload["real_forms"], "compiler.payload.real_forms");
    std::map<std::string, RealFormMatrix> real_forms;
    for (std::size_t form_index = 0; form_index < payload["real_forms"].size(); ++form_index) {
      const YAML::Node form = payload["real_forms"][form_index];
      const std::string path = "compiler.payload.real_forms[" + std::to_string(form_index) + "]";
      require_mapping(form, path);
      require_string(form["convention_id"],
                     "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1",
                     path + ".convention_id");
      require_string(form["algebraic_reverse"], "transpose", path + ".algebraic_reverse");
      require_string(form["polynomial_pairing"], "complex_multilinear_no_conjugation",
                     path + ".polynomial_pairing");
      const int angular = integer(form["angular_l"], path + ".angular_l", 0);
      const int width = 2 * angular + 1;
      std::vector<int> expected_magnetic_order;
      expected_magnetic_order.reserve(static_cast<std::size_t>(width));
      for (int magnetic = -angular; magnetic <= angular; ++magnetic)
        expected_magnetic_order.push_back(magnetic);
      if (integer_sequence(form["magnetic_order"], path + ".magnetic_order", -angular, angular) !=
          expected_magnetic_order)
        fail(path, "unsupported native real-form angular ordering");
      require_sequence(form["real_coordinate_order"], path + ".real_coordinate_order");
      if (form["real_coordinate_order"].size() != static_cast<std::size_t>(width))
        fail(path + ".real_coordinate_order",
             "native real-form width disagrees with angular momentum");
      for (int component = 0; component < width; ++component) {
        const YAML::Node coordinate = form["real_coordinate_order"][component];
        std::string expected_kind = "m_zero";
        int expected_m = 0;
        if (component < angular) {
          expected_kind = "cosine";
          expected_m = angular - component;
        } else if (component > angular) {
          expected_kind = "sine";
          expected_m = component - angular;
        }
        if (scalar_string(coordinate["kind"], path + ".real_coordinate_order.kind") !=
                expected_kind ||
            integer(coordinate["m"], path + ".real_coordinate_order.m", 0, angular) != expected_m)
          fail(path + ".real_coordinate_order", "unsupported native real-coordinate ordering");
      }
      require_sequence(form["real_to_complex_matrix"], path + ".real_to_complex_matrix");
      if (form["real_to_complex_matrix"].size() != static_cast<std::size_t>(width))
        fail(path + ".real_to_complex_matrix", "matrix width disagrees with angular momentum");
      RealFormMatrix matrix(static_cast<std::size_t>(width),
                            std::vector<std::complex<double>>(static_cast<std::size_t>(width)));
      for (int magnetic = 0; magnetic < width; ++magnetic) {
        const YAML::Node row = form["real_to_complex_matrix"][magnetic];
        require_sequence(row, path + ".real_to_complex_matrix.row");
        if (row.size() != static_cast<std::size_t>(width))
          fail(path + ".real_to_complex_matrix",
               "matrix row width disagrees with angular momentum");
        for (int component = 0; component < width; ++component)
          matrix[magnetic][component] =
              binary_complex(row[component], path + ".real_to_complex_matrix.entry");
      }
      const std::string id = scalar_string(form["real_form_id"], path + ".real_form_id");
      if (!real_forms.emplace(id, std::move(matrix)).second)
        fail("compiler.payload.real_forms", "duplicate real-form identifier");
    }

    require_sequence(payload["channel_real_form_ids"], "compiler.payload.channel_real_form_ids");
    if (payload["channel_real_form_ids"].size() != channel_positions.size())
      fail("compiler.payload.channel_real_form_ids",
           "real-form bindings do not cover every channel");
    std::map<int, const RealFormMatrix *> channel_forms;
    for (std::size_t index = 0; index < payload["channel_real_form_ids"].size(); ++index) {
      const YAML::Node binding = payload["channel_real_form_ids"][index];
      const int channel =
          integer(binding["channel_index"], "compiler.channel_real_form_ids.channel_index", 0);
      const std::string id =
          scalar_string(binding["real_form_id"], "compiler.channel_real_form_ids.real_form_id");
      const auto form = real_forms.find(id);
      if (channel_positions.count(channel) == 0 || form == real_forms.end() ||
          !channel_forms.emplace(channel, &form->second).second)
        fail("compiler.payload.channel_real_form_ids",
             "invalid or duplicate channel real-form binding");
    }

    require_mapping(payload["physical_scalar_reality_report"],
                    "compiler.payload.physical_scalar_reality_report");
    require_string(payload["physical_scalar_reality_report"]["basis"],
                   "real_tesseral_from_complex_condon_shortley_young_orthogonal_v1",
                   "compiler.payload.physical_scalar_reality_report.basis");
    require_bool(payload["physical_scalar_reality_report"]["exactly_real"], true,
                 "compiler.payload.physical_scalar_reality_report.exactly_real");
    require_string(payload["physical_scalar_reality_report"]["proof"],
                   "exact_canonical_polynomial_realification",
                   "compiler.payload.physical_scalar_reality_report.proof");
    if (integer(payload["physical_scalar_reality_report"]["descriptor_count"],
                "compiler.payload.physical_scalar_reality_report.descriptor_count",
                0) != descriptor_count ||
        finite_number(
            payload["physical_scalar_reality_report"]["maximum_absolute_imaginary_scalar_residual"],
            "compiler.payload.physical_scalar_reality_report."
            "maximum_absolute_imaginary_scalar_residual") != 0.0)
      fail("compiler.payload.physical_scalar_reality_report",
           "compiler scalar-reality certificate is inconsistent");

    require_sequence(payload["descriptors"], "compiler.payload.descriptors");
    if (payload["descriptors"].size() != static_cast<std::size_t>(descriptor_count))
      fail("compiler.payload.descriptors",
           "compiler descriptor count differs from the model readout");
    CompilerPolynomialBinding result;
    result.rows.reserve(static_cast<std::size_t>(descriptor_count));
    std::int64_t realification_visits = 0;
    for (int descriptor_index = 0; descriptor_index < descriptor_count; ++descriptor_index) {
      const YAML::Node descriptor = payload["descriptors"][descriptor_index];
      const std::string path =
          "compiler.payload.descriptors[" + std::to_string(descriptor_index) + "]";
      if (integer(descriptor["descriptor_index"], path + ".descriptor_index", 0) !=
          descriptor_index)
        fail(path, "descriptor indices must be dense and ordered");
      const int tensor_rank = integer(descriptor["label"]["rank"], path + ".label.rank", 0);
      if (integer(descriptor["label"]["target_L"], path + ".label.target_L", 0) != 0 ||
          integer(descriptor["label"]["target_parity"], path + ".label.target_parity", -1, 1) != 1)
        fail(path + ".label", "native energy requires an even L=0 descriptor");
      require_sequence(descriptor["canonical_terms"], path + ".canonical_terms");
      std::map<MonomialKey, std::complex<double>> physical;
      for (std::size_t term_index = 0; term_index < descriptor["canonical_terms"].size();
           ++term_index) {
        const YAML::Node term = descriptor["canonical_terms"][term_index];
        const std::string term_path = path + ".canonical_terms[" + std::to_string(term_index) + "]";
        require_sequence(term["coordinates"], term_path + ".coordinates");
        if (term["coordinates"].size() != static_cast<std::size_t>(tensor_rank))
          fail(term_path + ".coordinates", "canonical term degree disagrees with descriptor rank");
        std::map<MonomialKey, std::complex<double>> terms;
        terms.emplace(MonomialKey{},
                      binary_complex(term["coefficient"], term_path + ".coefficient"));
        for (std::size_t factor = 0; factor < term["coordinates"].size(); ++factor) {
          const YAML::Node coordinate = term["coordinates"][factor];
          const auto indices = integer_sequence(coordinate, term_path + ".coordinates.factor", 0,
                                                std::numeric_limits<int>::max(), 3);
          if (indices.size() != 3)
            fail(term_path + ".coordinates", "canonical coordinate must have three indices");
          const auto binding = source_bindings.find(indices[0]);
          if (indices[1] >= role_dimension || binding == source_bindings.end() ||
              indices[2] >= binding->second.component_count)
            fail(term_path + ".coordinates",
                 "canonical coordinate is outside the native source space");
          const RealFormMatrix &matrix = *channel_forms.at(indices[0]);
          std::vector<std::pair<std::int64_t, std::complex<double>>> factors;
          for (int component = 0; component < binding->second.component_count; ++component) {
            const std::complex<double> value = matrix[indices[2]][component];
            if (value == std::complex<double>{}) continue;
            const std::int64_t source_index = binding->second.source_offset +
                static_cast<std::int64_t>(indices[1]) * binding->second.component_count + component;
            if (source_index < 0 || source_index >= source_count)
              fail(term_path + ".coordinates",
                   "realified source index is outside the native source space");
            factors.emplace_back(source_index, value);
          }
          if (factors.empty()) {
            terms.clear();
            break;
          }
          if (terms.size() > static_cast<std::size_t>(MAX_REALIFICATION_VISITS) / factors.size())
            fail(term_path, "physical-real expansion exceeds the visit bound");
          realification_visits += static_cast<std::int64_t>(terms.size() * factors.size());
          if (realification_visits > MAX_REALIFICATION_VISITS)
            fail(term_path, "physical-real expansion exceeds the visit bound");
          std::map<MonomialKey, std::complex<double>> next;
          for (const auto &partial : terms)
            for (const auto &item : factors) {
              MonomialKey key = partial.first;
              key.push_back(item.first);
              std::sort(key.begin(), key.end());
              next[key] += partial.second * item.second;
            }
          if (next.size() > static_cast<std::size_t>(MAX_TERMS))
            fail(term_path, "physical-real expansion exceeds the term bound");
          terms = std::move(next);
        }
        for (const auto &item : terms) physical[item.first] += item.second;
        if (physical.size() > static_cast<std::size_t>(MAX_TERMS))
          fail(path, "physical-real descriptor row exceeds the term bound");
      }

      DescriptorRow row;
      for (const auto &item : physical) {
        if (item.second == std::complex<double>{}) continue;
        result.maximum_imaginary_residual =
            std::max(result.maximum_imaginary_residual, std::abs(item.second.imag()));
        result.maximum_coefficient_scale =
            std::max(result.maximum_coefficient_scale, std::abs(item.second));
        if (item.second.real() != 0.0) row.emplace(item.first, item.second.real());
      }
      if (result.maximum_imaginary_residual > 5.0e-11 * result.maximum_coefficient_scale)
        fail(path, "physical-real compiler row has a material imaginary residual");
      if (result.physical_term_count > MAX_TERMS - static_cast<std::int64_t>(row.size()))
        fail(path, "physical-real compiler rows exceed the term bound");
      result.physical_term_count += static_cast<std::int64_t>(row.size());
      result.rows.push_back(std::move(row));
    }
    if (integer64(payload["physical_scalar_reality_report"]["transformed_coefficient_count"],
                  "compiler.payload.physical_scalar_reality_report."
                  "transformed_coefficient_count",
                  0, MAX_TERMS) != result.physical_term_count)
      fail("compiler.payload.physical_scalar_reality_report",
           "compiler realified-term certificate is inconsistent");
    return result;
  }

  LiftedCauchySparsePolynomial lower_compiler_readout(const CompilerPolynomialBinding &binding,
                                                      const std::vector<double> &weights,
                                                      double offset, const std::string &path)
  {
    if (binding.rows.size() != weights.size())
      fail(path, "compiler row and readout dimensions differ");
    std::map<MonomialKey, double> combined;
    double constant = offset;
    for (std::size_t descriptor = 0; descriptor < binding.rows.size(); ++descriptor) {
      if (weights[descriptor] == 0.0) continue;
      for (const auto &item : binding.rows[descriptor]) {
        const double value = weights[descriptor] * item.second;
        if (item.first.empty())
          constant += value;
        else
          combined[item.first] += value;
      }
    }

    LiftedCauchySparsePolynomial result;
    result.offset = constant;
    result.factor_offsets.push_back(0);
    for (const auto &item : combined) {
      if (item.second == 0.0) continue;
      if (result.monomial_coefficients.size() >= static_cast<std::size_t>(MAX_TERMS))
        fail(path, "lowered readout exceeds the term bound");
      if (item.first.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        fail(path, "lowered readout tensor rank exceeds the native bound");
      result.maximum_tensor_rank =
          std::max(result.maximum_tensor_rank, static_cast<int>(item.first.size()));
      std::int64_t previous = -1;
      for (const std::int64_t source_index : item.first) {
        if (source_index == previous) {
          ++result.factor_exponents.back();
        } else {
          result.factor_indices.push_back(source_index);
          result.factor_exponents.push_back(1);
          previous = source_index;
        }
      }
      if (result.factor_indices.size() > static_cast<std::size_t>(MAX_FACTOR_REFERENCES))
        fail(path, "lowered readout exceeds the factor-reference bound");
      const std::int64_t factor_count =
          static_cast<std::int64_t>(result.factor_indices.size()) - result.factor_offsets.back();
      result.maximum_factor_count = std::max(result.maximum_factor_count, factor_count);
      result.factor_offsets.push_back(static_cast<std::int64_t>(result.factor_indices.size()));
      result.monomial_coefficients.push_back(item.second);
    }
    return result;
  }

  void require_same_polynomial(const LiftedCauchySparsePolynomial &actual,
                               const LiftedCauchySparsePolynomial &expected,
                               const std::string &path)
  {
    if (actual.factor_offsets != expected.factor_offsets ||
        actual.factor_indices != expected.factor_indices ||
        actual.factor_exponents != expected.factor_exponents ||
        actual.maximum_tensor_rank != expected.maximum_tensor_rank ||
        actual.maximum_factor_count != expected.maximum_factor_count ||
        actual.monomial_coefficients.size() != expected.monomial_coefficients.size() ||
        !same_binary64(actual.offset, expected.offset))
      fail(path, "native polynomial disagrees with compiler/readout lowering");
    for (std::size_t term = 0; term < actual.monomial_coefficients.size(); ++term)
      if (!same_binary64(actual.monomial_coefficients[term], expected.monomial_coefficients[term]))
        fail(path, "native polynomial disagrees with compiler/readout lowering");
  }

  LiftedCauchySparsePolynomial parse_polynomial(const YAML::Node &node, std::int64_t source_count,
                                                const std::string &path)
  {
    require_exact_fields(node,
                         {"factor_exponents", "factor_indices", "factor_offsets",
                          "maximum_factor_count", "maximum_tensor_rank", "monomial_coefficients",
                          "offset", "term_count"},
                         path);
    const std::int64_t term_count =
        integer64(node["term_count"], path + ".term_count", 0, MAX_TERMS);
    LiftedCauchySparsePolynomial result;
    result.offset = finite_number(node["offset"], path + ".offset");
    result.maximum_tensor_rank =
        integer(node["maximum_tensor_rank"], path + ".maximum_tensor_rank", 0);
    result.maximum_factor_count = integer64(
        node["maximum_factor_count"], path + ".maximum_factor_count", 0, MAX_SOURCE_VARIABLES);
    result.factor_offsets =
        integer64_sequence(node["factor_offsets"], path + ".factor_offsets", 0,
                           MAX_FACTOR_REFERENCES, static_cast<std::size_t>(term_count + 1));
    if (result.factor_offsets.size() != static_cast<std::size_t>(term_count + 1) ||
        result.factor_offsets.front() != 0 ||
        !std::is_sorted(result.factor_offsets.begin(), result.factor_offsets.end()))
      fail(path + ".factor_offsets", "invalid sparse term offsets");
    const std::int64_t factor_count = result.factor_offsets.back();
    result.factor_indices = integer64_sequence(node["factor_indices"], path + ".factor_indices", 0,
                                               source_count == 0 ? 0 : source_count - 1,
                                               static_cast<std::size_t>(MAX_FACTOR_REFERENCES));
    result.factor_exponents = integer64_sequence(
        node["factor_exponents"], path + ".factor_exponents", 1, std::numeric_limits<int>::max(),
        static_cast<std::size_t>(MAX_FACTOR_REFERENCES));
    if (result.factor_indices.size() != static_cast<std::size_t>(factor_count) ||
        result.factor_exponents.size() != static_cast<std::size_t>(factor_count))
      fail(path, "sparse factor arrays disagree with offsets");
    result.monomial_coefficients =
        finite_sequence(node["monomial_coefficients"], path + ".monomial_coefficients",
                        static_cast<std::size_t>(MAX_TERMS));
    if (result.monomial_coefficients.size() != static_cast<std::size_t>(term_count))
      fail(path + ".monomial_coefficients", "term count mismatch");

    std::int64_t observed_factor_count = 0;
    std::int64_t observed_rank = 0;
    std::set<std::vector<std::pair<std::int64_t, std::int64_t>>> observed_terms;
    for (std::int64_t term = 0; term < term_count; ++term) {
      const std::int64_t begin = result.factor_offsets[term];
      const std::int64_t end = result.factor_offsets[term + 1];
      if (begin == end)
        fail(path + ".factor_offsets", "constant terms must be folded into the polynomial offset");
      observed_factor_count = std::max(observed_factor_count, end - begin);
      std::int64_t rank = 0;
      std::int64_t previous = -1;
      std::vector<std::pair<std::int64_t, std::int64_t>> current_term;
      current_term.reserve(static_cast<std::size_t>(end - begin));
      for (std::int64_t cursor = begin; cursor < end; ++cursor) {
        if (result.factor_indices[cursor] <= previous)
          fail(path + ".factor_indices",
               "factor indices must be strictly increasing within a term");
        previous = result.factor_indices[cursor];
        current_term.emplace_back(result.factor_indices[cursor], result.factor_exponents[cursor]);
        if (rank > std::numeric_limits<int>::max() - result.factor_exponents[cursor])
          fail(path, "tensor rank overflows the native bound");
        rank += result.factor_exponents[cursor];
      }
      if (!observed_terms.insert(std::move(current_term)).second)
        fail(path, "sparse monomials must be unique");
      observed_rank = std::max(observed_rank, rank);
    }
    if (observed_factor_count != result.maximum_factor_count ||
        observed_rank != result.maximum_tensor_rank)
      fail(path, "sparse polynomial maxima do not match the executable table");
    return result;
  }

}    // namespace

LiftedCauchyModel LiftedCauchyModel::load(const std::string &supplied_path)
{
  namespace fs = std::filesystem;
  const fs::path supplied(supplied_path);
  std::error_code error;
  const bool is_directory = fs::is_directory(supplied, error);
  if (error) fail(supplied_path, "could not inspect bundle path");
  const fs::path root = is_directory ? supplied : supplied.parent_path();
  const fs::path model_path = is_directory ? root / "model.ye3t.json" : supplied;
  if (model_path.filename() != "model.ye3t.json")
    fail(model_path.string(), "lifted model filename must be model.ye3t.json");
  const fs::path manifest_path = root / "YE3T_LIFTED_BUNDLE_MANIFEST.sha256";
  const auto manifest = read_manifest(manifest_path);
  const auto model_entry = manifest.find("model.ye3t.json");
  if (model_entry == manifest.end()) fail(manifest_path.string(), "manifest omits model.ye3t.json");
  require_manifest_file(manifest, root, "model.ye3t.json", model_entry->second,
                        manifest_path.string());

  const YAML::Node model = load_json(model_path, MAX_MODEL_BYTES);
  require_mapping(model, "model");
  const std::string model_schema = scalar_string(model["schema"], "model.schema");
  const bool composite_bundle = model_schema == "ye3t_lifted_cauchy_composite_linear_bundle_v1";
  if (composite_bundle)
    require_exact_fields(model,
                         {"central_species_order", "compiler_binding", "conventions",
                          "default_realization", "default_source_realization", "fit_metadata",
                          "model_family", "native_runtime_reference", "ordinary_reference",
                          "readout", "schema", "self_hash", "source", "source_plan_hash",
                          "type_map"},
                         "model");
  else {
    require_exact_fields(model,
                         {"central_species_order", "compiler_artifact", "conventions",
                          "default_realization", "default_source_realization", "fit_metadata",
                          "model_family", "native_runtime_reference", "ordinary_reference",
                          "readout", "schema", "self_hash", "source", "source_plan_hash",
                          "type_map"},
                         "model");
    require_string(model["schema"], "ye3t_lifted_cauchy_linear_bundle_v3", "model.schema");
  }
  require_string(model["model_family"], "linear_lifted_cauchy_scalar", "model.model_family");
  if (!model["ordinary_reference"].IsNull())
    fail("model.ordinary_reference",
         "additive ordinary models are not supported by this native slice");
  validate_conventions(model["conventions"], "model.conventions");
  require_mapping(model["fit_metadata"], "model.fit_metadata");
  const std::string model_self_hash = sha256(model["self_hash"], "model.self_hash");
  if (canonical_json_hash_without_root_member(model_path.string(), "self_hash") != model_self_hash)
    fail(model_path.string(), "model self-hash mismatch");
  const std::string default_realization =
      scalar_string(model["default_realization"], "model.default_realization");
  if (default_realization != "canonical" && default_realization != "factored")
    fail("model.default_realization", "unsupported descriptor realization");
  const std::string default_source =
      scalar_string(model["default_source_realization"], "model.default_source_realization");
  if (default_source != "direct" && default_source != "factorized" && default_source != "auto")
    fail("model.default_source_realization", "unsupported source realization");

  const std::vector<std::string> species =
      string_sequence(model["central_species_order"], "model.central_species_order");
  if (species.empty() ||
      std::set<std::string>(species.begin(), species.end()).size() != species.size())
    fail("model.central_species_order", "species must be nonempty and unique");
  const std::vector<int> type_map = parse_type_map(model["type_map"], species, "model.type_map");
  const std::string source_plan_hash = sha256(model["source_plan_hash"], "model.source_plan_hash");

  require_exact_fields(model["source"],
                       {"angular_scope", "certificates", "cutoff_A", "density_normalization",
                        "envelope", "groups", "neighbor_backend", "periodic_image_mode",
                        "radial_channel_semantics", "radial_coordinate", "radial_measure",
                        "role_dimension", "roles", "schema", "source_family_id",
                        "source_plan_hash"},
                       "model.source");
  const std::string source_schema = scalar_string(model["source"]["schema"], "model.source.schema");
  const bool mixed_l_source = source_schema == "ye3t_lifted_cauchy_joint_source_v3";
  if (!mixed_l_source && source_schema != "ye3t_lifted_cauchy_joint_source_v2")
    fail("model.source.schema", "unsupported joint source schema");
  require_string(model["source"]["angular_scope"],
                 mixed_l_source ? "l>=0_origin_regular_real_tesseral"
                                : "l=1_origin_regular_solid_harmonic",
                 "model.source.angular_scope");
  require_string(model["source"]["density_normalization"], "none",
                 "model.source.density_normalization");
  require_string(model["source"]["radial_coordinate"], "x=r/r_c", "model.source.radial_coordinate");
  require_string(model["source"]["radial_measure"], "x^2_dx", "model.source.radial_measure");
  require_string(model["source"]["radial_channel_semantics"], "q=2*n+s",
                 "model.source.radial_channel_semantics");
  const std::string expected_source_family = mixed_l_source
      ? "orthogonal_shifted_jacobi_origin_regular_v1"
      : "orthogonal_shifted_jacobi_l1_v1";
  require_string(model["source"]["source_family_id"], expected_source_family,
                 "model.source.source_family_id");
  require_string(model["source"]["envelope"], "(1-x)^2", "model.source.envelope");
  if (integer(model["source"]["role_dimension"], "model.source.role_dimension", 1) != 2)
    fail("model.source.role_dimension", "native source requires two roles");
  require_exact_fields(model["source"]["certificates"],
                       {"cutoff_first_derivative", "cutoff_value", "factorized_forward",
                        "factorized_reverse", "radial_source_gram", "runtime_gram_solve",
                        "source_span_dimension"},
                       "model.source.certificates");
  require_string(model["source"]["certificates"]["factorized_forward"], "A_Q=T*A_f",
                 "model.source.certificates.factorized_forward");
  require_string(model["source"]["certificates"]["factorized_reverse"], "bar_A_f=T^T*bar_A_Q",
                 "model.source.certificates.factorized_reverse");
  require_string(model["source"]["certificates"]["radial_source_gram"], "identity_exact",
                 "model.source.certificates.radial_source_gram");
  require_bool(model["source"]["certificates"]["runtime_gram_solve"], false,
               "model.source.certificates.runtime_gram_solve");
  require_string(model["source"]["certificates"]["cutoff_value"], "zero_exact",
                 "model.source.certificates.cutoff_value");
  require_string(model["source"]["certificates"]["cutoff_first_derivative"], "zero_exact",
                 "model.source.certificates.cutoff_first_derivative");
  require_string(model["source"]["certificates"]["source_span_dimension"], "2*C_per_species",
                 "model.source.certificates.source_span_dimension");
  require_sequence(model["source"]["roles"], "model.source.roles");
  if (model["source"]["roles"].size() != 2)
    fail("model.source.roles", "native source requires exactly two roles");
  const std::vector<std::string> role_ids{"even_shell_difference", "odd_shell"};
  for (int role = 0; role < 2; ++role) {
    const YAML::Node entry = model["source"]["roles"][role];
    require_exact_fields(entry, {"id", "index"},
                         "model.source.roles[" + std::to_string(role) + "]");
    if (integer(entry["index"], "model.source.roles.index", 0, 1) != role ||
        scalar_string(entry["id"], "model.source.roles.id") != role_ids[role])
      fail("model.source.roles", "role identity/order mismatch");
  }
  if (sha256(model["source"]["source_plan_hash"], "model.source.source_plan_hash") !=
      source_plan_hash)
    fail("model.source", "source-plan identity mismatch");
  const double model_cutoff = finite_number(model["source"]["cutoff_A"], "model.source.cutoff_A");
  if (!(model_cutoff > 0.0)) fail("model.source.cutoff_A", "cutoff must be positive");

  require_exact_fields(model["readout"],
                       {"coefficient_shape", "coefficients", "dtype", "offset_shape", "offsets"},
                       "model.readout");
  require_string(model["readout"]["dtype"], "float64", "model.readout.dtype");
  const auto coefficient_shape =
      integer_sequence(model["readout"]["coefficient_shape"], "model.readout.coefficient_shape", 0,
                       std::numeric_limits<int>::max(), 2);
  const auto offset_shape =
      integer_sequence(model["readout"]["offset_shape"], "model.readout.offset_shape", 0,
                       std::numeric_limits<int>::max(), 1);
  if (coefficient_shape.size() != 2 || offset_shape.size() != 1 ||
      coefficient_shape[0] != static_cast<int>(species.size()) ||
      offset_shape[0] != static_cast<int>(species.size()))
    fail("model.readout", "readout shapes disagree with species heads");
  require_sequence(model["readout"]["coefficients"], "model.readout.coefficients");
  if (model["readout"]["coefficients"].size() != species.size())
    fail("model.readout.coefficients", "head count mismatch");
  std::vector<std::vector<double>> readout_coefficients;
  readout_coefficients.reserve(species.size());
  for (std::size_t head = 0; head < species.size(); ++head) {
    auto row = finite_sequence(model["readout"]["coefficients"][head],
                               "model.readout.coefficients[" + std::to_string(head) + "]",
                               static_cast<std::size_t>(coefficient_shape[1]));
    if (row.size() != static_cast<std::size_t>(coefficient_shape[1]))
      fail("model.readout.coefficients", "coefficient shape mismatch");
    readout_coefficients.push_back(std::move(row));
  }
  const auto offsets =
      finite_sequence(model["readout"]["offsets"], "model.readout.offsets", species.size());
  if (offsets.size() != species.size()) fail("model.readout.offsets", "offset shape mismatch");

  std::string compiler_name;
  std::string compiler_self_hash;
  std::string compiler_binding_name;
  std::string compiler_binding_self_hash;
  std::string composite_artifact_hash;
  fs::path compiler_path;
  fs::path compiler_binding_path;
  CompactCompilerBinding compact_binding;
  std::set<std::string> expected_files{"model.ye3t.json"};
  if (composite_bundle) {
    require_exact_fields(model["compiler_binding"],
                         {"binding_self_hash", "composite_artifact_hash", "file", "file_sha256"},
                         "model.compiler_binding");
    compiler_binding_name =
        scalar_string(model["compiler_binding"]["file"], "model.compiler_binding.file");
    const std::string binding_file_hash =
        sha256(model["compiler_binding"]["file_sha256"], "model.compiler_binding.file_sha256");
    compiler_binding_self_hash = sha256(model["compiler_binding"]["binding_self_hash"],
                                        "model.compiler_binding.binding_self_hash");
    composite_artifact_hash = sha256(model["compiler_binding"]["composite_artifact_hash"],
                                     "model.compiler_binding.composite_artifact_hash");
    compiler_binding_path =
        bundle_member(root, compiler_binding_name, "model.compiler_binding.file");
    require_file_bound(compiler_binding_path, MAX_COMPILER_BINDING_DOM_BYTES);
    require_manifest_file(manifest, root, compiler_binding_name, binding_file_hash,
                          manifest_path.string());
    expected_files.insert(compiler_binding_name);
    compact_binding = validate_compact_compiler_binding(
        compiler_binding_path, model_path, root, manifest, expected_files,
        compiler_binding_self_hash, composite_artifact_hash, source_plan_hash,
        static_cast<int>(species.size()), coefficient_shape[1]);
  } else {
    require_exact_fields(model["compiler_artifact"], {"artifact_self_hash", "file", "file_sha256"},
                         "model.compiler_artifact");
    compiler_name =
        scalar_string(model["compiler_artifact"]["file"], "model.compiler_artifact.file");
    const std::string compiler_file_hash =
        sha256(model["compiler_artifact"]["file_sha256"], "model.compiler_artifact.file_sha256");
    compiler_self_hash = sha256(model["compiler_artifact"]["artifact_self_hash"],
                                "model.compiler_artifact.artifact_self_hash");
    compiler_path = bundle_member(root, compiler_name, "model.compiler_artifact.file");
    require_file_bound(compiler_path, MAX_COMPILER_BYTES);
    require_manifest_file(manifest, root, compiler_name, compiler_file_hash,
                          manifest_path.string());
    expected_files.insert(compiler_name);
    if (canonical_json_root_member_value(compiler_path.string(), "self_hash") !=
            quoted_json_string(compiler_self_hash) ||
        canonical_json_hash_without_root_member(compiler_path.string(), "self_hash") !=
            compiler_self_hash)
      fail(compiler_path.string(), "compiler self-hash identity mismatch");
  }

  require_exact_fields(model["native_runtime_reference"],
                       {"file", "file_sha256", "format", "plan_self_hash"},
                       "model.native_runtime_reference");
  const std::string native_format = scalar_string(model["native_runtime_reference"]["format"],
                                                  "model.native_runtime_reference.format");
  const bool native_v2 = native_format == "ye3t_lifted_cauchy_native_runtime_v2";
  const bool composite_native = native_format == "ye3t_lifted_cauchy_composite_native_runtime_v1";
  const bool packed_native = native_v2 || composite_native;
  if (!packed_native && native_format != "ye3t_lifted_cauchy_native_runtime_v1")
    fail("model.native_runtime_reference.format", "unsupported native runtime format");
  if (composite_bundle != composite_native)
    fail("model.native_runtime_reference.format", "model and native composite schemas disagree");
  if (!packed_native && mixed_l_source)
    fail("model.native_runtime_reference.format",
         "native runtime and joint source schemas disagree");
  const std::string native_name = scalar_string(model["native_runtime_reference"]["file"],
                                                "model.native_runtime_reference.file");
  const std::string native_file_hash = sha256(model["native_runtime_reference"]["file_sha256"],
                                              "model.native_runtime_reference.file_sha256");
  const std::string native_self_reference =
      sha256(model["native_runtime_reference"]["plan_self_hash"],
             "model.native_runtime_reference.plan_self_hash");
  const fs::path native_path =
      bundle_member(root, native_name, "model.native_runtime_reference.file");
  require_manifest_file(manifest, root, native_name, native_file_hash, manifest_path.string());
  expected_files.insert(native_name);
  std::set<std::string> manifest_files;
  for (const auto &entry : manifest) manifest_files.insert(entry.first);
  if (manifest_files != expected_files)
    fail(manifest_path.string(), "manifest contains missing or unexpected files");

  const YAML::Node native = load_json(native_path, MAX_NATIVE_BYTES);
  if (composite_native)
    require_exact_fields(native,
                         {"capabilities", "central_species_order", "certificates",
                          "compiler_binding_self_hash", "composite_artifact_hash",
                          "coordinate_convention", "cutoff_A", "deployment_identity_sha256",
                          "heads", "model_family", "real_component_count", "role_dimension",
                          "schema", "self_hash", "source_groups", "source_plan_hash",
                          "source_variable_count", "type_map"},
                         "native");
  else
    require_exact_fields(native,
                         {"capabilities", "central_species_order", "certificates",
                          "compiler_artifact_self_hash", "coordinate_convention", "cutoff_A",
                          "deployment_identity_sha256", "heads", "model_family",
                          "real_component_count", "role_dimension", "schema", "self_hash",
                          "source_groups", "source_plan_hash", "source_variable_count", "type_map"},
                         "native");
  require_string(native["schema"], native_format, "native.schema");
  require_string(native["model_family"], "linear_lifted_cauchy_scalar", "native.model_family");
  const std::string native_self_hash = sha256(native["self_hash"], "native.self_hash");
  if (native_self_hash != native_self_reference ||
      canonical_json_hash_without_root_member(native_path.string(), "self_hash") !=
          native_self_hash)
    fail(native_path.string(), "native runtime self-hash mismatch");
  if (composite_native) {
    if (sha256(native["compiler_binding_self_hash"], "native.compiler_binding_self_hash") !=
            compiler_binding_self_hash ||
        sha256(native["composite_artifact_hash"], "native.composite_artifact_hash") !=
            composite_artifact_hash)
      fail("native", "composite compiler-binding identity mismatch");
  } else if (sha256(native["compiler_artifact_self_hash"], "native.compiler_artifact_self_hash") !=
             compiler_self_hash) {
    fail("native", "compiler identity mismatch");
  }
  if (sha256(native["source_plan_hash"], "native.source_plan_hash") != source_plan_hash)
    fail("native", "source identity mismatch");
  const std::string deployment_hash =
      sha256(native["deployment_identity_sha256"], "native.deployment_identity_sha256");
  const std::string expected_deployment_hash = composite_native
      ? compute_composite_deployment_identity_hash(model_path.string())
      : compute_deployment_identity_hash(model_path.string());
  if (deployment_hash != expected_deployment_hash)
    fail("native.deployment_identity_sha256", "deployment identity mismatch");
  if (string_sequence(native["central_species_order"], "native.central_species_order") != species ||
      parse_type_map(native["type_map"], species, "native.type_map") != type_map)
    fail("native", "species/type ordering differs from the model");

  std::map<int, NativeChannelBlock> native_channel_blocks;
  if (packed_native)
    require_exact_fields(native["coordinate_convention"],
                         {"channel_blocks", "component_order", "id", "source_variable_order"},
                         "native.coordinate_convention");
  else
    require_exact_fields(native["coordinate_convention"],
                         {"component_order", "id", "l", "source_variable_order"},
                         "native.coordinate_convention");
  require_string(native["coordinate_convention"]["id"], "ye3t_physical_real_tesseral_v1",
                 "native.coordinate_convention.id");
  if (packed_native) {
    require_string(native["coordinate_convention"]["component_order"],
                   "cosine_l_to_1_then_m0_then_sine_1_to_l",
                   "native.coordinate_convention.component_order");
    require_string(native["coordinate_convention"]["source_variable_order"],
                   "packed_channel_then_role_then_real_component",
                   "native.coordinate_convention.source_variable_order");
    require_sequence(native["coordinate_convention"]["channel_blocks"],
                     "native.coordinate_convention.channel_blocks");
    std::int64_t next_offset = 0;
    const YAML::Node blocks = native["coordinate_convention"]["channel_blocks"];
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      const YAML::Node block = blocks[index];
      const std::string path =
          "native.coordinate_convention.channel_blocks[" + std::to_string(index) + "]";
      require_exact_fields(block,
                           {"channel_index", "channel_position", "l", "real_component_count",
                            "source_variable_offset"},
                           path);
      const int channel_index = integer(block["channel_index"], path + ".channel_index", 0);
      NativeChannelBlock parsed;
      parsed.position = integer(block["channel_position"], path + ".channel_position", 0);
      parsed.angular = integer(block["l"], path + ".l", 0);
      parsed.component_count =
          integer(block["real_component_count"], path + ".real_component_count", 1, 64);
      parsed.source_offset = integer64(block["source_variable_offset"],
                                       path + ".source_variable_offset", 0, MAX_SOURCE_VARIABLES);
      if (parsed.position != static_cast<int>(index) ||
          parsed.component_count != 2 * parsed.angular + 1 || parsed.source_offset != next_offset ||
          !native_channel_blocks.emplace(channel_index, parsed).second)
        fail(path, "invalid or noncontiguous packed channel block");
      next_offset += 2 * parsed.component_count;
      if (next_offset > MAX_SOURCE_VARIABLES)
        fail(path, "packed channel blocks exceed the source bound");
    }
  } else {
    if (integer(native["coordinate_convention"]["l"], "native.coordinate_convention.l", 0) != 1 ||
        string_sequence(native["coordinate_convention"]["component_order"],
                        "native.coordinate_convention.component_order") !=
            std::vector<std::string>{"x", "z", "minus_y"})
      fail("native.coordinate_convention", "unsupported physical-real convention");
    require_string(native["coordinate_convention"]["source_variable_order"],
                   "channel_then_role_then_real_component",
                   "native.coordinate_convention.source_variable_order");
  }

  if (composite_native)
    require_exact_fields(
        native["capabilities"],
        {"composite_compiler_binding", "direct_orthogonal_q_source",
         "factorized_reverse_uses_algebraic_transpose", "factorized_source_then_center_transform",
         "physical_real_sparse_polynomial", "runtime_compiler_dom", "runtime_gram_solve"},
        "native.capabilities");
  else
    require_exact_fields(native["capabilities"],
                         {"direct_orthogonal_q_source",
                          "factorized_reverse_uses_algebraic_transpose",
                          "factorized_source_then_center_transform",
                          "physical_real_sparse_polynomial", "runtime_gram_solve"},
                         "native.capabilities");
  require_bool(native["capabilities"]["direct_orthogonal_q_source"], true,
               "native.capabilities.direct_orthogonal_q_source");
  require_bool(native["capabilities"]["factorized_reverse_uses_algebraic_transpose"], true,
               "native.capabilities.factorized_reverse_uses_algebraic_transpose");
  require_bool(native["capabilities"]["factorized_source_then_center_transform"], true,
               "native.capabilities.factorized_source_then_center_transform");
  require_bool(native["capabilities"]["physical_real_sparse_polynomial"], true,
               "native.capabilities.physical_real_sparse_polynomial");
  require_bool(native["capabilities"]["runtime_gram_solve"], false,
               "native.capabilities.runtime_gram_solve");
  if (composite_native) {
    require_bool(native["capabilities"]["composite_compiler_binding"], true,
                 "native.capabilities.composite_compiler_binding");
    require_bool(native["capabilities"]["runtime_compiler_dom"], false,
                 "native.capabilities.runtime_compiler_dom");
  }

  const int role_dimension = integer(native["role_dimension"], "native.role_dimension", 1, 64);
  const int component_count =
      integer(native["real_component_count"], "native.real_component_count", 1, 64);
  if (role_dimension != 2 || (!packed_native && component_count != 3))
    fail("native", "native runtime has an unsupported source shape");
  const std::int64_t source_count = integer64(
      native["source_variable_count"], "native.source_variable_count", 1, MAX_SOURCE_VARIABLES);
  if (packed_native) {
    int maximum_component_count = 0;
    std::int64_t packed_count = 0;
    for (const auto &item : native_channel_blocks) {
      maximum_component_count = std::max(maximum_component_count, item.second.component_count);
      packed_count += 2 * item.second.component_count;
    }
    if (native_channel_blocks.empty() || maximum_component_count != component_count ||
        packed_count != source_count)
      fail("native.coordinate_convention.channel_blocks",
           "packed channel blocks disagree with native source dimensions");
  }
  const double cutoff = finite_number(native["cutoff_A"], "native.cutoff_A");
  if (cutoff != model_cutoff) fail("native.cutoff_A", "native and model cutoffs differ");

  require_sequence(native["source_groups"], "native.source_groups");
  require_sequence(model["source"]["groups"], "model.source.groups");
  if (native["source_groups"].size() == 0 || native["source_groups"].size() > 4096 ||
      native["source_groups"].size() != model["source"]["groups"].size())
    fail("native.source_groups", "source-group count is outside the supported bound");
  std::vector<bool> covered(static_cast<std::size_t>(source_count), false);
  std::set<int> global_channel_positions;
  std::vector<LiftedCauchySourceGroup> groups;
  groups.reserve(native["source_groups"].size());
  for (std::size_t group_index = 0; group_index < native["source_groups"].size(); ++group_index) {
    const YAML::Node entry = native["source_groups"][group_index];
    const YAML::Node model_entry = model["source"]["groups"][group_index];
    const std::string path = "native.source_groups[" + std::to_string(group_index) + "]";
    if (packed_native)
      require_exact_fields(entry,
                           {"channel_indices", "channel_positions", "direct_q_polynomials",
                            "factorized_radials", "l", "neighbor_species", "neighbor_species_index",
                            "q_source_variable_offsets", "real_component_count", "source_dimension",
                            "transform_q_from_f"},
                           path);
    else
      require_exact_fields(entry,
                           {"channel_indices", "channel_positions", "direct_q_polynomials",
                            "factorized_radials", "l", "neighbor_species", "neighbor_species_index",
                            "q_source_variable_offsets", "source_dimension", "transform_q_from_f"},
                           path);
    require_exact_fields(model_entry,
                         {"channel_indices", "coordinate_order", "factorized_lowering", "l",
                          "neighbor_species", "polynomials", "source_dimension",
                          "source_family_id"},
                         "model.source.groups[" + std::to_string(group_index) + "]");
    LiftedCauchySourceGroup group;
    group.neighbor_species = scalar_string(entry["neighbor_species"], path + ".neighbor_species");
    group.neighbor_species_index =
        integer(entry["neighbor_species_index"], path + ".neighbor_species_index", 0,
                static_cast<int>(species.size()) - 1);
    if (species[group.neighbor_species_index] != group.neighbor_species)
      fail(path, "neighbor species name/index mismatch");
    if (scalar_string(model_entry["neighbor_species"], "model.source.groups.neighbor_species") !=
        group.neighbor_species)
      fail(path, "native/model group species mismatch");
    group.angular = integer(entry["l"], path + ".l", 0);
    if (!packed_native && group.angular != 1)
      fail(path + ".l", "native v1 supports only l=1 sources");
    if (integer(model_entry["l"], "model.source.groups.l", 0) != group.angular)
      fail(path, "native/model group angular momentum mismatch");
    group.source_dimension = integer(entry["source_dimension"], path + ".source_dimension", 1,
                                     static_cast<int>(MAX_SOURCE_VARIABLES));
    group.channel_indices = integer_sequence(entry["channel_indices"], path + ".channel_indices", 0,
                                             std::numeric_limits<int>::max());
    group.channel_positions =
        integer_sequence(entry["channel_positions"], path + ".channel_positions", 0,
                         std::numeric_limits<int>::max());
    if (integer_sequence(model_entry["channel_indices"], "model.source.groups.channel_indices", 0,
                         std::numeric_limits<int>::max()) != group.channel_indices ||
        integer(model_entry["source_dimension"], "model.source.groups.source_dimension", 1) !=
            group.source_dimension)
      fail(path, "native/model group channel dimensions mismatch");
    require_string(model_entry["coordinate_order"], "q=2*n+s",
                   "model.source.groups.coordinate_order");
    group.source_family_id =
        scalar_string(model_entry["source_family_id"], "model.source.groups.source_family_id");
    if (group.source_family_id != expected_source_family)
      fail("model.source.groups.source_family_id",
           "source group family disagrees with the model source");
    group.real_component_count = packed_native
        ? integer(entry["real_component_count"], path + ".real_component_count", 1, 64)
        : component_count;
    if (group.real_component_count != 2 * group.angular + 1 ||
        group.real_component_count > component_count)
      fail(path + ".real_component_count", "source group width disagrees with angular momentum");
    if (group.channel_indices.size() != group.channel_positions.size() ||
        group.source_dimension != role_dimension * static_cast<int>(group.channel_positions.size()))
      fail(path, "channel dimensions disagree with the role/source dimension");
    group.q_source_variable_offsets =
        integer64_sequence(entry["q_source_variable_offsets"], path + ".q_source_variable_offsets",
                           0, source_count - group.real_component_count,
                           static_cast<std::size_t>(group.source_dimension));
    if (group.q_source_variable_offsets.size() != static_cast<std::size_t>(group.source_dimension))
      fail(path, "source offset count mismatch");
    for (const std::int64_t offset : group.q_source_variable_offsets) {
      if (!packed_native && offset % component_count != 0)
        fail(path, "source offsets must align with real-component blocks");
      for (int component = 0; component < group.real_component_count; ++component) {
        const std::size_t index = static_cast<std::size_t>(offset + component);
        if (covered[index]) fail(path, "source-variable blocks overlap");
        covered[index] = true;
      }
    }
    for (const int position : group.channel_positions)
      if (!global_channel_positions.insert(position).second)
        fail(path, "channel positions must be globally unique");

    require_sequence(entry["direct_q_polynomials"], path + ".direct_q_polynomials");
    require_sequence(entry["factorized_radials"], path + ".factorized_radials");
    require_sequence(entry["transform_q_from_f"], path + ".transform_q_from_f");
    require_sequence(model_entry["polynomials"], "model.source.groups.polynomials");
    require_exact_fields(model_entry["factorized_lowering"],
                         {"binary64_matrix", "binary64_minimum_singular_value",
                          "determinant_nonzero_exact", "factor_order", "forward_equation",
                          "reverse_equation", "unnormalized_integer_determinant",
                          "unnormalized_integer_rows"},
                         "model.source.groups.factorized_lowering");
    require_sequence(model_entry["factorized_lowering"]["binary64_matrix"],
                     "model.source.groups.factorized_lowering.binary64_matrix");
    require_sequence(model_entry["factorized_lowering"]["unnormalized_integer_rows"],
                     "model.source.groups.factorized_lowering.unnormalized_integer_rows");
    require_bool(model_entry["factorized_lowering"]["determinant_nonzero_exact"], true,
                 "model.source.groups.factorized_lowering.determinant_nonzero_exact");
    require_string(model_entry["factorized_lowering"]["factor_order"],
                   "p_even_minus_p_odd,p_odd_by_n",
                   "model.source.groups.factorized_lowering.factor_order");
    require_string(model_entry["factorized_lowering"]["forward_equation"], "A_Q=T*A_f",
                   "model.source.groups.factorized_lowering.forward_equation");
    require_string(model_entry["factorized_lowering"]["reverse_equation"], "bar_A_f=T^T*bar_A_Q",
                   "model.source.groups.factorized_lowering.reverse_equation");
    const double minimum_singular_value =
        finite_number(model_entry["factorized_lowering"]["binary64_minimum_singular_value"],
                      "model.source.groups.factorized_lowering."
                      "binary64_minimum_singular_value");
    if (!(minimum_singular_value > 0.0))
      fail("model.source.groups.factorized_lowering",
           "transform must have a positive certified minimum singular value");
    if (!nonzero_decimal_integer(
            scalar_string(model_entry["factorized_lowering"]["unnormalized_integer_determinant"],
                          "model.source.groups.factorized_lowering."
                          "unnormalized_integer_determinant")))
      fail("model.source.groups.factorized_lowering",
           "exact transform determinant must be a nonzero decimal integer");
    if (entry["direct_q_polynomials"].size() != static_cast<std::size_t>(group.source_dimension) ||
        entry["factorized_radials"].size() != static_cast<std::size_t>(group.source_dimension) ||
        entry["transform_q_from_f"].size() != static_cast<std::size_t>(group.source_dimension) ||
        model_entry["polynomials"].size() != static_cast<std::size_t>(group.source_dimension) ||
        model_entry["factorized_lowering"]["binary64_matrix"].size() !=
            static_cast<std::size_t>(group.source_dimension) ||
        model_entry["factorized_lowering"]["unnormalized_integer_rows"].size() !=
            static_cast<std::size_t>(group.source_dimension))
      fail(path, "direct, factorized, and transform dimensions disagree");
    for (int q = 0; q < group.source_dimension; ++q) {
      const YAML::Node direct = entry["direct_q_polynomials"][q];
      require_exact_fields(direct, {"coefficients", "q"},
                           path + ".direct_q_polynomials[" + std::to_string(q) + "]");
      LiftedCauchyDirectPolynomial polynomial;
      polynomial.q = integer(direct["q"], path + ".direct_q_polynomials.q", 0);
      if (polynomial.q != q) fail(path + ".direct_q_polynomials", "q identifiers must be dense");
      polynomial.coefficients =
          finite_sequence(direct["coefficients"], path + ".direct_q_polynomials.coefficients",
                          static_cast<std::size_t>(group.source_dimension + 1));
      if (polynomial.coefficients.size() != static_cast<std::size_t>(q + 1))
        fail(path + ".direct_q_polynomials", "coefficient degree disagrees with q");

      const YAML::Node model_polynomial = model_entry["polynomials"][q];
      const std::string model_polynomial_path = "model.source.groups[" +
          std::to_string(group_index) + "].polynomials[" + std::to_string(q) + "]";
      require_exact_fields(model_polynomial,
                           {"channel_radial_index", "jacobi_degree", "normalization_squared", "q",
                            "role_index", "shifted_jacobi_power_coefficients",
                            "total_radial_polynomial_degree"},
                           model_polynomial_path);
      if (integer(model_polynomial["q"], model_polynomial_path + ".q", 0) != q ||
          integer(model_polynomial["jacobi_degree"], model_polynomial_path + ".jacobi_degree", 0) !=
              q ||
          integer(model_polynomial["role_index"], model_polynomial_path + ".role_index", 0, 1) !=
              q % role_dimension ||
          integer(model_polynomial["channel_radial_index"],
                  model_polynomial_path + ".channel_radial_index", 0) != q / role_dimension ||
          integer(model_polynomial["total_radial_polynomial_degree"],
                  model_polynomial_path + ".total_radial_polynomial_degree",
                  0) != group.angular + q + 2)
        fail(model_polynomial_path, "polynomial labels disagree with q");
      require_exact_fields(model_polynomial["normalization_squared"], {"denominator", "numerator"},
                           model_polynomial_path + ".normalization_squared");
      const std::int64_t numerator =
          integer64(model_polynomial["normalization_squared"]["numerator"],
                    model_polynomial_path + ".normalization_squared.numerator", 1);
      const std::int64_t denominator =
          integer64(model_polynomial["normalization_squared"]["denominator"],
                    model_polynomial_path + ".normalization_squared.denominator", 1);
      const auto exact_coefficients = integer64_sequence(
          model_polynomial["shifted_jacobi_power_coefficients"],
          model_polynomial_path + ".shifted_jacobi_power_coefficients",
          std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(),
          static_cast<std::size_t>(q + 1));
      if (exact_coefficients.size() != static_cast<std::size_t>(q + 1))
        fail(model_polynomial_path, "exact polynomial degree disagrees with q");
      const double normalization =
          std::sqrt(static_cast<double>(numerator) / static_cast<double>(denominator));
      for (int degree = 0; degree <= q; ++degree)
        if (!same_binary64(
                polynomial.coefficients[static_cast<std::size_t>(degree)],
                normalization *
                    static_cast<double>(exact_coefficients[static_cast<std::size_t>(degree)])))
          fail(path + ".direct_q_polynomials",
               "native polynomial disagrees with the exact source certificate");
      group.direct_q_polynomials.push_back(std::move(polynomial));

      const YAML::Node factorized = entry["factorized_radials"][q];
      require_exact_fields(factorized, {"envelope_power", "q", "x_power"},
                           path + ".factorized_radials[" + std::to_string(q) + "]");
      LiftedCauchyFactorizedRadial radial;
      radial.q = integer(factorized["q"], path + ".factorized_radials.q", 0);
      radial.x_power = integer(factorized["x_power"], path + ".factorized_radials.x_power", 0);
      radial.envelope_power =
          integer(factorized["envelope_power"], path + ".factorized_radials.envelope_power", 2);
      if (radial.q != q) fail(path + ".factorized_radials", "q identifiers must be dense");
      if (radial.x_power != q || radial.envelope_power != (q % role_dimension == 0 ? 3 : 2))
        fail(path + ".factorized_radials",
             "factorized radial disagrees with the certified source family");
      group.factorized_radials.push_back(radial);

      const auto row = finite_sequence(entry["transform_q_from_f"][q],
                                       path + ".transform_q_from_f[" + std::to_string(q) + "]",
                                       static_cast<std::size_t>(group.source_dimension));
      if (row.size() != static_cast<std::size_t>(group.source_dimension))
        fail(path + ".transform_q_from_f", "transform must be square");
      const auto certified_row = finite_sequence(
          model_entry["factorized_lowering"]["binary64_matrix"][q],
          "model.source.groups.factorized_lowering.binary64_matrix[" + std::to_string(q) + "]",
          static_cast<std::size_t>(group.source_dimension));
      if (certified_row.size() != row.size())
        fail(path + ".transform_q_from_f", "certified transform must be square");
      const auto exact_transform_row = integer64_sequence(
          model_entry["factorized_lowering"]["unnormalized_integer_rows"][q],
          "model.source.groups.factorized_lowering."
          "unnormalized_integer_rows[" +
              std::to_string(q) + "]",
          std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(),
          static_cast<std::size_t>(group.source_dimension));
      if (exact_transform_row.size() != row.size())
        fail(path + ".transform_q_from_f", "exact transform row must be square");
      for (int column = 0; column < group.source_dimension; ++column) {
        const double exact_value = normalization *
            static_cast<double>(exact_transform_row[static_cast<std::size_t>(column)]);
        if (!same_binary64(certified_row[static_cast<std::size_t>(column)], exact_value))
          fail("model.source.groups.factorized_lowering.binary64_matrix",
               "serialized transform disagrees with the exact certificate");
        if (!same_binary64(row[static_cast<std::size_t>(column)], exact_value))
          fail(path + ".transform_q_from_f",
               "native transform disagrees with the source certificate");
      }
      group.transform_q_from_f.insert(group.transform_q_from_f.end(), row.begin(), row.end());
    }
    for (int q = 0; q < group.source_dimension; ++q) {
      const int channel = q / role_dimension;
      const int role = q % role_dimension;
      std::int64_t expected_offset = 0;
      if (packed_native) {
        const auto block = native_channel_blocks.find(group.channel_indices[channel]);
        if (block == native_channel_blocks.end() ||
            block->second.position != group.channel_positions[channel] ||
            block->second.angular != group.angular ||
            block->second.component_count != group.real_component_count)
          fail(path + ".channel_indices", "source group disagrees with its packed channel block");
        expected_offset = block->second.source_offset +
            static_cast<std::int64_t>(role) * group.real_component_count;
      } else {
        expected_offset =
            (static_cast<std::int64_t>(group.channel_positions[channel]) * role_dimension + role) *
            component_count;
      }
      if (group.q_source_variable_offsets[static_cast<std::size_t>(q)] != expected_offset)
        fail(path + ".q_source_variable_offsets",
             "offsets disagree with channel/role/component ordering");
    }
    groups.push_back(std::move(group));
  }
  if (std::find(covered.begin(), covered.end(), false) != covered.end())
    fail("native.source_groups", "source groups do not cover every source variable");
  if (!packed_native &&
      static_cast<std::int64_t>(global_channel_positions.size()) * role_dimension *
              component_count !=
          source_count)
    fail("native.source_groups", "global channel count is inconsistent");
  int expected_channel_position = 0;
  for (const int position : global_channel_positions)
    if (position != expected_channel_position++)
      fail("native.source_groups", "global channel positions must be contiguous from zero");

  CompilerPolynomialBinding compiler_binding;
  if (composite_native) {
    if (compact_binding.source_variable_count != source_count)
      fail("compiler_binding.physical_real_lowering.source_variable_count",
           "compiler binding and native source dimensions differ");
  } else {
    compiler_binding = reconstruct_compiler_rows(compiler_path, role_dimension, source_count,
                                                 packed_native, groups, coefficient_shape[1]);
  }

  require_sequence(native["heads"], "native.heads");
  if (native["heads"].size() != species.size())
    fail("native.heads", "head count differs from central species order");
  std::vector<LiftedCauchyHead> heads;
  heads.reserve(species.size());
  std::int64_t total_terms = 0;
  std::int64_t total_static_array_bytes = 0;
  for (std::size_t head_index = 0; head_index < species.size(); ++head_index) {
    const YAML::Node entry = native["heads"][head_index];
    const std::string path = "native.heads[" + std::to_string(head_index) + "]";
    require_exact_fields(entry, {"central_species", "central_species_index", "polynomial"}, path);
    LiftedCauchyHead head;
    head.central_species = scalar_string(entry["central_species"], path + ".central_species");
    head.central_species_index =
        integer(entry["central_species_index"], path + ".central_species_index", 0,
                static_cast<int>(species.size()) - 1);
    if (head.central_species_index != static_cast<int>(head_index) ||
        head.central_species != species[head_index])
      fail(path, "head species name/index ordering mismatch");
    head.polynomial = parse_polynomial(entry["polynomial"], source_count, path + ".polynomial");
    if (!composite_native) {
      const LiftedCauchySparsePolynomial expected =
          lower_compiler_readout(compiler_binding, readout_coefficients[head_index],
                                 offsets[head_index], path + ".polynomial");
      require_same_polynomial(head.polynomial, expected, path + ".polynomial");
    }
    total_terms += static_cast<std::int64_t>(head.polynomial.monomial_coefficients.size());
    const std::int64_t static_bytes = 8LL *
            static_cast<std::int64_t>(head.polynomial.factor_offsets.size() +
                                      head.polynomial.factor_indices.size() +
                                      head.polynomial.factor_exponents.size() +
                                      head.polynomial.monomial_coefficients.size()) +
        8LL;
    if (static_bytes > MAX_STATIC_ARRAY_BYTES - total_static_array_bytes)
      fail(path + ".polynomial", "native polynomial exceeds the static-array memory bound");
    total_static_array_bytes += static_bytes;
    heads.push_back(std::move(head));
  }

  if (composite_native)
    require_exact_fields(native["certificates"],
                         {"channel_source_variable_offsets", "component_row_certificate_sha256",
                          "descriptor_count", "estimated_static_array_bytes",
                          "maximum_absolute_imaginary_residual", "maximum_coefficient_scale",
                          "physical_real_term_count_before_readout", "readout_lowering",
                          "source_forward", "source_gram", "source_reverse",
                          "source_variable_count", "total_lowered_term_count"},
                         "native.certificates");
  else if (packed_native) {
    std::set<std::string> certificate_fields = {"channel_source_variable_offsets",
                                                "descriptor_count",
                                                "maximum_absolute_imaginary_residual",
                                                "maximum_coefficient_scale",
                                                "physical_real_term_count_before_readout",
                                                "readout_lowering",
                                                "source_forward",
                                                "source_gram",
                                                "source_reverse",
                                                "source_variable_count",
                                                "total_lowered_term_count"};
    if (native_v2 && native["certificates"]["physical_scalar_reality_report_hash"])
      certificate_fields.insert("physical_scalar_reality_report_hash");
    require_exact_fields(native["certificates"], certificate_fields, "native.certificates");
  } else
    require_exact_fields(native["certificates"],
                         {"descriptor_count", "maximum_absolute_imaginary_residual",
                          "maximum_coefficient_scale", "physical_real_term_count_before_readout",
                          "readout_lowering", "source_forward", "source_gram", "source_reverse",
                          "total_lowered_term_count"},
                         "native.certificates");
  if (native_v2 && native["certificates"]["physical_scalar_reality_report_hash"]) {
    const std::string reality_hash =
        sha256(native["certificates"]["physical_scalar_reality_report_hash"],
               "native.certificates.physical_scalar_reality_report_hash");
    const std::string compiler_payload =
        canonical_json_root_member_value(compiler_path.string(), "payload");
    const std::string equivalence =
        canonical_json_value_root_member(compiler_payload, "equivalence_certificate");
    if (canonical_json_value_root_member(equivalence, "physical_scalar_reality_report_hash") !=
        quoted_json_string(reality_hash))
      fail("native.certificates.physical_scalar_reality_report_hash",
           "compiler physical-reality certificate mismatch");
  }
  if (integer(native["certificates"]["descriptor_count"], "native.certificates.descriptor_count",
              0) != coefficient_shape[1] ||
      integer64(native["certificates"]["total_lowered_term_count"],
                "native.certificates.total_lowered_term_count", 0, MAX_TERMS) != total_terms)
    fail("native.certificates", "descriptor or term certificate mismatch");
  if (packed_native) {
    if (integer64(native["certificates"]["source_variable_count"],
                  "native.certificates.source_variable_count", 1,
                  MAX_SOURCE_VARIABLES) != source_count)
      fail("native.certificates.source_variable_count",
           "source-size certificate disagrees with the native model");
    std::vector<std::int64_t> expected_offsets(native_channel_blocks.size());
    for (const auto &item : native_channel_blocks)
      expected_offsets[static_cast<std::size_t>(item.second.position)] = item.second.source_offset;
    if (integer64_sequence(native["certificates"]["channel_source_variable_offsets"],
                           "native.certificates.channel_source_variable_offsets", 0,
                           source_count - 1, expected_offsets.size()) != expected_offsets)
      fail("native.certificates.channel_source_variable_offsets",
           "packed-offset certificate disagrees with channel blocks");
  }
  if (composite_native) {
    if (sha256(native["certificates"]["component_row_certificate_sha256"],
               "native.certificates.component_row_certificate_sha256") !=
        compact_binding.component_row_certificate_hash)
      fail("native.certificates.component_row_certificate_sha256",
           "native and compiler-binding component rows differ");
    if (integer64(native["certificates"]["estimated_static_array_bytes"],
                  "native.certificates.estimated_static_array_bytes", 0,
                  MAX_STATIC_ARRAY_BYTES) != total_static_array_bytes)
      fail("native.certificates.estimated_static_array_bytes",
           "native static-array size certificate differs from parsed arrays");
  }
  const double maximum_imaginary_residual =
      finite_number(native["certificates"]["maximum_absolute_imaginary_residual"],
                    "native.certificates.maximum_absolute_imaginary_residual");
  const double maximum_coefficient_scale =
      finite_number(native["certificates"]["maximum_coefficient_scale"],
                    "native.certificates.maximum_coefficient_scale");
  if (maximum_imaginary_residual < 0.0 || maximum_coefficient_scale < 0.0 ||
      maximum_imaginary_residual > 5.0e-11 * std::max(1.0, maximum_coefficient_scale) ||
      !same_binary64(maximum_imaginary_residual,
                     composite_native ? compact_binding.maximum_imaginary_residual
                                      : compiler_binding.maximum_imaginary_residual) ||
      !same_binary64(maximum_coefficient_scale,
                     composite_native ? compact_binding.maximum_coefficient_scale
                                      : compiler_binding.maximum_coefficient_scale))
    fail("native.certificates.maximum_absolute_imaginary_residual",
         "physical-real lowering certificate disagrees with the compiler rows");
  if (integer64(native["certificates"]["physical_real_term_count_before_readout"],
                "native.certificates.physical_real_term_count_before_readout", 0, MAX_TERMS) !=
      (composite_native ? compact_binding.physical_term_count
                        : compiler_binding.physical_term_count))
    fail("native.certificates.physical_real_term_count_before_readout",
         "physical-real term certificate disagrees with the compiler rows");
  require_string(native["certificates"]["readout_lowering"],
                 composite_native ? "fit_coordinates_to_component_pivots_then_shared_physical_real_"
                                    "polynomial"
                                  : "fit_coordinates_to_pivot_then_physical_real_polynomial",
                 "native.certificates.readout_lowering");
  require_string(native["certificates"]["source_forward"], "A_Q=T*A_f",
                 "native.certificates.source_forward");
  require_string(native["certificates"]["source_reverse"], "bar_A_f=T^T*bar_A_Q",
                 "native.certificates.source_reverse");
  require_string(native["certificates"]["source_gram"], "identity_exact",
                 "native.certificates.source_gram");

  LiftedCauchyModel result;
  result.bundle_root = root.string();
  result.model_path = model_path.string();
  result.native_runtime_path = native_path.string();
  result.compiler_artifact_path = compiler_path.string();
  result.compiler_binding_path = compiler_binding_path.string();
  result.model_self_hash = model_self_hash;
  result.native_self_hash = native_self_hash;
  result.compiler_artifact_self_hash = compiler_self_hash;
  result.compiler_binding_self_hash = compiler_binding_self_hash;
  result.composite_artifact_hash = composite_artifact_hash;
  result.source_plan_hash = source_plan_hash;
  result.deployment_identity_hash = deployment_hash;
  result.default_source_realization = default_source;
  result.composite_compiler_binding = composite_native;
  result.exclude_zero_separation = packed_native;
  result.cutoff = cutoff;
  result.role_dimension = role_dimension;
  result.real_component_count = component_count;
  result.source_variable_count = source_count;
  result.central_species_order = species;
  result.type_map = type_map;
  result.source_groups = std::move(groups);
  result.heads = std::move(heads);
  return result;
}

double LiftedCauchyModel::memory_usage() const
{
  double bytes = sizeof(*this);
  bytes += bundle_root.capacity() + model_path.capacity() + native_runtime_path.capacity() +
      compiler_artifact_path.capacity() + compiler_binding_path.capacity() +
      model_self_hash.capacity() + native_self_hash.capacity() +
      compiler_artifact_self_hash.capacity() + compiler_binding_self_hash.capacity() +
      composite_artifact_hash.capacity() + source_plan_hash.capacity() +
      deployment_identity_hash.capacity() + default_source_realization.capacity();
  bytes += central_species_order.capacity() * sizeof(std::string);
  for (const auto &species : central_species_order) bytes += species.capacity();
  bytes += type_map.capacity() * sizeof(int);
  bytes += source_groups.capacity() * sizeof(LiftedCauchySourceGroup);
  for (const auto &group : source_groups) {
    bytes += group.neighbor_species.capacity();
    bytes += group.source_family_id.capacity();
    bytes += group.channel_indices.capacity() * sizeof(int);
    bytes += group.channel_positions.capacity() * sizeof(int);
    bytes += group.q_source_variable_offsets.capacity() * sizeof(std::int64_t);
    bytes += group.direct_q_polynomials.capacity() * sizeof(LiftedCauchyDirectPolynomial);
    for (const auto &polynomial : group.direct_q_polynomials)
      bytes += polynomial.coefficients.capacity() * sizeof(double);
    bytes += group.factorized_radials.capacity() * sizeof(LiftedCauchyFactorizedRadial);
    bytes += group.transform_q_from_f.capacity() * sizeof(double);
  }
  bytes += heads.capacity() * sizeof(LiftedCauchyHead);
  for (const auto &head : heads) {
    bytes += head.central_species.capacity();
    bytes += head.polynomial.factor_offsets.capacity() * sizeof(std::int64_t);
    bytes += head.polynomial.factor_indices.capacity() * sizeof(std::int64_t);
    bytes += head.polynomial.factor_exponents.capacity() * sizeof(std::int64_t);
    bytes += head.polynomial.monomial_coefficients.capacity() * sizeof(double);
  }
  return bytes;
}

}    // namespace YE3T_LAMMPS
