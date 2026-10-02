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

#include "ye3t_yace_model.h"

#include "ye3t_canonical_json_hash.h"
#include "ye3t_sha256.h"

#include "ye3t_runtime_core.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace YE3T_LAMMPS {
namespace {

  using ChannelKey = std::tuple<int, int, int, int, int>;
  using MonomialKey = std::vector<std::pair<int, int>>;
  using PowerKey = std::pair<std::int64_t, std::int64_t>;

  struct MappedBlockDescriptor {
    int central_species = -1;
    int function_index = -1;
    int channel_index = -1;
    int tableau_index = -1;
    int magnetic_index = -1;
    std::string feature_id;
    std::string instruction_id;
    std::string direct_candidate_id;
    std::string block_candidate_id;
    std::string scalar_candidate_id;
    std::string scalar_factorization_id;
    std::complex<double> scale;
    double equivalence_absolute_tolerance = 0.0;
    double equivalence_relative_tolerance = 0.0;
  };

  struct ScalarPowerScheduleStep {
    std::int64_t output_exponent = 0;
    std::int64_t left_exponent = 0;
    std::int64_t right_exponent = 0;
  };

  struct ScalarRouteCandidate {
    YACEScalarInvariantBase base;
    std::vector<ScalarPowerScheduleStep> schedule;
    std::int64_t outer_power = 0;
    std::int64_t function_index = -1;
    std::complex<double> scale;
    std::string factorization_id;
    std::int64_t operation_estimate = 0;
  };

  struct BlockRouteCandidate {
    YACEBlockRoute route;
    std::vector<YACEBlockPowerPlan> power_plans;
    std::string direct_candidate_id;
    std::string block_candidate_id;
    std::string scalar_candidate_id;
    bool scalar_available = false;
    ScalarRouteCandidate scalar;
  };

  struct CoupledProductRouteCandidate {
    int central_species = -1;
    int function_index = -1;
    std::string direct_candidate_id;
    std::string candidate_id;
    YACECoupledProductDAGPlan plan;
  };

  struct CompiledEvaluatorCandidate {
    YACEEvaluatorKind evaluator = YACEEvaluatorKind::EXPLICIT_CTILDE;
    std::string candidate_id;
    std::int64_t operation_estimate = 0;
    std::vector<BlockRouteCandidate> block_terms;
    ScalarRouteCandidate scalar_route;
    YACECoupledProductDAGPlan coupled_product_plan;
  };

  struct FunctionCandidatePortfolio {
    int central_species = -1;
    int function_index = -1;
    std::string feature_id;
    std::vector<CompiledEvaluatorCandidate> candidates;
  };

  struct BlockSelection {
    std::vector<std::size_t> choices;
    std::int64_t operation_estimate = 0;
    std::string algorithm;
    std::string status = "selected";
    std::int64_t score_evaluations = 0;
    bool optimal = false;
  };

  [[noreturn]] void fail(const std::string &path, const std::string &message)
  {
    throw std::runtime_error(path + ": " + message);
  }

  std::string scalar_string(const YAML::Node &node, const std::string &path);
  std::complex<double> complex_number(const YAML::Node &node, const std::string &path);

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
      if (!entry.first.IsScalar()) fail(path, "expected scalar field names");
      actual.insert(entry.first.as<std::string>());
    }
    for (const auto &field : expected)
      if (actual.count(field) == 0) fail(path, "missing required field '" + field + "'");
    for (const auto &field : actual)
      if (expected.count(field) == 0) fail(path, "unsupported field '" + field + "'");
  }

  int integer(const YAML::Node &node, const std::string &path, int minimum)
  {
    if (!node || !node.IsScalar()) fail(path, "expected an integer");
    try {
      const int value = node.as<int>();
      if (value < minimum) fail(path, "value is below the supported minimum");
      return value;
    } catch (const YAML::Exception &) {
      fail(path, "expected an integer");
    }
  }

  int signed_integer(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsScalar()) fail(path, "expected an integer");
    try {
      return node.as<int>();
    } catch (const YAML::Exception &) {
      fail(path, "expected an integer");
    }
  }

  bool lower_hex_digest(const std::string &value, std::size_t width)
  {
    return value.size() == width && std::all_of(value.begin(), value.end(), [](char character) {
             return (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
           });
  }

  bool lower_sha256(const std::string &value)
  {
    return lower_hex_digest(value, 64);
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

  using WideSigned = __int128_t;
  using WideUnsigned = __uint128_t;

  WideUnsigned wide_magnitude(WideSigned value)
  {
    if (value >= 0) return static_cast<WideUnsigned>(value);
    return static_cast<WideUnsigned>(-(value + 1)) + 1;
  }

  WideUnsigned wide_gcd(WideUnsigned left, WideUnsigned right)
  {
    while (right != 0) {
      const WideUnsigned remainder = left % right;
      left = right;
      right = remainder;
    }
    return left;
  }

  std::uint64_t integer_magnitude(std::int64_t value)
  {
    if (value >= 0) return static_cast<std::uint64_t>(value);
    return static_cast<std::uint64_t>(-(value + 1)) + 1;
  }

  std::int64_t exact_integer(const YAML::Node &node, const std::string &path)
  {
    if (!node || !node.IsScalar()) fail(path, "expected an exact integer");
    const std::string value = node.Scalar();
    if (value.empty()) fail(path, "expected an exact integer");
    std::size_t position = 0;
    bool negative = false;
    if (value[position] == '-') {
      negative = true;
      ++position;
    }
    if (position == value.size() || (value[position] == '0' && position + 1 != value.size()))
      fail(path, "exact integer is not canonically encoded");
    const std::uint64_t limit = negative
        ? std::uint64_t{1} << 63
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t magnitude = 0;
    for (; position < value.size(); ++position) {
      const char character = value[position];
      if (character < '0' || character > '9') fail(path, "expected an exact integer");
      const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
      if (magnitude > (limit - digit) / 10)
        fail(path, "exact integer exceeds the native certificate bound");
      magnitude = 10 * magnitude + digit;
    }
    if (negative && magnitude == 0) fail(path, "negative zero is not a canonical exact integer");
    if (!negative) return static_cast<std::int64_t>(magnitude);
    if (magnitude == (std::uint64_t{1} << 63)) return std::numeric_limits<std::int64_t>::min();
    return -static_cast<std::int64_t>(magnitude);
  }

  struct ExactRational {
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;

    static ExactRational from_wide(WideSigned numerator, WideUnsigned denominator,
                                   const std::string &path)
    {
      if (denominator == 0) fail(path, "exact rational has a zero denominator");
      if (numerator == 0) return {};
      const WideUnsigned divisor = wide_gcd(wide_magnitude(numerator), denominator);
      numerator /= static_cast<WideSigned>(divisor);
      denominator /= divisor;
      if (numerator < std::numeric_limits<std::int64_t>::min() ||
          numerator > std::numeric_limits<std::int64_t>::max() ||
          denominator > static_cast<WideUnsigned>(std::numeric_limits<std::int64_t>::max()))
        fail(path, "exact rational exceeds the native certificate bound");
      return {static_cast<std::int64_t>(numerator), static_cast<std::int64_t>(denominator)};
    }

    bool zero() const { return numerator == 0; }

    long double value() const
    {
      return static_cast<long double>(numerator) / static_cast<long double>(denominator);
    }
  };

  bool operator==(const ExactRational &left, const ExactRational &right)
  {
    return left.numerator == right.numerator && left.denominator == right.denominator;
  }

  ExactRational add_exact(const ExactRational &left, const ExactRational &right,
                          const std::string &path)
  {
    const std::uint64_t divisor = std::gcd(static_cast<std::uint64_t>(left.denominator),
                                           static_cast<std::uint64_t>(right.denominator));
    const WideSigned left_scale = right.denominator / divisor;
    const WideSigned right_scale = left.denominator / divisor;
    return ExactRational::from_wide(
        static_cast<WideSigned>(left.numerator) * left_scale +
            static_cast<WideSigned>(right.numerator) * right_scale,
        static_cast<WideUnsigned>(left.denominator) * static_cast<WideUnsigned>(left_scale), path);
  }

  ExactRational multiply_exact(const ExactRational &left, const ExactRational &right,
                               const std::string &path)
  {
    const std::uint64_t left_cancel =
        std::gcd(integer_magnitude(left.numerator), static_cast<std::uint64_t>(right.denominator));
    const std::uint64_t right_cancel =
        std::gcd(integer_magnitude(right.numerator), static_cast<std::uint64_t>(left.denominator));
    const WideSigned numerator =
        (static_cast<WideSigned>(left.numerator) / static_cast<WideSigned>(left_cancel)) *
        (static_cast<WideSigned>(right.numerator) / static_cast<WideSigned>(right_cancel));
    const WideUnsigned denominator =
        static_cast<WideUnsigned>(left.denominator / static_cast<std::int64_t>(right_cancel)) *
        static_cast<WideUnsigned>(right.denominator / static_cast<std::int64_t>(left_cancel));
    return ExactRational::from_wide(numerator, denominator, path);
  }

  ExactRational parse_exact_rational(const YAML::Node &node, const std::string &path)
  {
    require_exact_fields(node, {"denominator", "numerator"}, path);
    const std::int64_t numerator = exact_integer(node["numerator"], path + ".numerator");
    const std::int64_t denominator = exact_integer(node["denominator"], path + ".denominator");
    if (denominator <= 0 || numerator == std::numeric_limits<std::int64_t>::min() ||
        std::gcd(integer_magnitude(numerator), static_cast<std::uint64_t>(denominator)) != 1 ||
        (numerator == 0 && denominator != 1))
      fail(path, "exact rational is not in reduced canonical form");
    return {numerator, denominator};
  }

  bool squarefree(std::uint64_t value)
  {
    if (value == 0) return false;
    for (std::uint64_t factor = 2; factor <= value / factor; factor += factor == 2 ? 1 : 2)
      if (value % (factor * factor) == 0) return false;
    return true;
  }

  struct ExactRadical {
    std::map<std::uint64_t, ExactRational> terms;

    void add_term(std::uint64_t radicand, const ExactRational &coefficient, const std::string &path)
    {
      if (coefficient.zero()) return;
      const auto found = terms.find(radicand);
      if (found == terms.end()) {
        terms.emplace(radicand, coefficient);
        return;
      }
      found->second = add_exact(found->second, coefficient, path);
      if (found->second.zero()) terms.erase(found);
    }

    bool zero() const { return terms.empty(); }

    long double value() const
    {
      long double result = 0.0L;
      for (const auto &[radicand, coefficient] : terms)
        result += coefficient.value() * std::sqrt(static_cast<long double>(radicand));
      return result;
    }
  };

  bool operator==(const ExactRadical &left, const ExactRadical &right)
  {
    return left.terms == right.terms;
  }

  ExactRadical add_exact(const ExactRadical &left, const ExactRadical &right,
                         const std::string &path)
  {
    ExactRadical result = left;
    for (const auto &[radicand, coefficient] : right.terms)
      result.add_term(radicand, coefficient, path);
    return result;
  }

  ExactRadical negate_exact(const ExactRadical &value)
  {
    ExactRadical result;
    for (const auto &[radicand, coefficient] : value.terms)
      result.terms.emplace(radicand,
                           ExactRational{-coefficient.numerator, coefficient.denominator});
    return result;
  }

  ExactRadical multiply_exact(const ExactRadical &left, const ExactRadical &right,
                              const std::string &path)
  {
    ExactRadical result;
    for (const auto &[left_radicand, left_coefficient] : left.terms)
      for (const auto &[right_radicand, right_coefficient] : right.terms) {
        const std::uint64_t common = std::gcd(left_radicand, right_radicand);
        const WideUnsigned radicand = static_cast<WideUnsigned>(left_radicand / common) *
            static_cast<WideUnsigned>(right_radicand / common);
        if (radicand > 1000000000ULL)
          fail(path, "exact radical product exceeds the native radicand bound");
        ExactRational coefficient = multiply_exact(left_coefficient, right_coefficient, path);
        coefficient =
            multiply_exact(coefficient, ExactRational{static_cast<std::int64_t>(common), 1}, path);
        result.add_term(static_cast<std::uint64_t>(radicand), coefficient, path);
      }
    return result;
  }

  ExactRadical parse_exact_radical(const YAML::Node &node, const std::string &path)
  {
    require_exact_fields(node, {"terms"}, path);
    const YAML::Node terms = node["terms"];
    require_sequence(terms, path + ".terms");
    if (terms.size() > 4096) fail(path, "exact radical term count exceeds the native bound");
    ExactRadical result;
    std::uint64_t previous = 0;
    for (std::size_t index = 0; index < terms.size(); ++index) {
      const std::string term_path = path + ".terms[" + std::to_string(index) + "]";
      require_exact_fields(terms[index], {"coefficient", "radicand"}, term_path);
      const ExactRational radicand =
          parse_exact_rational(terms[index]["radicand"], term_path + ".radicand");
      const ExactRational coefficient =
          parse_exact_rational(terms[index]["coefficient"], term_path + ".coefficient");
      if (radicand.denominator != 1 || radicand.numerator <= 0 ||
          radicand.numerator > 1000000000LL || coefficient.zero())
        fail(term_path, "exact radical term is outside the native canonical form");
      const std::uint64_t key = static_cast<std::uint64_t>(radicand.numerator);
      if ((index > 0 && key <= previous) || !squarefree(key))
        fail(term_path, "exact radical terms are not canonical and squarefree");
      previous = key;
      result.terms.emplace(key, coefficient);
    }
    return result;
  }

  struct ExactComplex {
    ExactRadical real;
    ExactRadical imaginary;

    bool zero() const { return real.zero() && imaginary.zero(); }
  };

  bool operator==(const ExactComplex &left, const ExactComplex &right)
  {
    return left.real == right.real && left.imaginary == right.imaginary;
  }

  ExactComplex add_exact(const ExactComplex &left, const ExactComplex &right,
                         const std::string &path)
  {
    return {add_exact(left.real, right.real, path),
            add_exact(left.imaginary, right.imaginary, path)};
  }

  ExactComplex multiply_exact(const ExactComplex &left, const ExactComplex &right,
                              const std::string &path)
  {
    return {add_exact(multiply_exact(left.real, right.real, path),
                      negate_exact(multiply_exact(left.imaginary, right.imaginary, path)), path),
            add_exact(multiply_exact(left.real, right.imaginary, path),
                      multiply_exact(left.imaginary, right.real, path), path)};
  }

  ExactComplex parse_exact_complex(const YAML::Node &node, const std::string &path)
  {
    require_exact_fields(node, {"imag", "real"}, path);
    return {parse_exact_radical(node["real"], path + ".real"),
            parse_exact_radical(node["imag"], path + ".imag")};
  }

  struct ExactSparseEntry {
    int row = 0;
    int column = 0;
    ExactComplex value;
  };

  struct ExactSparseMatrix {
    int rows = 0;
    int columns = 0;
    std::vector<ExactSparseEntry> entries;
  };

  ExactSparseMatrix parse_exact_matrix(const YAML::Node &node, const std::string &path)
  {
    require_exact_fields(node, {"entries", "shape"}, path);
    const YAML::Node shape = node["shape"];
    require_sequence(shape, path + ".shape");
    if (shape.size() != 2) fail(path, "exact matrix shape must have two dimensions");
    ExactSparseMatrix result;
    result.rows = integer(shape[0], path + ".shape[0]", 1);
    result.columns = integer(shape[1], path + ".shape[1]", 1);
    if (result.rows > 512 || result.columns > 512)
      fail(path, "exact matrix shape exceeds the native certificate bound");
    const YAML::Node entries = node["entries"];
    require_sequence(entries, path + ".entries");
    if (entries.size() > 262144)
      fail(path, "exact matrix entry count exceeds the native certificate bound");
    std::pair<int, int> previous{-1, -1};
    for (std::size_t index = 0; index < entries.size(); ++index) {
      const std::string entry_path = path + ".entries[" + std::to_string(index) + "]";
      require_exact_fields(entries[index], {"column", "row", "value"}, entry_path);
      ExactSparseEntry entry;
      entry.row = integer(entries[index]["row"], entry_path + ".row", 0);
      entry.column = integer(entries[index]["column"], entry_path + ".column", 0);
      entry.value = parse_exact_complex(entries[index]["value"], entry_path + ".value");
      if (entry.row >= result.rows || entry.column >= result.columns ||
          std::make_pair(entry.row, entry.column) <= previous || entry.value.zero())
        fail(entry_path, "exact matrix entry is out of range or noncanonical");
      previous = {entry.row, entry.column};
      result.entries.push_back(std::move(entry));
    }
    return result;
  }

  std::vector<ExactComplex> exact_matrix_column(const ExactSparseMatrix &matrix,
                                                const std::string &path)
  {
    if (matrix.columns != 1) fail(path, "exact vector must have one column");
    std::vector<ExactComplex> result(static_cast<std::size_t>(matrix.rows));
    for (const auto &entry : matrix.entries)
      result[static_cast<std::size_t>(entry.row)] = entry.value;
    return result;
  }

  void validate_exact_image_certificate(const YAML::Node &metadata, const std::string &path)
  {
    const YAML::Node certificate = metadata["exact_image_certificate"];
    require_exact_fields(certificate,
                         {"active_independent_column_indices",
                          "augmented_rank",
                          "certificate_sha256",
                          "coefficient_materialization",
                          "enumeration_complete",
                          "enumeration_scope",
                          "exact_residual_zero",
                          "independent_product_column_indices",
                          "independent_product_rank",
                          "matrix_rank",
                          "max_M_inconsistency",
                          "missing_basis_indices",
                          "missing_rank",
                          "product_column_order_sha256",
                          "product_matrix",
                          "product_root_node_ids",
                          "raw_product_column_count",
                          "readout_solution",
                          "schema",
                          "search_policy",
                          "target_basis_index",
                          "target_basis_indices",
                          "target_basis_order_sha256",
                          "target_coordinate"},
                         path + ".exact_image_certificate");
    const std::string certificate_path = path + ".exact_image_certificate";
    if (scalar_string(certificate["schema"], certificate_path + ".schema") !=
            "ye3t_ace_coupled_product_image_certificate_v1" ||
        !boolean(certificate["exact_residual_zero"], certificate_path + ".exact_residual_zero") ||
        !boolean(certificate["enumeration_complete"], certificate_path + ".enumeration_complete") ||
        scalar_string(certificate["enumeration_scope"], certificate_path + ".enumeration_scope") !=
            "complete_for_recorded_search_policy" ||
        scalar_string(certificate["coefficient_materialization"],
                      certificate_path + ".coefficient_materialization") != "binary64" ||
        !lower_sha256(scalar_string(certificate["certificate_sha256"],
                                    certificate_path + ".certificate_sha256")) ||
        !lower_sha256(scalar_string(certificate["target_basis_order_sha256"],
                                    certificate_path + ".target_basis_order_sha256")) ||
        !lower_sha256(scalar_string(certificate["product_column_order_sha256"],
                                    certificate_path + ".product_column_order_sha256")) ||
        !parse_exact_complex(certificate["max_M_inconsistency"],
                             certificate_path + ".max_M_inconsistency")
             .zero())
      fail(certificate_path, "exact-image certificate declarations are invalid");

    const YAML::Node search = certificate["search_policy"];
    require_mapping(search, certificate_path + ".search_policy");
    if (boolean(search["include_target_primitive"],
                certificate_path + ".search_policy.include_target_primitive"))
      fail(certificate_path, "exact-image search must exclude the target primitive");

    const ExactSparseMatrix product =
        parse_exact_matrix(certificate["product_matrix"], certificate_path + ".product_matrix");
    const ExactSparseMatrix target = parse_exact_matrix(certificate["target_coordinate"],
                                                        certificate_path + ".target_coordinate");
    const ExactSparseMatrix readout =
        parse_exact_matrix(certificate["readout_solution"], certificate_path + ".readout_solution");
    if (target.rows != product.rows || target.columns != 1 || readout.rows != product.columns ||
        readout.columns != 1)
      fail(certificate_path, "exact-image matrix shapes are inconsistent");

    const std::vector<ExactComplex> readout_values =
        exact_matrix_column(readout, certificate_path + ".readout_solution");
    const std::vector<ExactComplex> target_values =
        exact_matrix_column(target, certificate_path + ".target_coordinate");
    std::vector<ExactComplex> reconstructed(static_cast<std::size_t>(product.rows));
    for (const auto &entry : product.entries) {
      const ExactComplex contribution =
          multiply_exact(entry.value, readout_values[static_cast<std::size_t>(entry.column)],
                         certificate_path + ".P_times_r");
      reconstructed[static_cast<std::size_t>(entry.row)] =
          add_exact(reconstructed[static_cast<std::size_t>(entry.row)], contribution,
                    certificate_path + ".P_times_r");
    }
    if (reconstructed != target_values)
      fail(certificate_path, "native exact-image validation failed P r = c");

    const int selected_basis = integer(metadata["target"]["selected_basis_index"],
                                       path + ".target.selected_basis_index", 0);
    if (integer(certificate["target_basis_index"], certificate_path + ".target_basis_index", 0) !=
            selected_basis ||
        selected_basis >= target.rows)
      fail(certificate_path, "exact-image target basis binding is inconsistent");
    for (int row = 0; row < target.rows; ++row) {
      ExactComplex expected;
      if (row == selected_basis) expected.real.terms.emplace(1, ExactRational{1, 1});
      if (!(target_values[static_cast<std::size_t>(row)] == expected))
        fail(certificate_path, "exact-image target is not the selected unit basis vector");
    }
    const int matrix_rank =
        integer(certificate["matrix_rank"], certificate_path + ".matrix_rank", 0);
    const int augmented_rank =
        integer(certificate["augmented_rank"], certificate_path + ".augmented_rank", 0);
    if (matrix_rank != augmented_rank || matrix_rank <= 0)
      fail(certificate_path, "exact-image rank declarations are inconsistent");

    const YAML::Node root_ids = certificate["product_root_node_ids"];
    require_sequence(root_ids, certificate_path + ".product_root_node_ids");
    if (root_ids.size() != static_cast<std::size_t>(product.columns))
      fail(certificate_path, "exact-image root count does not match P columns");
    std::map<std::string, ExactComplex> solution_by_root;
    for (std::size_t index = 0; index < root_ids.size(); ++index) {
      const std::string root =
          scalar_string(root_ids[index], certificate_path + ".product_root_node_ids");
      if (root.empty() || !solution_by_root.emplace(root, readout_values[index]).second)
        fail(certificate_path, "exact-image root IDs are empty or duplicate");
    }
    const YAML::Node outputs = metadata["outputs"];
    require_sequence(outputs, path + ".outputs");
    if (outputs.size() != 1) fail(path, "coupled-product plan must have one output");
    const YAML::Node terms = outputs[0]["terms"];
    require_sequence(terms, path + ".outputs[0].terms");
    if (terms.size() != solution_by_root.size())
      fail(path, "coupled-product output does not cover the exact readout");
    std::set<std::string> seen_roots;
    for (std::size_t index = 0; index < terms.size(); ++index) {
      const std::string term_path = path + ".outputs[0].terms[" + std::to_string(index) + "]";
      const std::string root =
          scalar_string(terms[index]["root_node_id"], term_path + ".root_node_id");
      const auto solution = solution_by_root.find(root);
      if (solution == solution_by_root.end() || !seen_roots.insert(root).second ||
          !(parse_exact_complex(terms[index]["coefficient_exact"],
                                term_path + ".coefficient_exact") == solution->second))
        fail(term_path, "output readout disagrees with the exact solution");
      const std::complex<double> binary =
          complex_number(terms[index]["coefficient_binary64"], term_path + ".coefficient_binary64");
      const std::complex<long double> exact(solution->second.real.value(),
                                            solution->second.imaginary.value());
      const long double residual =
          std::abs(exact - std::complex<long double>(binary.real(), binary.imag()));
      const long double scale = 1.0L + std::abs(exact);
      if (residual > 8.0L * std::numeric_limits<double>::epsilon() * scale)
        fail(term_path, "binary64 output coefficient disagrees with the exact solution");
    }
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

  YAML::Node record_by_id(const YAML::Node &records, const std::string &field,
                          const std::string &identifier, const std::string &path)
  {
    require_sequence(records, path);
    YAML::Node selected;
    bool found = false;
    for (std::size_t index = 0; index < records.size(); ++index) {
      require_mapping(records[index], path + "[" + std::to_string(index) + "]");
      if (scalar_string(records[index][field], path + "." + field) != identifier) continue;
      if (found) fail(path, "duplicate identifier '" + identifier + "'");
      selected = records[index];
      found = true;
    }
    if (!found) fail(path, "missing identifier '" + identifier + "'");
    return selected;
  }

  std::string payload_path(const std::string &manifest_path, const YAML::Node &payload,
                           const std::string &path)
  {
    require_exact_fields(payload, {"path", "schema", "sha256"}, path);
    const std::filesystem::path relative(scalar_string(payload["path"], path + ".path"));
    if (relative.empty() || relative.is_absolute())
      fail(path + ".path", "payload paths must be nonempty and relative");
    for (const auto &component : relative)
      if (component == "..") fail(path + ".path", "payload path escapes its bundle");
    const std::filesystem::path manifest(manifest_path);
    return (manifest.parent_path() / relative).lexically_normal().string();
  }

  std::vector<int> integer_sequence(const YAML::Node &node, const std::string &path,
                                    std::size_t expected_size, int minimum = 0)
  {
    require_sequence(node, path);
    if (node.size() != expected_size) fail(path, "sequence has the wrong length");
    std::vector<int> values;
    values.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      values.push_back(integer(node[index], path + "[" + std::to_string(index) + "]", minimum));
    return values;
  }

  std::vector<double> number_sequence(const YAML::Node &node, const std::string &path,
                                      std::size_t expected_size)
  {
    require_sequence(node, path);
    if (node.size() != expected_size) fail(path, "sequence has the wrong length");
    std::vector<double> values;
    values.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index)
      values.push_back(finite_number(node[index], path + "[" + std::to_string(index) + "]"));
    return values;
  }

  std::complex<double> complex_number(const YAML::Node &node, const std::string &path)
  {
    const auto values = number_sequence(node, path, 2);
    return {values[0], values[1]};
  }

  void append_semantic_field(std::string &payload, const std::string &value)
  {
    payload += std::to_string(value.size());
    payload.push_back(':');
    payload += value;
  }

  void append_semantic_integer(std::string &payload, std::int64_t value)
  {
    append_semantic_field(payload, std::to_string(value));
  }

  void append_semantic_double(std::string &payload, double value)
  {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_semantic_field(payload, std::to_string(bits));
  }

  void append_semantic_complex(std::string &payload, const std::complex<double> &value)
  {
    append_semantic_double(payload, value.real());
    append_semantic_double(payload, value.imag());
  }

  std::string candidate_binding_identity(const YAML::Node &term, const std::string &path)
  {
    std::string payload;
    append_semantic_field(payload, "ye3t_candidate_binding_v1");
    append_semantic_field(payload, scalar_string(term["instruction_id"], path + ".instruction_id"));
    append_semantic_integer(payload, integer(term["channel_index"], path + ".channel_index", 0));
    append_semantic_integer(payload, integer(term["tableau_index"], path + ".tableau_index", 0));
    append_semantic_integer(payload, integer(term["magnetic_index"], path + ".magnetic_index", 0));
    append_semantic_complex(payload, complex_number(term["scale"], path + ".scale"));
    return sha256_string(payload);
  }

  std::string candidate_readout_identity(const YAML::Node &record, const std::string &path)
  {
    const YAML::Node equivalence = record["equivalence"];
    const YAML::Node terms = record["terms"];
    std::string payload;
    append_semantic_field(payload, "ye3t_candidate_readout_identity_v1");
    append_semantic_field(payload, scalar_string(record["schema"], path + ".schema"));
    append_semantic_integer(payload, integer(record["central_type"], path + ".central_type", 0));
    append_semantic_integer(payload,
                            integer(record["function_index"], path + ".function_index", 0));
    append_semantic_field(payload, scalar_string(record["feature_id"], path + ".feature_id"));
    append_semantic_field(
        payload, scalar_string(record["source_yace_sha256"], path + ".source_yace_sha256"));
    append_semantic_field(
        payload, scalar_string(record["variable_order_hash"], path + ".variable_order_hash"));
    const std::string method = scalar_string(equivalence["method"], path + ".equivalence.method");
    append_semantic_field(payload, method);
    append_semantic_field(
        payload,
        scalar_string(equivalence["derivative_rule"], path + ".equivalence.derivative_rule"));
    append_semantic_integer(payload, boolean(equivalence["passed"], path + ".equivalence.passed"));
    append_semantic_integer(
        payload, boolean(equivalence["support_equal"], path + ".equivalence.support_equal"));
    append_semantic_integer(
        payload,
        boolean(equivalence["adjoint_certified_by_coefficient_identity"],
                path + ".equivalence.adjoint_certified_by_coefficient_identity"));
    append_semantic_double(
        payload,
        finite_number(equivalence["absolute_tolerance"], path + ".equivalence.absolute_tolerance"));
    append_semantic_double(
        payload,
        finite_number(equivalence["relative_tolerance"], path + ".equivalence.relative_tolerance"));
    append_semantic_double(
        payload,
        finite_number(equivalence["maximum_absolute_coefficient_residual"],
                      path + ".equivalence.maximum_absolute_coefficient_residual"));
    append_semantic_double(payload,
                           finite_number(equivalence["relative_l2_residual"],
                                         path + ".equivalence.relative_l2_residual"));
    if (method == "coefficientwise_sparse_polynomial_mixed_v1") {
      append_semantic_field(
          payload,
          scalar_string(equivalence["tolerance_rule"], path + ".equivalence.tolerance_rule"));
      append_semantic_double(payload,
                             finite_number(equivalence["maximum_mixed_tolerance_ratio"],
                                           path + ".equivalence.maximum_mixed_tolerance_ratio"));
      append_semantic_double(payload,
                             finite_number(equivalence["maximum_reference_coefficient_magnitude"],
                                           path +
                                               ".equivalence.maximum_reference_coefficient_"
                                               "magnitude"));
    }
    append_semantic_field(payload,
                          scalar_string(equivalence["plan_polynomial_sha256"],
                                        path + ".equivalence.plan_polynomial_sha256"));
    append_semantic_field(payload,
                          scalar_string(equivalence["yace_polynomial_sha256"],
                                        path + ".equivalence.yace_polynomial_sha256"));
    require_sequence(terms, path + ".terms");
    append_semantic_integer(payload, static_cast<std::int64_t>(terms.size()));
    for (std::size_t index = 0; index < terms.size(); ++index) {
      const YAML::Node term = terms[index];
      const std::string term_path = path + ".terms[" + std::to_string(index) + "]";
      append_semantic_field(payload, scalar_string(term["binding_id"], term_path + ".binding_id"));
      append_semantic_field(payload,
                            scalar_string(term["instruction_id"], term_path + ".instruction_id"));
      append_semantic_integer(payload,
                              integer(term["channel_index"], term_path + ".channel_index", 0));
      append_semantic_integer(payload,
                              integer(term["tableau_index"], term_path + ".tableau_index", 0));
      append_semantic_integer(payload,
                              integer(term["magnetic_index"], term_path + ".magnetic_index", 0));
      append_semantic_complex(payload, complex_number(term["scale"], term_path + ".scale"));
    }
    return sha256_string(payload);
  }

  std::string evaluator_candidate_identity(const YAML::Node &alternative,
                                           const std::set<std::string> &capabilities,
                                           const std::string &path)
  {
    const std::string evaluator = scalar_string(alternative["evaluator"], path + ".evaluator");
    const YAML::Node availability = alternative["availability"];
    std::string payload;
    append_semantic_field(payload, "ye3t_evaluator_candidate_identity_v1");
    append_semantic_field(payload, evaluator);
    append_semantic_field(payload,
                          scalar_string(availability["status"], path + ".availability.status"));
    append_semantic_field(payload,
                          scalar_string(availability["reason"], path + ".availability.reason"));
    append_semantic_integer(payload, static_cast<std::int64_t>(capabilities.size()));
    for (const auto &capability : capabilities) append_semantic_field(payload, capability);
    if (evaluator == "explicit_ctilde") {
      append_semantic_complex(payload, complex_number(alternative["scale"], path + ".scale"));
      const YAML::Node source = alternative["source_binding"];
      append_semantic_integer(
          payload, integer(source["central_type"], path + ".source_binding.central_type", 0));
      append_semantic_integer(
          payload, integer(source["function_index"], path + ".source_binding.function_index", 0));
      append_semantic_field(
          payload, scalar_string(source["feature_id"], path + ".source_binding.feature_id"));
      append_semantic_field(
          payload,
          scalar_string(source["source_yace_sha256"], path + ".source_binding.source_yace_sha256"));
    } else if (evaluator == "execution_plan_readout") {
      append_semantic_field(
          payload, scalar_string(alternative["compiler_plan_hash"], path + ".compiler_plan_hash"));
      append_semantic_field(payload,
                            scalar_string(alternative["readout_id"], path + ".readout_id"));
    } else {
      fail(path, "unsupported evaluator candidate identity");
    }
    return sha256_string(payload);
  }

  YACEChannel make_channel(const ChannelKey &key)
  {
    YACEChannel channel;
    channel.kind = std::get<0>(key) == 0 ? YACEChannel::RADIAL_BASE
                                         : YACEChannel::CONTRACTED_RADIAL_ANGULAR;
    channel.neighbor_species = std::get<1>(key);
    channel.radial = std::get<2>(key);
    channel.angular = std::get<3>(key);
    channel.magnetic = std::get<4>(key);
    return channel;
  }

  void build_splines(YACEBond &bond, double requested_spacing)
  {
    bond.spline_interval_count =
        ye3t::runtime::pace_uniform_spline_interval_count<double>(requested_spacing, bond.cutoff);
    const std::int64_t node_count = bond.spline_interval_count;
    const double spacing = bond.cutoff / static_cast<double>(node_count);
    std::vector<double> node_radii(static_cast<std::size_t>(node_count));
    std::vector<double> cutoffs(node_radii.size(), bond.cutoff);
    std::vector<double> cutoff_widths(node_radii.size(), bond.cutoff_width);
    std::vector<double> lambdas(node_radii.size(), bond.radial_lambda);
    for (std::int64_t node = 0; node < node_count; ++node)
      node_radii[static_cast<std::size_t>(node)] = spacing * static_cast<double>(node + 1);

    std::vector<double> base_values(node_radii.size() *
                                    static_cast<std::size_t>(bond.radial_base_count));
    std::vector<double> base_derivatives(base_values.size());
    ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
        node_radii.data(), cutoffs.data(), cutoff_widths.data(), lambdas.data(), node_count,
        bond.radial_base_count, base_values.data(), base_derivatives.data());

    bond.radial_base_spline.resize(
        static_cast<std::size_t>((node_count + 1) * bond.radial_base_count * 4));
    ye3t::runtime::pace_uniform_cubic_spline_build<double>(
        base_values.data(), base_derivatives.data(), node_count, bond.radial_base_count,
        bond.cutoff, bond.radial_base_spline.data());

    const int contracted_width = bond.contracted_width();
    std::vector<double> contracted_values(node_radii.size() *
                                          static_cast<std::size_t>(contracted_width));
    std::vector<double> contracted_derivatives(contracted_values.size());
    ye3t::runtime::pace_radial_channel_contraction_with_derivative<double>(
        base_values.data(), base_derivatives.data(), bond.radial_coefficients.data(), node_count,
        bond.radial_base_count, contracted_width, contracted_values.data(),
        contracted_derivatives.data());
    bond.contracted_spline.resize(
        static_cast<std::size_t>((node_count + 1) * contracted_width * 4));
    ye3t::runtime::pace_uniform_cubic_spline_build<double>(
        contracted_values.data(), contracted_derivatives.data(), node_count, contracted_width,
        bond.cutoff, bond.contracted_spline.data());
  }

  struct MonomialDAG {
    std::vector<std::int64_t> power_channels;
    std::vector<std::int64_t> power_exponents;
    std::vector<std::int64_t> node_parents;
    std::vector<std::int64_t> node_powers;
    std::int64_t root_node_count = 0;
    std::vector<std::int64_t> monomial_nodes;
  };

  MonomialDAG compile_monomial_dag(const YACESparsePolynomial &plan, bool frequent_factors_first)
  {
    std::map<PowerKey, std::int64_t> frequencies;
    for (std::size_t factor = 0; factor < plan.factor_indices.size(); ++factor) {
      const PowerKey key(plan.factor_indices[factor], plan.factor_exponents[factor]);
      ++frequencies[key];
    }

    MonomialDAG result;
    std::map<PowerKey, std::int64_t> powers;
    std::vector<std::map<std::int64_t, std::int64_t>> children(1);
    result.node_parents.push_back(-1);
    result.node_powers.push_back(-1);

    const std::int64_t monomial_count =
        static_cast<std::int64_t>(plan.monomial_coefficients.size());
    result.monomial_nodes.reserve(static_cast<std::size_t>(monomial_count));
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      std::int64_t node = 0;
      const std::int64_t begin = plan.factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t end = plan.factor_offsets[static_cast<std::size_t>(term + 1)];
      std::vector<PowerKey> factors;
      factors.reserve(static_cast<std::size_t>(end - begin));
      for (std::int64_t factor = begin; factor < end; ++factor)
        factors.emplace_back(plan.factor_indices[static_cast<std::size_t>(factor)],
                             plan.factor_exponents[static_cast<std::size_t>(factor)]);
      if (frequent_factors_first) {
        std::sort(factors.begin(), factors.end(),
                  [&frequencies](const PowerKey &left, const PowerKey &right) {
                    const std::int64_t left_frequency = frequencies.at(left);
                    const std::int64_t right_frequency = frequencies.at(right);
                    if (left_frequency != right_frequency) return left_frequency > right_frequency;
                    return left < right;
                  });
      }
      for (const auto &key : factors) {
        const auto inserted = powers.emplace(key, static_cast<std::int64_t>(powers.size()));
        const std::int64_t power = inserted.first->second;
        if (inserted.second) {
          result.power_channels.push_back(key.first);
          result.power_exponents.push_back(key.second);
        }

        const auto child = children[static_cast<std::size_t>(node)].find(power);
        if (child != children[static_cast<std::size_t>(node)].end()) {
          node = child->second;
          continue;
        }
        const std::int64_t next = static_cast<std::int64_t>(children.size());
        children[static_cast<std::size_t>(node)].emplace(power, next);
        children.emplace_back();
        result.node_parents.push_back(node);
        result.node_powers.push_back(power);
        node = next;
      }
      result.monomial_nodes.push_back(node);
    }

    std::vector<std::int64_t> node_order{0};
    for (std::int64_t node = 1; node < static_cast<std::int64_t>(result.node_parents.size());
         ++node) {
      if (result.node_parents[static_cast<std::size_t>(node)] == 0) node_order.push_back(node);
    }
    result.root_node_count = static_cast<std::int64_t>(node_order.size()) - 1;
    for (std::int64_t node = 1; node < static_cast<std::int64_t>(result.node_parents.size());
         ++node) {
      if (result.node_parents[static_cast<std::size_t>(node)] != 0) node_order.push_back(node);
    }
    std::vector<std::int64_t> old_to_new(node_order.size());
    for (std::size_t node = 0; node < node_order.size(); ++node)
      old_to_new[static_cast<std::size_t>(node_order[node])] = static_cast<std::int64_t>(node);
    std::vector<std::int64_t> ordered_parents(node_order.size());
    std::vector<std::int64_t> ordered_powers(node_order.size());
    ordered_parents[0] = -1;
    ordered_powers[0] = -1;
    for (std::size_t node = 1; node < node_order.size(); ++node) {
      const std::int64_t old_node = node_order[node];
      ordered_parents[node] = old_to_new[static_cast<std::size_t>(
          result.node_parents[static_cast<std::size_t>(old_node)])];
      ordered_powers[node] = result.node_powers[static_cast<std::size_t>(old_node)];
    }
    for (auto &node : result.monomial_nodes) node = old_to_new[static_cast<std::size_t>(node)];
    result.node_parents = std::move(ordered_parents);
    result.node_powers = std::move(ordered_powers);
    return result;
  }

  struct BinaryMonomialDAG {
    std::vector<std::int64_t> power_channels;
    std::vector<std::int64_t> power_exponents;
    std::vector<std::int64_t> node_left;
    std::vector<std::int64_t> node_right;
    std::vector<std::int64_t> monomial_operands;
  };

  BinaryMonomialDAG compile_binary_monomial_dag(const YACESparsePolynomial &plan,
                                                int factor_ordering)
  {
    std::map<PowerKey, std::int64_t> frequencies;
    std::map<PowerKey, std::int64_t> powers;
    BinaryMonomialDAG result;
    for (std::size_t factor = 0; factor < plan.factor_indices.size(); ++factor) {
      const PowerKey key(plan.factor_indices[factor], plan.factor_exponents[factor]);
      ++frequencies[key];
      const auto inserted = powers.emplace(key, static_cast<std::int64_t>(powers.size()));
      if (inserted.second) {
        result.power_channels.push_back(key.first);
        result.power_exponents.push_back(key.second);
      }
    }

    const std::int64_t power_count = static_cast<std::int64_t>(result.power_channels.size());
    std::map<std::vector<PowerKey>, std::int64_t> products;
    std::function<std::int64_t(const std::vector<PowerKey> &)> materialize =
        [&](const std::vector<PowerKey> &factors) {
          if (factors.empty()) return std::int64_t(-1);
          if (factors.size() == 1) return powers.at(factors.front());
          const auto cached = products.find(factors);
          if (cached != products.end()) return cached->second;

          const std::int64_t total_rank =
              std::accumulate(factors.begin(), factors.end(), std::int64_t(0),
                              [](std::int64_t rank, const PowerKey &factor) {
                                return rank + factor.second;
                              });
          std::int64_t prefix_rank = 0;
          std::size_t split = 1;
          std::int64_t imbalance = total_rank;
          for (std::size_t candidate = 1; candidate < factors.size(); ++candidate) {
            prefix_rank += factors[candidate - 1].second;
            const std::int64_t candidate_imbalance = std::abs(total_rank - 2 * prefix_rank);
            if (candidate_imbalance < imbalance) {
              imbalance = candidate_imbalance;
              split = candidate;
            }
          }
          const std::vector<PowerKey> left_factors(
              factors.begin(), factors.begin() + static_cast<std::ptrdiff_t>(split));
          const std::vector<PowerKey> right_factors(
              factors.begin() + static_cast<std::ptrdiff_t>(split), factors.end());
          const std::int64_t left = materialize(left_factors);
          const std::int64_t right = materialize(right_factors);
          const std::int64_t operand =
              power_count + static_cast<std::int64_t>(result.node_left.size());
          products.emplace(factors, operand);
          result.node_left.push_back(left);
          result.node_right.push_back(right);
          return operand;
        };

    const std::int64_t monomial_count =
        static_cast<std::int64_t>(plan.monomial_coefficients.size());
    result.monomial_operands.reserve(static_cast<std::size_t>(monomial_count));
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const std::int64_t begin = plan.factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t end = plan.factor_offsets[static_cast<std::size_t>(term + 1)];
      std::vector<PowerKey> factors;
      factors.reserve(static_cast<std::size_t>(end - begin));
      for (std::int64_t factor = begin; factor < end; ++factor)
        factors.emplace_back(plan.factor_indices[static_cast<std::size_t>(factor)],
                             plan.factor_exponents[static_cast<std::size_t>(factor)]);
      if (factor_ordering == 1) {
        std::reverse(factors.begin(), factors.end());
      } else if (factor_ordering == 2) {
        std::sort(factors.begin(), factors.end(),
                  [&frequencies](const PowerKey &left, const PowerKey &right) {
                    const std::int64_t left_frequency = frequencies.at(left);
                    const std::int64_t right_frequency = frequencies.at(right);
                    if (left_frequency != right_frequency) return left_frequency > right_frequency;
                    return left < right;
                  });
      }
      result.monomial_operands.push_back(materialize(factors));
    }
    return result;
  }

  void build_monomial_dag(YACESparsePolynomial &plan)
  {
    MonomialDAG selected = compile_monomial_dag(plan, false);
    MonomialDAG frequent = compile_monomial_dag(plan, true);
    std::string prefix_ordering = "channel-order";
    if (frequent.node_parents.size() < selected.node_parents.size()) {
      selected = std::move(frequent);
      prefix_ordering = "frequent-first";
    }

    BinaryMonomialDAG binary = compile_binary_monomial_dag(plan, 0);
    std::string binary_ordering = "balanced-channel-order";
    BinaryMonomialDAG reverse = compile_binary_monomial_dag(plan, 1);
    if (reverse.node_left.size() < binary.node_left.size()) {
      binary = std::move(reverse);
      binary_ordering = "balanced-reverse-channel-order";
    }
    BinaryMonomialDAG binary_frequent = compile_binary_monomial_dag(plan, 2);
    if (binary_frequent.node_left.size() < binary.node_left.size()) {
      binary = std::move(binary_frequent);
      binary_ordering = "balanced-frequent-first";
    }

    const std::int64_t prefix_product_count =
        static_cast<std::int64_t>(selected.node_parents.size()) - 1 - selected.root_node_count;
    if (static_cast<std::int64_t>(binary.node_left.size()) < prefix_product_count) {
      plan.binary_dag = true;
      plan.power_channels = std::move(binary.power_channels);
      plan.power_exponents = std::move(binary.power_exponents);
      plan.binary_node_left = std::move(binary.node_left);
      plan.binary_node_right = std::move(binary.node_right);
      plan.monomial_nodes = std::move(binary.monomial_operands);
      plan.dag_factor_ordering = std::move(binary_ordering);
      return;
    }

    plan.dag_factor_ordering = std::move(prefix_ordering);
    plan.power_channels = std::move(selected.power_channels);
    plan.power_exponents = std::move(selected.power_exponents);
    plan.dag_node_parents = std::move(selected.node_parents);
    plan.dag_node_powers = std::move(selected.node_powers);
    plan.dag_root_node_count = selected.root_node_count;
    plan.monomial_nodes = std::move(selected.monomial_nodes);
  }

  int full_channel_index(const YACESpecies &species, int neighbor_species, int radial, int angular,
                         int magnetic, const std::string &path)
  {
    int selected = -1;
    for (std::size_t index = 0; index < species.channels.size(); ++index) {
      const auto &channel = species.channels[index];
      if (channel.kind != YACEChannel::CONTRACTED_RADIAL_ANGULAR ||
          channel.neighbor_species != neighbor_species || channel.radial != radial ||
          channel.angular != angular || channel.magnetic != magnetic)
        continue;
      if (selected >= 0) fail(path, "duplicate YACE channel binding");
      selected = static_cast<int>(index);
    }
    if (selected < 0) fail(path, "sidecar source channel is absent from the YACE model");
    return selected;
  }

  std::int64_t direct_descriptor_operations(const YACESparsePolynomial &plan, int function_index)
  {
    const std::int64_t begin = plan.descriptor_offsets[static_cast<std::size_t>(function_index)];
    const std::int64_t end = plan.descriptor_offsets[static_cast<std::size_t>(function_index + 1)];
    std::int64_t operations = 0;
    for (std::int64_t row = begin; row < end; ++row) {
      const std::int64_t term = plan.descriptor_terms[static_cast<std::size_t>(row)];
      const std::int64_t factor_begin = plan.factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t factor_end = plan.factor_offsets[static_cast<std::size_t>(term + 1)];
      const std::int64_t factor_count = factor_end - factor_begin;
      operations += 2 + 3 * std::max<std::int64_t>(factor_count - 1, 0);
      for (std::int64_t factor = factor_begin; factor < factor_end; ++factor)
        operations +=
            std::max<std::int64_t>(plan.factor_exponents[static_cast<std::size_t>(factor)] - 1, 0);
    }
    return operations;
  }

  std::int64_t direct_plan_operations(const YACESparsePolynomial &plan)
  {
    std::int64_t power_work = 0;
    for (const std::int64_t exponent : plan.power_exponents)
      power_work += std::max<std::int64_t>(exponent - 1, 0);
    const std::int64_t products = plan.binary_dag
        ? static_cast<std::int64_t>(plan.binary_node_left.size())
        : static_cast<std::int64_t>(plan.dag_node_parents.size()) - 1 - plan.dag_root_node_count;
    return 2 * power_work + 3 * std::max<std::int64_t>(products, 0) +
        2 * static_cast<std::int64_t>(plan.monomial_coefficients.size());
  }

  std::int64_t block_plan_power_operations(const YACEBlockPowerPlan &plan)
  {
    if (plan.direct_input_plan) return 0;
    std::int64_t power_work = 0;
    for (std::int64_t component = 0; component < plan.input_dimension; ++component) {
      std::int64_t maximum = 0;
      for (std::int64_t term = 0; term < plan.monomial_count; ++term)
        maximum = std::max(maximum,
                           plan.monomial_counts[static_cast<std::size_t>(
                               term * plan.input_dimension + component)]);
      power_work += std::max<std::int64_t>(maximum - 1, 0);
    }
    return power_work;
  }

  std::int64_t block_plan_contraction_operations(const YACEBlockPowerPlan &plan)
  {
    if (plan.direct_input_plan)
      return 2 *
          static_cast<std::int64_t>(std::count_if(plan.direct_input_channels.begin(),
                                                  plan.direct_input_channels.end(),
                                                  [](std::int64_t channel) {
                                                    return channel >= 0;
                                                  }));
    std::int64_t monomial_work = 0;
    std::int64_t derivative_work = 0;
    for (std::int64_t term = 0; term < plan.monomial_count; ++term) {
      const std::int64_t begin = plan.monomial_factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t end = plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
      const std::int64_t support = end - begin;
      monomial_work += std::max<std::int64_t>(support - 1, 0);
      derivative_work += 6 * support;
    }
    const std::int64_t coefficients = static_cast<std::int64_t>(plan.coefficient_values.size());
    return monomial_work + derivative_work + 4 * coefficients;
  }

  std::int64_t block_plan_operations(const YACEBlockPowerPlan &plan)
  {
    return block_plan_power_operations(plan) + block_plan_contraction_operations(plan);
  }

  bool same_block_plan(const YACEBlockPowerPlan &left, const YACEBlockPowerPlan &right)
  {
    return left.input_channels == right.input_channels &&
        left.monomial_counts == right.monomial_counts &&
        left.output_offsets == right.output_offsets &&
        left.coefficient_terms == right.coefficient_terms &&
        left.coefficient_values == right.coefficient_values &&
        left.input_dimension == right.input_dimension &&
        left.output_dimension == right.output_dimension && left.output_L == right.output_L &&
        left.conjugate_half_output == right.conjugate_half_output &&
        left.direct_input_plan == right.direct_input_plan &&
        left.direct_input_channels == right.direct_input_channels &&
        left.direct_input_scales == right.direct_input_scales;
  }

  YACEBlockPowerPlan parse_block_power_plan(const YAML::Node &record, const YAML::Node &block,
                                            int central_species,
                                            const std::vector<YACESpecies> &species,
                                            const std::string &path)
  {
    require_mapping(record, path);
    const YAML::Node plan_node = record["plan"];
    require_mapping(plan_node, path + ".plan");
    if (scalar_string(plan_node["backend"], path + ".plan.backend") !=
        "symmetric_power_exponent_vector")
      fail(path, "unsupported block power backend");
    if (scalar_string(plan_node["carrier"], path + ".plan.carrier") != "ACE_density" ||
        scalar_string(plan_node["factor_basis"], path + ".plan.factor_basis") != "A" ||
        scalar_string(plan_node["normalization_convention"],
                      path + ".plan.normalization_convention") != "none")
      fail(path, "block power factor semantics do not match ordinary YACE density");

    require_mapping(block, path + ".block");
    const int block_index = integer(block["block_index"], path + ".block.block_index", 0);
    if (integer(record["block_index"], path + ".block_index", 0) != block_index)
      fail(path, "block power plan is bound to the wrong block");
    const int block_power = integer(block["power"], path + ".block.power", 1);
    const auto block_partition =
        integer_sequence(block["block_partition"], path + ".block.block_partition", 1);
    const auto block_slots = integer_sequence(block["slot_indices"], path + ".block.slot_indices",
                                              static_cast<std::size_t>(block_power));
    if (block_partition[0] != block_power)
      fail(path, "ordinary-density block is not in its fully symmetric partition");
    const YAML::Node binding = block["source_binding"];
    require_mapping(binding, path + ".block.source_binding");
    const YAML::Node yace = binding["yace"];
    require_mapping(yace, path + ".block.source_binding.yace");
    if (integer(yace["central_type"], path + ".central_type", 0) != central_species)
      fail(path, "block source central type does not match its function");
    if (scalar_string(yace["factor_basis"], path + ".factor_basis") != "A" ||
        scalar_string(yace["magnetic_order"], path + ".magnetic_order") != "minus_l_to_plus_l")
      fail(path, "unsupported block source basis or magnetic order");
    const int neighbor_species = integer(yace["neighbor_type"], path + ".neighbor_type", 0);
    if (neighbor_species >= static_cast<int>(species.size()))
      fail(path, "block source neighbor type is out of range");
    const int radial = integer(yace["radial_n"], path + ".radial_n", 1) - 1;
    const int angular = integer(yace["angular_l"], path + ".angular_l", 0);
    const int input_dimension = 2 * angular + 1;
    if (integer(binding["input_L"], path + ".source_binding.input_L", 0) != angular ||
        integer(block["input_L"], path + ".block.input_L", 0) != angular ||
        integer(plan_node["channel_count"], path + ".plan.channel_count", 1) != input_dimension)
      fail(path, "block source angular dimension is inconsistent");
    const YAML::Node plan_validation = plan_node["validation_report"];
    require_mapping(plan_validation, path + ".plan.validation_report");
    if (!boolean(plan_validation["passed"], path + ".plan.validation_report.passed") ||
        !boolean(plan_validation["supports_descriptor_adjoint"],
                 path + ".plan.validation_report.supports_descriptor_adjoint") ||
        !boolean(plan_validation["ordinary_density_selection_enforced"],
                 path + ".plan.validation_report.ordinary_density_selection_enforced") ||
        scalar_string(plan_validation["source_kind"],
                      path + ".plan.validation_report.source_kind") != "ordinary_density" ||
        integer_sequence(plan_validation["slot_indices"],
                         path + ".plan.validation_report.slot_indices",
                         block_slots.size()) != block_slots)
      fail(path, "block power plan validation does not match its declared block");

    YACEBlockPowerPlan result;
    result.input_dimension = input_dimension;
    result.output_dimension =
        integer(plan_node["descriptor_count"], path + ".plan.descriptor_count", 1);
    const int record_output_L = integer(record["output_L"], path + ".output_L", 0);
    result.output_L = record_output_L;
    const YAML::Node target = plan_node["target"];
    const YAML::Node target_rotation = target["rotation"];
    require_mapping(target, path + ".plan.target");
    require_mapping(target_rotation, path + ".plan.target.rotation");
    if (scalar_string(target["permutation"], path + ".plan.target.permutation") !=
            "young:" + std::to_string(block_power) ||
        scalar_string(target_rotation["group"], path + ".plan.target.rotation.group") != "O3" ||
        integer(target_rotation["L_R"], path + ".plan.target.rotation.L_R", 0) != record_output_L ||
        result.output_dimension % (2 * record_output_L + 1) != 0)
      fail(path, "block target carrier does not match its selected power and output L");
    const YAML::Node magnetic_values = target_rotation["M_R_values"];
    require_sequence(magnetic_values, path + ".plan.target.rotation.M_R_values");
    if (magnetic_values.size() != static_cast<std::size_t>(2 * record_output_L + 1))
      fail(path, "block target magnetic range has the wrong length");
    for (int component = 0; component <= 2 * record_output_L; ++component)
      if (integer(magnetic_values[static_cast<std::size_t>(component)],
                  path + ".plan.target.rotation.M_R_values",
                  -record_output_L) != component - record_output_L)
        fail(path, "block target magnetic range is not ordered from -L to +L");
    for (int magnetic = -angular; magnetic <= angular; ++magnetic)
      result.input_channels.push_back(
          full_channel_index(species[static_cast<std::size_t>(central_species)], neighbor_species,
                             radial, angular, magnetic, path));

    std::vector<std::map<std::vector<std::int64_t>, std::complex<double>>> outputs(
        static_cast<std::size_t>(result.output_dimension));
    std::vector<bool> seen(static_cast<std::size_t>(result.output_dimension), false);
    const YAML::Node entries = plan_node["entries"];
    require_sequence(entries, path + ".plan.entries");
    for (std::size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
      const YAML::Node entry = entries[entry_index];
      const std::string entry_path = path + ".plan.entries[" + std::to_string(entry_index) + "]";
      const int descriptor =
          integer(entry["descriptor_index"], entry_path + ".descriptor_index", 0);
      if (descriptor >= result.output_dimension || seen[static_cast<std::size_t>(descriptor)])
        fail(entry_path, "duplicate or out-of-range block descriptor index");
      seen[static_cast<std::size_t>(descriptor)] = true;
      const int power = integer(entry["power"], entry_path + ".power", 1);
      if (power != block_power)
        fail(entry_path, "block entry degree does not match its declared block");
      result.maximum_power = std::max<std::int64_t>(result.maximum_power, power);
      if (integer(entry["input_L"], entry_path + ".input_L", 0) != angular)
        fail(entry_path, "block entry input_L mismatch");
      const int output_L = integer(entry["output_L"], entry_path + ".output_L", 0);
      if (output_L != record_output_L)
        fail(entry_path, "block entry output_L does not match its selected plan");
      const int component = integer(entry["component_index"], entry_path + ".component_index", 0);
      const int multiplicity =
          integer(entry["multiplicity_index"], entry_path + ".multiplicity_index", 0);
      if (component > 2 * output_L || descriptor != multiplicity * (2 * output_L + 1) + component)
        fail(entry_path, "block output layout is not multiplicity-major magnetic order");
      const YAML::Node entry_validation = entry["validation_report"];
      require_mapping(entry_validation, entry_path + ".validation_report");
      if (!boolean(entry_validation["passed"], entry_path + ".validation_report.passed") ||
          scalar_string(entry_validation["basis_convention"],
                        entry_path + ".validation_report.basis_convention") != "complex_magnetic")
        fail(entry_path, "block descriptor validation did not pass");
      const auto channels =
          integer_sequence(entry["channel_indices"], entry_path + ".channel_indices",
                           static_cast<std::size_t>(input_dimension));
      for (int channel = 0; channel < input_dimension; ++channel)
        if (channels[static_cast<std::size_t>(channel)] != channel)
          fail(entry_path, "block plan channel indices are not canonical");
      const YAML::Node terms = entry["component_terms"];
      require_sequence(terms, entry_path + ".component_terms");
      for (std::size_t term_index = 0; term_index < terms.size(); ++term_index) {
        const std::string term_path =
            entry_path + ".component_terms[" + std::to_string(term_index) + "]";
        const YAML::Node term = terms[term_index];
        const auto exponents = integer_sequence(term["exponents"], term_path + ".exponents",
                                                static_cast<std::size_t>(input_dimension));
        if (std::accumulate(exponents.begin(), exponents.end(), 0) != power)
          fail(term_path, "block monomial degree does not match its power");
        int magnetic_weight = 0;
        for (int input_component = 0; input_component < input_dimension; ++input_component)
          magnetic_weight +=
              (input_component - angular) * exponents[static_cast<std::size_t>(input_component)];
        if (magnetic_weight != component - output_L)
          fail(term_path,
               "block monomial magnetic weight does not match its "
               "output component");
        std::vector<std::int64_t> counts(exponents.begin(), exponents.end());
        outputs[static_cast<std::size_t>(descriptor)][counts] +=
            complex_number(term["coefficient"], term_path + ".coefficient");
      }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
      fail(path, "block power plan does not cover every output component");

    if (record_output_L > 0) {
      const int magnetic_width = 2 * record_output_L + 1;
      bool conjugate_symmetric = true;
      const double epsilon = 128.0 * std::numeric_limits<double>::epsilon();
      const int multiplicity_count = result.output_dimension / magnetic_width;
      for (int multiplicity = 0; multiplicity < multiplicity_count; ++multiplicity) {
        const int base = multiplicity * magnetic_width;
        for (int magnetic = 1; magnetic <= record_output_L; ++magnetic) {
          const auto &positive =
              outputs[static_cast<std::size_t>(base + record_output_L + magnetic)];
          const auto &negative =
              outputs[static_cast<std::size_t>(base + record_output_L - magnetic)];
          for (const auto &term : positive) {
            std::vector<std::int64_t> reversed(term.first.rbegin(), term.first.rend());
            const auto match = negative.find(reversed);
            const std::complex<double> value =
                match == negative.end() ? std::complex<double>(0.0, 0.0) : match->second;
            if (std::abs(value - std::conj(term.second)) >
                epsilon * (1.0 + std::max(std::abs(value), std::abs(term.second))))
              conjugate_symmetric = false;
          }
          for (const auto &term : negative) {
            std::vector<std::int64_t> reversed(term.first.rbegin(), term.first.rend());
            const auto match = positive.find(reversed);
            const std::complex<double> value =
                match == positive.end() ? std::complex<double>(0.0, 0.0) : match->second;
            if (std::abs(value - std::conj(term.second)) >
                epsilon * (1.0 + std::max(std::abs(value), std::abs(term.second))))
              conjugate_symmetric = false;
          }
        }
      }
      result.conjugate_half_output = conjugate_symmetric;
    }

    std::map<std::vector<std::int64_t>, std::int64_t> monomial_indices;
    result.output_offsets.push_back(0);
    const int magnetic_width = 2 * record_output_L + 1;
    for (std::size_t output_index = 0; output_index < outputs.size(); ++output_index) {
      const bool retained_component = !result.conjugate_half_output ||
          static_cast<int>(output_index % static_cast<std::size_t>(magnetic_width)) >=
              record_output_L;
      if (retained_component) {
        for (const auto &term : outputs[output_index]) {
          if (term.second == std::complex<double>(0.0, 0.0)) continue;
          const auto inserted = monomial_indices.emplace(
              term.first, static_cast<std::int64_t>(monomial_indices.size()));
          if (inserted.second)
            result.monomial_counts.insert(result.monomial_counts.end(), term.first.begin(),
                                          term.first.end());
          result.coefficient_terms.push_back(inserted.first->second);
          result.coefficient_values.push_back(term.second);
          if (term.second.imag() != 0.0) result.real_coefficients = false;
        }
      }
      result.output_offsets.push_back(static_cast<std::int64_t>(result.coefficient_values.size()));
    }
    result.monomial_count = static_cast<std::int64_t>(monomial_indices.size());
    if (result.monomial_count == 0) fail(path, "block power plan contains no nonzero monomials");
    result.monomial_factor_offsets.push_back(0);
    for (std::int64_t term = 0; term < result.monomial_count; ++term) {
      for (std::int64_t component = 0; component < result.input_dimension; ++component) {
        const std::int64_t exponent = result.monomial_counts[static_cast<std::size_t>(
            term * result.input_dimension + component)];
        if (exponent == 0) continue;
        result.monomial_factor_components.push_back(component);
        result.monomial_factor_exponents.push_back(exponent);
      }
      result.monomial_factor_offsets.push_back(
          static_cast<std::int64_t>(result.monomial_factor_components.size()));
    }
    if (block_power == 1 && result.real_coefficients) {
      result.direct_input_channels.assign(static_cast<std::size_t>(result.output_dimension), -1);
      result.direct_input_scales.assign(static_cast<std::size_t>(result.output_dimension), 0.0);
      bool direct_input = true;
      for (std::int64_t component = 0; component < result.output_dimension; ++component) {
        const std::int64_t magnetic_component = component % (2 * result.output_L + 1);
        if (result.conjugate_half_output && magnetic_component < result.output_L) continue;
        const std::int64_t begin = result.output_offsets[static_cast<std::size_t>(component)];
        const std::int64_t end = result.output_offsets[static_cast<std::size_t>(component + 1)];
        if (end - begin != 1) {
          direct_input = false;
          break;
        }
        const std::int64_t term = result.coefficient_terms[static_cast<std::size_t>(begin)];
        const std::int64_t factor_begin =
            result.monomial_factor_offsets[static_cast<std::size_t>(term)];
        const std::int64_t factor_end =
            result.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
        if (factor_end - factor_begin != 1 ||
            result.monomial_factor_exponents[static_cast<std::size_t>(factor_begin)] != 1) {
          direct_input = false;
          break;
        }
        const std::int64_t input_component =
            result.monomial_factor_components[static_cast<std::size_t>(factor_begin)];
        result.direct_input_channels[static_cast<std::size_t>(component)] =
            result.input_channels[static_cast<std::size_t>(input_component)];
        result.direct_input_scales[static_cast<std::size_t>(component)] =
            result.coefficient_values[static_cast<std::size_t>(begin)].real();
      }
      result.direct_input_plan = direct_input;
      if (!direct_input) {
        result.direct_input_channels.clear();
        result.direct_input_scales.clear();
      }
    }
    return result;
  }

  std::int64_t append_block_plan(YACEBlockProgram &program, YACEBlockPowerPlan candidate)
  {
    for (std::size_t index = 0; index < program.power_plans.size(); ++index)
      if (same_block_plan(program.power_plans[index], candidate))
        return static_cast<std::int64_t>(index);
    program.power_plans.push_back(std::move(candidate));
    return static_cast<std::int64_t>(program.power_plans.size() - 1);
  }

  void build_shared_block_power_cache(YACEBlockProgram &program)
  {
    std::map<std::int64_t, std::int64_t> maximum_powers;
    for (const auto &plan : program.power_plans)
      if (!plan.direct_input_plan)
        for (std::int64_t component = 0; component < plan.input_dimension; ++component) {
          std::int64_t maximum = 0;
          for (std::int64_t term = 0; term < plan.monomial_count; ++term)
            maximum = std::max(maximum,
                               plan.monomial_counts[static_cast<std::size_t>(
                                   term * plan.input_dimension + component)]);
          const std::int64_t channel = plan.input_channels[static_cast<std::size_t>(component)];
          maximum_powers[channel] = std::max(maximum_powers[channel], maximum);
        }

    std::map<std::int64_t, std::int64_t> offsets;
    for (const auto &[channel, maximum] : maximum_powers) {
      offsets[channel] = program.power_storage_size;
      program.power_channels.push_back(channel);
      program.power_maximum_exponents.push_back(maximum);
      program.power_offsets.push_back(program.power_storage_size);
      program.power_storage_size += maximum + 1;
    }
    for (auto &plan : program.power_plans) {
      plan.input_power_offsets.clear();
      if (plan.direct_input_plan) continue;
      plan.input_power_offsets.reserve(plan.input_channels.size());
      for (const std::int64_t channel : plan.input_channels)
        plan.input_power_offsets.push_back(offsets.at(channel));
    }
  }

  bool same_scalar_base(const YACEScalarInvariantBase &left, const YACEScalarInvariantBase &right)
  {
    return left.base_id == right.base_id && left.left_channels == right.left_channels &&
        left.right_channels == right.right_channels && left.coefficients == right.coefficients;
  }

  void build_scalar_power_program(YACEBlockProgram &program,
                                  const std::vector<FunctionCandidatePortfolio> &portfolios,
                                  const std::vector<std::size_t> &choices)
  {
    auto &output = program.scalar_program;
    std::map<std::string, std::int64_t> base_indices;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &candidate = portfolios[index].candidates.at(choices.at(index));
      if (candidate.evaluator != YACEEvaluatorKind::SCALAR_INVARIANT_POWER) continue;
      const auto &base = candidate.scalar_route.base;
      const auto found = base_indices.find(base.base_id);
      if (found != base_indices.end()) {
        if (!same_scalar_base(output.bases[static_cast<std::size_t>(found->second)], base))
          fail("sidecar", "one scalar base ID names conflicting payloads");
        continue;
      }
      const std::int64_t base_index = static_cast<std::int64_t>(output.bases.size());
      base_indices.emplace(base.base_id, base_index);
      output.bases.push_back(base);
    }

    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> values;
    for (std::size_t index = 0; index < output.bases.size(); ++index)
      values.emplace(std::make_pair(static_cast<std::int64_t>(index), 1),
                     static_cast<std::int64_t>(index));
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &selected = portfolios[index].candidates.at(choices.at(index));
      if (selected.evaluator != YACEEvaluatorKind::SCALAR_INVARIANT_POWER) continue;
      const auto &candidate = selected.scalar_route;
      const std::int64_t base_index = base_indices.at(candidate.base.base_id);
      for (const auto &step : candidate.schedule) {
        const auto key = std::make_pair(base_index, step.output_exponent);
        if (values.count(key) != 0) continue;
        const auto left = values.find(std::make_pair(base_index, step.left_exponent));
        const auto right = values.find(std::make_pair(base_index, step.right_exponent));
        if (left == values.end() || right == values.end())
          fail("sidecar", "scalar power schedule is not topological");
        YACEScalarPowerNode node;
        node.left_value = left->second;
        node.right_value = right->second;
        node.exponent = step.output_exponent;
        node.base_index = base_index;
        const std::int64_t value_index =
            static_cast<std::int64_t>(output.bases.size() + output.nodes.size());
        output.nodes.push_back(node);
        values.emplace(key, value_index);
      }
      const auto route_value = values.find(std::make_pair(base_index, candidate.outer_power));
      if (route_value == values.end())
        fail("sidecar", "scalar route output is absent from its schedule");
      output.routes.push_back({candidate.function_index, route_value->second, candidate.scale,
                               candidate.factorization_id});
    }
    output.value_count = static_cast<std::int64_t>(output.bases.size() + output.nodes.size());
  }

  YACESparsePolynomial without_descriptors(const YACESparsePolynomial &source,
                                           const std::set<int> &removed)
  {
    std::vector<double> coefficients(source.monomial_coefficients.size(), 0.0);
    const int function_count = static_cast<int>(source.descriptor_offsets.size()) - 1;
    for (int function = 0; function < function_count; ++function) {
      if (removed.count(function) != 0) continue;
      const std::int64_t begin = source.descriptor_offsets[static_cast<std::size_t>(function)];
      const std::int64_t end = source.descriptor_offsets[static_cast<std::size_t>(function + 1)];
      for (std::int64_t row = begin; row < end; ++row) {
        const std::int64_t term = source.descriptor_terms[static_cast<std::size_t>(row)];
        coefficients[static_cast<std::size_t>(term)] +=
            source.descriptor_coefficients[static_cast<std::size_t>(row)];
      }
    }

    YACESparsePolynomial result;
    result.maximum_rank = source.maximum_rank;
    result.factor_offsets.push_back(0);
    for (std::size_t old_term = 0; old_term < coefficients.size(); ++old_term) {
      if (coefficients[old_term] == 0.0) continue;
      const std::int64_t begin = source.factor_offsets[old_term];
      const std::int64_t end = source.factor_offsets[old_term + 1];
      result.maximum_term_factors = std::max(result.maximum_term_factors, end - begin);
      for (std::int64_t factor = begin; factor < end; ++factor) {
        result.factor_indices.push_back(source.factor_indices[static_cast<std::size_t>(factor)]);
        result.factor_exponents.push_back(
            source.factor_exponents[static_cast<std::size_t>(factor)]);
      }
      result.factor_offsets.push_back(static_cast<std::int64_t>(result.factor_indices.size()));
      result.monomial_coefficients.push_back(coefficients[old_term]);
    }
    if (!result.monomial_coefficients.empty()) build_monomial_dag(result);
    return result;
  }

  std::int64_t add_operations(std::int64_t left, std::int64_t right)
  {
    if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right)
      return std::numeric_limits<std::int64_t>::max();
    return left + right;
  }

  std::set<int> removed_function_set(const std::vector<FunctionCandidatePortfolio> &portfolios,
                                     const std::vector<std::size_t> &choices)
  {
    std::set<int> removed;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &candidate = portfolios[index].candidates.at(choices.at(index));
      if (candidate.evaluator != YACEEvaluatorKind::EXPLICIT_CTILDE)
        removed.insert(portfolios[index].function_index);
    }
    return removed;
  }

  // Per-function share of the complete direct plan's operation estimate: the
  // power leaves, product nodes, and readout terms consumed by exactly one
  // function, attributed by one owner pass over the loader's DAG. A subtree
  // shared by several functions is freed only when all of them leave the
  // direct residual, so the marginal sum is a lower bound on the saving. The
  // coordinate-descent planner uses these marginals only to rank neighbouring
  // selections; every accepted move and the reported estimate are still scored
  // exactly against a recompiled residual.
  struct DirectOperationMarginals {
    std::int64_t full = 0;
    std::vector<std::int64_t> per_function;
  };

  DirectOperationMarginals exclusive_direct_operations(const YACESparsePolynomial &source,
                                                       int function_count)
  {
    DirectOperationMarginals result;
    result.full = direct_plan_operations(source);
    result.per_function.assign(static_cast<std::size_t>(std::max(function_count, 0)), 0);
    constexpr int unassigned = -2;
    constexpr int shared = -1;
    auto merge = [](int &owner, int function) {
      if (owner == unassigned)
        owner = function;
      else if (owner != function)
        owner = shared;
    };
    const std::size_t term_count = source.monomial_coefficients.size();
    std::vector<int> term_owner(term_count, unassigned);
    for (int function = 0; function < function_count; ++function) {
      const std::int64_t begin = source.descriptor_offsets.at(static_cast<std::size_t>(function));
      const std::int64_t end = source.descriptor_offsets.at(static_cast<std::size_t>(function + 1));
      for (std::int64_t row = begin; row < end; ++row)
        merge(term_owner.at(static_cast<std::size_t>(
                  source.descriptor_terms.at(static_cast<std::size_t>(row)))),
              function);
    }
    auto term_consumer = [&](std::size_t term) {
      return term_owner[term] == unassigned ? shared : term_owner[term];
    };
    const std::size_t power_count = source.power_channels.size();
    std::vector<int> power_owner(power_count, unassigned);
    if (source.binary_dag) {
      const std::size_t node_count = source.binary_node_left.size();
      std::vector<int> value_owner(power_count + node_count, unassigned);
      for (std::size_t term = 0; term < term_count; ++term) {
        const std::int64_t operand = source.monomial_nodes.at(term);
        if (operand >= 0)
          merge(value_owner.at(static_cast<std::size_t>(operand)), term_consumer(term));
      }
      for (std::size_t node = node_count; node-- > 0;) {
        const int owner = value_owner[power_count + node];
        if (owner == unassigned) continue;
        merge(value_owner.at(static_cast<std::size_t>(source.binary_node_left[node])), owner);
        merge(value_owner.at(static_cast<std::size_t>(source.binary_node_right[node])), owner);
        if (owner >= 0) result.per_function[static_cast<std::size_t>(owner)] += 3;
      }
      for (std::size_t power = 0; power < power_count; ++power)
        power_owner[power] = value_owner[power];
    } else {
      const std::size_t node_count = source.dag_node_parents.size();
      std::vector<int> node_owner(node_count, unassigned);
      for (std::size_t term = 0; term < term_count; ++term) {
        const std::int64_t operand = source.monomial_nodes.at(term);
        if (operand > 0)
          merge(node_owner.at(static_cast<std::size_t>(operand)), term_consumer(term));
      }
      for (std::size_t node = node_count; node-- > 1;) {
        const int owner = node_owner[node];
        if (owner == unassigned) continue;
        merge(power_owner.at(static_cast<std::size_t>(source.dag_node_powers[node])), owner);
        const std::int64_t parent = source.dag_node_parents[node];
        if (parent > 0) {
          merge(node_owner.at(static_cast<std::size_t>(parent)), owner);
          if (owner >= 0) result.per_function[static_cast<std::size_t>(owner)] += 3;
        }
      }
    }
    for (std::size_t power = 0; power < power_count; ++power)
      if (power_owner[power] >= 0)
        result.per_function[static_cast<std::size_t>(power_owner[power])] +=
            2 * std::max<std::int64_t>(source.power_exponents[power] - 1, 0);
    for (std::size_t term = 0; term < term_count; ++term)
      if (term_owner[term] >= 0)
        result.per_function[static_cast<std::size_t>(term_owner[term])] += 2;
    return result;
  }

  // Operation estimate of every selected non-direct route (the direct residual
  // is accounted separately).
  std::int64_t
  selection_non_direct_operations(const std::vector<FunctionCandidatePortfolio> &portfolios,
                                  const std::vector<std::size_t> &choices)
  {
    std::int64_t operations = 0;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &candidate = portfolios[index].candidates.at(choices.at(index));
      if (candidate.evaluator == YACEEvaluatorKind::COUPLED_PRODUCT_DAG)
        operations = add_operations(operations, candidate.operation_estimate);
    }

    std::vector<const YACEBlockPowerPlan *> unique_plans;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &candidate = portfolios[index].candidates.at(choices.at(index));
      if (candidate.evaluator != YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER) continue;
      for (const auto &term : candidate.block_terms)
        for (const auto &plan : term.power_plans) {
          const bool duplicate = std::any_of(unique_plans.begin(), unique_plans.end(),
                                             [&plan](const YACEBlockPowerPlan *existing) {
                                               return same_block_plan(*existing, plan);
                                             });
          if (!duplicate) unique_plans.push_back(&plan);
        }
    }

    std::map<std::int64_t, std::int64_t> maximum_powers;
    for (const auto *plan : unique_plans) {
      operations = add_operations(operations, block_plan_contraction_operations(*plan));
      if (plan->direct_input_plan) continue;
      for (std::int64_t component = 0; component < plan->input_dimension; ++component) {
        std::int64_t maximum = 0;
        for (std::int64_t term = 0; term < plan->monomial_count; ++term)
          maximum = std::max(maximum,
                             plan->monomial_counts[static_cast<std::size_t>(
                                 term * plan->input_dimension + component)]);
        const std::int64_t channel = plan->input_channels[static_cast<std::size_t>(component)];
        maximum_powers[channel] = std::max(maximum_powers[channel], maximum);
      }
    }
    for (const auto &[channel, maximum] : maximum_powers) {
      (void) channel;
      operations = add_operations(operations, std::max<std::int64_t>(maximum - 1, 0));
    }
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &candidate = portfolios[index].candidates.at(choices.at(index));
      if (candidate.evaluator != YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER) continue;
      for (const auto &term : candidate.block_terms) {
        const auto &route = term.route;
        if (!route.term_factor_offsets.empty())
          operations = add_operations(
              operations, 6 * static_cast<std::int64_t>(route.term_factor_plans.size()));
        else
          operations = add_operations(operations,
                                      (route.right_plan < 0 ? 2 : 6) *
                                          static_cast<std::int64_t>(route.coefficients.size()));
      }
    }

    std::map<std::string, const YACEScalarInvariantBase *> scalar_bases;
    std::set<std::pair<std::string, std::int64_t>> scalar_nodes;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const auto &selected = portfolios[index].candidates.at(choices.at(index));
      if (selected.evaluator != YACEEvaluatorKind::SCALAR_INVARIANT_POWER) continue;
      const auto &scalar = selected.scalar_route;
      const auto inserted = scalar_bases.emplace(scalar.base.base_id, &scalar.base);
      if (!inserted.second) {
        const auto *existing = inserted.first->second;
        if (existing->left_channels != scalar.base.left_channels ||
            existing->right_channels != scalar.base.right_channels ||
            existing->coefficients != scalar.base.coefficients)
          fail("sidecar", "one scalar base ID names conflicting payloads");
      }
      for (const auto &step : scalar.schedule)
        scalar_nodes.emplace(scalar.base.base_id, step.output_exponent);
      operations = add_operations(operations, 2);
    }
    for (const auto &[base_id, base] : scalar_bases) {
      (void) base_id;
      operations =
          add_operations(operations, 5 * static_cast<std::int64_t>(base->coefficients.size()));
    }
    operations = add_operations(operations, 3 * static_cast<std::int64_t>(scalar_nodes.size()));
    return operations;
  }

  std::int64_t
  selection_operation_estimate(const YACESparsePolynomial &source,
                               const std::vector<FunctionCandidatePortfolio> &portfolios,
                               const std::vector<std::size_t> &choices)
  {
    const std::set<int> removed = removed_function_set(portfolios, choices);
    const YACESparsePolynomial residual = removed.empty() ? source
                                                          : without_descriptors(source, removed);
    return add_operations(direct_plan_operations(residual),
                          selection_non_direct_operations(portfolios, choices));
  }

  std::size_t selected_count(const std::vector<FunctionCandidatePortfolio> &portfolios,
                             const std::vector<std::size_t> &choices)
  {
    std::size_t count = 0;
    for (std::size_t index = 0; index < portfolios.size(); ++index)
      if (portfolios[index].candidates.at(choices.at(index)).evaluator !=
          YACEEvaluatorKind::EXPLICIT_CTILDE)
        ++count;
    return count;
  }

  bool better_selection(const std::vector<FunctionCandidatePortfolio> &portfolios,
                        const BlockSelection &candidate, const BlockSelection &reference)
  {
    if (candidate.operation_estimate != reference.operation_estimate)
      return candidate.operation_estimate < reference.operation_estimate;
    const std::size_t candidate_count = selected_count(portfolios, candidate.choices);
    const std::size_t reference_count = selected_count(portfolios, reference.choices);
    if (candidate_count != reference_count) return candidate_count < reference_count;
    for (std::size_t index = 0; index < portfolios.size(); ++index) {
      const std::string &candidate_id =
          portfolios[index].candidates.at(candidate.choices.at(index)).candidate_id;
      const std::string &reference_id =
          portfolios[index].candidates.at(reference.choices.at(index)).candidate_id;
      if (candidate_id != reference_id) return candidate_id < reference_id;
    }
    return false;
  }

  BlockSelection score_selection(const YACESparsePolynomial &source,
                                 const std::vector<FunctionCandidatePortfolio> &portfolios,
                                 std::vector<std::size_t> choices)
  {
    BlockSelection result;
    result.operation_estimate = selection_operation_estimate(source, portfolios, choices);
    result.choices = std::move(choices);
    return result;
  }

  std::size_t direct_candidate_index(const FunctionCandidatePortfolio &portfolio)
  {
    for (std::size_t index = 0; index < portfolio.candidates.size(); ++index)
      if (portfolio.candidates[index].evaluator == YACEEvaluatorKind::EXPLICIT_CTILDE) return index;
    fail("sidecar", "candidate portfolio has no direct evaluator");
  }

  std::vector<std::size_t> available_choices(const FunctionCandidatePortfolio &portfolio,
                                             YACEBlockPolicy policy)
  {
    std::vector<std::size_t> result;
    const std::size_t direct = direct_candidate_index(portfolio);
    if (policy == YACEBlockPolicy::AUTO || policy == YACEBlockPolicy::GPU_AUTO)
      for (std::size_t index = 0; index < portfolio.candidates.size(); ++index)
        result.push_back(index);
    else {
      YACEEvaluatorKind required = YACEEvaluatorKind::SCALAR_INVARIANT_POWER;
      if (policy == YACEBlockPolicy::BLOCK)
        required = YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER;
      else if (policy == YACEBlockPolicy::COUPLED_PRODUCT)
        required = YACEEvaluatorKind::COUPLED_PRODUCT_DAG;
      for (std::size_t index = 0; index < portfolio.candidates.size(); ++index)
        if (portfolio.candidates[index].evaluator == required) result.push_back(index);
      if (result.empty()) {
        if (portfolio.candidates.size() == 1) {
          result.push_back(direct);
        } else if (policy == YACEBlockPolicy::BLOCK) {
          fail("sidecar",
               "forced block policy requires one certified block "
               "alternative for every mapped function");
        } else if (policy == YACEBlockPolicy::COUPLED_PRODUCT) {
          fail("sidecar",
               "forced coupled-product policy requires one certified DAG "
               "alternative for every mapped function");
        } else {
          fail("sidecar",
               "forced scalar-power policy requires one certified "
               "scalar alternative for every mapped function");
        }
      }
    }
    return result;
  }

  BlockSelection choose_block_selection(const YACESparsePolynomial &source,
                                        const std::vector<FunctionCandidatePortfolio> &portfolios,
                                        YACEBlockPolicy policy)
  {
    if (policy != YACEBlockPolicy::AUTO && policy != YACEBlockPolicy::GPU_AUTO &&
        policy != YACEBlockPolicy::BLOCK && policy != YACEBlockPolicy::SCALAR_POWER &&
        policy != YACEBlockPolicy::COUPLED_PRODUCT)
      fail("sidecar", "block selection received an unsupported policy");

    constexpr std::size_t candidates_per_function_limit = 64;
    constexpr std::size_t total_candidate_limit = 16384;
    constexpr std::size_t planner_temporary_byte_limit = 64 * 1024 * 1024;
    constexpr std::uint64_t exhaustive_combination_limit = 4096;
    constexpr std::int64_t score_evaluation_limit = 200000;
    const std::size_t count = portfolios.size();
    const bool automatic = policy == YACEBlockPolicy::AUTO || policy == YACEBlockPolicy::GPU_AUTO;
    auto direct_selection = [&]() {
      std::vector<std::size_t> choices;
      choices.reserve(count);
      for (const auto &portfolio : portfolios) choices.push_back(direct_candidate_index(portfolio));
      BlockSelection result = score_selection(source, portfolios, std::move(choices));
      if (policy == YACEBlockPolicy::GPU_AUTO) {
        result.algorithm = "conservative_direct_fallback_v1";
        result.status = "selected_direct_no_authorized_profile";
        result.score_evaluations = 1;
        result.optimal = false;
      } else {
        result.algorithm = "budget_direct_fallback_v1";
        result.status = "budget_direct_fallback";
        result.optimal = false;
      }
      return result;
    };

    std::size_t total_candidates = 0;
    for (const auto &portfolio : portfolios) {
      if (portfolio.candidates.empty()) fail("sidecar", "candidate portfolio is empty");
      if (portfolio.candidates.size() > candidates_per_function_limit) {
        if (automatic) return direct_selection();
        fail("sidecar", "candidate count per function exceeds the planner limit");
      }
      if (total_candidates > total_candidate_limit - portfolio.candidates.size()) {
        if (automatic) return direct_selection();
        fail("sidecar", "total candidate count exceeds the planner limit");
      }
      total_candidates += portfolio.candidates.size();
    }
    const std::size_t temporary_bytes =
        total_candidates * sizeof(std::size_t) + count * sizeof(std::vector<std::size_t>);
    if (temporary_bytes > planner_temporary_byte_limit) {
      if (automatic) return direct_selection();
      fail("sidecar", "candidate planner temporary storage exceeds its limit");
    }

    std::vector<std::vector<std::size_t>> options;
    if (policy == YACEBlockPolicy::GPU_AUTO) return direct_selection();
    options.reserve(count);
    for (const auto &portfolio : portfolios)
      options.push_back(available_choices(portfolio, policy));

    std::int64_t score_evaluations = 0;
    auto score = [&](std::vector<std::size_t> choices, BlockSelection &result) {
      if (score_evaluations >= score_evaluation_limit) return false;
      result = score_selection(source, portfolios, std::move(choices));
      ++score_evaluations;
      return true;
    };

    std::uint64_t combination_count = 1;
    for (const auto &row_options : options) {
      const std::uint64_t option_count = row_options.size();
      if (combination_count > exhaustive_combination_limit / option_count) {
        combination_count = exhaustive_combination_limit + 1;
        break;
      }
      combination_count *= option_count;
    }
    if (combination_count <= exhaustive_combination_limit) {
      BlockSelection best;
      bool have_best = false;
      for (std::uint64_t encoded = 0; encoded < combination_count; ++encoded) {
        std::uint64_t remaining = encoded;
        std::vector<std::size_t> choices(count, 0);
        for (std::size_t index = 0; index < count; ++index) {
          choices[index] = options[index][remaining % options[index].size()];
          remaining /= options[index].size();
        }
        BlockSelection candidate;
        if (!score(std::move(choices), candidate)) {
          if (policy == YACEBlockPolicy::AUTO) {
            BlockSelection fallback = direct_selection();
            fallback.score_evaluations = score_evaluations;
            return fallback;
          }
          fail("sidecar", "forced evaluator selection exceeded its score budget");
        }
        if (!have_best || better_selection(portfolios, candidate, best)) {
          best = std::move(candidate);
          have_best = true;
        }
      }
      best.algorithm = policy == YACEBlockPolicy::AUTO ? "exhaustive_candidate_vector_catalogue_v2"
                                                       : "forced_candidate_vector_catalogue_v2";
      best.status = "selected";
      best.score_evaluations = score_evaluations;
      best.optimal = true;
      return best;
    }

    std::set<std::vector<std::size_t>> unique_seeds;
    std::vector<std::size_t> first(count, 0);
    std::vector<std::size_t> row_local(count, 0);
    for (std::size_t index = 0; index < count; ++index) {
      first[index] = options[index].front();
      row_local[index] = *std::min_element(
          options[index].begin(), options[index].end(), [&](std::size_t left, std::size_t right) {
            const auto &left_candidate = portfolios[index].candidates[left];
            const auto &right_candidate = portfolios[index].candidates[right];
            if (left_candidate.operation_estimate != right_candidate.operation_estimate)
              return left_candidate.operation_estimate < right_candidate.operation_estimate;
            return left_candidate.candidate_id < right_candidate.candidate_id;
          });
    }
    unique_seeds.insert(std::move(first));
    unique_seeds.insert(std::move(row_local));
    if (policy == YACEBlockPolicy::AUTO) {
      std::vector<std::size_t> all_direct;
      all_direct.reserve(count);
      for (const auto &portfolio : portfolios)
        all_direct.push_back(direct_candidate_index(portfolio));
      unique_seeds.insert(std::move(all_direct));
    }

    std::vector<BlockSelection> seeds;
    seeds.reserve(unique_seeds.size());
    for (const auto &seed : unique_seeds) {
      BlockSelection scored;
      if (!score(seed, scored)) {
        if (policy == YACEBlockPolicy::AUTO) {
          BlockSelection fallback = direct_selection();
          fallback.score_evaluations = score_evaluations;
          return fallback;
        }
        fail("sidecar", "forced evaluator selection exceeded its score budget");
      }
      seeds.push_back(std::move(scored));
    }
    std::sort(seeds.begin(), seeds.end(),
              [&](const BlockSelection &left, const BlockSelection &right) {
                return better_selection(portfolios, left, right);
              });

    BlockSelection best = seeds.front();
    // Neighbours of one selection differ in a single function, so their direct
    // residual cost is ranked from precomputed per-function marginals instead
    // of recompiling the whole residual DAG for every neighbour. The best
    // ranked neighbour of each pass is rescored exactly before acceptance, so
    // the accepted trajectory and the reported estimate remain exact.
    const int function_count = static_cast<int>(source.descriptor_offsets.size()) - 1;
    bool incremental = function_count > 0;
    for (const auto &portfolio : portfolios)
      incremental =
          incremental && portfolio.function_index >= 0 && portfolio.function_index < function_count;
    const DirectOperationMarginals marginals = incremental
        ? exclusive_direct_operations(source, function_count)
        : DirectOperationMarginals{};
    auto estimate = [&](std::vector<std::size_t> choices) {
      BlockSelection result;
      std::int64_t direct = marginals.full;
      for (std::size_t index = 0; index < count; ++index) {
        const auto &candidate = portfolios[index].candidates.at(choices.at(index));
        if (candidate.evaluator != YACEEvaluatorKind::EXPLICIT_CTILDE)
          direct -=
              marginals.per_function[static_cast<std::size_t>(portfolios[index].function_index)];
      }
      result.operation_estimate = add_operations(
          std::max<std::int64_t>(direct, 0), selection_non_direct_operations(portfolios, choices));
      result.choices = std::move(choices);
      return result;
    };
    auto exact_or_fallback = [&](BlockSelection &selection) {
      BlockSelection exact;
      if (score(selection.choices, exact)) {
        selection = std::move(exact);
        return true;
      }
      return false;
    };
    const std::size_t start_count = std::min<std::size_t>(seeds.size(), 4);
    const std::size_t pass_limit = std::min<std::size_t>(count, 16);
    for (std::size_t start = 0; start < start_count; ++start) {
      BlockSelection current = seeds[start];
      for (std::size_t pass = 0; pass < pass_limit; ++pass) {
        BlockSelection neighbor = current;
        bool have_neighbor = false;
        for (std::size_t index = 0; index < count; ++index) {
          for (const std::size_t choice : options[index]) {
            if (choice == current.choices[index]) continue;
            std::vector<std::size_t> changed = current.choices;
            changed[index] = choice;
            BlockSelection candidate;
            if (incremental) {
              candidate = estimate(std::move(changed));
            } else if (!score(std::move(changed), candidate)) {
              if (policy == YACEBlockPolicy::AUTO) {
                BlockSelection fallback = direct_selection();
                fallback.score_evaluations = score_evaluations;
                return fallback;
              }
              fail("sidecar", "forced evaluator selection exceeded its score budget");
            }
            if (!have_neighbor || better_selection(portfolios, candidate, neighbor)) {
              neighbor = std::move(candidate);
              have_neighbor = true;
            }
          }
        }
        if (!have_neighbor) break;
        if (incremental && !exact_or_fallback(neighbor)) {
          if (policy == YACEBlockPolicy::AUTO) {
            BlockSelection fallback = direct_selection();
            fallback.score_evaluations = score_evaluations;
            return fallback;
          }
          fail("sidecar", "forced evaluator selection exceeded its score budget");
        }
        if (!better_selection(portfolios, neighbor, current)) break;
        current = std::move(neighbor);
      }
      if (better_selection(portfolios, current, best)) best = std::move(current);
    }
    best.algorithm = incremental ? "candidate_vector_coordinate_descent_v4_marginal"
                                 : "candidate_vector_coordinate_descent_v3";
    best.status = "selected";
    best.score_evaluations = score_evaluations;
    best.optimal = false;
    return best;
  }

  BlockSelection choose_replay_selection(const YACESparsePolynomial &source,
                                         const std::vector<FunctionCandidatePortfolio> &portfolios,
                                         const std::vector<std::string> &candidate_ids)
  {
    if (candidate_ids.size() != portfolios.size())
      fail("auto_replay.selections",
           "candidate vector does not cover the complete function catalogue");
    std::vector<std::size_t> choices;
    choices.reserve(portfolios.size());
    for (std::size_t function = 0; function < portfolios.size(); ++function) {
      const auto &portfolio = portfolios[function];
      const auto match = std::find_if(portfolio.candidates.begin(), portfolio.candidates.end(),
                                      [&](const CompiledEvaluatorCandidate &candidate) {
                                        return candidate.candidate_id == candidate_ids[function];
                                      });
      if (match == portfolio.candidates.end())
        fail("auto_replay.selections", "candidate ID is absent from its compiler-issued portfolio");
      choices.push_back(
          static_cast<std::size_t>(std::distance(portfolio.candidates.begin(), match)));
    }
    BlockSelection result = score_selection(source, portfolios, std::move(choices));
    result.algorithm = "device_bound_candidate_replay_v1";
    result.status = "selected";
    result.score_evaluations = 0;
    result.optimal = false;
    return result;
  }

  void append_identity_field(std::string &payload, const std::string &value)
  {
    payload += std::to_string(value.size());
    payload.push_back(':');
    payload += value;
  }

  void append_identity_integer(std::string &payload, std::int64_t value)
  {
    append_identity_field(payload, std::to_string(value));
  }

  void append_identity_double(std::string &payload, double value)
  {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    append_identity_field(payload, std::to_string(bits));
  }

  void append_identity_complex(std::string &payload, const std::complex<double> &value)
  {
    append_identity_double(payload, value.real());
    append_identity_double(payload, value.imag());
  }

  template <class Value>
  void append_identity_integers(std::string &payload, const std::vector<Value> &values)
  {
    append_identity_integer(payload, static_cast<std::int64_t>(values.size()));
    for (const Value value : values)
      append_identity_integer(payload, static_cast<std::int64_t>(value));
  }

  void append_identity_doubles(std::string &payload, const std::vector<double> &values)
  {
    append_identity_integer(payload, static_cast<std::int64_t>(values.size()));
    for (const double value : values) append_identity_double(payload, value);
  }

  void append_identity_complexes(std::string &payload,
                                 const std::vector<std::complex<double>> &values)
  {
    append_identity_integer(payload, static_cast<std::int64_t>(values.size()));
    for (const auto &value : values) append_identity_complex(payload, value);
  }

  const char *evaluator_name(YACEEvaluatorKind evaluator)
  {
    switch (evaluator) {
      case YACEEvaluatorKind::EXPLICIT_CTILDE:
        return "explicit_ctilde";
      case YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER:
        return "block_symmetric_power";
      case YACEEvaluatorKind::SCALAR_INVARIANT_POWER:
        return "scalar_invariant_power";
      case YACEEvaluatorKind::COUPLED_PRODUCT_DAG:
        return "coupled_product_dag";
    }
    return "unknown";
  }

  std::string evaluator_plan_hash(const std::string &source_hash, const YACEBlockProgram &program,
                                  const YACESparsePolynomial &residual)
  {
    std::string payload;
    if (program.planner_calibration_hash.empty()) {
      append_identity_field(payload, "ye3t_cpu_evaluator_candidate_vector_v2");
      append_identity_field(payload, "cpu_float64_complex_soa_tile8_abi_v1");
    } else {
      append_identity_field(payload, "ye3t_kokkos_gpu_evaluator_candidate_vector_v1");
      append_identity_field(payload, "kokkos_device_float64_abi_v1");
    }
    append_identity_field(payload, source_hash);
    append_identity_field(payload, program.plan_hash);
    append_identity_field(payload, program.planner_profile);
    append_identity_field(payload, program.planner_algorithm);
    append_identity_field(payload, program.planner_status);
    if (!program.planner_calibration_hash.empty()) {
      append_identity_field(payload, program.planner_calibration_hash);
      append_identity_field(payload, program.planner_decision_reason);
    }
    append_identity_field(payload, program.dispatch);
    append_identity_integer(payload, program.candidate_count);
    append_identity_field(payload, std::to_string(program.candidate_route_count));
    append_identity_integer(payload, program.candidate_function_count);
    append_identity_field(payload, std::to_string(program.catalogue_function_count));
    append_identity_integer(payload, program.planner_score_evaluations);
    append_identity_field(payload, std::to_string(program.direct_operation_estimate));
    append_identity_field(payload, std::to_string(program.selected_operation_estimate));
    append_identity_field(payload, program.planner_optimal ? "optimal" : "heuristic");
    for (const auto &decision : program.decisions) {
      append_identity_field(payload, std::to_string(decision.function_index));
      append_identity_integer(payload, static_cast<std::int64_t>(decision.candidates.size()));
      for (const auto &candidate : decision.candidates) {
        append_identity_field(payload, candidate.candidate_id);
        append_identity_field(payload, evaluator_name(candidate.evaluator));
        append_identity_integer(payload, candidate.operation_estimate);
      }
      append_identity_field(payload, decision.selected_candidate_id);
      append_identity_integer(payload, decision.selected_candidate_index);
      append_identity_field(payload, evaluator_name(decision.selected_evaluator));
    }
    append_identity_integers(payload, residual.factor_offsets);
    append_identity_integers(payload, residual.factor_indices);
    append_identity_integers(payload, residual.factor_exponents);
    append_identity_doubles(payload, residual.monomial_coefficients);
    append_identity_integers(payload, residual.descriptor_offsets);
    append_identity_integers(payload, residual.descriptor_terms);
    append_identity_doubles(payload, residual.descriptor_coefficients);
    append_identity_integers(payload, residual.power_channels);
    append_identity_integers(payload, residual.power_exponents);
    append_identity_integers(payload, residual.dag_node_parents);
    append_identity_integers(payload, residual.dag_node_powers);
    append_identity_integer(payload, residual.dag_root_node_count);
    append_identity_integer(payload, residual.binary_dag ? 1 : 0);
    append_identity_integers(payload, residual.binary_node_left);
    append_identity_integers(payload, residual.binary_node_right);
    append_identity_integers(payload, residual.monomial_nodes);
    append_identity_field(payload, residual.dag_factor_ordering);
    append_identity_integer(payload, residual.maximum_rank);
    append_identity_integer(payload, residual.maximum_term_factors);
    for (const auto &plan : program.power_plans) {
      append_identity_integers(payload, plan.input_channels);
      append_identity_integers(payload, plan.input_power_offsets);
      append_identity_integers(payload, plan.monomial_counts);
      append_identity_integers(payload, plan.monomial_factor_offsets);
      append_identity_integers(payload, plan.monomial_factor_components);
      append_identity_integers(payload, plan.monomial_factor_exponents);
      append_identity_integers(payload, plan.output_offsets);
      append_identity_integers(payload, plan.coefficient_terms);
      append_identity_complexes(payload, plan.coefficient_values);
      append_identity_integers(payload, plan.direct_input_channels);
      append_identity_doubles(payload, plan.direct_input_scales);
      append_identity_integer(payload, plan.input_dimension);
      append_identity_integer(payload, plan.output_dimension);
      append_identity_integer(payload, plan.monomial_count);
      append_identity_integer(payload, plan.maximum_power);
      append_identity_integer(payload, plan.output_storage_offset);
      append_identity_integer(payload, plan.output_L);
      append_identity_integer(payload, plan.real_coefficients ? 1 : 0);
      append_identity_integer(payload, plan.conjugate_half_output ? 1 : 0);
      append_identity_integer(payload, plan.direct_input_plan ? 1 : 0);
    }
    for (const auto &route : program.routes) {
      append_identity_integers(payload, route.left_components);
      append_identity_integers(payload, route.right_components);
      append_identity_integers(payload, route.term_factor_offsets);
      append_identity_integers(payload, route.term_factor_plans);
      append_identity_integers(payload, route.term_factor_components);
      append_identity_complexes(payload, route.coefficients);
      append_identity_integer(payload, route.left_plan);
      append_identity_integer(payload, route.right_plan);
      append_identity_integer(payload, route.function_index);
      append_identity_integer(payload, route.direct_operation_estimate);
      append_identity_integer(payload, route.block_operation_estimate);
      append_identity_integer(payload, route.real_coefficients ? 1 : 0);
    }
    for (const auto &base : program.scalar_program.bases) {
      append_identity_field(payload, base.base_id);
      append_identity_integers(payload, base.left_channels);
      append_identity_integers(payload, base.right_channels);
      append_identity_complexes(payload, base.coefficients);
    }
    for (const auto &node : program.scalar_program.nodes) {
      append_identity_field(payload, std::to_string(node.base_index));
      append_identity_field(payload, std::to_string(node.exponent));
      append_identity_field(payload, std::to_string(node.left_value));
      append_identity_field(payload, std::to_string(node.right_value));
    }
    for (const auto &route : program.scalar_program.routes) {
      append_identity_field(payload, std::to_string(route.function_index));
      append_identity_field(payload, std::to_string(route.value_index));
      append_identity_complex(payload, route.scale);
      append_identity_field(payload, route.factorization_id);
    }
    append_identity_integer(payload, program.scalar_program.value_count);
    for (const auto &plan : program.coupled_product_plans) {
      append_identity_integers(payload, plan.node_offsets);
      append_identity_integers(payload, plan.node_dimensions);
      append_identity_integers(payload, plan.node_leaf_offsets);
      append_identity_integers(payload, plan.leaf_input_components);
      append_identity_integers(payload, plan.node_coefficient_offsets);
      append_identity_integers(payload, plan.coefficient_left_components);
      append_identity_integers(payload, plan.coefficient_right_components);
      append_identity_integers(payload, plan.coefficient_output_components);
      append_identity_doubles(payload, plan.coefficient_values);
      append_identity_integers(payload, plan.readout_components);
      append_identity_doubles(payload, plan.readout_coefficients);
      append_identity_integer(payload, plan.function_index);
      append_identity_integer(payload, plan.operation_estimate);
      append_identity_integer(payload, plan.total_node_components);
    }
    append_identity_integers(payload, program.power_channels);
    append_identity_integers(payload, program.power_maximum_exponents);
    append_identity_integers(payload, program.power_offsets);
    append_identity_integer(payload, program.power_storage_size);
    return sha256_string(payload);
  }

  void validate_sparse_synthesis_table(const YAML::Node &table, const std::string &path)
  {
    require_mapping(table, path);
    if (scalar_string(table["orientation"], path + ".orientation") != "synthesis" ||
        scalar_string(table["analysis_orientation"], path + ".analysis_orientation") !=
            "conjugate_transpose")
      fail(path, "unsupported synthesis-table orientation");
    if (scalar_string(table["convention_id"], path + ".convention_id") !=
        "o3:complex_condon_shortley_young_orthogonal_v1")
      fail(path, "unsupported synthesis-table convention");
    const YAML::Node validation = table["validation_report"];
    require_mapping(validation, path + ".validation_report");
    if (!boolean(validation["passed"], path + ".validation_report.passed"))
      fail(path, "synthesis-table validation did not pass");
    const int input_dimension = integer(table["input_dimension"], path + ".input_dimension", 1);
    const int output_dimension = integer(table["output_dimension"], path + ".output_dimension", 1);
    const YAML::Node values = table["values"];
    require_sequence(values, path + ".values");
    const auto rows = integer_sequence(table["row_indices"], path + ".row_indices", values.size());
    const auto columns =
        integer_sequence(table["column_indices"], path + ".column_indices", values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (rows[index] >= input_dimension || columns[index] >= output_dimension)
        fail(path, "synthesis-table coordinate is out of range");
      (void) complex_number(values[index], path + ".values");
    }
  }

  int species_index(const std::vector<YACESpecies> &species, const std::string &element,
                    const std::string &path)
  {
    int result = -1;
    for (std::size_t index = 0; index < species.size(); ++index)
      if (species[index].element == element) {
        if (result >= 0) fail(path, "element name is not unique in the YACE model");
        result = static_cast<int>(index);
      }
    if (result < 0) fail(path, "element is absent from the YACE model");
    return result;
  }

  struct CoupledSourceChannel {
    int neighbor_species = -1;
    int radial = -1;
    int angular = -1;
  };

  using CoupledLoadPolynomial = std::map<MonomialKey, long double>;

  MonomialKey multiply_monomial_keys(const MonomialKey &left, const MonomialKey &right,
                                     const std::string &path)
  {
    MonomialKey result;
    result.reserve(left.size() + right.size());
    std::size_t left_index = 0;
    std::size_t right_index = 0;
    while (left_index < left.size() || right_index < right.size()) {
      if (right_index == right.size() ||
          (left_index < left.size() && left[left_index].first < right[right_index].first)) {
        result.push_back(left[left_index++]);
      } else if (left_index == left.size() || right[right_index].first < left[left_index].first) {
        result.push_back(right[right_index++]);
      } else {
        if (left[left_index].second > std::numeric_limits<int>::max() - right[right_index].second)
          fail(path, "coupled-product monomial exponent overflow");
        result.emplace_back(left[left_index].first,
                            left[left_index].second + right[right_index].second);
        ++left_index;
        ++right_index;
      }
    }
    return result;
  }

  void add_coupled_load_term(CoupledLoadPolynomial &polynomial, MonomialKey monomial,
                             long double coefficient, std::size_t &stored_terms,
                             const std::string &path)
  {
    if (!std::isfinite(coefficient))
      fail(path, "coupled-product load-time polynomial became non-finite");
    if (coefficient == 0.0L) return;
    const auto inserted = polynomial.emplace(std::move(monomial), 0.0L);
    if (inserted.second && ++stored_terms > 262144)
      fail(path, "coupled-product load-time polynomial exceeds its term bound");
    const long double accumulated = inserted.first->second + coefficient;
    if (!std::isfinite(accumulated))
      fail(path, "coupled-product load-time polynomial became non-finite");
    inserted.first->second = accumulated;
  }

  void validate_coupled_product_direct_polynomial(const YACECoupledProductDAGPlan &plan,
                                                  const YACESparsePolynomial &direct,
                                                  int function_index, double absolute_tolerance,
                                                  double relative_tolerance,
                                                  const std::string &path)
  {
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
        !std::isfinite(relative_tolerance) || relative_tolerance < 0.0)
      fail(path, "coupled-product equivalence tolerances are invalid");
    if (absolute_tolerance > 5.0e-11 || relative_tolerance > 1.0e-12)
      fail(path, "coupled-product equivalence tolerances exceed native bounds");
    if (function_index < 0 ||
        static_cast<std::size_t>(function_index + 1) >= direct.descriptor_offsets.size())
      fail(path, "coupled-product direct descriptor index is out of range");

    std::vector<CoupledLoadPolynomial> components(
        static_cast<std::size_t>(plan.total_node_components));
    std::size_t stored_terms = 0;
    for (std::size_t node = 0; node < plan.node_offsets.size(); ++node) {
      const std::int64_t offset = plan.node_offsets[node];
      const std::int64_t dimension = plan.node_dimensions[node];
      const std::int64_t leaf_offset = plan.node_leaf_offsets[node];
      if (leaf_offset >= 0) {
        if (leaf_offset + dimension > static_cast<std::int64_t>(plan.leaf_input_components.size()))
          fail(path, "coupled-product primitive polynomial is out of range");
        for (std::int64_t component = 0; component < dimension; ++component)
          add_coupled_load_term(
              components[static_cast<std::size_t>(offset + component)],
              MonomialKey{{static_cast<int>(plan.leaf_input_components[static_cast<std::size_t>(
                               leaf_offset + component)]),
                           1}},
              1.0L, stored_terms, path);
        continue;
      }

      const std::int64_t begin = plan.node_coefficient_offsets[node];
      const std::int64_t end = plan.node_coefficient_offsets[node + 1];
      for (std::int64_t coefficient = begin; coefficient < end; ++coefficient) {
        const std::int64_t left =
            plan.coefficient_left_components[static_cast<std::size_t>(coefficient)];
        const std::int64_t right =
            plan.coefficient_right_components[static_cast<std::size_t>(coefficient)];
        const std::int64_t output =
            plan.coefficient_output_components[static_cast<std::size_t>(coefficient)];
        if (left < 0 || right < 0 || left >= offset || right >= offset || output < offset ||
            output >= offset + dimension)
          fail(path, "coupled-product polynomial topology is inconsistent");
        const auto &left_polynomial = components[static_cast<std::size_t>(left)];
        const auto &right_polynomial = components[static_cast<std::size_t>(right)];
        if (!left_polynomial.empty() && right_polynomial.size() > 262144 / left_polynomial.size())
          fail(path, "coupled-product polynomial product exceeds its term bound");
        for (const auto &[left_key, left_value] : left_polynomial)
          for (const auto &[right_key, right_value] : right_polynomial)
            add_coupled_load_term(
                components[static_cast<std::size_t>(output)],
                multiply_monomial_keys(left_key, right_key, path),
                static_cast<long double>(
                    plan.coefficient_values[static_cast<std::size_t>(coefficient)]) *
                    left_value * right_value,
                stored_terms, path);
      }
    }

    CoupledLoadPolynomial actual;
    for (std::size_t readout = 0; readout < plan.readout_components.size(); ++readout) {
      const std::int64_t component = plan.readout_components[readout];
      if (component < 0 || component >= plan.total_node_components)
        fail(path, "coupled-product polynomial readout is out of range");
      for (const auto &[key, value] : components[static_cast<std::size_t>(component)])
        add_coupled_load_term(actual, key,
                              static_cast<long double>(plan.readout_coefficients[readout]) * value,
                              stored_terms, path);
    }

    CoupledLoadPolynomial expected;
    const std::int64_t descriptor_begin =
        direct.descriptor_offsets[static_cast<std::size_t>(function_index)];
    const std::int64_t descriptor_end =
        direct.descriptor_offsets[static_cast<std::size_t>(function_index + 1)];
    for (std::int64_t row = descriptor_begin; row < descriptor_end; ++row) {
      const std::int64_t term = direct.descriptor_terms[static_cast<std::size_t>(row)];
      MonomialKey key;
      const std::int64_t factor_begin = direct.factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t factor_end = direct.factor_offsets[static_cast<std::size_t>(term + 1)];
      key.reserve(static_cast<std::size_t>(factor_end - factor_begin));
      for (std::int64_t factor = factor_begin; factor < factor_end; ++factor)
        key.emplace_back(
            static_cast<int>(direct.factor_indices[static_cast<std::size_t>(factor)]),
            static_cast<int>(direct.factor_exponents[static_cast<std::size_t>(factor)]));
      add_coupled_load_term(
          expected, std::move(key),
          static_cast<long double>(direct.descriptor_coefficients[static_cast<std::size_t>(row)]),
          stored_terms, path);
    }

    auto actual_entry = actual.begin();
    auto expected_entry = expected.begin();
    while (actual_entry != actual.end() || expected_entry != expected.end()) {
      long double actual_value = 0.0L;
      long double expected_value = 0.0L;
      if (expected_entry == expected.end() ||
          (actual_entry != actual.end() && actual_entry->first < expected_entry->first)) {
        actual_value = actual_entry->second;
        ++actual_entry;
      } else if (actual_entry == actual.end() || expected_entry->first < actual_entry->first) {
        expected_value = expected_entry->second;
        ++expected_entry;
      } else {
        actual_value = actual_entry->second;
        expected_value = expected_entry->second;
        ++actual_entry;
        ++expected_entry;
      }
      const long double difference = actual_value - expected_value;
      const long double tolerance = static_cast<long double>(absolute_tolerance) +
          static_cast<long double>(relative_tolerance) * std::abs(expected_value) +
          64.0L * std::numeric_limits<double>::epsilon() * (1.0L + std::abs(expected_value));
      if (!std::isfinite(actual_value) || !std::isfinite(expected_value) ||
          !std::isfinite(difference) || !std::isfinite(tolerance))
        fail(path, "coupled-product polynomial comparison became non-finite");
      if (std::abs(difference) > tolerance)
        fail(path, "native coupled-product polynomial does not match direct C-tilde");
    }
  }

  YACECoupledProductDAGPlan parse_coupled_product_dag_plan(const YAML::Node &plan,
                                                           const MappedBlockDescriptor &mapped,
                                                           const std::vector<YACESpecies> &species,
                                                           const std::string &source_hash,
                                                           const std::string &path)
  {
    const YAML::Node instructions = plan["instructions"];
    require_sequence(instructions, path + ".instructions");
    if (instructions.size() != 1)
      fail(path, "coupled-product execution plan must contain one instruction");
    const YAML::Node instruction = instructions[0];
    if (scalar_string(instruction["opcode"], path + ".instruction.opcode") !=
            "ace_coupled_product_dag" ||
        scalar_string(instruction["analysis_orientation"],
                      path + ".instruction.analysis_orientation") != "conjugate_transpose" ||
        scalar_string(instruction["instruction_id"], path + ".instruction.instruction_id") !=
            mapped.instruction_id)
      fail(path, "coupled-product instruction contract is inconsistent");
    const YAML::Node metadata = instruction["metadata"];
    require_mapping(metadata, path + ".metadata");
    if (scalar_string(metadata["schema"], path + ".metadata.schema") !=
            "ye3t_ace_coupled_product_dag_v1" ||
        scalar_string(metadata["execution_kind"], path + ".metadata.execution_kind") !=
            "coupled_product_dag" ||
        boolean(metadata["runtime_path_discovery"], path + ".metadata.runtime_path_discovery") ||
        !lower_sha256(
            scalar_string(metadata["semantic_sha256"], path + ".metadata.semantic_sha256")))
      fail(path, "coupled-product metadata contract is inconsistent");
    const YAML::Node top_certificate = plan["certificate"]["ace_coupled_product_dag"];
    if (!boolean(top_certificate["passed"], path + ".certificate.ace_coupled_product_dag.passed") ||
        scalar_string(top_certificate["semantic_sha256"],
                      path +
                          ".certificate.ace_coupled_product_dag.semantic_"
                          "sha256") !=
            scalar_string(metadata["semantic_sha256"], path + ".metadata.semantic_sha256"))
      fail(path, "coupled-product compiler certificate did not pass");
    const YAML::Node target = metadata["target"];
    if (integer(target["L"], path + ".metadata.target.L", 0) != 0 ||
        integer(target["parity"], path + ".metadata.target.parity", 1) != 1 ||
        scalar_string(target["convention_id"], path + ".metadata.target.convention_id") !=
            "o3:complex_condon_shortley_young_orthogonal_v1" ||
        scalar_string(target["central_species"], path + ".metadata.target.central_species") !=
            species[static_cast<std::size_t>(mapped.central_species)].element)
      fail(path, "coupled-product target is not the mapped ordinary scalar");
    validate_exact_image_certificate(metadata, path + ".metadata");

    std::map<std::string, CoupledSourceChannel> source_channels;
    const YAML::Node source_records = metadata["source_channels"];
    require_sequence(source_records, path + ".metadata.source_channels");
    if (source_records.size() == 0 || source_records.size() > 4096)
      fail(path, "coupled-product source channel count is outside the bound");
    for (std::size_t index = 0; index < source_records.size(); ++index) {
      const YAML::Node record = source_records[index];
      const std::string record_path =
          path + ".metadata.source_channels[" + std::to_string(index) + "]";
      require_exact_fields(record,
                           {"central_species", "channel_id", "content_token", "convention_id", "l",
                            "neighbor_species", "radial_basis_id", "radial_index"},
                           record_path);
      const std::string channel_id =
          scalar_string(record["channel_id"], record_path + ".channel_id");
      if (channel_id.empty() ||
          scalar_string(record["central_species"], record_path + ".central_species") !=
              species[static_cast<std::size_t>(mapped.central_species)].element ||
          scalar_string(record["convention_id"], record_path + ".convention_id") !=
              "o3:complex_condon_shortley_young_orthogonal_v1" ||
          scalar_string(record["radial_basis_id"], record_path + ".radial_basis_id") !=
              "pace_cheb_exp_cos" ||
          !source_channels
               .emplace(channel_id,
                        CoupledSourceChannel{
                            species_index(species,
                                          scalar_string(record["neighbor_species"],
                                                        record_path + ".neighbor_species"),
                                          record_path),
                            integer(record["radial_index"], record_path + ".radial_index", 0),
                            integer(record["l"], record_path + ".l", 0)})
               .second)
        fail(record_path, "coupled-product source binding is invalid");
    }

    const YAML::Node nodes = metadata["nodes"];
    require_sequence(nodes, path + ".metadata.nodes");
    if (nodes.size() == 0 || nodes.size() > 2048)
      fail(path, "coupled-product node count is outside the native bound");
    const YAML::Node input_carriers = instruction["input_carriers"];
    require_sequence(input_carriers, path + ".instruction.input_carriers");
    std::set<int> input_indices;
    std::map<std::string, std::size_t> node_indices;
    std::vector<int> node_angular_momenta;
    YACECoupledProductDAGPlan result;
    result.function_index = mapped.function_index;
    result.node_offsets.reserve(nodes.size());
    result.node_dimensions.reserve(nodes.size());
    result.node_leaf_offsets.reserve(nodes.size());
    result.node_coefficient_offsets.reserve(nodes.size() + 1);
    result.node_coefficient_offsets.push_back(0);
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      const YAML::Node node = nodes[index];
      const std::string node_path = path + ".metadata.nodes[" + std::to_string(index) + "]";
      const std::string node_id = scalar_string(node["node_id"], node_path + ".node_id");
      const std::string kind = scalar_string(node["kind"], node_path + ".kind");
      const int angular = integer(node["L"], node_path + ".L", 0);
      const std::int64_t dimension = 2 * static_cast<std::int64_t>(angular) + 1;
      if (node_id.empty() || !node_indices.emplace(node_id, index).second ||
          result.total_node_components > 65536 - dimension)
        fail(node_path, "coupled-product node identity or storage is invalid");
      result.node_offsets.push_back(result.total_node_components);
      result.node_dimensions.push_back(dimension);
      result.total_node_components += dimension;
      node_angular_momenta.push_back(angular);
      if (kind == "primitive") {
        const int input_index = integer(node["input_index"], node_path + ".input_index", 0);
        if (input_index >= static_cast<int>(input_carriers.size()) ||
            !input_indices.insert(input_index).second ||
            integer(node["basis_index"], node_path + ".basis_index", 0) != 0)
          fail(node_path, "primitive input binding is invalid");
        const std::string channel_id =
            scalar_string(node["source_channel_id"], node_path + ".source_channel_id");
        const auto channel = source_channels.find(channel_id);
        if (channel == source_channels.end() || channel->second.angular != angular)
          fail(node_path, "primitive source channel is invalid");
        const ExactSparseMatrix coordinate =
            parse_exact_matrix(node["basis_coordinate"], node_path + ".basis_coordinate");
        if (coordinate.rows != 1 || coordinate.columns != 1 || coordinate.entries.size() != 1 ||
            coordinate.entries[0].row != 0 || coordinate.entries[0].column != 0)
          fail(node_path, "primitive basis coordinate is not scalar identity");
        ExactComplex identity;
        identity.real.terms.emplace(1, ExactRational{1, 1});
        if (!(coordinate.entries[0].value == identity))
          fail(node_path, "primitive basis coordinate is not scalar identity");
        result.node_leaf_offsets.push_back(
            static_cast<std::int64_t>(result.leaf_input_components.size()));
        for (int magnetic = -angular; magnetic <= angular; ++magnetic)
          result.leaf_input_components.push_back(
              full_channel_index(species[static_cast<std::size_t>(mapped.central_species)],
                                 channel->second.neighbor_species, channel->second.radial, angular,
                                 magnetic, node_path));
      } else if (kind == "product") {
        result.node_leaf_offsets.push_back(-1);
        const std::string left_id =
            scalar_string(node["left_node_id"], node_path + ".left_node_id");
        const std::string right_id =
            scalar_string(node["right_node_id"], node_path + ".right_node_id");
        const auto left = node_indices.find(left_id);
        const auto right = node_indices.find(right_id);
        if (left == node_indices.end() || right == node_indices.end() || left->second >= index ||
            right->second >= index)
          fail(node_path, "product operands are not topologically available");
        const YAML::Node table = record_by_id(
            plan["synthesis_tables"], "table_id",
            scalar_string(node["synthesis_table_id"], node_path + ".synthesis_table_id"),
            path + ".synthesis_tables");
        validate_sparse_synthesis_table(table, node_path + ".table");
        const std::int64_t left_dimension = result.node_dimensions[left->second];
        const std::int64_t right_dimension = result.node_dimensions[right->second];
        if (integer(table["input_dimension"], node_path + ".table.input_dimension", 1) !=
                left_dimension * right_dimension ||
            integer(table["output_dimension"], node_path + ".table.output_dimension", 1) !=
                dimension)
          fail(node_path, "product synthesis table has the wrong dimensions");
        const YAML::Node values = table["values"];
        const auto rows =
            integer_sequence(table["row_indices"], node_path + ".table.row_indices", values.size());
        const auto columns = integer_sequence(table["column_indices"],
                                              node_path + ".table.column_indices", values.size());
        for (std::size_t coefficient = 0; coefficient < values.size(); ++coefficient) {
          const std::complex<double> value =
              complex_number(values[coefficient], node_path + ".table.values");
          if (value.imag() != 0.0)
            fail(node_path, "native coupled-product path requires real CG coefficients");
          const std::int64_t row = rows[coefficient];
          result.coefficient_left_components.push_back(result.node_offsets[left->second] +
                                                       row / right_dimension);
          result.coefficient_right_components.push_back(result.node_offsets[right->second] +
                                                        row % right_dimension);
          result.coefficient_output_components.push_back(result.node_offsets[index] +
                                                         columns[coefficient]);
          result.coefficient_values.push_back(value.real());
        }
      } else {
        fail(node_path, "unsupported coupled-product node kind");
      }
      result.node_coefficient_offsets.push_back(
          static_cast<std::int64_t>(result.coefficient_values.size()));
    }
    if (input_indices.size() != input_carriers.size())
      fail(path, "coupled-product primitive nodes do not cover every input");

    if (mapped.scale.imag() != 0.0)
      fail(path, "native coupled-product path requires a real YACE scale");
    const YAML::Node outputs = metadata["outputs"];
    if (!outputs.IsSequence() || outputs.size() != 1)
      fail(path, "coupled-product plan must contain one output");
    const YAML::Node terms = outputs[0]["terms"];
    require_sequence(terms, path + ".metadata.outputs[0].terms");
    if (terms.size() == 0 || terms.size() > 512)
      fail(path, "coupled-product readout count is outside the native bound");
    for (std::size_t index = 0; index < terms.size(); ++index) {
      const std::string term_path =
          path + ".metadata.outputs[0].terms[" + std::to_string(index) + "]";
      const std::string root_id =
          scalar_string(terms[index]["root_node_id"], term_path + ".root_node_id");
      const auto root = node_indices.find(root_id);
      if (root == node_indices.end() || node_angular_momenta[root->second] != 0 ||
          result.node_dimensions[root->second] != 1)
        fail(term_path, "coupled-product readout root is not scalar");
      const std::complex<double> coefficient =
          complex_number(terms[index]["coefficient_binary64"], term_path + ".coefficient_binary64");
      if (coefficient.imag() != 0.0)
        fail(term_path, "native coupled-product path requires real readout coefficients");
      const double scaled_coefficient = mapped.scale.real() * coefficient.real();
      if (!std::isfinite(scaled_coefficient))
        fail(term_path, "coupled-product scaled readout coefficient overflow");
      result.readout_components.push_back(result.node_offsets[root->second]);
      result.readout_coefficients.push_back(scaled_coefficient);
    }
    const YAML::Node adjoint = metadata["adjoint_contract"];
    if (scalar_string(adjoint["cotangent_pairing"],
                      path + ".metadata.adjoint_contract.cotangent_pairing") !=
            "real_part_conjugate_pairing_v1" ||
        scalar_string(adjoint["coefficient_adjoint"],
                      path + ".metadata.adjoint_contract.coefficient_adjoint") !=
            "conjugate_transpose" ||
        scalar_string(adjoint["product_adjoint"],
                      path + ".metadata.adjoint_contract.product_adjoint") !=
            "bilinear_product_transpose_v1" ||
        scalar_string(adjoint["alias_accumulation"],
                      path + ".metadata.adjoint_contract.alias_accumulation") !=
            "sum_all_occurrences" ||
        !boolean(adjoint["division_free"], path + ".metadata.adjoint_contract.division_free") ||
        !boolean(adjoint["reverse_topological"],
                 path + ".metadata.adjoint_contract.reverse_topological"))
      fail(path, "coupled-product adjoint contract is unsupported");
    const YAML::Node resources = metadata["resource_report"];
    if (scalar_string(resources["status"], path + ".metadata.resource_report.status") !=
            "eligible" ||
        !boolean(resources["enumeration_complete"],
                 path + ".metadata.resource_report.enumeration_complete"))
      fail(path, "coupled-product resource report is not eligible");
    result.operation_estimate = integer(resources["operation_estimate"],
                                        path + ".metadata.resource_report.operation_estimate", 0);
    const YAML::Node fallback = metadata["direct_fallback"];
    if (!boolean(fallback["required"], path + ".metadata.direct_fallback.required") ||
        scalar_string(fallback["evaluator_kind"],
                      path + ".metadata.direct_fallback.evaluator_kind") != "direct_ctilde" ||
        scalar_string(fallback["binding_id"], path + ".metadata.direct_fallback.binding_id") !=
            mapped.direct_candidate_id ||
        source_hash.empty())
      fail(path, "coupled-product direct fallback binding is inconsistent");
    validate_coupled_product_direct_polynomial(
        result, species[static_cast<std::size_t>(mapped.central_species)].polynomial,
        mapped.function_index, mapped.equivalence_absolute_tolerance,
        mapped.equivalence_relative_tolerance, path);
    return result;
  }

  std::complex<double> sparse_table_value(const YAML::Node &table, int row, int column,
                                          const std::string &path)
  {
    validate_sparse_synthesis_table(table, path);
    const YAML::Node values = table["values"];
    const auto rows = integer_sequence(table["row_indices"], path + ".row_indices", values.size());
    const auto columns =
        integer_sequence(table["column_indices"], path + ".column_indices", values.size());
    std::complex<double> result(0.0, 0.0);
    for (std::size_t index = 0; index < values.size(); ++index)
      if (rows[index] == row && columns[index] == column)
        result += complex_number(values[index], path + ".values");
    return result;
  }

  std::complex<double> sparse_assembly_value(const YAML::Node &assembly, int row, int column,
                                             const std::string &path)
  {
    require_mapping(assembly, path);
    const int source_dimension =
        integer(assembly["source_dimension"], path + ".source_dimension", 1);
    const int induced_dimension =
        integer(assembly["induced_dimension"], path + ".induced_dimension", 1);
    const YAML::Node values = assembly["values"];
    require_sequence(values, path + ".values");
    const auto rows =
        integer_sequence(assembly["row_indices"], path + ".row_indices", values.size());
    const auto columns =
        integer_sequence(assembly["column_indices"], path + ".column_indices", values.size());
    std::complex<double> result(0.0, 0.0);
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (rows[index] >= source_dimension || columns[index] >= induced_dimension)
        fail(path, "source-assembly coordinate is out of range");
      if (rows[index] == row && columns[index] == column)
        result += complex_number(values[index], path + ".values");
    }
    return result;
  }

  void validate_output_binding(const YAML::Node &plan, const MappedBlockDescriptor &mapped,
                               const std::string &source_hash, const std::string &path)
  {
    const YAML::Node records = plan["certificate"]["yace_output_bindings"]["records"];
    require_sequence(records, path + ".certificate.yace_output_bindings.records");
    int matches = 0;
    for (std::size_t index = 0; index < records.size(); ++index) {
      const YAML::Node record = records[index];
      if (integer(record["central_type"], path + ".central_type", 0) != mapped.central_species ||
          integer(record["function_index"], path + ".function_index", 0) != mapped.function_index)
        continue;
      ++matches;
      if (scalar_string(record["feature_id"], path + ".feature_id") != mapped.feature_id ||
          scalar_string(record["instruction_id"], path + ".instruction_id") !=
              mapped.instruction_id ||
          integer(record["channel_index"], path + ".channel_index", 0) != mapped.channel_index ||
          integer(record["tableau_index"], path + ".tableau_index", 0) != mapped.tableau_index ||
          integer(record["magnetic_index"], path + ".magnetic_index", 0) != mapped.magnetic_index ||
          complex_number(record["scale"], path + ".scale") != mapped.scale ||
          scalar_string(record["source_yace_sha256"], path + ".source_yace_sha256") != source_hash)
        fail(path, "function-map output binding disagrees with the plan certificate");
    }
    if (matches != 1) fail(path, "mapped function requires exactly one plan certificate binding");
  }

  std::vector<MappedBlockDescriptor>
  candidate_readout_terms(const YAML::Node &plan, const MappedBlockDescriptor &row,
                          const std::string &candidate_id, const std::string &readout_id,
                          const std::string &source_hash, const std::string &path)
  {
    if (!lower_sha256(readout_id)) fail(path, "candidate readout ID must be a SHA-256");
    const YAML::Node ledger = plan["certificate"]["yace_candidate_readouts"];
    require_exact_fields(ledger, {"records", "schema"},
                         "execution_plan.certificate.yace_candidate_readouts");
    if (scalar_string(ledger["schema"],
                      "execution_plan.certificate.yace_candidate_readouts.schema") !=
        "ye3t_yace_candidate_readouts_v1")
      fail(path, "unsupported candidate-readout certificate ledger");
    const YAML::Node record =
        record_by_id(ledger["records"], "readout_id", readout_id,
                     "execution_plan.certificate.yace_candidate_readouts.records");
    require_exact_fields(record,
                         {"central_type", "equivalence", "feature_id", "function_index",
                          "readout_id", "schema", "source_yace_sha256", "terms",
                          "variable_order_hash"},
                         path + ".certificate_readout");
    if (candidate_readout_identity(record, path + ".certificate_readout") != readout_id ||
        scalar_string(record["schema"], path + ".certificate_readout.schema") !=
            "ye3t_yace_candidate_readout_v1" ||
        integer(record["central_type"], path + ".certificate_readout.central_type", 0) !=
            row.central_species ||
        integer(record["function_index"], path + ".certificate_readout.function_index", 0) !=
            row.function_index ||
        scalar_string(record["feature_id"], path + ".certificate_readout.feature_id") !=
            row.feature_id ||
        scalar_string(record["source_yace_sha256"],
                      path + ".certificate_readout.source_yace_sha256") != source_hash ||
        !lower_sha256(scalar_string(record["variable_order_hash"],
                                    path + ".certificate_readout.variable_order_hash")))
      fail(path, "candidate readout is not bound to this exact YACE row");

    const YAML::Node equivalence = record["equivalence"];
    const std::string equivalence_path = path + ".certificate_readout.equivalence";
    const std::string method = scalar_string(equivalence["method"], equivalence_path + ".method");
    std::set<std::string> equivalence_fields = {"absolute_tolerance",
                                                "adjoint_certified_by_coefficient_identity",
                                                "derivative_rule",
                                                "maximum_absolute_coefficient_residual",
                                                "method",
                                                "passed",
                                                "plan_polynomial_sha256",
                                                "relative_l2_residual",
                                                "relative_tolerance",
                                                "support_equal",
                                                "yace_polynomial_sha256"};
    if (method == "coefficientwise_sparse_polynomial_mixed_v1") {
      equivalence_fields.insert("maximum_mixed_tolerance_ratio");
      equivalence_fields.insert("maximum_reference_coefficient_magnitude");
      equivalence_fields.insert("tolerance_rule");
    } else if (method != "coefficientwise_sparse_polynomial_v1") {
      fail(equivalence_path, "unsupported coefficient-identity method");
    }
    require_exact_fields(equivalence, equivalence_fields, equivalence_path);
    const double absolute_tolerance =
        finite_number(equivalence["absolute_tolerance"],
                      path + ".certificate_readout.equivalence.absolute_tolerance");
    const double relative_tolerance =
        finite_number(equivalence["relative_tolerance"],
                      path + ".certificate_readout.equivalence.relative_tolerance");
    const double maximum_residual =
        finite_number(equivalence["maximum_absolute_coefficient_residual"],
                      path +
                          ".certificate_readout.equivalence.maximum_absolute_"
                          "coefficient_residual");
    const double relative_residual =
        finite_number(equivalence["relative_l2_residual"],
                      path + ".certificate_readout.equivalence.relative_l2_residual");
    if (!boolean(equivalence["passed"], path + ".certificate_readout.equivalence.passed") ||
        !boolean(equivalence["support_equal"],
                 path + ".certificate_readout.equivalence.support_equal") ||
        !boolean(equivalence["adjoint_certified_by_coefficient_identity"],
                 path +
                     ".certificate_readout.equivalence.adjoint_certified_by_"
                     "coefficient_identity") ||
        scalar_string(equivalence["derivative_rule"],
                      path + ".certificate_readout.equivalence.derivative_rule") !=
            "explicit_polynomial_product_rule" ||
        absolute_tolerance < 0.0 || relative_tolerance < 0.0 || maximum_residual < 0.0 ||
        relative_residual < 0.0 ||
        !lower_sha256(
            scalar_string(equivalence["plan_polynomial_sha256"],
                          path + ".certificate_readout.equivalence.plan_polynomial_sha256")) ||
        !lower_sha256(
            scalar_string(equivalence["yace_polynomial_sha256"],
                          path + ".certificate_readout.equivalence.yace_polynomial_sha256")))
      fail(path, "candidate readout coefficient-identity certificate failed");

    if (method == "coefficientwise_sparse_polynomial_mixed_v1") {
      const double maximum_mixed_ratio =
          finite_number(equivalence["maximum_mixed_tolerance_ratio"],
                        equivalence_path + ".maximum_mixed_tolerance_ratio");
      const double maximum_reference =
          finite_number(equivalence["maximum_reference_coefficient_magnitude"],
                        equivalence_path + ".maximum_reference_coefficient_magnitude");
      const double maximum_allowed_residual =
          absolute_tolerance + relative_tolerance * maximum_reference;
      const double maximum_allowed_residual_with_ulp =
          std::nextafter(maximum_allowed_residual, std::numeric_limits<double>::infinity());
      if (scalar_string(equivalence["tolerance_rule"], equivalence_path + ".tolerance_rule") !=
              "absolute_plus_relative_reference" ||
          absolute_tolerance <= 0.0 || maximum_mixed_ratio < 0.0 || maximum_mixed_ratio > 1.0 ||
          maximum_reference <= 0.0 || !std::isfinite(maximum_allowed_residual) ||
          maximum_residual > maximum_allowed_residual_with_ulp)
        fail(path, "candidate readout mixed coefficient certificate failed");
    } else if (maximum_residual > absolute_tolerance || relative_residual > relative_tolerance) {
      fail(path, "candidate readout coefficient residual exceeds tolerance");
    }

    const YAML::Node terms = record["terms"];
    require_sequence(terms, path + ".certificate_readout.terms");
    if (terms.size() == 0 || terms.size() > 256)
      fail(path, "candidate readout term count is outside the supported bound");
    std::vector<MappedBlockDescriptor> result;
    result.reserve(terms.size());
    std::set<std::string> binding_ids;
    for (std::size_t index = 0; index < terms.size(); ++index) {
      const YAML::Node term = terms[index];
      const std::string term_path =
          path + ".certificate_readout.terms[" + std::to_string(index) + "]";
      require_exact_fields(term,
                           {"binding_id", "channel_index", "instruction_id", "magnetic_index",
                            "scale", "tableau_index"},
                           term_path);
      const std::string binding_id = scalar_string(term["binding_id"], term_path + ".binding_id");
      if (!lower_sha256(binding_id) || candidate_binding_identity(term, term_path) != binding_id ||
          !binding_ids.insert(binding_id).second)
        fail(term_path, "candidate readout binding ID is invalid or duplicate");
      MappedBlockDescriptor mapped = row;
      mapped.block_candidate_id = candidate_id;
      mapped.instruction_id = scalar_string(term["instruction_id"], term_path + ".instruction_id");
      mapped.channel_index = integer(term["channel_index"], term_path + ".channel_index", 0);
      mapped.tableau_index = integer(term["tableau_index"], term_path + ".tableau_index", 0);
      mapped.magnetic_index = integer(term["magnetic_index"], term_path + ".magnetic_index", 0);
      mapped.scale = complex_number(term["scale"], term_path + ".scale");
      mapped.equivalence_absolute_tolerance = absolute_tolerance;
      mapped.equivalence_relative_tolerance = relative_tolerance;
      result.push_back(std::move(mapped));
    }
    return result;
  }

  std::vector<ScalarPowerScheduleStep> expected_scalar_power_schedule(std::int64_t exponent)
  {
    if (exponent <= 0) fail("execution_plan", "scalar invariant exponent must be positive");
    std::vector<ScalarPowerScheduleStep> result;
    std::int64_t highest = 1;
    while (2 * highest <= exponent) {
      result.push_back({2 * highest, highest, highest});
      highest *= 2;
    }
    std::int64_t accumulator = highest;
    for (std::int64_t bit = highest / 2; bit > 0; bit /= 2)
      if ((exponent & bit) != 0) {
        result.push_back({accumulator + bit, accumulator, bit});
        accumulator += bit;
      }
    return result;
  }

  using ScalarExponent = std::array<std::int64_t, 3>;
  using ScalarPolynomial = std::map<ScalarExponent, std::complex<double>>;

  ScalarPolynomial scalar_polynomial_power(const ScalarPolynomial &source, std::int64_t exponent)
  {
    ScalarPolynomial result{{ScalarExponent{0, 0, 0}, {1.0, 0.0}}};
    for (std::int64_t iteration = 0; iteration < exponent; ++iteration) {
      ScalarPolynomial following;
      for (const auto &[left_exponents, left_coefficient] : result)
        for (const auto &[right_exponents, right_coefficient] : source) {
          ScalarExponent output{};
          for (std::size_t index = 0; index < output.size(); ++index)
            output[index] = left_exponents[index] + right_exponents[index];
          following[output] += left_coefficient * right_coefficient;
        }
      result = std::move(following);
    }
    return result;
  }

  ScalarRouteCandidate
  parse_scalar_route(const YAML::Node &metadata, const YAML::Node &selected_plan,
                     const YACEBlockPowerPlan &power_plan, std::int64_t selected_component,
                     int parent_rank, const std::complex<double> &output_scale,
                     const MappedBlockDescriptor &mapped, const std::string &path)
  {
    const YAML::Node certificates = metadata["fast_route_certificates"];
    require_sequence(certificates, path + ".fast_route_certificates");
    YAML::Node certificate;
    bool found = false;
    for (std::size_t index = 0; index < certificates.size(); ++index) {
      const YAML::Node candidate = certificates[index];
      require_mapping(candidate, path + ".fast_route_certificates");
      if (scalar_string(candidate["certificate_sha256"], path + ".certificate_sha256") !=
          mapped.scalar_factorization_id)
        continue;
      if (found) fail(path, "duplicate scalar-invariant-power certificate ID");
      certificate = candidate;
      found = true;
    }
    if (!found) fail(path, "scalar alternative names an unknown factorization certificate");

    require_exact_fields(certificate,
                         {"absolute_tolerance",
                          "adjoint_rule",
                          "analysis_orientation",
                          "basis_certificate_schema",
                          "basis_certificate_sha256",
                          "basis_convention",
                          "block_index",
                          "block_plan_id",
                          "carrier",
                          "certificate_sha256",
                          "channel_indices",
                          "coefficient_identity_passed",
                          "component_index",
                          "descriptor_index",
                          "factor_basis",
                          "forward_schedule",
                          "full_support_checked",
                          "input_L",
                          "intrinsic_output_scale",
                          "magnetic_order",
                          "maximum_absolute_residual",
                          "maximum_relative_residual",
                          "maximum_scaled_residual",
                          "multiplication_schedule",
                          "multiplicity_index",
                          "normalization_convention",
                          "outer_power",
                          "output_L",
                          "parent_partition",
                          "parent_rank",
                          "passed",
                          "physical_source_binding_sha256",
                          "quadratic_base_sha256",
                          "quadratic_source_component_sha256",
                          "quadratic_source_plan_convention_hash",
                          "quadratic_source_plan_sha256",
                          "quadratic_terms",
                          "relative_tolerance",
                          "reverse_schedule",
                          "runtime_path_discovery",
                          "schema",
                          "source_binding_sha256",
                          "source_power_plan_convention_hash",
                          "source_power_plan_sha256",
                          "support_equal",
                          "target_component_sha256",
                          "zero_safe_adjoint"},
                         path + ".scalar_certificate");
    if (scalar_string(certificate["schema"], path + ".schema") !=
            "ye3t_scalar_invariant_power_certificate_v1" ||
        !boolean(certificate["passed"], path + ".passed") ||
        !boolean(certificate["coefficient_identity_passed"],
                 path + ".coefficient_identity_passed") ||
        !boolean(certificate["full_support_checked"], path + ".full_support_checked") ||
        !boolean(certificate["support_equal"], path + ".support_equal") ||
        !boolean(certificate["zero_safe_adjoint"], path + ".zero_safe_adjoint") ||
        boolean(certificate["runtime_path_discovery"], path + ".runtime_path_discovery"))
      fail(path, "scalar-invariant-power compiler certificate did not pass");
    if (!lower_sha256(mapped.scalar_factorization_id) ||
        scalar_string(certificate["certificate_sha256"], path + ".certificate_sha256") !=
            mapped.scalar_factorization_id)
      fail(path, "scalar-invariant-power certificate ID is invalid");
    for (const std::string &field :
         {"basis_certificate_sha256", "physical_source_binding_sha256", "quadratic_base_sha256",
          "quadratic_source_component_sha256", "quadratic_source_plan_sha256",
          "source_binding_sha256", "source_power_plan_sha256", "target_component_sha256"})
      if (!lower_sha256(scalar_string(certificate[field], path + "." + field)))
        fail(path, "scalar-invariant-power certificate contains an invalid hash");
    if (scalar_string(certificate["block_plan_id"], path + ".block_plan_id") !=
            scalar_string(selected_plan["plan_id"], path + ".plan_id") ||
        scalar_string(certificate["source_power_plan_convention_hash"],
                      path + ".source_power_plan_convention_hash") !=
            scalar_string(selected_plan["plan"]["convention_hash"], path + ".plan.convention_hash"))
      fail(path, "scalar-invariant-power source plan binding is inconsistent");
    const std::string quadratic_convention =
        scalar_string(certificate["quadratic_source_plan_convention_hash"],
                      path + ".quadratic_source_plan_convention_hash");
    if (quadratic_convention.size() != 16 ||
        !std::all_of(quadratic_convention.begin(), quadratic_convention.end(), [](char character) {
          return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
        }))
      fail(path, "scalar-invariant-power quadratic convention hash is invalid");
    const std::string basis_certificate_schema =
        scalar_string(certificate["basis_certificate_schema"], path + ".basis_certificate_schema");
    if (basis_certificate_schema != "ye3t_exact_symbolic_occupancy_basis_v1" &&
        basis_certificate_schema != "ye3t_homogeneous_numeric_basis_validation_v1")
      fail(path, "scalar-invariant-power basis certificate is unsupported");
    if (integer(certificate["block_index"], path + ".block_index", 0) != 0 ||
        integer(certificate["descriptor_index"], path + ".descriptor_index", 0) !=
            selected_component ||
        integer(certificate["parent_rank"], path + ".parent_rank", 1) != parent_rank ||
        integer(certificate["input_L"], path + ".input_L", 0) != 1 ||
        integer(certificate["output_L"], path + ".output_L", 0) != 0 ||
        integer(certificate["multiplicity_index"], path + ".multiplicity_index", 0) != 0 ||
        integer(certificate["component_index"], path + ".component_index", 0) != 0 ||
        scalar_string(certificate["carrier"], path + ".carrier") != "ACE_density" ||
        scalar_string(certificate["factor_basis"], path + ".factor_basis") != "A" ||
        scalar_string(certificate["normalization_convention"],
                      path + ".normalization_convention") != "none" ||
        scalar_string(certificate["basis_convention"], path + ".basis_convention") !=
            "complex_magnetic" ||
        scalar_string(certificate["analysis_orientation"], path + ".analysis_orientation") !=
            "conjugate_transpose" ||
        scalar_string(certificate["adjoint_rule"], path + ".adjoint_rule") !=
            "reverse_multiplication_schedule_and_quadratic_product_without_"
            "division")
      fail(path, "scalar-invariant-power certificate semantics are unsupported");
    const auto parent_partition =
        integer_sequence(certificate["parent_partition"], path + ".parent_partition", 1, 1);
    if (parent_partition[0] != parent_rank)
      fail(path, "scalar-invariant-power parent partition is inconsistent");
    const auto channels =
        integer_sequence(certificate["channel_indices"], path + ".channel_indices", 3);
    if (channels != std::vector<int>({0, 1, 2}) || power_plan.input_dimension != 3)
      fail(path, "scalar-invariant-power channel order is unsupported");
    require_sequence(certificate["magnetic_order"], path + ".magnetic_order");
    if (certificate["magnetic_order"].size() != 3 ||
        signed_integer(certificate["magnetic_order"][0], path + ".magnetic_order") != -1 ||
        signed_integer(certificate["magnetic_order"][1], path + ".magnetic_order") != 0 ||
        signed_integer(certificate["magnetic_order"][2], path + ".magnetic_order") != 1)
      fail(path, "scalar-invariant-power magnetic order is unsupported");

    const std::int64_t outer_power = integer(certificate["outer_power"], path + ".outer_power", 1);
    constexpr std::int64_t maximum_outer_power = 64;
    if (parent_rank % 2 != 0 || outer_power != parent_rank / 2 || outer_power > maximum_outer_power)
      fail(path,
           "scalar-invariant-power exponent is inconsistent with the "
           "parent rank or exceeds the bounded CPU loader limit");
    const auto expected_schedule = expected_scalar_power_schedule(outer_power);
    const YAML::Node schedule = certificate["multiplication_schedule"];
    require_sequence(schedule, path + ".multiplication_schedule");
    if (schedule.size() != expected_schedule.size())
      fail(path, "scalar-invariant-power schedule length is inconsistent");
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      require_exact_fields(schedule[index],
                           {"left_exponent", "node_id", "output_exponent", "right_exponent"},
                           path + ".multiplication_schedule");
      const auto &expected = expected_schedule[index];
      if (integer(schedule[index]["output_exponent"], path + ".output_exponent", 1) !=
              expected.output_exponent ||
          integer(schedule[index]["left_exponent"], path + ".left_exponent", 1) !=
              expected.left_exponent ||
          integer(schedule[index]["right_exponent"], path + ".right_exponent", 1) !=
              expected.right_exponent ||
          scalar_string(schedule[index]["node_id"], path + ".node_id") !=
              "q_pow_" + std::to_string(expected.output_exponent))
        fail(path, "scalar-invariant-power multiplication schedule is invalid");
    }
    const YAML::Node forward = certificate["forward_schedule"];
    const YAML::Node reverse = certificate["reverse_schedule"];
    require_sequence(forward, path + ".forward_schedule");
    require_sequence(reverse, path + ".reverse_schedule");
    if (forward.size() != expected_schedule.size() + 2 ||
        reverse.size() != expected_schedule.size() + 2 ||
        scalar_string(forward[0], path + ".forward_schedule") != "quadratic_base" ||
        scalar_string(forward[forward.size() - 1], path + ".forward_schedule") !=
            "intrinsic_scale" ||
        scalar_string(reverse[0], path + ".reverse_schedule") != "intrinsic_scale" ||
        scalar_string(reverse[reverse.size() - 1], path + ".reverse_schedule") != "quadratic_base")
      fail(path, "scalar-invariant-power forward/reverse schedule is invalid");
    for (std::size_t index = 0; index < expected_schedule.size(); ++index) {
      const std::string expected =
          "q_pow_" + std::to_string(expected_schedule[index].output_exponent);
      const std::string expected_reverse = "q_pow_" +
          std::to_string(expected_schedule[expected_schedule.size() - 1 - index].output_exponent);
      if (scalar_string(forward[index + 1], path + ".forward_schedule") != expected ||
          scalar_string(reverse[index + 1], path + ".reverse_schedule") != expected_reverse)
        fail(path, "scalar-invariant-power forward/reverse order is invalid");
    }

    const double absolute_tolerance =
        finite_number(certificate["absolute_tolerance"], path + ".absolute_tolerance");
    const double relative_tolerance =
        finite_number(certificate["relative_tolerance"], path + ".relative_tolerance");
    const double reported_absolute_residual = finite_number(
        certificate["maximum_absolute_residual"], path + ".maximum_absolute_residual");
    const double reported_relative_residual = finite_number(
        certificate["maximum_relative_residual"], path + ".maximum_relative_residual");
    const double reported_scaled_residual =
        finite_number(certificate["maximum_scaled_residual"], path + ".maximum_scaled_residual");
    if (absolute_tolerance <= 0.0 || absolute_tolerance > 2.0e-12 || relative_tolerance <= 0.0 ||
        relative_tolerance > 2.0e-11 || reported_scaled_residual < 0.0 ||
        reported_scaled_residual > 1.0 + 1.0e-12 || reported_absolute_residual < 0.0 ||
        reported_relative_residual < 0.0)
      fail(path, "scalar-invariant-power residual certificate is invalid");

    ScalarPolynomial quadratic;
    const YAML::Node quadratic_terms = certificate["quadratic_terms"];
    require_sequence(quadratic_terms, path + ".quadratic_terms");
    YACEScalarInvariantBase base;
    base.base_id =
        scalar_string(certificate["quadratic_base_sha256"], path + ".quadratic_base_sha256");
    for (std::size_t index = 0; index < quadratic_terms.size(); ++index) {
      const YAML::Node term = quadratic_terms[index];
      require_exact_fields(term, {"coefficient", "exponents"}, path + ".quadratic_terms");
      const auto exponents =
          integer_sequence(term["exponents"], path + ".quadratic_terms.exponents", 3);
      if (std::accumulate(exponents.begin(), exponents.end(), 0) != 2)
        fail(path, "scalar-invariant-power base term is not quadratic");
      ScalarExponent key{exponents[0], exponents[1], exponents[2]};
      const auto coefficient =
          complex_number(term["coefficient"], path + ".quadratic_terms.coefficient");
      if (!quadratic.emplace(key, coefficient).second)
        fail(path, "scalar-invariant-power base contains duplicate terms");
      std::vector<int> factors;
      for (int component = 0; component < 3; ++component)
        for (int copy = 0; copy < exponents[static_cast<std::size_t>(component)]; ++copy)
          factors.push_back(component);
      if (factors.size() != 2) fail(path, "scalar-invariant-power base factor count is invalid");
      base.left_channels.push_back(power_plan.input_channels[static_cast<std::size_t>(factors[0])]);
      base.right_channels.push_back(
          power_plan.input_channels[static_cast<std::size_t>(factors[1])]);
      base.coefficients.push_back(coefficient);
    }
    if (quadratic.empty()) fail(path, "scalar-invariant-power base is empty");
    ScalarPolynomial target;
    const std::int64_t target_begin =
        power_plan.output_offsets.at(static_cast<std::size_t>(selected_component));
    const std::int64_t target_end =
        power_plan.output_offsets.at(static_cast<std::size_t>(selected_component + 1));
    for (std::int64_t index = target_begin; index < target_end; ++index) {
      const std::int64_t term = power_plan.coefficient_terms.at(static_cast<std::size_t>(index));
      ScalarExponent exponents{};
      for (std::int64_t component = 0; component < 3; ++component)
        exponents[static_cast<std::size_t>(component)] = power_plan.monomial_counts.at(
            static_cast<std::size_t>(term * power_plan.input_dimension + component));
      target[exponents] += power_plan.coefficient_values.at(static_cast<std::size_t>(index));
    }
    const auto intrinsic_scale =
        complex_number(certificate["intrinsic_output_scale"], path + ".intrinsic_output_scale");
    if (std::abs(intrinsic_scale) <= 1.0e-30 || std::abs(intrinsic_scale.imag()) > 2.0e-12)
      fail(path, "scalar-invariant-power intrinsic scale is unsupported");
    const ScalarPolynomial reconstructed = scalar_polynomial_power(quadratic, outer_power);
    if (target.size() != reconstructed.size())
      fail(path, "scalar-invariant-power coefficient support differs");
    double maximum_absolute_residual = 0.0;
    double maximum_relative_residual = 0.0;
    double maximum_scaled_residual = 0.0;
    for (const auto &[exponents, target_value] : target) {
      const auto found_term = reconstructed.find(exponents);
      if (found_term == reconstructed.end())
        fail(path, "scalar-invariant-power coefficient support differs");
      const std::complex<double> expected = intrinsic_scale * found_term->second;
      const double residual = std::abs(target_value - expected);
      const double scale = std::max(std::abs(target_value), std::abs(expected));
      const double gate = absolute_tolerance + relative_tolerance * scale;
      maximum_absolute_residual = std::max(maximum_absolute_residual, residual);
      maximum_relative_residual =
          std::max(maximum_relative_residual, residual / std::max(scale, absolute_tolerance));
      maximum_scaled_residual = std::max(maximum_scaled_residual, residual / gate);
      if (residual > gate) fail(path, "scalar-invariant-power coefficient identity failed");
    }
    const auto residual_matches = [](double supplied, double recomputed) {
      return std::abs(supplied - recomputed) <=
          2.0e-15 + 2.0e-14 * std::max(std::abs(supplied), std::abs(recomputed));
    };
    if (!residual_matches(reported_absolute_residual, maximum_absolute_residual) ||
        !residual_matches(reported_relative_residual, maximum_relative_residual) ||
        !residual_matches(reported_scaled_residual, maximum_scaled_residual))
      fail(path, "scalar-invariant-power residual report is inconsistent");

    ScalarRouteCandidate result;
    result.base = std::move(base);
    result.schedule = expected_schedule;
    result.outer_power = outer_power;
    result.function_index = mapped.function_index;
    result.scale = output_scale * intrinsic_scale;
    result.factorization_id = mapped.scalar_factorization_id;
    result.operation_estimate = 5 * static_cast<std::int64_t>(result.base.coefficients.size()) +
        3 * static_cast<std::int64_t>(result.schedule.size()) + 2;
    return result;
  }

  YACEAutoReplay load_auto_replay(const std::string &path, const std::string &model_path,
                                  const std::string &manifest_path)
  {
    const YAML::Node root = YAML::LoadFile(path);
    require_exact_fields(root,
                         {"calibration", "canonical_encoding", "compiler", "device",
                          "replay_sha256", "runtime", "schema", "selections", "source_yace_sha256",
                          "workload"},
                         "auto_replay");
    if (scalar_string(root["schema"], "auto_replay.schema") != "ye3t_kokkos_auto_replay_v2")
      fail("auto_replay.schema", "unsupported AUTO replay schema");
    if (scalar_string(root["canonical_encoding"], "auto_replay.canonical_encoding") !=
        "ye3t_sorted_json_indent2_lf_v1")
      fail("auto_replay.canonical_encoding", "unsupported canonical encoding");

    YACEAutoReplay result;
    result.path = path;
    result.replay_hash = scalar_string(root["replay_sha256"], "auto_replay.replay_sha256");
    if (!lower_sha256(result.replay_hash) ||
        canonical_json_hash_without_root_member(path, "replay_sha256") != result.replay_hash)
      fail("auto_replay.replay_sha256", "canonical replay self-hash does not match");
    result.source_yace_hash =
        scalar_string(root["source_yace_sha256"], "auto_replay.source_yace_sha256");
    if (!lower_sha256(result.source_yace_hash) ||
        result.source_yace_hash != sha256_file(model_path))
      fail("auto_replay.source_yace_sha256", "replay does not bind the supplied YACE bytes");

    const YAML::Node calibration = root["calibration"];
    require_exact_fields(calibration,
                         {"evidence_sha256", "method", "selected_evaluator", "selection_rule"},
                         "auto_replay.calibration");
    result.calibration_evidence_hash =
        scalar_string(calibration["evidence_sha256"], "auto_replay.calibration.evidence_sha256");
    result.calibration_method =
        scalar_string(calibration["method"], "auto_replay.calibration.method");
    result.selected_evaluator = scalar_string(calibration["selected_evaluator"],
                                              "auto_replay.calibration.selected_evaluator");
    result.selection_rule =
        scalar_string(calibration["selection_rule"], "auto_replay.calibration.selection_rule");
    if (!lower_sha256(result.calibration_evidence_hash) ||
        result.calibration_method != "interleaved_paired_bootstrap_v1" ||
        (result.selected_evaluator != "direct" && result.selected_evaluator != "block" &&
         result.selected_evaluator != "scalar_power" &&
         result.selected_evaluator != "coupled_product") ||
        result.selection_rule != "worst_case_lower_confidence_across_workload_grid_v1")
      fail("auto_replay.calibration", "unsupported calibration evidence or method");

    const YAML::Node compiler = root["compiler"];
    require_exact_fields(compiler, {"plan_hash", "sidecar_manifest_sha256"},
                         "auto_replay.compiler");
    result.compiler_plan_hash =
        scalar_string(compiler["plan_hash"], "auto_replay.compiler.plan_hash");
    result.sidecar_manifest_hash = scalar_string(compiler["sidecar_manifest_sha256"],
                                                 "auto_replay.compiler.sidecar_manifest_sha256");
    if (!lower_sha256(result.compiler_plan_hash) || !lower_sha256(result.sidecar_manifest_hash) ||
        result.sidecar_manifest_hash != sha256_file(manifest_path))
      fail("auto_replay.compiler", "replay compiler or sidecar identity does not match");

    const YAML::Node device = root["device"];
    require_exact_fields(device, {"device_class_sha256", "execution_space"}, "auto_replay.device");
    result.device_class_hash =
        scalar_string(device["device_class_sha256"], "auto_replay.device.device_class_sha256");
    result.execution_space =
        scalar_string(device["execution_space"], "auto_replay.device.execution_space");
    if (!lower_sha256(result.device_class_hash) ||
        (result.execution_space != "Cuda" && result.execution_space != "HIP"))
      fail("auto_replay.device", "unsupported device identity");

    const YAML::Node runtime = root["runtime"];
    require_exact_fields(runtime, {"lammps_executable_sha256"}, "auto_replay.runtime");
    result.lammps_executable_hash = scalar_string(runtime["lammps_executable_sha256"],
                                                  "auto_replay.runtime.lammps_executable_sha256");
    if (!lower_sha256(result.lammps_executable_hash))
      fail("auto_replay.runtime", "invalid LAMMPS executable identity");

    const YAML::Node workload = root["workload"];
    require_exact_fields(workload,
                         {"block_schedule", "block_scratch_bytes", "block_scratch_layout",
                          "block_team_size", "chunksize", "kernel_abi", "layout",
                          "maximum_centers_per_rank", "minimum_centers_per_rank", "precision",
                          "source_policy", "vjp_policy"},
                         "auto_replay.workload");
    result.block_schedule =
        scalar_string(workload["block_schedule"], "auto_replay.workload.block_schedule");
    result.layout = scalar_string(workload["layout"], "auto_replay.workload.layout");
    result.block_scratch_layout = scalar_string(workload["block_scratch_layout"],
                                                "auto_replay.workload.block_scratch_layout");
    result.kernel_abi = scalar_string(workload["kernel_abi"], "auto_replay.workload.kernel_abi");
    result.precision = scalar_string(workload["precision"], "auto_replay.workload.precision");
    result.source_policy =
        scalar_string(workload["source_policy"], "auto_replay.workload.source_policy");
    result.vjp_policy = scalar_string(workload["vjp_policy"], "auto_replay.workload.vjp_policy");
    result.chunksize = integer(workload["chunksize"], "auto_replay.workload.chunksize", 1);
    result.block_team_size =
        integer(workload["block_team_size"], "auto_replay.workload.block_team_size", 0);
    result.block_scratch_bytes =
        exact_integer(workload["block_scratch_bytes"], "auto_replay.workload.block_scratch_bytes");
    result.minimum_centers_per_rank = integer(workload["minimum_centers_per_rank"],
                                              "auto_replay.workload.minimum_centers_per_rank", 1);
    result.maximum_centers_per_rank = integer(workload["maximum_centers_per_rank"],
                                              "auto_replay.workload.maximum_centers_per_rank", 1);
    const bool supported_source_policy = result.source_policy == "center_tiled_neighbor_major_v1" ||
        result.source_policy == "edge_team_atomic_v1" ||
        result.source_policy == "component_reference_v1";
    const bool supported_vjp_policy =
        result.vjp_policy == "edge_grouped_harmonic_cached_radial_v1" ||
        result.vjp_policy == "edge_reference_v1";
    const bool source_vjp_pair_is_consistent = (result.source_policy == "component_reference_v1") ==
        (result.vjp_policy == "edge_reference_v1");
    if ((result.block_schedule != "fused_center_lane_v2" &&
         result.block_schedule != "plan_route_major_v1" &&
         result.block_schedule != "not_selected") ||
        (result.layout != "legacy" && result.layout != "default") ||
        result.kernel_abi != "ye3t_kokkos_candidate_runtime_v2" || result.precision != "fp64" ||
        !supported_source_policy || !supported_vjp_policy || !source_vjp_pair_is_consistent ||
        result.block_scratch_bytes < 0 ||
        result.maximum_centers_per_rank < result.minimum_centers_per_rank)
      fail("auto_replay.workload", "unsupported replay workload contract");
    if (result.block_schedule == "plan_route_major_v1") {
      if ((result.block_team_size != 32 && result.block_team_size != 64 &&
           result.block_team_size != 128 && result.block_team_size != 256) ||
          result.block_scratch_layout != "monomial_tile16_lane_fast_v2" ||
          result.block_scratch_bytes == 0)
        fail("auto_replay.workload", "work-major replay has an invalid team or scratch contract");
    } else if (result.block_team_size != 0 || result.block_scratch_bytes != 0 ||
               result.block_scratch_layout != "not_selected") {
      fail("auto_replay.workload", "non-work-major replay must not reserve team scratch");
    }

    const YAML::Node selections = root["selections"];
    require_sequence(selections, "auto_replay.selections");
    if (selections.size() == 0 || selections.size() > 1000000)
      fail("auto_replay.selections", "selection count is outside the supported bound");
    std::pair<int, int> previous{-1, -1};
    for (std::size_t index = 0; index < selections.size(); ++index) {
      const YAML::Node record = selections[index];
      const std::string record_path = "auto_replay.selections[" + std::to_string(index) + "]";
      require_exact_fields(
          record, {"candidate_id", "central_type", "element", "feature_id", "function_index"},
          record_path);
      YACEAutoReplaySelection selection;
      selection.candidate_id = scalar_string(record["candidate_id"], record_path + ".candidate_id");
      selection.central_species = integer(record["central_type"], record_path + ".central_type", 0);
      selection.element = scalar_string(record["element"], record_path + ".element");
      selection.feature_id = scalar_string(record["feature_id"], record_path + ".feature_id");
      selection.function_index =
          integer(record["function_index"], record_path + ".function_index", 0);
      const std::pair<int, int> coordinate{selection.central_species, selection.function_index};
      if (!lower_sha256(selection.candidate_id) || !lower_sha256(selection.feature_id) ||
          selection.element.empty() || coordinate <= previous)
        fail(record_path, "selection IDs or strictly ordered function coordinate are invalid");
      previous = coordinate;
      result.selections.push_back(std::move(selection));
    }
    return result;
  }

  void apply_block_sidecar(const std::string &model_path, const std::string &manifest_path,
                           YACEBlockPolicy policy, const YACEAutoReplay *auto_replay,
                           std::vector<YACESpecies> &species)
  {
    const YAML::Node manifest = YAML::LoadFile(manifest_path);
    require_mapping(manifest, "sidecar");
    const std::string sidecar_schema = scalar_string(manifest["schema"], "sidecar.schema");
    const bool legacy_portfolio_schema = sidecar_schema == "ye3t_lammps_sidecar_v2";
    const bool candidate_vector_schema = sidecar_schema == "ye3t_lammps_sidecar_v3";
    const bool portfolio_schema = legacy_portfolio_schema || candidate_vector_schema;
    if (auto_replay != nullptr && (policy != YACEBlockPolicy::GPU_AUTO || !candidate_vector_schema))
      fail("auto_replay", "device replay requires GPU AUTO and a v3 candidate sidecar");
    const YAML::Node compiler = manifest["compiler"];
    require_mapping(compiler, "sidecar.compiler");
    const std::string compiler_plan_schema =
        scalar_string(compiler["execution_plan_schema"], "sidecar.compiler.execution_plan_schema");
    const bool coupled_product_schema = compiler_plan_schema == "ye3t_execution_plan_v3";
    if (!portfolio_schema && sidecar_schema != "ye3t_lammps_sidecar_v1")
      fail("sidecar.schema", "unsupported sidecar schema");
    if (coupled_product_schema && !candidate_vector_schema)
      fail("sidecar.schema", "coupled-product plans require a v3 candidate-vector sidecar");
    if ((policy == YACEBlockPolicy::AUTO || policy == YACEBlockPolicy::GPU_AUTO) &&
        !portfolio_schema)
      fail("sidecar",
           "automatic route selection requires explicit alternative "
           "routes in a v2 sidecar");
    if (scalar_string(manifest["canonical_encoding"], "sidecar.canonical_encoding") !=
        "ye3t_sorted_json_indent2_lf_v1")
      fail("sidecar.canonical_encoding", "unsupported canonical encoding");
    if (scalar_string(manifest["semantic_ledger"], "sidecar.semantic_ledger") !=
        "task55_semantic_ledger_v1")
      fail("sidecar.semantic_ledger", "unsupported semantic ledger");

    const std::set<std::string> supported_capabilities{"block_symmetric_power_adjoint_v1",
                                                       "block_symmetric_power_forward_v1",
                                                       "ace_coupled_product_dag_adjoint_v1",
                                                       "ace_coupled_product_dag_forward_v1",
                                                       "energy_v1",
                                                       "evaluator_candidate_vector_v1",
                                                       "evaluator_portfolio_v1",
                                                       "explicit_ctilde_forward_adjoint_v1",
                                                       "force_v1",
                                                       "linear_readout_v1",
                                                       "ordinary_density_v1",
                                                       "pace_cheb_exp_cos_uniform_cubic_hermite_v1",
                                                       "pace_complex_magnetic_y00_1",
                                                       "pace_linear_embedding_v1",
                                                       "symmetric_power_adjoint_v1",
                                                       "symmetric_power_forward_v1",
                                                       "scalar_invariant_power_adjoint_v1",
                                                       "scalar_invariant_power_forward_v1",
                                                       "virial_v1",
                                                       "yace_candidate_readout_v1",
                                                       "ye3t_execution_plan_v2",
                                                       "ye3t_execution_plan_v3"};
    const YAML::Node required = manifest["capabilities"]["required"];
    require_sequence(required, "sidecar.capabilities.required");
    std::set<std::string> declared_capabilities;
    for (std::size_t index = 0; index < required.size(); ++index) {
      const std::string capability =
          scalar_string(required[index], "sidecar.capabilities.required");
      if (supported_capabilities.count(capability) == 0)
        fail("sidecar.capabilities.required", "unsupported capability '" + capability + "'");
      if (!declared_capabilities.insert(capability).second)
        fail("sidecar.capabilities.required", "duplicate required capability");
    }
    const std::vector<std::string> runtime_capabilities = coupled_product_schema
        ? std::vector<std::string>{"ace_coupled_product_dag_adjoint_v1",
                                   "ace_coupled_product_dag_forward_v1", "ordinary_density_v1",
                                   "ye3t_execution_plan_v3"}
        : std::vector<std::string>{"block_symmetric_power_adjoint_v1",
                                   "block_symmetric_power_forward_v1", "ordinary_density_v1",
                                   "ye3t_execution_plan_v2"};
    for (const std::string &capability : runtime_capabilities)
      if (declared_capabilities.count(capability) == 0)
        fail("sidecar.capabilities.required", "missing required compiled-runtime capability");
    if (legacy_portfolio_schema && declared_capabilities.count("evaluator_portfolio_v1") == 0)
      fail("sidecar.capabilities.required", "v2 sidecars require evaluator_portfolio_v1");
    if (candidate_vector_schema &&
        (declared_capabilities.count("evaluator_candidate_vector_v1") == 0 ||
         declared_capabilities.count("yace_candidate_readout_v1") == 0))
      fail("sidecar.capabilities.required",
           "v3 sidecars require candidate-vector and readout capabilities");

    const YAML::Node conventions = manifest["conventions"];
    require_mapping(conventions, "sidecar.conventions");
    if (scalar_string(conventions["analysis_orientation"], "sidecar.analysis_orientation") !=
            "conjugate_transpose" ||
        scalar_string(conventions["angular_source_basis"], "sidecar.angular_source_basis") !=
            "pace_complex_magnetic_y00_1" ||
        scalar_string(conventions["atomic_base_normalization"],
                      "sidecar.atomic_base_normalization") != "none" ||
        scalar_string(conventions["factor_normalization"], "sidecar.factor_normalization") !=
            "none" ||
        scalar_string(conventions["radial_basis"], "sidecar.radial_basis") !=
            "pace_cheb_exp_cos_uniform_cubic_hermite_v1" ||
        scalar_string(conventions["scalar_type"], "sidecar.scalar_type") != "float64" ||
        scalar_string(conventions["byte_order"], "sidecar.byte_order") != "little" ||
        scalar_string(conventions["angular_transform"]["kind"], "sidecar.angular_transform") !=
            "none" ||
        scalar_string(conventions["linear_feature_transform"]["kind"],
                      "sidecar.linear_feature_transform") != "none")
      fail("sidecar.conventions", "sidecar conventions do not match the CPU compatibility path");
    const std::string requested_route =
        scalar_string(manifest["dispatch"]["requested_route"], "sidecar.dispatch");
    const std::string fallback_policy =
        scalar_string(manifest["dispatch"]["fallback_policy"], "sidecar.dispatch");
    if ((!portfolio_schema &&
         (requested_route != "ye3t_execution_plan_v2" || fallback_policy != "forbid")) ||
        (legacy_portfolio_schema &&
         (requested_route != "ye3t_evaluator_portfolio_v1" ||
          fallback_policy != "forbid_unlisted")) ||
        (candidate_vector_schema &&
         (requested_route != "ye3t_evaluator_candidate_vector_v1" ||
          fallback_policy != "forbid_unlisted")))
      fail("sidecar.dispatch", "unsupported requested route or fallback policy");

    const std::string expected_compiler_api = coupled_product_schema
        ? "ye3t.couplings.compile_execution_plan"
        : "ye3t.couplings.execution_plan_from_repeated_angular_blocks";
    if (scalar_string(compiler["api"], "sidecar.compiler.api") != expected_compiler_api ||
        (!coupled_product_schema && compiler_plan_schema != "ye3t_execution_plan_v2") ||
        scalar_string(compiler["convention_id"], "sidecar.compiler.convention_id") !=
            "o3:complex_condon_shortley_young_orthogonal_v1")
      fail("sidecar.compiler", "unsupported compiler contract");

    const std::string source_hash = sha256_file(model_path);
    const YAML::Node source = manifest["source_yace"];
    if (scalar_string(source["sha256"], "sidecar.source_yace.sha256") != source_hash ||
        scalar_string(source["compatibility_profile"],
                      "sidecar.source_yace.compatibility_profile") != "lammps_pace_linear_v1" ||
        scalar_string(source["lammps_units"], "sidecar.source_yace.lammps_units") != "metal")
      fail("sidecar.source_yace", "sidecar does not bind the supplied YACE bytes");
    const YAML::Node ordered_elements = source["ordered_elements"];
    require_sequence(ordered_elements, "sidecar.source_yace.ordered_elements");
    if (ordered_elements.size() != species.size())
      fail("sidecar.source_yace.ordered_elements", "element count does not match YACE");
    for (std::size_t index = 0; index < species.size(); ++index)
      if (scalar_string(ordered_elements[index], "sidecar.source_yace.ordered_elements") !=
          species[index].element)
        fail("sidecar.source_yace.ordered_elements", "ordered elements do not match YACE");

    const YAML::Node plan_payload = manifest["payloads"]["execution_plan"];
    const YAML::Node map_payload = manifest["payloads"]["yace_function_map"];
    const std::string expected_map_schema = candidate_vector_schema
        ? "ye3t_yace_function_map_v3"
        : (legacy_portfolio_schema ? "ye3t_yace_function_map_v2" : "ye3t_yace_function_map_v1");
    const std::string plan_path =
        payload_path(manifest_path, plan_payload, "sidecar.payloads.execution_plan");
    const std::string map_path =
        payload_path(manifest_path, map_payload, "sidecar.payloads.yace_function_map");
    if (scalar_string(plan_payload["schema"], "sidecar.execution_plan.schema") !=
            compiler_plan_schema ||
        scalar_string(map_payload["schema"], "sidecar.yace_function_map.schema") !=
            expected_map_schema ||
        sha256_file(plan_path) !=
            scalar_string(plan_payload["sha256"], "sidecar.execution_plan.sha256") ||
        sha256_file(map_path) !=
            scalar_string(map_payload["sha256"], "sidecar.yace_function_map.sha256"))
      fail("sidecar.payloads", "sidecar payload schema or SHA-256 mismatch");

    CanonicalExecutionPlanHashes computed_plan_hashes;
    if (candidate_vector_schema) computed_plan_hashes = canonical_execution_plan_hashes(plan_path);
    const YAML::Node plan = YAML::LoadFile(plan_path);
    const YAML::Node mapping = YAML::LoadFile(map_path);
    if (scalar_string(plan["schema_version"], "execution_plan.schema_version") !=
            compiler_plan_schema ||
        scalar_string(plan["convention_id"], "execution_plan.convention_id") !=
            "o3:complex_condon_shortley_young_orthogonal_v1" ||
        scalar_string(mapping["schema"], "function_map.schema") != expected_map_schema)
      fail("sidecar", "loaded payload schema mismatch");
    if (portfolio_schema) {
      require_exact_fields(mapping,
                           {"catalogue_hash", "coverage", "entries", "plan_hash", "schema",
                            "selection_contract", "source_yace_sha256"},
                           "function_map");
      const YAML::Node contract = mapping["selection_contract"];
      require_exact_fields(contract, {"fallback_policy", "mode", "schema"},
                           "function_map.selection_contract");
      const std::string expected_selection_schema = candidate_vector_schema
          ? "ye3t_evaluator_candidates_v2"
          : "ye3t_evaluator_alternatives_v1";
      if (scalar_string(contract["schema"], "function_map.selection_contract.schema") !=
              expected_selection_schema ||
          scalar_string(contract["mode"], "function_map.selection_contract.mode") !=
              "load_time_portfolio_allowed" ||
          scalar_string(contract["fallback_policy"],
                        "function_map.selection_contract.fallback_policy") != "forbid_unlisted")
        fail("function_map.selection_contract",
             "unsupported evaluator-alternative selection contract");
    }
    const std::string plan_hash = scalar_string(plan["plan_hash"], "execution_plan.plan_hash");
    const std::string coefficient_hash =
        scalar_string(plan["coefficient_hash"], "execution_plan.coefficient_hash");
    if (candidate_vector_schema && computed_plan_hashes.coefficient_hash != coefficient_hash)
      fail("execution_plan.coefficient_hash",
           "native reconstruction does not match the synthesis tables");
    if (candidate_vector_schema && computed_plan_hashes.plan_hash != plan_hash)
      fail("execution_plan.plan_hash", "native reconstruction does not match the plan payload");
    if (plan_hash != scalar_string(mapping["plan_hash"], "function_map.plan_hash") ||
        plan_hash !=
            scalar_string(manifest["compiler"]["plan_hash"], "sidecar.compiler.plan_hash") ||
        coefficient_hash !=
            scalar_string(manifest["compiler"]["coefficient_hash"],
                          "sidecar.compiler.coefficient_hash") ||
        scalar_string(mapping["source_yace_sha256"], "function_map.source_yace_sha256") !=
            source_hash)
      fail("sidecar", "plan, map, compiler, or source hashes disagree");
    if (auto_replay != nullptr && auto_replay->compiler_plan_hash != plan_hash)
      fail("auto_replay.compiler.plan_hash", "replay does not bind the loaded compiler plan");
    if (!coupled_product_schema &&
        (!boolean(plan["certificate"]["passed"], "execution_plan.certificate.passed") ||
         !boolean(plan["certificate"]["all_paths_compiled_before_runtime"],
                  "execution_plan.certificate.all_paths_compiled_before_runtime")))
      fail("execution_plan.certificate", "compiler certificate did not pass");
    std::set<std::string> certified_readout_ids;
    if (candidate_vector_schema) {
      const YAML::Node ledger = plan["certificate"]["yace_candidate_readouts"];
      require_exact_fields(ledger, {"records", "schema"},
                           "execution_plan.certificate.yace_candidate_readouts");
      if (scalar_string(ledger["schema"],
                        "execution_plan.certificate.yace_candidate_readouts.schema") !=
          "ye3t_yace_candidate_readouts_v1")
        fail("execution_plan.certificate.yace_candidate_readouts",
             "unsupported candidate-readout ledger");
      const YAML::Node records = ledger["records"];
      require_sequence(records, "execution_plan.certificate.yace_candidate_readouts.records");
      for (std::size_t index = 0; index < records.size(); ++index) {
        const std::string readout_id =
            scalar_string(records[index]["readout_id"],
                          "execution_plan.certificate.yace_candidate_readouts.readout_id");
        if (!lower_sha256(readout_id) || !certified_readout_ids.insert(readout_id).second)
          fail("execution_plan.certificate.yace_candidate_readouts",
               "candidate-readout IDs must be unique SHA-256 values");
      }
    }
    if (scalar_string(mapping["coverage"], "function_map.coverage") != "complete")
      fail("function_map.coverage", "function map must cover the complete YACE catalogue");

    std::vector<std::vector<bool>> seen;
    std::vector<std::vector<std::string>> direct_candidate_ids;
    std::vector<std::vector<std::string>> feature_ids;
    std::size_t total_function_count = 0;
    for (const auto &entry : species) {
      const std::size_t count = entry.polynomial.descriptor_offsets.size() - 1;
      seen.emplace_back(count, false);
      direct_candidate_ids.emplace_back(count);
      feature_ids.emplace_back(count);
      total_function_count += count;
    }
    if (auto_replay != nullptr && auto_replay->selections.size() != total_function_count)
      fail("auto_replay.selections", "replay does not cover the complete YACE catalogue");
    const YAML::Node entries = mapping["entries"];
    require_sequence(entries, "function_map.entries");
    if (entries.size() != total_function_count)
      fail("function_map.entries", "complete function map has the wrong length");
    std::vector<MappedBlockDescriptor> mapped_entries;
    std::vector<CoupledProductRouteCandidate> coupled_product_candidates;
    std::set<std::string> alternative_ids;
    std::set<std::string> used_readout_ids;
    bool has_explicit_ctilde = false;
    for (std::size_t index = 0; index < entries.size(); ++index) {
      const YAML::Node entry = entries[index];
      const std::string path = "function_map.entries[" + std::to_string(index) + "]";
      const int central = integer(entry["central_type"], path + ".central_type", 0);
      const int function = integer(entry["function_index"], path + ".function_index", 0);
      if (central >= static_cast<int>(species.size()) ||
          function >= static_cast<int>(seen[static_cast<std::size_t>(central)].size()) ||
          seen[static_cast<std::size_t>(central)][static_cast<std::size_t>(function)])
        fail(path, "duplicate or out-of-range function coordinate");
      seen[static_cast<std::size_t>(central)][static_cast<std::size_t>(function)] = true;
      MappedBlockDescriptor mapped;
      mapped.central_species = central;
      mapped.function_index = function;
      mapped.feature_id = scalar_string(entry["feature_id"], path + ".feature_id");
      feature_ids[static_cast<std::size_t>(central)][static_cast<std::size_t>(function)] =
          mapped.feature_id;
      if (!portfolio_schema) {
        const std::string dispatch = scalar_string(entry["dispatch"], path + ".dispatch");
        if (dispatch == "explicit_ctilde") {
          if (complex_number(entry["scale"], path + ".scale") != std::complex<double>(1.0, 0.0) ||
              (entry["plan_output"] && !entry["plan_output"].IsNull()))
            fail(path,
                 "explicit C-tilde entries require identity scale and "
                 "null plan output");
          has_explicit_ctilde = true;
          direct_candidate_ids[static_cast<std::size_t>(central)][static_cast<std::size_t>(
              function)] = "v1-direct-" + std::to_string(central) + "-" + std::to_string(function);
          continue;
        }
        if (dispatch != "execution_plan_output") fail(path, "unsupported function dispatch");
        const YAML::Node output = entry["plan_output"];
        require_mapping(output, path + ".plan_output");
        mapped.instruction_id =
            scalar_string(output["instruction_id"], path + ".plan_output.instruction_id");
        mapped.channel_index =
            integer(output["channel_index"], path + ".plan_output.channel_index", 0);
        mapped.tableau_index =
            integer(output["tableau_index"], path + ".plan_output.tableau_index", 0);
        mapped.magnetic_index =
            integer(output["magnetic_index"], path + ".plan_output.magnetic_index", 0);
        mapped.scale = complex_number(entry["scale"], path + ".scale");
        mapped.direct_candidate_id =
            "v1-direct-" + std::to_string(central) + "-" + std::to_string(function);
        direct_candidate_ids[static_cast<std::size_t>(central)]
                            [static_cast<std::size_t>(function)] = mapped.direct_candidate_id;
        mapped.block_candidate_id =
            "v1-block-" + mapped.instruction_id + "-" + std::to_string(mapped.channel_index);
        validate_output_binding(plan, mapped, source_hash, path);
        mapped_entries.push_back(std::move(mapped));
        continue;
      }

      if (candidate_vector_schema) {
        if (!lower_sha256(mapped.feature_id)) fail(path, "v3 feature ID must be a SHA-256");
        require_exact_fields(
            entry, {"alternatives", "central_type", "dispatch", "feature_id", "function_index"},
            path);
        if (scalar_string(entry["dispatch"], path + ".dispatch") != "listed_candidates")
          fail(path, "v3 function-map rows must list evaluator candidates");
        const YAML::Node alternatives = entry["alternatives"];
        require_sequence(alternatives, path + ".alternatives");
        if (alternatives.size() == 0 || alternatives.size() > 64)
          fail(path, "v3 candidate count is outside the supported bound");
        bool direct_found = false;
        std::vector<MappedBlockDescriptor> row_mapped;
        for (std::size_t alternative_index = 0; alternative_index < alternatives.size();
             ++alternative_index) {
          const YAML::Node alternative = alternatives[alternative_index];
          const std::string alternative_path =
              path + ".alternatives[" + std::to_string(alternative_index) + "]";
          const std::string evaluator =
              scalar_string(alternative["evaluator"], alternative_path + ".evaluator");
          const std::string alternative_id =
              scalar_string(alternative["alternative_id"], alternative_path + ".alternative_id");
          if (!lower_sha256(alternative_id) || !alternative_ids.insert(alternative_id).second)
            fail(alternative_path, "candidate ID must be a globally unique SHA-256");
          const YAML::Node availability = alternative["availability"];
          require_exact_fields(availability, {"reason", "status"},
                               alternative_path + ".availability");
          const std::string availability_status =
              scalar_string(availability["status"], alternative_path + ".availability.status");
          const std::string availability_reason =
              scalar_string(availability["reason"], alternative_path + ".availability.reason");
          if (availability_status != "available" || !availability_reason.empty())
            fail(alternative_path,
                 "v3 runtime accepts compiler-certified "
                 "available candidates with an empty reason "
                 "only");
          const YAML::Node required_capabilities = alternative["required_capabilities"];
          require_sequence(required_capabilities, alternative_path + ".required_capabilities");
          std::set<std::string> alternative_capabilities;
          for (std::size_t capability_index = 0; capability_index < required_capabilities.size();
               ++capability_index) {
            const std::string capability =
                scalar_string(required_capabilities[capability_index],
                              alternative_path + ".required_capabilities");
            if (!alternative_capabilities.insert(capability).second ||
                declared_capabilities.count(capability) == 0)
              fail(alternative_path, "candidate capabilities are duplicate or undeclared");
          }
          if (evaluator == "explicit_ctilde") {
            require_exact_fields(alternative,
                                 {"alternative_id", "availability", "evaluator",
                                  "required_capabilities", "scale", "source_binding"},
                                 alternative_path);
            if (direct_found ||
                alternative_capabilities !=
                    std::set<std::string>{"explicit_ctilde_forward_adjoint_v1"} ||
                complex_number(alternative["scale"], alternative_path + ".scale") !=
                    std::complex<double>(1.0, 0.0))
              fail(alternative_path, "each v3 row requires one identity direct candidate");
            const YAML::Node source_binding = alternative["source_binding"];
            require_exact_fields(
                source_binding,
                {"central_type", "feature_id", "function_index", "source_yace_sha256"},
                alternative_path + ".source_binding");
            if (integer(source_binding["central_type"],
                        alternative_path + ".source_binding.central_type", 0) != central ||
                integer(source_binding["function_index"],
                        alternative_path + ".source_binding.function_index", 0) != function ||
                scalar_string(source_binding["feature_id"],
                              alternative_path + ".source_binding.feature_id") !=
                    mapped.feature_id ||
                scalar_string(source_binding["source_yace_sha256"],
                              alternative_path + ".source_binding.source_yace_sha256") !=
                    source_hash ||
                evaluator_candidate_identity(alternative, alternative_capabilities,
                                             alternative_path) != alternative_id)
              fail(alternative_path, "direct candidate is not bound to this YACE function");
            mapped.direct_candidate_id = alternative_id;
            direct_found = true;
            has_explicit_ctilde = true;
            continue;
          }
          if (evaluator != "execution_plan_readout")
            fail(alternative_path, "unsupported v3 evaluator candidate");
          require_exact_fields(alternative,
                               {"alternative_id", "availability", "compiler_plan_hash", "evaluator",
                                "readout_id", "required_capabilities"},
                               alternative_path);
          const std::set<std::string> block_capabilities{"block_symmetric_power_adjoint_v1",
                                                         "block_symmetric_power_forward_v1",
                                                         "yace_candidate_readout_v1"};
          const std::set<std::string> coupled_capabilities{"ace_coupled_product_dag_adjoint_v1",
                                                           "ace_coupled_product_dag_forward_v1",
                                                           "yace_candidate_readout_v1"};
          if ((alternative_capabilities != block_capabilities &&
               alternative_capabilities != coupled_capabilities) ||
              scalar_string(alternative["compiler_plan_hash"],
                            alternative_path + ".compiler_plan_hash") != plan_hash ||
              evaluator_candidate_identity(alternative, alternative_capabilities,
                                           alternative_path) != alternative_id)
            fail(alternative_path, "compiled candidate capabilities or plan do not match");
          const std::string readout_id =
              scalar_string(alternative["readout_id"], alternative_path + ".readout_id");
          if (!used_readout_ids.insert(readout_id).second)
            fail(alternative_path, "candidate readout must be referenced exactly once");
          auto terms = candidate_readout_terms(plan, mapped, alternative_id, readout_id,
                                               source_hash, alternative_path);
          if (alternative_capabilities == coupled_capabilities) {
            if (!coupled_product_schema || !direct_found || terms.size() != 1)
              fail(alternative_path,
                   "coupled-product candidate requires one readout after its "
                   "direct fallback");
            terms[0].direct_candidate_id = mapped.direct_candidate_id;
            CoupledProductRouteCandidate candidate;
            candidate.central_species = central;
            candidate.function_index = function;
            candidate.direct_candidate_id = mapped.direct_candidate_id;
            candidate.candidate_id = alternative_id;
            candidate.plan = parse_coupled_product_dag_plan(plan, terms[0], species, source_hash,
                                                            alternative_path);
            coupled_product_candidates.push_back(std::move(candidate));
            continue;
          }
          if (coupled_product_schema)
            fail(alternative_path, "plan-v3 sidecars cannot lower legacy block candidates");
          row_mapped.insert(row_mapped.end(), std::make_move_iterator(terms.begin()),
                            std::make_move_iterator(terms.end()));
        }
        if (!direct_found) fail(path, "v3 function-map row has no available direct candidate");
        direct_candidate_ids[static_cast<std::size_t>(central)]
                            [static_cast<std::size_t>(function)] = mapped.direct_candidate_id;
        for (auto &term : row_mapped) {
          term.direct_candidate_id = mapped.direct_candidate_id;
          mapped_entries.push_back(std::move(term));
        }
        continue;
      }

      require_exact_fields(
          entry, {"alternatives", "central_type", "dispatch", "feature_id", "function_index"},
          path);
      if (scalar_string(entry["dispatch"], path + ".dispatch") != "listed_alternatives")
        fail(path, "v2 function-map rows must list evaluator alternatives");
      const YAML::Node alternatives = entry["alternatives"];
      require_sequence(alternatives, path + ".alternatives");
      bool direct_found = false;
      bool block_found = false;
      bool scalar_found = false;
      std::string scalar_instruction_id;
      int scalar_channel_index = -1;
      int scalar_tableau_index = -1;
      int scalar_magnetic_index = -1;
      std::complex<double> scalar_scale;
      for (std::size_t alternative_index = 0; alternative_index < alternatives.size();
           ++alternative_index) {
        const YAML::Node alternative = alternatives[alternative_index];
        const std::string alternative_path =
            path + ".alternatives[" + std::to_string(alternative_index) + "]";
        const std::string evaluator =
            scalar_string(alternative["evaluator"], alternative_path + ".evaluator");
        const std::string alternative_id =
            scalar_string(alternative["alternative_id"], alternative_path + ".alternative_id");
        if (alternative_id.empty() || !alternative_ids.insert(alternative_id).second)
          fail(alternative_path, "alternative_id must be nonempty and globally unique");
        const YAML::Node availability = alternative["availability"];
        require_exact_fields(availability, {"reason", "status"},
                             alternative_path + ".availability");
        if (scalar_string(availability["status"], alternative_path + ".availability.status") !=
            "available")
          fail(alternative_path, "the first portfolio runtime accepts available alternatives only");
        (void) scalar_string(availability["reason"], alternative_path + ".availability.reason");
        const YAML::Node required_capabilities = alternative["required_capabilities"];
        require_sequence(required_capabilities, alternative_path + ".required_capabilities");
        std::set<std::string> alternative_capabilities;
        for (std::size_t capability_index = 0; capability_index < required_capabilities.size();
             ++capability_index) {
          const std::string capability = scalar_string(required_capabilities[capability_index],
                                                       alternative_path + ".required_capabilities");
          if (!alternative_capabilities.insert(capability).second ||
              declared_capabilities.count(capability) == 0)
            fail(alternative_path, "alternative capabilities are duplicate or undeclared");
        }
        if (evaluator == "explicit_ctilde") {
          require_exact_fields(alternative,
                               {"alternative_id", "availability", "evaluator",
                                "required_capabilities", "scale", "source_binding"},
                               alternative_path);
          if (direct_found ||
              alternative_capabilities !=
                  std::set<std::string>{"explicit_ctilde_forward_adjoint_v1"} ||
              complex_number(alternative["scale"], alternative_path + ".scale") !=
                  std::complex<double>(1.0, 0.0))
            fail(alternative_path, "each v2 row requires one identity direct alternative");
          const YAML::Node source_binding = alternative["source_binding"];
          require_exact_fields(
              source_binding,
              {"central_type", "feature_id", "function_index", "source_yace_sha256"},
              alternative_path + ".source_binding");
          if (integer(source_binding["central_type"],
                      alternative_path + ".source_binding.central_type", 0) != central ||
              integer(source_binding["function_index"],
                      alternative_path + ".source_binding.function_index", 0) != function ||
              scalar_string(source_binding["feature_id"],
                            alternative_path + ".source_binding.feature_id") != mapped.feature_id ||
              scalar_string(source_binding["source_yace_sha256"],
                            alternative_path + ".source_binding.source_yace_sha256") != source_hash)
            fail(alternative_path, "direct alternative is not bound to this YACE function");
          mapped.direct_candidate_id = alternative_id;
          direct_found = true;
          has_explicit_ctilde = true;
          continue;
        }
        if (evaluator == "scalar_invariant_power") {
          require_exact_fields(alternative,
                               {"alternative_id", "availability", "compiler_plan_hash", "evaluator",
                                "factorization_id", "plan_output", "required_capabilities",
                                "scale"},
                               alternative_path);
          if (scalar_found ||
              alternative_capabilities !=
                  std::set<std::string>{"scalar_invariant_power_adjoint_v1",
                                        "scalar_invariant_power_forward_v1"} ||
              scalar_string(alternative["compiler_plan_hash"],
                            alternative_path + ".compiler_plan_hash") != plan_hash)
            fail(alternative_path, "scalar alternative is duplicate or bound to another plan");
          const std::string factorization_id = scalar_string(
              alternative["factorization_id"], alternative_path + ".factorization_id");
          if (!lower_sha256(factorization_id))
            fail(alternative_path, "scalar factorization ID must be a SHA-256");
          const YAML::Node output = alternative["plan_output"];
          require_exact_fields(output,
                               {"channel_index", "instruction_id", "magnetic_index",
                                "output_binding_hash", "tableau_index"},
                               alternative_path + ".plan_output");
          MappedBlockDescriptor scalar_mapped = mapped;
          scalar_mapped.instruction_id = scalar_string(
              output["instruction_id"], alternative_path + ".plan_output.instruction_id");
          scalar_mapped.channel_index =
              integer(output["channel_index"], alternative_path + ".plan_output.channel_index", 0);
          scalar_mapped.tableau_index =
              integer(output["tableau_index"], alternative_path + ".plan_output.tableau_index", 0);
          scalar_mapped.magnetic_index = integer(
              output["magnetic_index"], alternative_path + ".plan_output.magnetic_index", 0);
          scalar_mapped.scale = complex_number(alternative["scale"], alternative_path + ".scale");
          validate_output_binding(plan, scalar_mapped, source_hash, alternative_path);
          scalar_instruction_id = scalar_mapped.instruction_id;
          scalar_channel_index = scalar_mapped.channel_index;
          scalar_tableau_index = scalar_mapped.tableau_index;
          scalar_magnetic_index = scalar_mapped.magnetic_index;
          scalar_scale = scalar_mapped.scale;
          mapped.scalar_candidate_id = alternative_id;
          mapped.scalar_factorization_id = factorization_id;
          scalar_found = true;
          continue;
        }
        if (evaluator != "execution_plan_output")
          fail(alternative_path, "unsupported evaluator alternative");
        require_exact_fields(alternative,
                             {"alternative_id", "availability", "compiler_plan_hash", "evaluator",
                              "plan_output", "required_capabilities", "scale"},
                             alternative_path);
        if (block_found ||
            alternative_capabilities !=
                std::set<std::string>{"block_symmetric_power_adjoint_v1",
                                      "block_symmetric_power_forward_v1"} ||
            scalar_string(alternative["compiler_plan_hash"],
                          alternative_path + ".compiler_plan_hash") != plan_hash)
          fail(alternative_path, "block alternative is duplicate or bound to another plan");
        const YAML::Node output = alternative["plan_output"];
        require_exact_fields(output,
                             {"channel_index", "instruction_id", "magnetic_index",
                              "output_binding_hash", "tableau_index"},
                             alternative_path + ".plan_output");
        mapped.instruction_id = scalar_string(output["instruction_id"],
                                              alternative_path + ".plan_output.instruction_id");
        mapped.channel_index =
            integer(output["channel_index"], alternative_path + ".plan_output.channel_index", 0);
        mapped.tableau_index =
            integer(output["tableau_index"], alternative_path + ".plan_output.tableau_index", 0);
        mapped.magnetic_index =
            integer(output["magnetic_index"], alternative_path + ".plan_output.magnetic_index", 0);
        mapped.scale = complex_number(alternative["scale"], alternative_path + ".scale");
        mapped.block_candidate_id = alternative_id;
        validate_output_binding(plan, mapped, source_hash, alternative_path);
        block_found = true;
      }
      if (!direct_found) fail(path, "v2 function-map row has no available direct alternative");
      direct_candidate_ids[static_cast<std::size_t>(central)][static_cast<std::size_t>(function)] =
          mapped.direct_candidate_id;
      if (scalar_found) {
        if (!block_found || scalar_instruction_id != mapped.instruction_id ||
            scalar_channel_index != mapped.channel_index ||
            scalar_tableau_index != mapped.tableau_index ||
            scalar_magnetic_index != mapped.magnetic_index || scalar_scale != mapped.scale)
          fail(path, "scalar and block alternatives must bind the same plan output");
      }
      if (block_found || scalar_found) { mapped_entries.push_back(std::move(mapped)); }
    }
    for (const auto &bucket : seen)
      if (std::find(bucket.begin(), bucket.end(), false) != bucket.end())
        fail("function_map.entries", "function map omitted a YACE function");
    if (candidate_vector_schema && used_readout_ids != certified_readout_ids)
      fail("execution_plan.certificate.yace_candidate_readouts",
           "every certified candidate readout must be referenced exactly once");
    if (has_explicit_ctilde &&
        declared_capabilities.count("explicit_ctilde_forward_adjoint_v1") == 0)
      fail("sidecar.capabilities.required", "explicit C-tilde fallback is not declared");
    for (std::size_t central = 0; central < direct_candidate_ids.size(); ++central)
      for (std::size_t function = 0; function < direct_candidate_ids[central].size(); ++function)
        if (direct_candidate_ids[central][function].empty() ||
            feature_ids[central][function].empty())
          fail("function_map.entries",
               "function map did not bind a direct candidate and feature ID");

    std::vector<std::vector<BlockRouteCandidate>> candidates(species.size());
    for (const auto &mapped : mapped_entries) {
      const std::string path = "execution_plan.instructions." + mapped.instruction_id;
      const YAML::Node instruction =
          record_by_id(plan["instructions"], "instruction_id", mapped.instruction_id,
                       "execution_plan.instructions");
      if (scalar_string(instruction["opcode"], path + ".opcode") != "block_symmetric_power" ||
          scalar_string(instruction["analysis_orientation"], path + ".analysis_orientation") !=
              "conjugate_transpose")
        fail(path, "unsupported block instruction");
      const YAML::Node metadata = instruction["metadata"];
      const std::string hierarchical_schema =
          scalar_string(metadata["schema"], path + ".metadata.schema");
      const bool factorized_outer =
          hierarchical_schema == "ye3t_hierarchical_repeated_angular_blocks_v2";
      if ((hierarchical_schema != "ye3t_hierarchical_repeated_angular_blocks_v1" &&
           !factorized_outer) ||
          boolean(metadata["runtime_path_discovery"], path + ".runtime_path_discovery") ||
          boolean(metadata["raw_angular_tree_forest_materialized"],
                  path + ".raw_angular_tree_forest_materialized"))
        fail(path, "block instruction is not a fully compiled factorized plan");
      if (integer(metadata["target_L"], path + ".target_L", 0) != 0 ||
          integer(metadata["target_parity"], path + ".target_parity", 0) != 1 ||
          integer(metadata["parent_tableau_count"], path + ".parent_tableau_count", 1) != 1 ||
          mapped.tableau_index != 0 || mapped.magnetic_index != 0)
        fail(path, "the first PairYE3T block runtime supports scalar outputs only");

      const YAML::Node source_assembly = record_by_id(
          plan["source_assemblies"], "assembly_id",
          scalar_string(instruction["source_assembly_id"], path + ".source_assembly_id"),
          "execution_plan.source_assemblies");
      const YAML::Node source_realization = source_assembly["source_realization"];
      const YAML::Node source_validation = source_assembly["validation_report"];
      require_mapping(source_realization, path + ".source_realization");
      require_mapping(source_validation, path + ".source_assembly.validation_report");
      if (scalar_string(source_realization["kind"], path + ".source_realization.kind") !=
              "ordinary_density" ||
          boolean(source_realization["retain_role_order"],
                  path + ".source_realization.retain_role_order") ||
          !boolean(source_validation["passed"],
                   path + ".source_assembly.validation_report.passed") ||
          !boolean(source_validation["slot_coverage_complete"],
                   path + ".source_assembly.validation_report.slot_coverage_complete") ||
          integer(source_assembly["source_dimension"], path + ".source_dimension", 1) != 1 ||
          integer(source_assembly["induced_dimension"], path + ".induced_dimension", 1) != 1 ||
          sparse_assembly_value(source_assembly, 0, 0, path + ".source_assembly") !=
              std::complex<double>(1.0, 0.0))
        fail(path, "unsupported ordinary-density source assembly");

      const int lr_multiplicity =
          integer(metadata["lr_multiplicity"], path + ".lr_multiplicity", 1);
      if (lr_multiplicity != 1) fail(path, "ordinary symmetric ACE requires one LR coordinate");
      if (scalar_string(instruction["synthesis_table_id"], path + ".synthesis_table_id") !=
          scalar_string(metadata["lr_synthesis_table_id"], path + ".lr_synthesis_table_id"))
        fail(path, "instruction and metadata disagree on the LR table");
      const int route_index = mapped.channel_index / lr_multiplicity;
      const int lr_index = mapped.channel_index % lr_multiplicity;
      const YAML::Node routes = metadata["routes"];
      require_sequence(routes, path + ".routes");
      const int joint_multiplicity =
          integer(metadata["joint_multiplicity"], path + ".joint_multiplicity", 1);
      if (joint_multiplicity != static_cast<int>(routes.size()) ||
          mapped.channel_index >= joint_multiplicity)
        fail(path, "joint multiplicity does not match the compiled block routes");
      for (std::size_t index = 0; index < routes.size(); ++index)
        if (integer(routes[index]["route_index"], path + ".route_index", 0) !=
            static_cast<int>(index))
          fail(path, "block route indices are not canonical and contiguous");
      YAML::Node selected_route;
      bool route_found = false;
      for (std::size_t index = 0; index < routes.size(); ++index)
        if (integer(routes[index]["route_index"], path + ".route_index", 0) == route_index) {
          if (route_found) fail(path, "duplicate block route index");
          selected_route = routes[index];
          route_found = true;
        }
      if (!route_found) fail(path, "mapped output channel has no block route");
      const YAML::Node blocks = metadata["blocks"];
      if (!blocks.IsSequence() || blocks.size() < 1 || (!factorized_outer && blocks.size() > 2))
        fail(path,
             "legacy PairYE3T block routes require one or two blocks; higher "
             "arity requires a compiler-factorized v2 schedule");
      if (factorized_outer) {
        const YAML::Node outer = metadata["factorized_outer_schedule"];
        require_mapping(outer, path + ".factorized_outer_schedule");
        const int declared_term_count =
            integer(outer["term_count"], path + ".factorized_outer_schedule.term_count", 1);
        if (scalar_string(outer["schema"], path + ".factorized_outer_schedule.schema") !=
                "ye3t_factorized_outer_schedule_v1" ||
            integer(outer["block_count"], path + ".factorized_outer_schedule.block_count", 1) !=
                static_cast<int>(blocks.size()) ||
            integer(outer["route_count"], path + ".factorized_outer_schedule.route_count", 1) !=
                static_cast<int>(routes.size()) ||
            boolean(outer["raw_magnetic_tree_expansion_materialized"],
                    path +
                        ".factorized_outer_schedule.raw_magnetic_tree_"
                        "expansion_materialized"))
          fail(path, "invalid compiler-factorized outer schedule metadata");
        const YAML::Node label_records = metadata["compiler_label_records"];
        require_sequence(label_records, path + ".compiler_label_records");
        if (label_records.size() != routes.size())
          fail(path, "compiler-label provenance does not cover every route");
        std::int64_t actual_term_count = 0;
        for (std::size_t index = 0; index < routes.size(); ++index) {
          const YAML::Node route = routes[index];
          const YAML::Node label = label_records[index];
          require_mapping(label, path + ".compiler_label_records");
          const std::string route_hash =
              scalar_string(route["compiler_label_hash"], path + ".compiler_label_hash");
          const YAML::Node internal_ls = label["internal_Ls"];
          require_sequence(internal_ls, path + ".compiler_label_records.internal_Ls");
          if (integer(route["compiler_label_index"], path + ".compiler_label_index", 0) !=
                  static_cast<int>(index) ||
              integer(label["label_index"], path + ".label_index", 0) != static_cast<int>(index) ||
              scalar_string(label["route_hash"], path + ".route_hash") != route_hash ||
              !lower_hex_digest(route_hash, 16) ||
              scalar_string(label["angular_key"], path + ".angular_key").empty() ||
              !label["basis_key"] || internal_ls.size() == 0 ||
              integer(internal_ls[internal_ls.size() - 1],
                      path + ".compiler_label_records.internal_Ls", 0) != 0)
            fail(path, "invalid compiler-label route provenance");
          const YAML::Node table = record_by_id(plan["synthesis_tables"], "table_id",
                                                scalar_string(route["angular_synthesis_table_id"],
                                                              path + ".angular_synthesis_table_id"),
                                                "execution_plan.synthesis_tables");
          validate_sparse_synthesis_table(table, path + ".factorized_cg");
          const YAML::Node values = table["values"];
          require_sequence(values, path + ".factorized_cg.values");
          const YAML::Node report = table["validation_report"];
          require_mapping(report, path + ".factorized_cg.validation_report");
          if (scalar_string(report["scope"], path + ".factorized_cg.validation_report.scope") !=
                  "hierarchical_factorized_CG" ||
              integer(report["block_count"], path + ".factorized_cg.validation_report.block_count",
                      1) != static_cast<int>(blocks.size()) ||
              integer(report["term_count"], path + ".factorized_cg.validation_report.term_count",
                      1) != static_cast<int>(values.size()) ||
              integer(report["compiler_schedule_term_count"],
                      path +
                          ".factorized_cg.validation_report.compiler_schedule_"
                          "term_count",
                      1) != static_cast<int>(values.size()))
            fail(path, "invalid factorized-route term certificate");
          actual_term_count += static_cast<std::int64_t>(values.size());
        }
        if (actual_term_count != declared_term_count)
          fail(path, "factorized outer term count does not match route tables");
      }
      const auto output_Ls = integer_sequence(selected_route["block_output_Ls"],
                                              path + ".block_output_Ls", blocks.size());
      const auto multiplicities =
          integer_sequence(selected_route["block_multiplicity_indices"],
                           path + ".block_multiplicity_indices", blocks.size());
      const YAML::Node output_carrier = instruction["output_carrier"];
      require_mapping(output_carrier, path + ".output_carrier");
      const int parent_rank = integer(output_carrier["rank"], path + ".output_carrier.rank", 1);
      const auto parent_partition =
          integer_sequence(output_carrier["partition"], path + ".output_carrier.partition", 1);
      if (scalar_string(output_carrier["group"], path + ".output_carrier.group") != "S_N_x_O3" ||
          scalar_string(output_carrier["convention_id"], path + ".output_carrier.convention_id") !=
              "o3:complex_condon_shortley_young_orthogonal_v1" ||
          integer(output_carrier["parity"], path + ".output_carrier.parity", 0) != 1 ||
          parent_partition[0] != parent_rank ||
          integer(output_carrier["rotation_L"], path + ".output_carrier.rotation_L", 0) != 0)
        fail(path,
             "ordinary-density block instruction must target the symmetric "
             "scalar parent");
      if (integer(source_realization["rank"], path + ".source_realization.rank", 1) != parent_rank)
        fail(path, "ordinary-density source rank does not match the parent carrier");
      const YAML::Node layouts = plan["carrier_layouts"];
      require_sequence(layouts, "execution_plan.carrier_layouts");
      int layout_matches = 0;
      for (std::size_t index = 0; index < layouts.size(); ++index) {
        const YAML::Node layout = layouts[index];
        const YAML::Node key = layout["key"];
        require_mapping(layout, "execution_plan.carrier_layouts");
        require_mapping(key, "execution_plan.carrier_layouts.key");
        const auto partition =
            integer_sequence(key["partition"], "execution_plan.carrier_layouts.key.partition", 1);
        if (integer(key["rank"], "execution_plan.carrier_layouts.key.rank", 1) != parent_rank ||
            partition != parent_partition ||
            integer(key["rotation_L"], "execution_plan.carrier_layouts.key.rotation_L", 0) != 0 ||
            integer(key["parity"], "execution_plan.carrier_layouts.key.parity", 0) != 1 ||
            scalar_string(key["group"], "execution_plan.carrier_layouts.key.group") != "S_N_x_O3" ||
            scalar_string(key["convention_id"],
                          "execution_plan.carrier_layouts.key.convention_id") !=
                "o3:complex_condon_shortley_young_orthogonal_v1" ||
            integer(layout["channel_count"], "execution_plan.carrier_layouts.channel_count", 1) !=
                joint_multiplicity)
          continue;
        ++layout_matches;
        const YAML::Node axes = layout["axis_order"];
        require_sequence(axes, "execution_plan.carrier_layouts.axis_order");
        if (axes.size() != 3 ||
            scalar_string(axes[0], "execution_plan.carrier_layouts.axis_order") !=
                "channel_or_multiplicity" ||
            scalar_string(axes[1], "execution_plan.carrier_layouts.axis_order") != "tableau_t" ||
            scalar_string(axes[2], "execution_plan.carrier_layouts.axis_order") != "magnetic_M" ||
            integer(layout["tableau_count"], "execution_plan.carrier_layouts.tableau_count", 1) !=
                1 ||
            integer(layout["magnetic_count"], "execution_plan.carrier_layouts.magnetic_count", 1) !=
                1 ||
            integer(layout["width"], "execution_plan.carrier_layouts.width", 1) !=
                joint_multiplicity)
          fail(path, "ordinary symmetric carrier layout is inconsistent");
      }
      if (layout_matches != 1)
        fail(path, "ordinary symmetric output requires one matching carrier layout");
      std::vector<int> covered_slots;
      for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        const YAML::Node block = blocks[block_index];
        if (integer(block["block_index"], path + ".blocks.block_index", 0) !=
            static_cast<int>(block_index))
          fail(path, "block records are not in canonical order");
        const int power = integer(block["power"], path + ".blocks.power", 1);
        const auto partition =
            integer_sequence(block["block_partition"], path + ".blocks.block_partition", 1);
        if (partition[0] != power)
          fail(path, "ordinary-density block partition is not fully symmetric");
        const auto slots = integer_sequence(block["slot_indices"], path + ".blocks.slot_indices",
                                            static_cast<std::size_t>(power));
        covered_slots.insert(covered_slots.end(), slots.begin(), slots.end());
      }
      std::sort(covered_slots.begin(), covered_slots.end());
      for (int slot = 0; slot < parent_rank; ++slot)
        if (static_cast<std::size_t>(slot) >= covered_slots.size() ||
            covered_slots[static_cast<std::size_t>(slot)] != slot)
          fail(path, "block slot indices do not cover the parent rank exactly once");
      if (covered_slots.size() != static_cast<std::size_t>(parent_rank))
        fail(path, "block slot indices do not match the parent rank");

      std::vector<YACEBlockPowerPlan> candidate_plans(blocks.size());
      std::vector<YAML::Node> selected_plan_records(blocks.size());
      for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        YAML::Node selected_plan;
        bool plan_found = false;
        const YAML::Node records = metadata["block_power_plans"];
        require_sequence(records, path + ".block_power_plans");
        for (std::size_t index = 0; index < records.size(); ++index)
          if (integer(records[index]["block_index"], path + ".block_index", 0) ==
                  static_cast<int>(block_index) &&
              integer(records[index]["output_L"], path + ".output_L", 0) ==
                  output_Ls[block_index]) {
            if (plan_found) fail(path, "duplicate selected block power plan");
            selected_plan = records[index];
            plan_found = true;
          }
        if (!plan_found) fail(path, "selected route has no block power plan");
        selected_plan_records[block_index] = selected_plan;
        candidate_plans[block_index] =
            parse_block_power_plan(selected_plan, blocks[block_index], mapped.central_species,
                                   species, path + ".block_power_plans");
      }

      const YAML::Node lr_table = record_by_id(
          plan["synthesis_tables"], "table_id",
          scalar_string(metadata["lr_synthesis_table_id"], path + ".lr_synthesis_table_id"),
          "execution_plan.synthesis_tables");
      validate_sparse_synthesis_table(lr_table, path + ".lr_synthesis_table");
      if (integer(lr_table["input_dimension"], path + ".lr.input_dimension", 1) != 1 ||
          integer(lr_table["output_dimension"], path + ".lr.output_dimension", 1) != 1 ||
          scalar_string(lr_table["validation_report"]["scope"],
                        path + ".lr.validation_report.scope") != "canonical_hierarchical_lr_row")
        fail(path, "ordinary symmetric ACE requires a scalar LR table");
      const std::complex<double> lr = sparse_table_value(lr_table, lr_index, mapped.tableau_index,
                                                         path + ".lr_synthesis_table");
      if (lr == std::complex<double>(0.0, 0.0))
        fail(path, "mapped LR output coordinate is structurally zero");

      const int left_width = 2 * output_Ls[0] + 1;
      int right_width = 0;
      YAML::Node cg_values;
      std::vector<int> cg_rows;
      std::vector<int> cg_columns;
      if (factorized_outer) {
        const YAML::Node cg_table =
            record_by_id(plan["synthesis_tables"], "table_id",
                         scalar_string(selected_route["angular_synthesis_table_id"],
                                       path + ".angular_synthesis_table_id"),
                         "execution_plan.synthesis_tables");
        validate_sparse_synthesis_table(cg_table, path + ".factorized_cg");
        std::int64_t input_width = 1;
        for (const int output_L : output_Ls) {
          const std::int64_t width = 2 * static_cast<std::int64_t>(output_L) + 1;
          if (input_width > static_cast<std::int64_t>(std::numeric_limits<int>::max()) / width)
            fail(path, "factorized angular input dimension exceeds native bounds");
          input_width *= width;
        }
        if (integer(cg_table["input_dimension"], path + ".factorized_cg.input_dimension", 1) !=
                input_width ||
            integer(cg_table["output_dimension"], path + ".factorized_cg.output_dimension", 1) !=
                1 ||
            scalar_string(cg_table["validation_report"]["scope"],
                          path + ".factorized_cg.validation_report.scope") !=
                "hierarchical_factorized_CG")
          fail(path,
               "factorized angular table dimensions or provenance do not match "
               "the selected block route");
        cg_values = cg_table["values"];
        require_sequence(cg_values, path + ".factorized_cg.values");
        cg_rows = integer_sequence(cg_table["row_indices"], path + ".factorized_cg.row_indices",
                                   cg_values.size());
        cg_columns = integer_sequence(cg_table["column_indices"],
                                      path + ".factorized_cg.column_indices", cg_values.size());
      } else if (blocks.size() == 1) {
        const YAML::Node angular_table = selected_route["angular_synthesis_table_id"];
        if (angular_table && !angular_table.IsNull())
          fail(path,
               "a one-block scalar route must not declare an angular "
               "coupling table");
      } else {
        const YAML::Node cg_table =
            record_by_id(plan["synthesis_tables"], "table_id",
                         scalar_string(selected_route["angular_synthesis_table_id"],
                                       path + ".angular_synthesis_table_id"),
                         "execution_plan.synthesis_tables");
        validate_sparse_synthesis_table(cg_table, path + ".cg");
        right_width = 2 * output_Ls[1] + 1;
        if (integer(cg_table["input_dimension"], path + ".cg.input_dimension", 1) !=
                left_width * right_width ||
            integer(cg_table["output_dimension"], path + ".cg.output_dimension", 1) != 1 ||
            scalar_string(cg_table["validation_report"]["scope"],
                          path + ".cg.validation_report.scope") != "hierarchical_binary_CG")
          fail(path,
               "angular table orientation or dimensions do not match the "
               "selected block route");
        cg_values = cg_table["values"];
        require_sequence(cg_values, path + ".cg.values");
        cg_rows =
            integer_sequence(cg_table["row_indices"], path + ".cg.row_indices", cg_values.size());
        cg_columns = integer_sequence(cg_table["column_indices"], path + ".cg.column_indices",
                                      cg_values.size());
      }

      YACEBlockRoute route;
      route.function_index = mapped.function_index;
      route.direct_operation_estimate = direct_descriptor_operations(
          species[static_cast<std::size_t>(mapped.central_species)].polynomial,
          mapped.function_index);
      for (const auto &candidate : candidate_plans)
        route.block_operation_estimate += block_plan_operations(candidate);
      if (factorized_outer)
        route.block_operation_estimate += 6 * static_cast<std::int64_t>(blocks.size()) *
            static_cast<std::int64_t>(cg_values.size());
      else if (blocks.size() == 1)
        route.block_operation_estimate += 2;
      else
        route.block_operation_estimate += 6 * static_cast<std::int64_t>(cg_values.size());
      if (factorized_outer) {
        route.term_factor_offsets.push_back(0);
        for (std::size_t index = 0; index < cg_values.size(); ++index) {
          if (cg_columns[index] != mapped.magnetic_index) continue;
          std::int64_t row = cg_rows[index];
          std::vector<std::int64_t> components(blocks.size());
          for (std::size_t reverse = blocks.size(); reverse-- > 0;) {
            const std::int64_t width = 2 * output_Ls[reverse] + 1;
            components[reverse] = row % width;
            row /= width;
          }
          if (row != 0) fail(path, "factorized angular table row is out of range");
          for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            const std::int64_t width = 2 * output_Ls[block_index] + 1;
            const std::int64_t component =
                multiplicities[block_index] * width + components[block_index];
            if (component < 0 || component >= candidate_plans[block_index].output_dimension)
              fail(path, "factorized route addresses an unavailable block component");
            route.term_factor_plans.push_back(static_cast<std::int64_t>(block_index));
            route.term_factor_components.push_back(component);
          }
          route.term_factor_offsets.push_back(
              static_cast<std::int64_t>(route.term_factor_plans.size()));
          route.coefficients.push_back(
              mapped.scale * std::conj(lr) *
              std::conj(complex_number(cg_values[index], path + ".factorized_cg.values")));
          if (route.coefficients.back().imag() != 0.0) route.real_coefficients = false;
        }
        if (route.coefficients.empty() ||
            route.term_factor_offsets.size() != route.coefficients.size() + 1)
          fail(path, "selected factorized angular route is empty");
      } else if (blocks.size() == 1) {
        route.left_plan = 0;
        if (output_Ls[0] != 0)
          fail(path, "a one-block scalar route requires a scalar block output");
        const std::int64_t component = multiplicities[0];
        const auto &left_plan = candidate_plans[0];
        if (component < 0 || component >= left_plan.output_dimension)
          fail(path, "one-block route addresses an unavailable block component");
        route.left_components.push_back(component);
        route.coefficients.push_back(mapped.scale * std::conj(lr));
        if (route.coefficients.back().imag() != 0.0) route.real_coefficients = false;
      } else {
        route.left_plan = 0;
        route.right_plan = 1;
        const auto &left_plan = candidate_plans[0];
        const auto &right_plan = candidate_plans[1];
        for (std::size_t index = 0; index < cg_values.size(); ++index) {
          if (cg_columns[index] != mapped.magnetic_index) continue;
          const int row = cg_rows[index];
          if (row < 0 || row >= left_width * right_width)
            fail(path, "angular table row is out of range");
          const std::int64_t left_component = multiplicities[0] * left_width + row / right_width;
          const std::int64_t right_component = multiplicities[1] * right_width + row % right_width;
          if (left_component >= left_plan.output_dimension ||
              right_component >= right_plan.output_dimension)
            fail(path, "angular route addresses an unavailable block component");
          route.left_components.push_back(left_component);
          route.right_components.push_back(right_component);
          route.coefficients.push_back(
              mapped.scale * std::conj(lr) *
              std::conj(complex_number(cg_values[index], path + ".cg.values")));
          if (route.coefficients.back().imag() != 0.0) route.real_coefficients = false;
        }
        if (route.coefficients.empty()) fail(path, "selected angular route is empty");
      }
      BlockRouteCandidate candidate;
      candidate.route = std::move(route);
      candidate.power_plans = std::move(candidate_plans);
      candidate.direct_candidate_id = mapped.direct_candidate_id;
      candidate.block_candidate_id = mapped.block_candidate_id;
      candidate.scalar_candidate_id = mapped.scalar_candidate_id;
      if (!mapped.scalar_candidate_id.empty()) {
        if (blocks.size() != 1 || candidate.route.left_components.size() != 1)
          fail(path, "scalar-invariant-power route requires one scalar block");
        candidate.scalar = parse_scalar_route(
            metadata, selected_plan_records[0], candidate.power_plans[0],
            candidate.route.left_components[0], parent_rank, candidate.route.coefficients[0],
            mapped, path + ".scalar_invariant_power");
        candidate.scalar_available = true;
      }
      candidates[static_cast<std::size_t>(mapped.central_species)].push_back(std::move(candidate));
    }

    for (std::size_t central = 0; central < species.size(); ++central) {
      auto &entry = species[central];
      auto &program = entry.block_program;
      auto &species_candidates = candidates[central];
      std::sort(species_candidates.begin(), species_candidates.end(),
                [](const BlockRouteCandidate &left, const BlockRouteCandidate &right) {
                  if (left.route.function_index != right.route.function_index)
                    return left.route.function_index < right.route.function_index;
                  return left.block_candidate_id < right.block_candidate_id;
                });

      const std::size_t function_count = entry.polynomial.descriptor_offsets.size() - 1;
      std::vector<FunctionCandidatePortfolio> portfolios(function_count);
      for (std::size_t function = 0; function < function_count; ++function) {
        auto &portfolio = portfolios[function];
        portfolio.central_species = static_cast<int>(central);
        portfolio.function_index = static_cast<int>(function);
        portfolio.feature_id = feature_ids[central][function];
        CompiledEvaluatorCandidate direct;
        direct.evaluator = YACEEvaluatorKind::EXPLICIT_CTILDE;
        direct.candidate_id = direct_candidate_ids[central][function];
        direct.operation_estimate =
            direct_descriptor_operations(entry.polynomial, static_cast<int>(function));
        portfolio.candidates.push_back(std::move(direct));
      }
      for (auto &route_candidate : species_candidates) {
        const std::size_t function = static_cast<std::size_t>(route_candidate.route.function_index);
        if (function >= portfolios.size())
          fail("sidecar", "compiled candidate function is out of range");
        auto &portfolio = portfolios[function];
        if (route_candidate.direct_candidate_id != portfolio.candidates.front().candidate_id)
          fail("sidecar", "one function names conflicting direct candidates");

        auto block =
            std::find_if(portfolio.candidates.begin(), portfolio.candidates.end(),
                         [&](const CompiledEvaluatorCandidate &candidate) {
                           return candidate.evaluator == YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER &&
                               candidate.candidate_id == route_candidate.block_candidate_id;
                         });
        if (block == portfolio.candidates.end()) {
          CompiledEvaluatorCandidate added;
          added.evaluator = YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER;
          added.candidate_id = route_candidate.block_candidate_id;
          portfolio.candidates.push_back(std::move(added));
          block = std::prev(portfolio.candidates.end());
        }
        block->operation_estimate = add_operations(block->operation_estimate,
                                                   route_candidate.route.block_operation_estimate);
        block->block_terms.push_back(route_candidate);

        if (route_candidate.scalar_available) {
          CompiledEvaluatorCandidate scalar;
          scalar.evaluator = YACEEvaluatorKind::SCALAR_INVARIANT_POWER;
          scalar.candidate_id = route_candidate.scalar_candidate_id;
          scalar.operation_estimate = route_candidate.scalar.operation_estimate;
          scalar.scalar_route = route_candidate.scalar;
          portfolio.candidates.push_back(std::move(scalar));
        }
      }
      for (const auto &route_candidate : coupled_product_candidates) {
        if (route_candidate.central_species != static_cast<int>(central)) continue;
        const std::size_t function = static_cast<std::size_t>(route_candidate.function_index);
        if (function >= portfolios.size())
          fail("sidecar", "coupled-product candidate function is out of range");
        auto &portfolio = portfolios[function];
        if (route_candidate.direct_candidate_id != portfolio.candidates.front().candidate_id)
          fail("sidecar", "coupled-product candidate names a conflicting direct fallback");
        CompiledEvaluatorCandidate coupled;
        coupled.evaluator = YACEEvaluatorKind::COUPLED_PRODUCT_DAG;
        coupled.candidate_id = route_candidate.candidate_id;
        coupled.operation_estimate = route_candidate.plan.operation_estimate;
        coupled.coupled_product_plan = route_candidate.plan;
        portfolio.candidates.push_back(std::move(coupled));
      }
      for (auto &portfolio : portfolios) {
        std::sort(
            portfolio.candidates.begin(), portfolio.candidates.end(),
            [](const CompiledEvaluatorCandidate &left, const CompiledEvaluatorCandidate &right) {
              if (left.evaluator != right.evaluator) return left.evaluator < right.evaluator;
              return left.candidate_id < right.candidate_id;
            });
        if (portfolio.candidates.empty() ||
            portfolio.candidates.front().evaluator != YACEEvaluatorKind::EXPLICIT_CTILDE ||
            std::count_if(portfolio.candidates.begin(), portfolio.candidates.end(),
                          [](const CompiledEvaluatorCandidate &candidate) {
                            return candidate.evaluator == YACEEvaluatorKind::EXPLICIT_CTILDE;
                          }) != 1)
          fail("sidecar", "each function requires exactly one direct candidate");
        for (std::size_t index = 1; index < portfolio.candidates.size(); ++index)
          if (portfolio.candidates[index - 1].candidate_id ==
              portfolio.candidates[index].candidate_id)
            fail("sidecar", "one function contains a duplicate candidate ID");
      }

      program.direct_operation_estimate = direct_plan_operations(entry.polynomial);
      program.catalogue_function_count = static_cast<std::int64_t>(function_count);
      program.plan_hash = plan_hash;
      program.source_manifest = manifest_path;
      for (const auto &portfolio : portfolios) {
        program.candidate_count += static_cast<std::int64_t>(portfolio.candidates.size());
        if (portfolio.candidates.size() > 1) ++program.candidate_function_count;
        program.candidate_route_count += static_cast<std::int64_t>(std::count_if(
            portfolio.candidates.begin(), portfolio.candidates.end(),
            [](const CompiledEvaluatorCandidate &candidate) {
              return candidate.evaluator == YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER ||
                  candidate.evaluator == YACEEvaluatorKind::COUPLED_PRODUCT_DAG;
            }));
      }
      std::vector<std::string> replay_candidate_ids;
      if (auto_replay != nullptr) {
        replay_candidate_ids.resize(function_count);
        for (const auto &record : auto_replay->selections) {
          if (record.central_species != static_cast<int>(central)) continue;
          if (record.function_index < 0 ||
              static_cast<std::size_t>(record.function_index) >= function_count)
            fail("auto_replay.selections", "function coordinate is out of range");
          const std::size_t function = static_cast<std::size_t>(record.function_index);
          if (!replay_candidate_ids[function].empty())
            fail("auto_replay.selections", "function coordinate is duplicated");
          if (record.element != entry.element ||
              record.feature_id != portfolios[function].feature_id)
            fail("auto_replay.selections",
                 "element or feature identity does not match its function");
          replay_candidate_ids[function] = record.candidate_id;
        }
        if (std::find(replay_candidate_ids.begin(), replay_candidate_ids.end(), std::string()) !=
            replay_candidate_ids.end())
          fail("auto_replay.selections", "replay omitted a function coordinate");
      }
      const BlockSelection selection = auto_replay != nullptr
          ? choose_replay_selection(entry.polynomial, portfolios, replay_candidate_ids)
          : choose_block_selection(entry.polynomial, portfolios, policy);
      program.planner_algorithm = selection.algorithm;
      program.planner_status = selection.status;
      program.planner_score_evaluations = selection.score_evaluations;
      program.planner_optimal = selection.optimal;
      program.selected_operation_estimate = selection.operation_estimate;
      if (policy == YACEBlockPolicy::GPU_AUTO) {
        if (auto_replay != nullptr) {
          program.planner_profile = "kokkos_gpu_device_replay_v1";
          program.planner_calibration_hash = auto_replay->replay_hash;
          program.planner_decision_reason = "offline_device_bound_interleaved_calibration";
        } else {
          program.planner_profile = "kokkos_gpu_conservative_direct_v1";
          program.planner_calibration_hash = "not_applicable";
          program.planner_decision_reason = selection.algorithm == "conservative_direct_fallback_v1"
              ? "no_authorized_non_direct_profile_match"
              : selection.status;
        }
      } else if (std::any_of(portfolios.begin(), portfolios.end(),
                             [](const FunctionCandidatePortfolio &portfolio) {
                               return std::any_of(portfolio.candidates.begin(),
                                                  portfolio.candidates.end(),
                                                  [](const CompiledEvaluatorCandidate &candidate) {
                                                    return candidate.evaluator ==
                                                        YACEEvaluatorKind::COUPLED_PRODUCT_DAG;
                                                  });
                             }))
        program.planner_profile = "cpu_structural_coupled_product_split_real_tile8_v1";
      else if (std::any_of(portfolios.begin(), portfolios.end(),
                           [](const FunctionCandidatePortfolio &portfolio) {
                             return std::any_of(portfolio.candidates.begin(),
                                                portfolio.candidates.end(),
                                                [](const CompiledEvaluatorCandidate &candidate) {
                                                  return candidate.evaluator ==
                                                      YACEEvaluatorKind::SCALAR_INVARIANT_POWER;
                                                });
                           }))
        program.planner_profile = "cpu_structural_candidate_vector_split_real_tile8_v2";
      std::set<int> removed;
      for (std::size_t index = 0; index < portfolios.size(); ++index) {
        const auto &portfolio = portfolios[index];
        const std::size_t selected_index = selection.choices.at(index);
        const auto &candidate = portfolio.candidates.at(selected_index);
        YACEEvaluatorDecision decision;
        decision.function_index = portfolio.function_index;
        decision.selected_candidate_id = candidate.candidate_id;
        decision.selected_candidate_index = static_cast<std::int64_t>(selected_index);
        decision.selected_evaluator = candidate.evaluator;
        for (const auto &available : portfolio.candidates)
          decision.candidates.push_back(
              {available.candidate_id, available.evaluator, available.operation_estimate});
        program.decisions.push_back(std::move(decision));
        if (candidate.evaluator == YACEEvaluatorKind::EXPLICIT_CTILDE) continue;
        removed.insert(portfolio.function_index);
        if (candidate.evaluator == YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER) {
          for (const auto &term : candidate.block_terms) {
            YACEBlockRoute route = term.route;
            std::vector<std::int64_t> mapped_plans;
            mapped_plans.reserve(term.power_plans.size());
            for (const auto &power_plan : term.power_plans)
              mapped_plans.push_back(append_block_plan(program, power_plan));
            if (!route.term_factor_plans.empty()) {
              for (auto &plan_index : route.term_factor_plans) {
                if (plan_index < 0 || static_cast<std::size_t>(plan_index) >= mapped_plans.size())
                  fail("sidecar", "factorized route references an unavailable local plan");
                plan_index = mapped_plans[static_cast<std::size_t>(plan_index)];
              }
              route.left_plan = -1;
              route.right_plan = -1;
            } else {
              route.left_plan = mapped_plans.at(0);
              route.right_plan = mapped_plans.size() == 2 ? mapped_plans[1] : -1;
            }
            program.routes.push_back(std::move(route));
          }
        } else if (candidate.evaluator == YACEEvaluatorKind::COUPLED_PRODUCT_DAG) {
          program.coupled_product_plans.push_back(candidate.coupled_product_plan);
        }
      }
      build_scalar_power_program(program, portfolios, selection.choices);
      if (!removed.empty()) entry.polynomial = without_descriptors(entry.polynomial, removed);
      if (!program.routes.empty()) build_shared_block_power_cache(program);
      std::int64_t output_offset = 0;
      for (auto &power_plan : program.power_plans) {
        power_plan.output_storage_offset = output_offset;
        output_offset += power_plan.output_dimension;
      }
      const bool has_direct = !entry.polynomial.monomial_coefficients.empty();
      const bool has_block = !program.routes.empty();
      const bool has_scalar = !program.scalar_program.routes.empty();
      const bool has_coupled = !program.coupled_product_plans.empty();
      if (policy == YACEBlockPolicy::GPU_AUTO)
        program.dispatch = "kokkos_gpu_auto_candidate_vector_v1";
      else if (has_coupled)
        program.dispatch = has_direct ? "cpu_ye3t_direct_coupled_product_portfolio_v1"
                                      : "cpu_ye3t_coupled_product_dag_v1";
      else if (has_scalar && !has_block)
        program.dispatch = has_direct ? "cpu_ye3t_direct_scalar_portfolio_v1"
                                      : "cpu_ye3t_scalar_invariant_power_v1";
      else if (has_block && has_scalar)
        program.dispatch = has_direct ? "cpu_ye3t_direct_block_scalar_portfolio_v1"
                                      : "cpu_ye3t_block_scalar_portfolio_v1";
      else if (has_block)
        program.dispatch = has_direct ? "cpu_ye3t_direct_block_portfolio_v1"
                                      : "cpu_ye3t_block_symmetric_power_v1";
      program.evaluator_plan_hash = evaluator_plan_hash(source_hash, program, entry.polynomial);
    }
  }

}    // namespace

const YACESpecies &YACEModel::species(int index) const
{
  if (index < 0 || index >= species_count())
    throw std::out_of_range("YACE species index is out of range");
  return species_[static_cast<std::size_t>(index)];
}

const YACEBond &YACEModel::bond(int central_species, int neighbor_species) const
{
  if (central_species < 0 || central_species >= species_count() || neighbor_species < 0 ||
      neighbor_species >= species_count())
    throw std::out_of_range("YACE bond species index is out of range");
  const int index = central_species * species_count() + neighbor_species;
  return bonds_[static_cast<std::size_t>(index)];
}

double YACEModel::memory_usage() const
{
  double bytes = source_path_.capacity() * sizeof(char);
  bytes += auto_replay_.selections.capacity() * sizeof(YACEAutoReplaySelection);
  for (const auto &selection : auto_replay_.selections) {
    bytes += selection.candidate_id.capacity() * sizeof(char);
    bytes += selection.element.capacity() * sizeof(char);
    bytes += selection.feature_id.capacity() * sizeof(char);
  }
  bytes += auto_replay_.path.capacity() * sizeof(char);
  bytes += auto_replay_.replay_hash.capacity() * sizeof(char);
  bytes += auto_replay_.source_yace_hash.capacity() * sizeof(char);
  bytes += auto_replay_.sidecar_manifest_hash.capacity() * sizeof(char);
  bytes += auto_replay_.compiler_plan_hash.capacity() * sizeof(char);
  bytes += auto_replay_.calibration_evidence_hash.capacity() * sizeof(char);
  bytes += auto_replay_.calibration_method.capacity() * sizeof(char);
  bytes += auto_replay_.selected_evaluator.capacity() * sizeof(char);
  bytes += auto_replay_.selection_rule.capacity() * sizeof(char);
  bytes += auto_replay_.device_class_hash.capacity() * sizeof(char);
  bytes += auto_replay_.execution_space.capacity() * sizeof(char);
  bytes += auto_replay_.lammps_executable_hash.capacity() * sizeof(char);
  bytes += auto_replay_.block_schedule.capacity() * sizeof(char);
  bytes += auto_replay_.block_scratch_layout.capacity() * sizeof(char);
  bytes += auto_replay_.kernel_abi.capacity() * sizeof(char);
  bytes += auto_replay_.layout.capacity() * sizeof(char);
  bytes += auto_replay_.precision.capacity() * sizeof(char);
  bytes += auto_replay_.source_policy.capacity() * sizeof(char);
  bytes += auto_replay_.vjp_policy.capacity() * sizeof(char);
  bytes += species_.capacity() * sizeof(YACESpecies);
  for (const auto &species : species_) {
    bytes += species.element.capacity() * sizeof(char);
    bytes += species.channels.capacity() * sizeof(YACEChannel);
    bytes += species.source_channels.capacity() * sizeof(YACEChannel);
    bytes += species.full_channel_sources.capacity() * sizeof(int);
    bytes += species.full_channel_transforms.capacity() * sizeof(int);
    const auto &plan = species.polynomial;
    bytes += plan.factor_offsets.capacity() * sizeof(std::int64_t);
    bytes += plan.factor_indices.capacity() * sizeof(std::int64_t);
    bytes += plan.factor_exponents.capacity() * sizeof(std::int64_t);
    bytes += plan.monomial_coefficients.capacity() * sizeof(double);
    bytes += plan.descriptor_offsets.capacity() * sizeof(std::int64_t);
    bytes += plan.descriptor_terms.capacity() * sizeof(std::int64_t);
    bytes += plan.descriptor_coefficients.capacity() * sizeof(double);
    bytes += plan.power_channels.capacity() * sizeof(std::int64_t);
    bytes += plan.power_exponents.capacity() * sizeof(std::int64_t);
    bytes += plan.dag_node_parents.capacity() * sizeof(std::int64_t);
    bytes += plan.dag_node_powers.capacity() * sizeof(std::int64_t);
    bytes += plan.binary_node_left.capacity() * sizeof(std::int64_t);
    bytes += plan.binary_node_right.capacity() * sizeof(std::int64_t);
    bytes += plan.monomial_nodes.capacity() * sizeof(std::int64_t);
    bytes += plan.dag_factor_ordering.capacity() * sizeof(char);
    const auto &block = species.block_program;
    bytes += block.power_plans.capacity() * sizeof(YACEBlockPowerPlan);
    for (const auto &power : block.power_plans) {
      bytes += power.input_channels.capacity() * sizeof(std::int64_t);
      bytes += power.input_power_offsets.capacity() * sizeof(std::int64_t);
      bytes += power.monomial_counts.capacity() * sizeof(std::int64_t);
      bytes += power.monomial_factor_offsets.capacity() * sizeof(std::int64_t);
      bytes += power.monomial_factor_components.capacity() * sizeof(std::int64_t);
      bytes += power.monomial_factor_exponents.capacity() * sizeof(std::int64_t);
      bytes += power.output_offsets.capacity() * sizeof(std::int64_t);
      bytes += power.coefficient_terms.capacity() * sizeof(std::int64_t);
      bytes += power.coefficient_values.capacity() * sizeof(std::complex<double>);
      bytes += power.direct_input_channels.capacity() * sizeof(std::int64_t);
      bytes += power.direct_input_scales.capacity() * sizeof(double);
    }
    bytes += block.power_channels.capacity() * sizeof(std::int64_t);
    bytes += block.power_maximum_exponents.capacity() * sizeof(std::int64_t);
    bytes += block.power_offsets.capacity() * sizeof(std::int64_t);
    bytes += block.routes.capacity() * sizeof(YACEBlockRoute);
    for (const auto &route : block.routes) {
      bytes += route.left_components.capacity() * sizeof(std::int64_t);
      bytes += route.right_components.capacity() * sizeof(std::int64_t);
      bytes += route.term_factor_offsets.capacity() * sizeof(std::int64_t);
      bytes += route.term_factor_plans.capacity() * sizeof(std::int64_t);
      bytes += route.term_factor_components.capacity() * sizeof(std::int64_t);
      bytes += route.coefficients.capacity() * sizeof(std::complex<double>);
    }
    const auto &scalar = block.scalar_program;
    bytes += scalar.bases.capacity() * sizeof(YACEScalarInvariantBase);
    for (const auto &base : scalar.bases) {
      bytes += base.base_id.capacity() * sizeof(char);
      bytes += base.left_channels.capacity() * sizeof(std::int64_t);
      bytes += base.right_channels.capacity() * sizeof(std::int64_t);
      bytes += base.coefficients.capacity() * sizeof(std::complex<double>);
    }
    bytes += scalar.nodes.capacity() * sizeof(YACEScalarPowerNode);
    bytes += scalar.routes.capacity() * sizeof(YACEScalarPowerRoute);
    for (const auto &route : scalar.routes)
      bytes += route.factorization_id.capacity() * sizeof(char);
    bytes += block.coupled_product_plans.capacity() * sizeof(YACECoupledProductDAGPlan);
    for (const auto &coupled : block.coupled_product_plans) {
      bytes += coupled.node_offsets.capacity() * sizeof(std::int64_t);
      bytes += coupled.node_dimensions.capacity() * sizeof(std::int64_t);
      bytes += coupled.node_leaf_offsets.capacity() * sizeof(std::int64_t);
      bytes += coupled.leaf_input_components.capacity() * sizeof(std::int64_t);
      bytes += coupled.node_coefficient_offsets.capacity() * sizeof(std::int64_t);
      bytes += coupled.coefficient_left_components.capacity() * sizeof(std::int64_t);
      bytes += coupled.coefficient_right_components.capacity() * sizeof(std::int64_t);
      bytes += coupled.coefficient_output_components.capacity() * sizeof(std::int64_t);
      bytes += coupled.coefficient_values.capacity() * sizeof(double);
      bytes += coupled.readout_components.capacity() * sizeof(std::int64_t);
      bytes += coupled.readout_coefficients.capacity() * sizeof(double);
    }
    bytes += block.plan_hash.capacity() * sizeof(char);
    bytes += block.source_manifest.capacity() * sizeof(char);
    bytes += block.dispatch.capacity() * sizeof(char);
    bytes += block.planner_profile.capacity() * sizeof(char);
    bytes += block.planner_algorithm.capacity() * sizeof(char);
    bytes += block.planner_status.capacity() * sizeof(char);
    bytes += block.planner_calibration_hash.capacity() * sizeof(char);
    bytes += block.planner_decision_reason.capacity() * sizeof(char);
    bytes += block.evaluator_plan_hash.capacity() * sizeof(char);
    bytes += block.decisions.capacity() * sizeof(YACEEvaluatorDecision);
    for (const auto &decision : block.decisions) {
      bytes += decision.candidates.capacity() * sizeof(YACEEvaluatorCandidateSummary);
      for (const auto &candidate : decision.candidates)
        bytes += candidate.candidate_id.capacity() * sizeof(char);
      bytes += decision.selected_candidate_id.capacity() * sizeof(char);
    }
  }
  bytes += bonds_.capacity() * sizeof(YACEBond);
  for (const auto &bond : bonds_) {
    bytes += bond.radial_coefficients.capacity() * sizeof(double);
    bytes += bond.radial_base_spline.capacity() * sizeof(double);
    bytes += bond.contracted_spline.capacity() * sizeof(double);
    bytes += bond.radial_channel_outputs.capacity() * sizeof(int);
    bytes += bond.radial_channel_indices.capacity() * sizeof(int);
    bytes += bond.angular_channel_outputs.capacity() * sizeof(int);
    bytes += bond.contracted_channel_indices.capacity() * sizeof(int);
    bytes += bond.angular_channel_nonnegative_indices.capacity() * sizeof(int);
  }
  return bytes;
}

YACEModel YACEModel::load(const std::string &path)
{
  return load(path, "", YACEBlockPolicy::DIRECT, "");
}

YACEModel YACEModel::load(const std::string &path, const std::string &sidecar_manifest,
                          YACEBlockPolicy block_policy)
{
  return load(path, sidecar_manifest, block_policy, "");
}

YACEModel YACEModel::load(const std::string &path, const std::string &sidecar_manifest,
                          YACEBlockPolicy block_policy, const std::string &auto_replay_path)
{
  const YAML::Node root = YAML::LoadFile(path);
  require_exact_fields(
      root, {"elements", "E0", "deltaSplineBins", "embeddings", "bonds", "functions"}, "yace");

  YACEModel model;
  model.source_path_ = path;
  require_sequence(root["elements"], "elements");
  if (root["elements"].size() == 0) fail("elements", "expected at least one element");
  std::set<std::string> element_names;
  for (std::size_t index = 0; index < root["elements"].size(); ++index) {
    if (!root["elements"][index].IsScalar())
      fail("elements[" + std::to_string(index) + "]", "expected a nonempty name");
    const std::string name = root["elements"][index].as<std::string>();
    if (name.empty()) fail("elements[" + std::to_string(index) + "]", "expected a nonempty name");
    if (!element_names.insert(name).second) fail("elements", "element names must be unique");
    YACESpecies species;
    species.element = name;
    model.species_.push_back(std::move(species));
  }
  const int species_count = model.species_count();

  const auto reference_energies =
      number_sequence(root["E0"], "E0", static_cast<std::size_t>(species_count));
  for (int species = 0; species < species_count; ++species)
    model.species_[static_cast<std::size_t>(species)].reference_energy =
        reference_energies[static_cast<std::size_t>(species)];

  model.spline_spacing_ = finite_number(root["deltaSplineBins"], "deltaSplineBins");
  if (model.spline_spacing_ <= 0.0) fail("deltaSplineBins", "must be positive");

  require_mapping(root["embeddings"], "embeddings");
  if (root["embeddings"].size() != static_cast<std::size_t>(species_count))
    fail("embeddings", "expected exactly one record per species");
  for (int species = 0; species < species_count; ++species) {
    const YAML::Node embedding = root["embeddings"][species];
    const std::string prefix = "embeddings." + std::to_string(species);
    require_exact_fields(
        embedding, {"ndensity", "FS_parameters", "npoti", "rho_core_cutoff", "drho_core_cutoff"},
        prefix);
    if (integer(embedding["ndensity"], prefix + ".ndensity", 1) != 1)
      fail(prefix + ".ndensity", "only one density per element is supported");
    const auto parameters =
        number_sequence(embedding["FS_parameters"], prefix + ".FS_parameters", 2);
    if (parameters[1] != 1.0)
      fail(prefix + ".FS_parameters[1]", "only the linear exponent is supported");
    const std::string embedding_name = embedding["npoti"].as<std::string>();
    if (embedding_name != "FinnisSinclair" && embedding_name != "FinnisSinclairShiftedScaled")
      fail(prefix + ".npoti", "unsupported embedding");
    const double rho_cutoff =
        finite_number(embedding["rho_core_cutoff"], prefix + ".rho_core_cutoff");
    const double drho_cutoff =
        finite_number(embedding["drho_core_cutoff"], prefix + ".drho_core_cutoff");
    if (rho_cutoff <= 0.0 || drho_cutoff < 0.0 || drho_cutoff >= rho_cutoff)
      fail(prefix, "invalid density cutoff interval");
    auto &destination = model.species_[static_cast<std::size_t>(species)];
    destination.embedding_scale = parameters[0];
    destination.density_safe_limit = rho_cutoff - drho_cutoff;
  }

  require_mapping(root["bonds"], "bonds");
  std::map<std::pair<int, int>, YAML::Node> bond_nodes;
  for (const auto &entry : root["bonds"]) {
    const YAML::Node key = entry.first;
    if (!key.IsSequence() || key.size() != 2) fail("bonds", "expected two-species sequence keys");
    const int central = integer(key[0], "bonds key central species", 0);
    const int neighbor = integer(key[1], "bonds key neighbor species", 0);
    if (central >= species_count || neighbor >= species_count)
      fail("bonds", "bond species index is out of range");
    if (!bond_nodes.emplace(std::make_pair(central, neighbor), entry.second).second)
      fail("bonds", "duplicate directed bond record");
  }
  if (bond_nodes.size() != static_cast<std::size_t>(species_count * species_count))
    fail("bonds", "expected every ordered species pair");

  model.bonds_.resize(static_cast<std::size_t>(species_count * species_count));
  for (int central = 0; central < species_count; ++central) {
    for (int neighbor = 0; neighbor < species_count; ++neighbor) {
      const auto position = bond_nodes.find(std::make_pair(central, neighbor));
      if (position == bond_nodes.end()) fail("bonds", "missing directed bond record");
      const YAML::Node node = position->second;
      const std::string prefix =
          "bonds.[" + std::to_string(central) + "," + std::to_string(neighbor) + "]";
      require_exact_fields(node,
                           {"nradmax", "lmax", "nradbasemax", "radbasename", "radparameters",
                            "radcoefficients", "prehc", "lambdahc", "rcut", "dcut", "rcut_in",
                            "dcut_in", "inner_cutoff_type"},
                           prefix);
      YACEBond bond;
      bond.central_species = central;
      bond.neighbor_species = neighbor;
      bond.radial_count = integer(node["nradmax"], prefix + ".nradmax", 1);
      bond.angular_maximum = integer(node["lmax"], prefix + ".lmax", 0);
      bond.radial_base_count = integer(node["nradbasemax"], prefix + ".nradbasemax", 1);
      if (node["radbasename"].as<std::string>() != "ChebExpCos")
        fail(prefix + ".radbasename", "only ChebExpCos is supported");
      const auto radial_parameters =
          number_sequence(node["radparameters"], prefix + ".radparameters", 1);
      if (radial_parameters[0] <= 0.0) fail(prefix + ".radparameters[0]", "must be positive");
      bond.radial_lambda = radial_parameters[0];
      bond.cutoff = finite_number(node["rcut"], prefix + ".rcut");
      bond.cutoff_width = finite_number(node["dcut"], prefix + ".dcut");
      if (bond.cutoff <= 0.0 || bond.cutoff_width <= 0.0 || bond.cutoff_width >= bond.cutoff)
        fail(prefix, "invalid outer cutoff interval");
      if (finite_number(node["prehc"], prefix + ".prehc") != 0.0)
        fail(prefix + ".prehc", "hard-core repulsion is not supported");
      finite_number(node["lambdahc"], prefix + ".lambdahc");
      if (finite_number(node["rcut_in"], prefix + ".rcut_in") != 0.0 ||
          finite_number(node["dcut_in"], prefix + ".dcut_in") != 0.0)
        fail(prefix, "active inner cutoffs are not supported");
      if (node["inner_cutoff_type"].as<std::string>() != "distance")
        fail(prefix + ".inner_cutoff_type", "unsupported inner cutoff type");

      const YAML::Node coefficients = node["radcoefficients"];
      require_sequence(coefficients, prefix + ".radcoefficients");
      if (coefficients.size() != static_cast<std::size_t>(bond.radial_count))
        fail(prefix + ".radcoefficients", "wrong radial dimension");
      bond.radial_coefficients.reserve(
          static_cast<std::size_t>(bond.contracted_width() * bond.radial_base_count));
      for (int radial = 0; radial < bond.radial_count; ++radial) {
        const YAML::Node angular_rows = coefficients[radial];
        require_sequence(angular_rows, prefix + ".radcoefficients radial row");
        if (angular_rows.size() != static_cast<std::size_t>(bond.angular_maximum + 1))
          fail(prefix + ".radcoefficients", "wrong angular dimension");
        for (int angular = 0; angular <= bond.angular_maximum; ++angular) {
          const auto row = number_sequence(angular_rows[angular], prefix + ".radcoefficients row",
                                           static_cast<std::size_t>(bond.radial_base_count));
          bond.radial_coefficients.insert(bond.radial_coefficients.end(), row.begin(), row.end());
        }
      }
      build_splines(bond, model.spline_spacing_);
      model.maximum_radial_base_count_ =
          std::max(model.maximum_radial_base_count_, bond.radial_base_count);
      model.maximum_contracted_width_ =
          std::max(model.maximum_contracted_width_, bond.contracted_width());
      model.maximum_angular_momentum_ =
          std::max(model.maximum_angular_momentum_, bond.angular_maximum);
      model.maximum_cutoff_ = std::max(model.maximum_cutoff_, bond.cutoff);
      model.bonds_[static_cast<std::size_t>(central * species_count + neighbor)] = std::move(bond);
    }
  }

  require_mapping(root["functions"], "functions");
  if (root["functions"].size() != static_cast<std::size_t>(species_count))
    fail("functions", "expected exactly one bucket per central species");
  const std::set<std::string> function_fields{"mu0", "rank", "ndensity", "num_ms_combs", "mus",
                                              "ns",  "ls",   "ms_combs", "ctildes"};
  for (int central = 0; central < species_count; ++central) {
    const YAML::Node functions = root["functions"][central];
    const std::string bucket_path = "functions." + std::to_string(central);
    require_sequence(functions, bucket_path);
    if (functions.size() == 0) fail(bucket_path, "expected at least one function");

    auto &destination = model.species_[static_cast<std::size_t>(central)];
    auto &plan = destination.polynomial;
    plan.descriptor_offsets.push_back(0);
    std::map<ChannelKey, int> channel_indices;
    std::map<MonomialKey, std::int64_t> monomial_indices;
    std::vector<MonomialKey> monomial_factors;

    for (std::size_t function_index = 0; function_index < functions.size(); ++function_index) {
      const YAML::Node function = functions[function_index];
      const std::string prefix = bucket_path + "[" + std::to_string(function_index) + "]";
      require_exact_fields(function, function_fields, prefix);
      if (integer(function["mu0"], prefix + ".mu0", 0) != central)
        fail(prefix + ".mu0", "does not match the central-species bucket");
      const int rank = integer(function["rank"], prefix + ".rank", 1);
      plan.maximum_rank = std::max(plan.maximum_rank, rank);
      if (integer(function["ndensity"], prefix + ".ndensity", 1) != 1)
        fail(prefix + ".ndensity", "only one density per element is supported");
      const int row_count = integer(function["num_ms_combs"], prefix + ".num_ms_combs", 1);
      const auto mus = integer_sequence(function["mus"], prefix + ".mus", rank);
      const auto ns = integer_sequence(function["ns"], prefix + ".ns", rank);
      const auto ls = integer_sequence(function["ls"], prefix + ".ls", rank);
      const auto ms = integer_sequence(function["ms_combs"], prefix + ".ms_combs",
                                       static_cast<std::size_t>(rank * row_count),
                                       std::numeric_limits<int>::min());
      const auto ctildes = number_sequence(function["ctildes"], prefix + ".ctildes",
                                           static_cast<std::size_t>(row_count));
      if (rank == 1 && (ls[0] != 0 || ms[0] != 0 || row_count != 1))
        fail(prefix, "rank-one functions require one l=0,m=0 row");

      for (int row = 0; row < row_count; ++row) {
        int magnetic_sum = 0;
        std::vector<int> row_channels;
        row_channels.reserve(static_cast<std::size_t>(rank));
        for (int slot = 0; slot < rank; ++slot) {
          const int neighbor = mus[static_cast<std::size_t>(slot)];
          if (neighbor < 0 || neighbor >= species_count)
            fail(prefix + ".mus", "neighbor species is out of range");
          const auto &bond = model.bond(central, neighbor);
          const int radial = ns[static_cast<std::size_t>(slot)] - 1;
          const int angular = ls[static_cast<std::size_t>(slot)];
          const int magnetic = ms[static_cast<std::size_t>(row * rank + slot)];
          const int radial_limit = rank == 1 ? bond.radial_base_count : bond.radial_count;
          if (radial < 0 || radial >= radial_limit)
            fail(prefix + ".ns", "one-based radial index is out of range");
          if (angular < 0 || angular > bond.angular_maximum || std::abs(magnetic) > angular)
            fail(prefix, "angular or magnetic index is out of range");
          magnetic_sum += magnetic;
          const ChannelKey key(rank == 1 ? 0 : 1, neighbor, radial, angular, magnetic);
          const auto inserted =
              channel_indices.emplace(key, static_cast<int>(channel_indices.size()));
          if (inserted.second) destination.channels.push_back(make_channel(key));
          row_channels.push_back(inserted.first->second);
        }
        if (magnetic_sum != 0) fail(prefix + ".ms_combs", "magnetic row does not couple to M=0");
        std::sort(row_channels.begin(), row_channels.end());
        MonomialKey monomial;
        for (std::size_t position = 0; position < row_channels.size();) {
          std::size_t finish = position + 1;
          while (finish < row_channels.size() && row_channels[finish] == row_channels[position])
            ++finish;
          monomial.emplace_back(row_channels[position], static_cast<int>(finish - position));
          position = finish;
        }
        const auto inserted =
            monomial_indices.emplace(monomial, static_cast<std::int64_t>(monomial_indices.size()));
        if (inserted.second) {
          monomial_factors.push_back(monomial);
          plan.monomial_coefficients.push_back(0.0);
        }
        const std::int64_t term = inserted.first->second;
        const double coefficient = ctildes[static_cast<std::size_t>(row)];
        plan.monomial_coefficients[static_cast<std::size_t>(term)] += coefficient;
        plan.descriptor_terms.push_back(term);
        plan.descriptor_coefficients.push_back(coefficient);
      }
      plan.descriptor_offsets.push_back(static_cast<std::int64_t>(plan.descriptor_terms.size()));
    }

    plan.factor_offsets.push_back(0);
    for (const auto &monomial : monomial_factors) {
      plan.maximum_term_factors =
          std::max(plan.maximum_term_factors, static_cast<std::int64_t>(monomial.size()));
      for (const auto &factor : monomial) {
        plan.factor_indices.push_back(factor.first);
        plan.factor_exponents.push_back(factor.second);
      }
      plan.factor_offsets.push_back(static_cast<std::int64_t>(plan.factor_indices.size()));
    }
    build_monomial_dag(plan);
    std::map<ChannelKey, int> source_indices;
    destination.full_channel_sources.reserve(destination.channels.size());
    destination.full_channel_transforms.reserve(destination.channels.size());
    for (const auto &channel : destination.channels) {
      const ChannelKey key(channel.kind == YACEChannel::RADIAL_BASE ? 0 : 1,
                           channel.neighbor_species, channel.radial, channel.angular,
                           std::abs(channel.magnetic));
      const auto inserted = source_indices.emplace(key, static_cast<int>(source_indices.size()));
      if (inserted.second) destination.source_channels.push_back(make_channel(key));
      destination.full_channel_sources.push_back(inserted.first->second);
      const int absolute_magnetic = std::abs(channel.magnetic);
      destination.full_channel_transforms.push_back(
          channel.magnetic < 0 ? (absolute_magnetic % 2 == 0 ? 1 : -1) : 0);
    }
    for (std::size_t channel_index = 0; channel_index < destination.source_channels.size();
         ++channel_index) {
      const auto &channel = destination.source_channels[channel_index];
      auto &bond =
          model
              .bonds_[static_cast<std::size_t>(central * species_count + channel.neighbor_species)];
      if (channel.kind == YACEChannel::RADIAL_BASE) {
        bond.radial_channel_outputs.push_back(static_cast<int>(channel_index));
        bond.radial_channel_indices.push_back(channel.radial);
      } else {
        bond.angular_channel_outputs.push_back(static_cast<int>(channel_index));
        bond.contracted_channel_indices.push_back(channel.radial * (bond.angular_maximum + 1) +
                                                  channel.angular);
        bond.angular_channel_nonnegative_indices.push_back(
            channel.angular * (channel.angular + 1) / 2 + channel.magnetic);
      }
    }
  }

  if (!auto_replay_path.empty() && block_policy != YACEBlockPolicy::GPU_AUTO)
    fail("auto_replay", "replay is supported only by GPU AUTO dispatch");
  if (block_policy != YACEBlockPolicy::DIRECT && sidecar_manifest.empty())
    fail("sidecar",
         "automatic or forced block dispatch requires a sidecar "
         "manifest");
  if (!auto_replay_path.empty())
    model.auto_replay_ = load_auto_replay(auto_replay_path, path, sidecar_manifest);
  if (block_policy != YACEBlockPolicy::DIRECT && !sidecar_manifest.empty())
    apply_block_sidecar(path, sidecar_manifest, block_policy,
                        model.auto_replay_.enabled() ? &model.auto_replay_ : nullptr,
                        model.species_);
  return model;
}

}    // namespace YE3T_LAMMPS
