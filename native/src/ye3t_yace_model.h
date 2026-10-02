/* -*- c++ -*- ----------------------------------------------------------
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

#ifndef LMP_YE3T_YACE_MODEL_H
#define LMP_YE3T_YACE_MODEL_H

#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace YE3T_LAMMPS {

struct YACEChannel {
  enum Kind { RADIAL_BASE, CONTRACTED_RADIAL_ANGULAR };

  Kind kind = CONTRACTED_RADIAL_ANGULAR;
  int neighbor_species = -1;
  int radial = -1;
  int angular = 0;
  int magnetic = 0;
};

struct YACESparsePolynomial {
  std::vector<std::int64_t> factor_offsets;
  std::vector<std::int64_t> factor_indices;
  std::vector<std::int64_t> factor_exponents;
  std::vector<double> monomial_coefficients;
  std::vector<std::int64_t> descriptor_offsets;
  std::vector<std::int64_t> descriptor_terms;
  std::vector<double> descriptor_coefficients;
  std::vector<std::int64_t> power_channels;
  std::vector<std::int64_t> power_exponents;
  std::vector<std::int64_t> dag_node_parents;
  std::vector<std::int64_t> dag_node_powers;
  std::int64_t dag_root_node_count = 0;
  bool binary_dag = false;
  std::vector<std::int64_t> binary_node_left;
  std::vector<std::int64_t> binary_node_right;
  std::vector<std::int64_t> monomial_nodes;
  std::string dag_factor_ordering;
  int maximum_rank = 0;
  std::int64_t maximum_term_factors = 0;
};

enum class YACEBlockPolicy { DIRECT, AUTO, GPU_AUTO, BLOCK, SCALAR_POWER, COUPLED_PRODUCT };

enum class YACEEvaluatorKind {
  EXPLICIT_CTILDE,
  BLOCK_SYMMETRIC_POWER,
  SCALAR_INVARIANT_POWER,
  COUPLED_PRODUCT_DAG
};

struct YACECoupledProductDAGPlan {
  std::vector<std::int64_t> node_offsets;
  std::vector<std::int64_t> node_dimensions;
  std::vector<std::int64_t> node_leaf_offsets;
  std::vector<std::int64_t> leaf_input_components;
  std::vector<std::int64_t> node_coefficient_offsets;
  std::vector<std::int64_t> coefficient_left_components;
  std::vector<std::int64_t> coefficient_right_components;
  std::vector<std::int64_t> coefficient_output_components;
  std::vector<double> coefficient_values;
  std::vector<std::int64_t> readout_components;
  std::vector<double> readout_coefficients;
  std::int64_t function_index = -1;
  std::int64_t operation_estimate = 0;
  std::int64_t total_node_components = 0;
};

struct YACEBlockPowerPlan {
  std::vector<std::int64_t> input_channels;
  std::vector<std::int64_t> input_power_offsets;
  std::vector<std::int64_t> monomial_counts;
  std::vector<std::int64_t> monomial_factor_offsets;
  std::vector<std::int64_t> monomial_factor_components;
  std::vector<std::int64_t> monomial_factor_exponents;
  std::vector<std::int64_t> output_offsets;
  std::vector<std::int64_t> coefficient_terms;
  std::vector<std::complex<double>> coefficient_values;
  std::vector<std::int64_t> direct_input_channels;
  std::vector<double> direct_input_scales;
  std::int64_t input_dimension = 0;
  std::int64_t output_dimension = 0;
  std::int64_t monomial_count = 0;
  std::int64_t maximum_power = 0;
  std::int64_t output_storage_offset = 0;
  int output_L = 0;
  bool real_coefficients = true;
  bool conjugate_half_output = false;
  bool direct_input_plan = false;
};

struct YACEBlockRoute {
  std::vector<std::int64_t> left_components;
  std::vector<std::int64_t> right_components;
  std::vector<std::int64_t> term_factor_offsets;
  std::vector<std::int64_t> term_factor_plans;
  std::vector<std::int64_t> term_factor_components;
  std::vector<std::complex<double>> coefficients;
  std::int64_t left_plan = -1;
  std::int64_t right_plan = -1;
  std::int64_t function_index = -1;
  std::int64_t direct_operation_estimate = 0;
  std::int64_t block_operation_estimate = 0;
  bool real_coefficients = true;
};

struct YACEScalarInvariantBase {
  std::string base_id;
  std::vector<std::int64_t> left_channels;
  std::vector<std::int64_t> right_channels;
  std::vector<std::complex<double>> coefficients;
};

struct YACEScalarPowerNode {
  std::int64_t left_value = -1;
  std::int64_t right_value = -1;
  std::int64_t exponent = 0;
  std::int64_t base_index = -1;
};

struct YACEScalarPowerRoute {
  std::int64_t function_index = -1;
  std::int64_t value_index = -1;
  std::complex<double> scale;
  std::string factorization_id;
};

struct YACEScalarPowerProgram {
  std::vector<YACEScalarInvariantBase> bases;
  std::vector<YACEScalarPowerNode> nodes;
  std::vector<YACEScalarPowerRoute> routes;
  std::int64_t value_count = 0;
};

struct YACEEvaluatorCandidateSummary {
  std::string candidate_id;
  YACEEvaluatorKind evaluator = YACEEvaluatorKind::EXPLICIT_CTILDE;
  std::int64_t operation_estimate = 0;
};

struct YACEEvaluatorDecision {
  std::int64_t function_index = -1;
  std::vector<YACEEvaluatorCandidateSummary> candidates;
  std::string selected_candidate_id;
  std::int64_t selected_candidate_index = -1;
  YACEEvaluatorKind selected_evaluator = YACEEvaluatorKind::EXPLICIT_CTILDE;
};

struct YACEAutoReplaySelection {
  std::string candidate_id;
  std::string element;
  std::string feature_id;
  int central_species = -1;
  int function_index = -1;
};

struct YACEAutoReplay {
  std::vector<YACEAutoReplaySelection> selections;
  std::string path;
  std::string replay_hash;
  std::string source_yace_hash;
  std::string sidecar_manifest_hash;
  std::string compiler_plan_hash;
  std::string calibration_evidence_hash;
  std::string calibration_method;
  std::string selected_evaluator;
  std::string selection_rule;
  std::string device_class_hash;
  std::string execution_space;
  std::string lammps_executable_hash;
  std::string block_schedule;
  std::string block_scratch_layout;
  std::string kernel_abi;
  std::string layout;
  std::string precision;
  std::string source_policy;
  std::string vjp_policy;
  int chunksize = 0;
  int block_team_size = 0;
  std::int64_t block_scratch_bytes = 0;
  int minimum_centers_per_rank = 0;
  int maximum_centers_per_rank = 0;

  bool enabled() const { return !path.empty(); }
};

struct YACEBlockProgram {
  std::vector<YACEBlockPowerPlan> power_plans;
  std::vector<YACEBlockRoute> routes;
  YACEScalarPowerProgram scalar_program;
  std::vector<YACECoupledProductDAGPlan> coupled_product_plans;
  std::vector<std::int64_t> power_channels;
  std::vector<std::int64_t> power_maximum_exponents;
  std::vector<std::int64_t> power_offsets;
  std::string plan_hash;
  std::string source_manifest;
  std::string dispatch = "cpu_yace_explicit_ctilde_v1";
  std::string planner_profile = "cpu_structural_split_real_tile8_v1";
  std::string planner_algorithm = "none";
  std::string planner_status = "not_run";
  std::string planner_calibration_hash;
  std::string planner_decision_reason;
  std::string evaluator_plan_hash;
  std::vector<YACEEvaluatorDecision> decisions;
  std::int64_t candidate_count = 0;
  std::int64_t candidate_route_count = 0;
  std::int64_t candidate_function_count = 0;
  std::int64_t catalogue_function_count = 0;
  std::int64_t planner_score_evaluations = 0;
  bool planner_optimal = true;
  std::int64_t direct_operation_estimate = 0;
  std::int64_t selected_operation_estimate = 0;
  std::int64_t power_storage_size = 0;
};

struct YACESpecies {
  std::string element;
  double reference_energy = 0.0;
  double embedding_scale = 1.0;
  double density_safe_limit = 0.0;
  std::vector<YACEChannel> channels;
  std::vector<YACEChannel> source_channels;
  std::vector<int> full_channel_sources;
  std::vector<int> full_channel_transforms;
  YACESparsePolynomial polynomial;
  YACEBlockProgram block_program;
};

struct YACEBond {
  int central_species = -1;
  int neighbor_species = -1;
  int radial_count = 0;
  int angular_maximum = 0;
  int radial_base_count = 0;
  double radial_lambda = 0.0;
  double cutoff = 0.0;
  double cutoff_width = 0.0;
  std::int64_t spline_interval_count = 0;
  std::vector<double> radial_coefficients;
  std::vector<double> radial_base_spline;
  std::vector<double> contracted_spline;
  std::vector<int> radial_channel_outputs;
  std::vector<int> radial_channel_indices;
  std::vector<int> angular_channel_outputs;
  std::vector<int> contracted_channel_indices;
  std::vector<int> angular_channel_nonnegative_indices;

  int contracted_width() const { return radial_count * (angular_maximum + 1); }
};

class YACEModel {
  friend struct YACEModelTestAccess;    // Native fixture tests; no public mutation API.
 public:
  static YACEModel load(const std::string &path);
  static YACEModel load(const std::string &path, const std::string &sidecar_manifest,
                        YACEBlockPolicy block_policy);
  static YACEModel load(const std::string &path, const std::string &sidecar_manifest,
                        YACEBlockPolicy block_policy, const std::string &auto_replay_path);

  const std::vector<YACESpecies> &species() const { return species_; }
  const YACESpecies &species(int index) const;
  const YACEBond &bond(int central_species, int neighbor_species) const;
  int species_count() const { return static_cast<int>(species_.size()); }
  int bond_count() const { return static_cast<int>(bonds_.size()); }
  int maximum_radial_base_count() const { return maximum_radial_base_count_; }
  int maximum_contracted_width() const { return maximum_contracted_width_; }
  int maximum_angular_momentum() const { return maximum_angular_momentum_; }
  double maximum_cutoff() const { return maximum_cutoff_; }
  double spline_spacing() const { return spline_spacing_; }
  const std::string &source_path() const { return source_path_; }
  const YACEAutoReplay &auto_replay() const { return auto_replay_; }
  double memory_usage() const;

 private:
  std::string source_path_;
  YACEAutoReplay auto_replay_;
  std::vector<YACESpecies> species_;
  std::vector<YACEBond> bonds_;
  int maximum_radial_base_count_ = 0;
  int maximum_contracted_width_ = 0;
  int maximum_angular_momentum_ = 0;
  double maximum_cutoff_ = 0.0;
  double spline_spacing_ = 0.0;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_YACE_MODEL_H
