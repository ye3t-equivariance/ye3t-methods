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

#ifndef LMP_YE3T_TAGGED_CAUCHY_MODEL_H
#define LMP_YE3T_TAGGED_CAUCHY_MODEL_H

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

enum class TaggedCauchyDeploymentKind {
  LegacyMomentV2,
  PhysicalImageV3,
  PhysicalImageV4,
};

enum class TaggedCauchySourceRealization {
  PaceChebExpCos,
  ExpandedPowerHornerV1,
  ShiftedJacobiThreeTermV1,
};

// One artifact `compiled_artifact.payload.real_forms[*]` entry: the
// (2l+1)x(2l+1) complex matrix U with `complex[row] = sum_a U[row][a] *
// real[a]` (row ordered by `magnetic_order`, column ordered by the
// artifact's own real-coordinate order) plus its precomputed inverse
// (`real = U^{-1} complex`).
struct TaggedCauchyRealForm {
  std::string real_form_id;
  int l = 0;
  int width = 0;                                // 2*l + 1
  std::vector<int> magnetic_order;              // width entries, the m value of each row
  std::vector<std::complex<double>> matrix;     // width*width row-major
  std::vector<std::complex<double>> inverse;    // width*width row-major
  // Exact nonzero entries of each inverse row.  These are derived once from
  // `inverse` without thresholding, so the CPU contraction skips only
  // coefficients that are exactly zero in the validated dense transform.
  std::vector<int> inverse_row_offsets;    // width+1 entries
  std::vector<int> inverse_columns;
  std::vector<std::complex<double>> inverse_values;
};

// One `channel_real_forms[*]` entry, with `neighbor_species` resolved to an
// index into `TaggedCauchyModel::species_order` and `real_form_id` resolved
// to an index into `TaggedCauchyModel::real_forms`.
struct TaggedCauchyChannel {
  int channel_index = 0;
  int l = 0;
  int radial_channel = 0;    // 0-based evaluator radial column (PACE n = radial_channel + 1)
  int neighbor_species_index = -1;
  int real_form_index = -1;
  int component_offset = 0;    // offset of this channel's components in the flat per-edge layout

  // V3 direct shifted-Jacobi source.  Coefficients are in increasing
  // powers of x=r/rc.  The full radial multiplier is
  // normalization*angular_scale*(1-x)^2*P_q(x)*x^l for 0<r<rc.
  std::vector<double> shifted_jacobi_power_coefficients;
  double normalization = 1.0;
  double angular_scale = 1.0;
};

// One `real_moment_program.terms[*]` row.
struct TaggedCauchyTerm {
  int feature_index = 0;
  double coefficient = 0.0;
  std::vector<int> density_factor_indices;    // indices into real_density_keys / A[]
  std::vector<int> moment_indices;            // indices into real_moment_keys / M[]
  int p = 0;
};

// One compiler-realified, division-free V3 reverse-schedule row.
struct TaggedCauchyAdjointTerm {
  int feature_index = 0;
  int source_index = 0;
  double coefficient = 0.0;
  std::vector<int> remaining_source_indices;
};

struct TaggedCauchyBinaryNode {
  int left_value = -1;
  int right_value = -1;
};

struct TaggedCauchyBinaryProductPlan {
  int base_count = 0;
  std::vector<TaggedCauchyBinaryNode> nodes;
  std::vector<int> roots;
  std::int64_t direct_multiplication_count = 0;
  std::int64_t binary_node_count = 0;
  int maximum_degree = 0;
  std::int64_t repeated_factor_product_count = 0;
};

struct TaggedCauchyExecutionRoute {
  int p = 0;
  int root_value = -1;
  double coefficient = 0.0;
};

struct TaggedCauchyExecutionPortfolio {
  bool present = false;
  std::string portfolio_hash;
  std::string program_hash;
  std::string readout_hash;
  TaggedCauchyBinaryProductPlan moment_plan;
  TaggedCauchyBinaryProductPlan outer_plan;
  std::vector<std::vector<TaggedCauchyExecutionRoute>> species_routes;
  bool direct_eligible = false;
  bool generic_dag_eligible = false;
  bool symmetric_power_eligible = false;
  bool block_eligible = false;
};

// Artifact-bound pair reference; constants use LAMMPS metal units. The
// additive C2 switch is precomputed at load, independently from its equations.
struct TaggedCauchyZBLPair {
  double inner = 0.0, outer = 0.0;
  double screening_length = 0.0, amplitude = 0.0;
  double cubic = 0.0, quartic = 0.0, constant = 0.0;
};

// A loaded, hash-verified `ye3t_tagged_cauchy_slice_v2` artifact, reduced to
// exactly what the native real-arithmetic evaluator needs: the compiled
// `real_moment_program` (already lowered to real tesseral components and
// already restricted to the fitted feature space, so `terms[*].feature_index`
// addresses `beta` directly), the per-channel real-form bindings, and the
// PACE ChebExpCos radial definition. `moment_program` (complex) and
// `execution_strategy` are intentionally not read; only the real program is
// implemented natively.
struct TaggedCauchyModel {
  static TaggedCauchyModel load(const std::string &path);
  double memory_usage() const;
  bool has_ordinary_backbone() const { return !ordinary_model_path.empty(); }
  bool is_physical_image() const
  { return deployment_kind != TaggedCauchyDeploymentKind::LegacyMomentV2; }

  std::string model_path;
  std::string self_hash;
  // A composite deployment keeps the independently readable ordinary .yace
  // and tagged JSON components beside one hash-bound manifest.  The tagged
  // loader verifies both byte hashes, then records the ordinary path for the
  // enclosing PairYE3T runtime.  Plain V2/V3 tagged slices leave these empty.
  std::string composite_manifest_path;
  std::string composite_self_hash;
  std::string ordinary_model_path;
  std::string ordinary_model_hash;
  std::string tagged_component_hash;
  TaggedCauchyDeploymentKind deployment_kind = TaggedCauchyDeploymentKind::LegacyMomentV2;
  TaggedCauchySourceRealization source_realization = TaggedCauchySourceRealization::PaceChebExpCos;
  std::string compiler_artifact_hash;
  std::string source_plan_hash;
  std::string schedule_hash;
  std::string readout_hash;
  std::string deployment_identity_hash;

  std::vector<std::string> species_order;
  double cutoff = 0.0;
  // Dense [central species][neighbor species], distinct from host cutoff.
  std::vector<double> pair_cutoffs;
  std::vector<TaggedCauchyZBLPair> zbl_pairs;
  int tag_count = 0;
  int feature_count = 0;

  std::vector<double> offsets;    // one per species_order index, eV
  std::string offset_mode;        // "fitted_species_offsets" or "composition_fixed";
                                  // both are one-value-per-species,
                                  // counted once per owned atom, no force/virial,
                                  // so the evaluator treats them identically --
                                  // this is recorded for provenance/logging only.

  // beta[species_index][feature_index]: the readout weights are
  // per-central-species (the real program/features stay species-
  // independent; only which beta vector is dotted with F depends on the
  // center's species). `beta.size() == species_order.size()`, every inner
  // vector has length `feature_count`. A JSON `beta` array (rather than a
  // {species: [...]} mapping) is only accepted when species_order has
  // exactly one entry, and is loaded as that single species' vector --
  // this is the one-species backward-compatible form of the schema.
  std::vector<std::vector<double>> beta;

  // radial_definition (kind == "pace_cheb_exp_cos" only)
  double radial_rc = 0.0;
  double radial_cutoff_width = 0.0;
  double radial_lambda = 0.0;
  int radial_count = 0;

  std::vector<TaggedCauchyRealForm> real_forms;
  std::vector<TaggedCauchyChannel> channels;    // dense by channel_index, 0..N-1
  int total_component_count = 0;                // sum of channel widths
  std::vector<int> distinct_angular_l;          // sorted unique l values used by channels

  std::vector<std::pair<int, int>> real_density_keys;    // (channel, a)
  std::vector<int> real_density_flat_index;              // per key, flat component index

  // real_moment_keys[m] is a multiset of (channel, a) factors, evaluated on
  // one edge; stored both as the original (possibly repeating) list and as
  // flat component indices in the same order for the per-edge product.
  std::vector<std::vector<std::pair<int, int>>> real_moment_keys;
  std::vector<std::vector<int>> real_moment_flat_indices;

  std::vector<TaggedCauchyTerm> terms;
  std::vector<TaggedCauchyAdjointTerm> adjoint_terms;
  TaggedCauchyExecutionPortfolio execution_portfolio;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_TAGGED_CAUCHY_MODEL_H
