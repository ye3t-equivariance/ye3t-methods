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

#ifndef LMP_YE3T_TAGGED_CAUCHY_CPU_H
#define LMP_YE3T_TAGGED_CAUCHY_CPU_H

#include "ye3t_tagged_cauchy_model.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace YE3T_LAMMPS {

enum class TaggedCauchyExecutionPolicy {
  COMPILED_DIRECT,
  GENERIC_DAG,
  SYMMETRIC_POWER,
  BLOCK,
  AUTO,
};

// Native CPU evaluator for a loaded `ye3t_tagged_cauchy_slice_v2` model's
// `real_moment_program`. Consumes the same edge-gathering convention as the
// other YE3T CPU evaluators: a full neighbor list (Newton on), periodic
// images as distinct edge occurrences, edges grouped contiguously by owned
// center in `edge_offsets` (CSR style), and `edge_vectors[e] = r_neighbor -
// r_center` per edge. Returns per-atom energies (species offset included)
// and, per edge, `g = dE_i/d(edge_vector)` in the same convention the
// caller already uses for the lifted-Cauchy path (force on center += g,
// force on neighbor -= g; virial via the caller's ev_tally_xyz).
class TaggedCauchyCPUEvaluator {
 public:
  explicit TaggedCauchyCPUEvaluator(
      const TaggedCauchyModel *model,
      TaggedCauchyExecutionPolicy policy = TaggedCauchyExecutionPolicy::COMPILED_DIRECT);

  void evaluate(int atom_count, const int *central_species_indices, const std::size_t *edge_offsets,
                const int *edge_neighbor_species, const double *edge_vectors,
                double *atomic_energies, double *edge_gradients,
                double *atomic_features = nullptr);

  double memory_usage() const;
  const char *selected_evaluator_name() const;
  bool auto_requested() const;
  bool auto_calibrated() const;
  bool policy_eligible(TaggedCauchyExecutionPolicy policy) const;

  // Run a bounded, model-specific initialization calibration on deterministic
  // source environments.  The four entries are elapsed seconds for
  // compiled-direct, generic-DAG, symmetric-power, and block, respectively;
  // an ineligible candidate is reported as infinity.  This routine does not
  // freeze a route: PairYE3T broadcasts the rank-zero decision before calling
  // freeze_auto_policy(), so every MPI rank executes the same schedule.
  std::array<double, 4> calibrate_auto(int center_count, int neighbors_per_center,
                                       int measured_repeats);
  bool auto_candidate_confidently_faster(TaggedCauchyExecutionPolicy policy,
                                         double retention_ratio) const;
  const std::array<double, 4> &auto_calibration_lower_seconds() const;
  const std::array<double, 4> &auto_calibration_upper_seconds() const;
  void freeze_auto_policy(TaggedCauchyExecutionPolicy policy);

 private:
  // Evaluate every channel's real component (and, if requested, its
  // Cartesian gradient with respect to the raw edge vector) for a batch of
  // edges. The source calculation is edge-local, so one batch may contain
  // contiguous edges from several owned centers. `values` is laid out
  // `[edge][component]` (component = channel.component_offset + a, flat
  // over `model_->total_component_count`); `gradients` is
  // `[edge][component][3]` and is left empty when `need_gradient` is false.
  void compute_edge_components(std::int64_t edge_count, const int *edge_neighbor_species,
                               const double *edge_vectors, const double *edge_cutoffs, bool need_gradient,
                               std::vector<double> &values, std::vector<double> &gradients);

  const TaggedCauchyModel *model_;
  TaggedCauchyExecutionPolicy requested_policy_ = TaggedCauchyExecutionPolicy::COMPILED_DIRECT;
  TaggedCauchyExecutionPolicy selected_policy_ = TaggedCauchyExecutionPolicy::COMPILED_DIRECT;
  bool auto_calibrated_ = false;
  bool auto_calibration_in_progress_ = false;
  std::array<double, 4> auto_calibration_lower_seconds_{};
  std::array<double, 4> auto_calibration_upper_seconds_{};

  int total_components_ = 0;
  int angular_maximum_ = -1;
  std::vector<int> l_list_;                  // distinct angular momenta, ascending
  std::vector<int> l_slot_of_l_;             // indexed by l value; position in l_list_
  std::vector<int> channel_angular_slot_;    // indexed by channel index
  std::vector<int> real_form_angular_slot_;
  std::vector<std::size_t> real_form_offsets_;
  std::vector<std::size_t> channel_real_form_offsets_;
  std::vector<int> jacobi_max_q_by_slot_;
  std::vector<std::size_t> jacobi_offsets_by_slot_;
  std::vector<double> jacobi_values_;
  std::vector<double> jacobi_derivatives_;

  // Per-evaluate-call scratch, grown on demand and reused across centers and
  // chunks to avoid steady-state heap churn.
  std::vector<double> radii_;
  std::vector<double> radial_cutoffs_;
  std::vector<double> radial_cutoff_widths_;
  std::vector<double> radial_lambdas_;
  std::vector<double> radial_values_;
  std::vector<double> radial_derivatives_;
  std::vector<double> unit_vectors_;
  std::vector<double> angular_plan_;
  std::vector<std::complex<double>> angular_values_;
  std::vector<std::complex<double>> angular_derivatives_;
  // Retained only for the zero-separation compatibility fallback. Physical
  // LAMMPS neighbor edges use the recurrence above.
  std::vector<std::vector<std::complex<double>>> angular_values_by_slot_;
  std::vector<std::vector<std::complex<double>>> angular_derivatives_by_slot_;
  std::vector<double> real_angular_values_;
  std::vector<double> real_angular_derivatives_;

  std::vector<double> component_values_;
  std::vector<double> component_gradients_;
  std::vector<double> source_batch_component_values_;
  std::vector<double> source_batch_component_gradients_;
  std::vector<double> source_batch_cutoffs_;
  std::vector<double> component_values_transposed_;

  std::vector<double> density_;
  std::vector<double> density_adjoint_;
  std::vector<double> moment_;
  std::vector<double> moment_adjoint_;
  std::vector<double> component_seed_;

  // Species-local product DAGs for the live per-edge moments. Each node is
  // `parent_value * component[factor]`, with `parent == -1` denoting the
  // constant one. Canonically ordered repeated factors become an exact
  // symmetric-power chain. Reversing the nodes accumulates multiplicities
  // without division by a possibly zero source value.
  std::vector<std::int64_t> species_moment_node_offsets_;
  std::vector<int> moment_node_parent_;
  std::vector<int> moment_node_factor_;
  std::vector<std::int64_t> moment_node_terminal_offsets_;
  std::vector<int> moment_node_terminal_moments_;
  std::vector<double> moment_node_values_;
  std::vector<double> moment_node_adjoint_;
  std::vector<double> edge_moment_node_values_;
  std::vector<double> batched_moment_node_values_;
  std::vector<double> moment_node_tile_adjoint_;
  std::vector<double> component_seed_tile_;
  std::vector<double> binary_moment_node_values_;
  std::vector<double> binary_moment_tile_adjoint_;
  std::vector<double> binary_outer_values_;
  std::vector<double> binary_outer_adjoint_;
  std::vector<double> binary_outer_bases_;
  std::vector<double> binary_outer_base_adjoint_;

  // Species-specific fixed-readout programs. Legacy V2 terms with identical
  // commutative density/moment factor multisets are coalesced at load time;
  // V3 retains compiler order because its separately certified adjoint is
  // the authoritative reverse schedule.
  std::vector<std::int64_t> species_term_offsets_;
  std::vector<std::int64_t> term_density_offsets_;
  std::vector<int> term_density_flat_;
  std::vector<std::int64_t> term_moment_offsets_;
  std::vector<int> term_moment_flat_;
  std::vector<int> term_p_;
  std::vector<double> term_coefficient_;
  std::vector<int> term_product_node_;
  int maximum_term_p_ = 0;
  std::vector<double> free_count_by_p_;

  // Species-local product DAGs for the coalesced V2 readout. Variables below
  // `density_key_count_` address density values; the remainder address
  // moment values. The same node schedule supplies the exact division-free
  // reverse pass.
  int density_key_count_ = 0;
  std::vector<std::int64_t> species_readout_node_offsets_;
  std::vector<int> readout_node_parent_;
  std::vector<int> readout_node_variable_;
  std::vector<double> readout_node_values_;
  std::vector<double> readout_node_adjoint_;

  // Only moments referenced by the species' live coalesced readout are
  // accumulated and pulled back for that center species.
  std::vector<std::int64_t> species_live_moment_offsets_;
  std::vector<int> species_live_moments_;
  std::vector<int> species_live_moment_nodes_;

  // V3 compiler-emitted division-free reverse schedule, flattened and
  // readout-folded at construction time just like the forward terms.
  std::vector<std::int64_t> adjoint_remaining_offsets_;
  std::vector<int> adjoint_remaining_flat_;
  std::vector<int> adjoint_source_index_;
  std::vector<std::vector<double>> folded_adjoint_coefficient_;

  // `species_has_channel_[s]` is true when at least one
  // channel has `neighbor_species_index == s`, precomputed once at
  // construction time so `compute_edge_components` can skip an edge's
  // entire per-channel combination (and its unit-vector computation)
  // before doing any of that work when the edge's neighbor species has no
  // channel at all (a no-op for single-species artifacts, where every
  // edge's species trivially has channels, but real for multi-species
  // artifacts where some species pairs have none).
  std::vector<bool> species_has_channel_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_TAGGED_CAUCHY_CPU_H
