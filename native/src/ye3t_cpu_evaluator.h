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

#ifndef LMP_YE3T_CPU_EVALUATOR_H
#define LMP_YE3T_CPU_EVALUATOR_H

// clang-format off
#include "ye3t_yace_model.h"
#include "ye3t_cpu_batching.h"
#include "ye3t_cpu_source_tiling.h"
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
// clang-format on

namespace YE3T_LAMMPS {

class YE3TCPUEvaluator {
 public:
  explicit YE3TCPUEvaluator(const YACEModel *model);

  void evaluate(int atom_count, const int *central_species, int edge_count, const int *edge_centers,
                const int *edge_neighbor_species, const double *edge_vectors,
                double *atomic_energies, double *edge_gradients);

  double memory_usage() const;
  std::size_t source_tile_target_bytes() const { return source_tile_policy_.target_bytes; }
  int source_tile_edge_capacity() const { return source_tile_policy_.edge_capacity; }
  double maximum_imaginary_density() const { return maximum_imaginary_density_; }

 private:
  // Test access changes only the schedule, never the mathematical model. There
  // is intentionally no runtime/user-facing source-budget override.
  friend struct YE3TCPUEvaluatorTestAccess;
  CPUSourceTilePolicy source_tile_policy_;
  // Production runs complete-center batches with one source preparation per
  // edge. Only the test friend can select the former replay schedule as oracle.
  bool reference_edge_tiling_ = false;
  int active_angular_maximum_ = 0;
  std::uint64_t last_source_edges_evaluated_ = 0;
  std::uint64_t last_complete_batches_ = 0;
  CPUSpeciesBatches center_batches_;
  std::vector<int> center_edge_offsets_, edge_cursors_, edge_order_;
  std::vector<int> complete_central_species_, complete_edge_centers_;
  std::vector<int> complete_neighbor_species_, complete_original_edges_;
  std::vector<double> complete_vectors_, complete_energies_, complete_gradients_;
  void evaluate_complete_batch(int atom_count, const int *central_species, int edge_count,
                               const int *edge_centers, const int *edge_neighbor_species,
                               const double *edge_vectors, double *atomic_energies,
                               double *edge_gradients, int center_index_base,
                               int source_edge_capacity);
  void prepare_source_tile(int atom_count, const int *central_species, int edge_count,
                           const int *edge_centers, const int *edge_neighbor_species,
                           const double *edge_vectors, int center_index_base);

  template <typename Value>
  static void resize_source_table(std::vector<Value> &values, std::size_t size)
  {
    // The outer readout chunksize must not determine edge-table capacity.
    // Explicit reserve avoids geometric growth beyond the internal tile size.
    // Capacity is reused after the first full-sized tile; allocator overhead is
    // implementation-defined and not part of the working-set estimate.
    if (values.capacity() < size) values.reserve(size);
    values.resize(size);
  }

  template <typename Value> static void resize_growing(std::vector<Value> &values, std::size_t size)
  {
    if (values.capacity() < size) {
      const std::size_t grown = values.capacity() + values.capacity() / 2 + 64;
      values.reserve(grown > size ? grown : size);
    }
    values.resize(size);
  }

  void evaluate_readout_batch(const YACESpecies &, int, const std::complex<double> *,
                              std::complex<double> *, std::complex<double> *);
  CPUSpeciesBatches readout_batches_;
  std::vector<std::complex<double>> batch_atomic_values_;
  std::vector<std::complex<double>> batch_atomic_adjoint_;
  std::vector<std::complex<double>> batch_density_;

  const YACEModel *model_;
  double maximum_imaginary_density_ = 0.0;

  void evaluate_block_program(const YACESpecies &species, const std::complex<double> *input,
                              std::complex<double> &output, std::complex<double> *input_adjoint);
  void evaluate_block_program_tiled(const YACESpecies &species, const std::complex<double> *input,
                                    std::int64_t batch_size, std::int64_t input_dimension,
                                    std::complex<double> *output,
                                    std::complex<double> *input_adjoint);
  void evaluate_scalar_power_program_tiled(const YACESpecies &species,
                                           const std::complex<double> *input,
                                           std::int64_t batch_size, std::int64_t input_dimension,
                                           std::complex<double> *output,
                                           std::complex<double> *input_adjoint);
  void evaluate_coupled_product_program_tiled(const YACESpecies &species,
                                              const std::complex<double> *input,
                                              std::int64_t batch_size, std::int64_t input_dimension,
                                              std::complex<double> *output,
                                              std::complex<double> *input_adjoint);

  std::vector<double> radii_;
  std::vector<double> radial_directions_;
  std::vector<std::int64_t> edge_bonds_;
  std::vector<std::int64_t> bond_counts_;
  std::vector<std::int64_t> bond_offsets_;
  std::vector<std::int64_t> bond_cursors_;
  std::vector<std::int64_t> bond_edges_;
  std::vector<double> gathered_radii_;
  std::vector<double> gathered_values_;
  std::vector<double> gathered_derivatives_;
  std::vector<double> radial_base_values_;
  std::vector<double> radial_base_derivatives_;
  std::vector<double> contracted_values_;
  std::vector<double> contracted_derivatives_;
  std::vector<double> angular_plan_;
  std::vector<std::complex<double>> angular_workspace_;
  std::vector<std::complex<double>> angular_values_;
  std::vector<std::complex<double>> angular_derivatives_;
  std::vector<std::int64_t> atomic_offsets_;
  std::vector<std::int64_t> source_offsets_;
  std::vector<std::complex<double>> source_values_;
  std::vector<std::complex<double>> source_adjoint_;
  std::vector<std::complex<double>> atomic_values_;
  std::vector<std::complex<double>> atomic_adjoint_;
  std::vector<std::complex<double>> density_;
  std::vector<std::complex<double>> monomial_workspace_;
  std::vector<double> tiled_monomial_workspace_;
  std::vector<std::complex<double>> block_outputs_;
  std::vector<std::complex<double>> block_output_adjoint_;
  std::vector<std::complex<double>> block_powers_;
  std::vector<std::complex<double>> block_prefix_;
  std::vector<std::complex<double>> block_monomials_;
  std::vector<std::complex<double>> block_monomial_adjoint_;
  std::vector<double> tiled_block_workspace_;
  std::vector<double> tiled_scalar_workspace_;
  std::vector<double> tiled_coupled_product_workspace_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_CPU_EVALUATOR_H
