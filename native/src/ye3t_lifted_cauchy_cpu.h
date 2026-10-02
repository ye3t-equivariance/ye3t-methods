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

#ifndef LMP_YE3T_LIFTED_CAUCHY_CPU_H
#define LMP_YE3T_LIFTED_CAUCHY_CPU_H

// clang-format off
#include "ye3t_lifted_cauchy_model.h"
#include "ye3t_cpu_batching.h"
#include <array>
#include <cstddef>
#include <vector>
// clang-format on

namespace YE3T_LAMMPS {

enum class LiftedCauchySourcePolicy { DIRECT_Q, FACTORIZED };

struct LiftedCauchyEdge {
  int neighbor_species_index = -1;
  std::array<double, 3> displacement{};
};

namespace detail {
  struct LiftedCauchyAngularPlan {
    std::vector<std::vector<double>> derivative_coefficients;
    std::vector<double> scales;
  };
}    // namespace detail

class LiftedCauchyCPUSource {
 public:
  explicit LiftedCauchyCPUSource(const LiftedCauchyModel *model);

  void accumulate(const std::vector<LiftedCauchyEdge> &edges, LiftedCauchySourcePolicy policy,
                  std::vector<double> &source_values) const;

  void vjp(const std::vector<LiftedCauchyEdge> &edges, const std::vector<double> &source_adjoint,
           LiftedCauchySourcePolicy policy,
           std::vector<std::array<double, 3>> &edge_gradients) const;

  double memory_usage() const;

 private:
  const LiftedCauchyModel *model_;
  // Built once, immutable afterwards. Model metadata must also remain immutable.
  std::vector<detail::LiftedCauchyAngularPlan> angular_plans_;
};

class LiftedCauchyCPULoweredReadout {
 public:
  explicit LiftedCauchyCPULoweredReadout(const LiftedCauchyModel *model);

  void evaluate(int atom_count, const int *central_species_indices, const double *source_values,
                double *atomic_energies, double *source_adjoint);

  double memory_usage() const;

 private:
  void evaluate_head(const LiftedCauchyHead &head, int atom_count, const double *source_values,
                     double *atomic_energies, double *source_adjoint);

  const LiftedCauchyModel *model_;
  std::vector<double> workspace_;
  CPUSpeciesBatches readout_batches_;
  std::vector<double> batch_source_;
  std::vector<double> batch_adjoint_;
  std::vector<double> batch_energy_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_LIFTED_CAUCHY_CPU_H
