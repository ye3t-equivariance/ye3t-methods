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

#ifndef LMP_YE3T_CPU_SOURCE_TILING_H
#define LMP_YE3T_CPU_SOURCE_TILING_H

#include "ye3t_yace_model.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace YE3T_LAMMPS {

// Internal implementation policy, not model data or pair_style input. The target
// covers edge-local tables/scratch, not the full evaluator, allocator overhead,
// LAMMPS storage, or immutable spline/readout plans. It does not reserve cache.
// Production evaluates complete 8/16-center batches. This is a SOFT target:
// a full eight-center tile (or fewer remaining centers) may exceed it; complete
// environments are never split for cache capacity and source tables are not replayed.
struct CPUSourceTilePolicy {
  std::size_t target_bytes = 0;
  std::size_t bytes_per_edge = 0;
  std::size_t fixed_bytes = 0;
  int edge_capacity = 1;
};

// Query once at evaluator construction. Zero means unavailable; no filesystem,
// MPI communication, environment override, or per-step hardware query is used.
std::size_t cpu_reported_l2_cache_bytes() noexcept;
CPUSourceTilePolicy make_cpu_source_tile_policy(int radial_base_width, int contracted_width,
                                                int angular_width, int bond_count,
                                                std::size_t angular_plan_bytes,
                                                std::size_t reported_l2_bytes);

// Low-level pass driver retained for reference/parity tests. Production passes
// the entire complete-batch edge count as capacity, so only one tile is used.
// With a smaller test capacity it exercises the multi-tile edge-replay policy.
// Ordered contiguous EDGE tiles, deliberately independent of the center/readout
// batch. A tile boundary may split one center's neighbor sum; no partial density
// is sent to readout. The last tile stays live across readout and is reused first
// by the VJP. Other tiles are rebuilt once, so one-tile batches never recompute.
// Readout runs once even for zero edges (isolated-atom reference energies).
// All callbacks are synchronous; readout must not overwrite edge-table scratch.
template <class Prepare, class Accumulate, class Readout, class Pullback>
void cpu_source_tiled_passes(int edge_count, int edge_capacity, Prepare prepare,
                             Accumulate accumulate, Readout readout, Pullback pullback)
{
  if (edge_count < 0 || edge_capacity <= 0)
    throw std::invalid_argument("invalid CPU source-tile dimensions");
  int last_begin = 0;
  for (int begin = 0; begin < edge_count;) {
    const int end = begin + std::min(edge_capacity, edge_count - begin);
    prepare(begin, end);
    accumulate(begin, end);
    last_begin = begin;
    begin = end;
  }
  readout();
  if (edge_count == 0) return;
  pullback(last_begin, edge_count);    // reuse the resident forward tile
  for (int end = last_begin; end > 0;) {
    const int begin = end - std::min(edge_capacity, end);
    prepare(begin, end);
    pullback(begin, end);
    end = begin;
  }
}

// These are the original CPU source arithmetic, with edge-local table pointers.
// The caller validates geometry/species, checks the physical cutoff, and supplies
// prevalidated model maps. Source accumulation order and complex VJP convention
// are unchanged. Sharing this code permits real contraction/scheduling tests
// without providing a replacement for the external YE3T runtime kernels.
inline void cpu_accumulate_ace_source_edge(const YACEBond &bond, const double *radial_base_values,
                                           const double *contracted_values,
                                           const std::complex<double> *angular_values,
                                           std::complex<double> *source)
{
  for (std::size_t term = 0; term < bond.radial_channel_outputs.size(); ++term) {
    source[static_cast<std::size_t>(bond.radial_channel_outputs[term])] +=
        radial_base_values[static_cast<std::size_t>(bond.radial_channel_indices[term])];
  }
  for (std::size_t term = 0; term < bond.angular_channel_outputs.size(); ++term) {
    source[static_cast<std::size_t>(bond.angular_channel_outputs[term])] +=
        contracted_values[static_cast<std::size_t>(bond.contracted_channel_indices[term])] *
        angular_values[static_cast<std::size_t>(bond.angular_channel_nonnegative_indices[term])];
  }
}

inline void cpu_pullback_ace_source_edge(
    const YACEBond &bond, const double *direction, const double *radial_base_derivatives,
    const double *contracted_values, const double *contracted_derivatives,
    const std::complex<double> *angular_values, const std::complex<double> *angular_derivatives,
    const std::complex<double> *source_adjoint, double *gradient)
{
  double gradient_x = 0.0;
  double gradient_y = 0.0;
  double gradient_z = 0.0;
  const double direction_x = direction[0];
  const double direction_y = direction[1];
  const double direction_z = direction[2];
  for (std::size_t term = 0; term < bond.radial_channel_outputs.size(); ++term) {
    const double root =
        source_adjoint[static_cast<std::size_t>(bond.radial_channel_outputs[term])].real();
    const double weight =
        root * radial_base_derivatives[static_cast<std::size_t>(bond.radial_channel_indices[term])];
    gradient_x += weight * direction_x;
    gradient_y += weight * direction_y;
    gradient_z += weight * direction_z;
  }
  for (std::size_t term = 0; term < bond.angular_channel_outputs.size(); ++term) {
    const std::complex<double> &root =
        source_adjoint[static_cast<std::size_t>(bond.angular_channel_outputs[term])];
    const double root_real = root.real();
    const double root_imaginary = root.imag();
    const std::size_t radial = static_cast<std::size_t>(bond.contracted_channel_indices[term]);
    const std::size_t angular =
        static_cast<std::size_t>(bond.angular_channel_nonnegative_indices[term]);
    const double radial_value = contracted_values[radial];
    const double radial_derivative = contracted_derivatives[radial];
    const std::complex<double> angular_value = angular_values[angular];
    const std::complex<double> *angular_gradient = angular_derivatives + angular * 3;
    const double radial_weight = radial_derivative *
        (root_real * angular_value.real() + root_imaginary * angular_value.imag());
    const double angular_weight_real = radial_value * root_real;
    const double angular_weight_imaginary = radial_value * root_imaginary;
    gradient_x += radial_weight * direction_x + angular_weight_real * angular_gradient[0].real() +
        angular_weight_imaginary * angular_gradient[0].imag();
    gradient_y += radial_weight * direction_y + angular_weight_real * angular_gradient[1].real() +
        angular_weight_imaginary * angular_gradient[1].imag();
    gradient_z += radial_weight * direction_z + angular_weight_real * angular_gradient[2].real() +
        angular_weight_imaginary * angular_gradient[2].imag();
  }
  gradient[0] = gradient_x;
  gradient[1] = gradient_y;
  gradient[2] = gradient_z;
}

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_CPU_SOURCE_TILING_H
