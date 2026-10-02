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

#include "ye3t_lifted_cauchy_cpu.h"

#include "ye3t_runtime_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

LiftedCauchyCPULoweredReadout::LiftedCauchyCPULoweredReadout(const LiftedCauchyModel *model) :
    model_(model)
{
  if (model_ == nullptr)
    throw std::invalid_argument("lifted-Cauchy lowered readout requires a model");
  if (model_->source_variable_count <= 0 || model_->heads.empty())
    throw std::invalid_argument(
        "lifted-Cauchy lowered readout requires source variables and heads");

  std::int64_t maximum_factor_count = 0;
  for (const auto &head : model_->heads)
    maximum_factor_count = std::max(maximum_factor_count, head.polynomial.maximum_factor_count);
  if (static_cast<std::uint64_t>(maximum_factor_count) >
      (static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) - 2) / 4)
    throw std::overflow_error("lifted-Cauchy lowered-readout workspace size overflows");
  workspace_.resize(static_cast<std::size_t>(4 * maximum_factor_count + 2));
}

void LiftedCauchyCPULoweredReadout::evaluate_head(const LiftedCauchyHead &head, int atom_count,
                                                  const double *source_values,
                                                  double *atomic_energies, double *source_adjoint)
{
  const auto &polynomial = head.polynomial;
  const std::int64_t source_count = model_->source_variable_count;
  const std::int64_t term_count =
      static_cast<std::int64_t>(polynomial.monomial_coefficients.size());
  if (term_count == 0) {
    std::fill(atomic_energies, atomic_energies + atom_count, polynomial.offset);
    std::fill(source_adjoint, source_adjoint + static_cast<std::int64_t>(atom_count) * source_count,
              0.0);
    return;
  }

  ye3t::runtime::symmetric_power_sparse_monomial_linear_forward_adjoint_prevalidated<double>(
      source_values, atom_count, source_count, polynomial.factor_offsets.data(),
      polynomial.factor_indices.data(), polynomial.factor_exponents.data(),
      static_cast<std::int64_t>(polynomial.factor_indices.size()),
      polynomial.monomial_coefficients.data(), term_count, polynomial.maximum_factor_count,
      workspace_.data(), static_cast<std::int64_t>(workspace_.size()), atomic_energies,
      source_adjoint);
  for (int atom = 0; atom < atom_count; ++atom) {
    atomic_energies[atom] += polynomial.offset;
    if (!std::isfinite(atomic_energies[atom]))
      throw std::runtime_error("lifted-Cauchy lowered readout produced a non-finite energy");
  }
}

void LiftedCauchyCPULoweredReadout::evaluate(int atom_count, const int *central_species_indices,
                                             const double *source_values, double *atomic_energies,
                                             double *source_adjoint)
{
  if (atom_count < 0)
    throw std::invalid_argument("lifted-Cauchy lowered readout received a negative atom count");
  if (atom_count == 0) return;
  if (central_species_indices == nullptr || source_values == nullptr ||
      atomic_energies == nullptr || source_adjoint == nullptr)
    throw std::invalid_argument("lifted-Cauchy lowered readout received a null array");
  const std::int64_t source_count = model_->source_variable_count;
  if (static_cast<std::uint64_t>(atom_count) >
      std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(source_count))
    throw std::overflow_error("lifted-Cauchy lowered readout batch size overflows");

  const int first_head = central_species_indices[0];
  bool homogeneous = first_head >= 0 && first_head < static_cast<int>(model_->heads.size());
  for (int atom = 0; atom < atom_count; ++atom) {
    const int head = central_species_indices[atom];
    if (head < 0 || head >= static_cast<int>(model_->heads.size()))
      throw std::invalid_argument("lifted-Cauchy lowered readout received an "
                                  "invalid central species index");
    homogeneous = homogeneous && head == first_head;
  }
  if (homogeneous) {
    evaluate_head(model_->heads[static_cast<std::size_t>(first_head)], atom_count, source_values,
                  atomic_energies, source_adjoint);
    return;
  }

  readout_batches_.build(atom_count, static_cast<int>(model_->heads.size()),
                         central_species_indices);
  const std::size_t width = static_cast<std::size_t>(source_count);
  for (std::size_t head = 0; head < model_->heads.size(); ++head) {
    const int finish = readout_batches_.offsets[head + 1];
    for (int begin = readout_batches_.offsets[head]; begin < finish;) {
      const int count = std::min(32, finish - begin);
      batch_source_.resize(static_cast<std::size_t>(count) * width);
      batch_adjoint_.resize(static_cast<std::size_t>(count) * width);
      batch_energy_.resize(static_cast<std::size_t>(count));
      for (int lane = 0; lane < count; ++lane) {
        const int atom = readout_batches_.atoms[begin + lane];
        std::copy_n(source_values + static_cast<std::size_t>(atom) * width, width,
                    batch_source_.data() + static_cast<std::size_t>(lane) * width);
      }
      evaluate_head(model_->heads[head], count, batch_source_.data(), batch_energy_.data(),
                    batch_adjoint_.data());
      for (int lane = 0; lane < count; ++lane) {
        const int atom = readout_batches_.atoms[begin + lane];
        atomic_energies[atom] = batch_energy_[lane];
        std::copy_n(batch_adjoint_.data() + static_cast<std::size_t>(lane) * width, width,
                    source_adjoint + static_cast<std::size_t>(atom) * width);
      }
      begin += count;
    }
  }
}

double LiftedCauchyCPULoweredReadout::memory_usage() const
{
  return readout_batches_.memory_usage() +
      static_cast<double>((workspace_.capacity() + batch_source_.capacity() +
                           batch_adjoint_.capacity() + batch_energy_.capacity()) *
                          sizeof(double));
}

}    // namespace YE3T_LAMMPS
