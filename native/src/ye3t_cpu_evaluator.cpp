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

#include "ye3t_cpu_evaluator.h"

#include "ye3t_runtime_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace YE3T_LAMMPS {
namespace {

// Lanes per split-real DAG tile. The tile workspace is (4 * values + 2) * tile
// doubles, so wide catalogues trade vector width against cache footprint;
// the override exists for developer measurements only.
#ifndef YE3T_CPU_DAG_TILE
#define YE3T_CPU_DAG_TILE 8
#endif
  constexpr std::int64_t binary_dag_tile_size = YE3T_CPU_DAG_TILE;
  constexpr std::int64_t block_tile_size = 8;
  constexpr std::int64_t scalar_power_tile_size = 8;
  constexpr std::int64_t coupled_product_tile_size = 8;

}    // namespace

YE3TCPUEvaluator::YE3TCPUEvaluator(const YACEModel *model) : model_(model)
{
  if (model_ == nullptr) throw std::invalid_argument("YE3T evaluator requires a model");
  for (const auto &species : model_->species())
    for (const auto &channel : species.source_channels)
      if (channel.kind == YACEChannel::CONTRACTED_RADIAL_ANGULAR)
        active_angular_maximum_ = std::max(active_angular_maximum_, channel.angular);
  const std::int64_t maximum_angular_momentum = active_angular_maximum_;
  const std::int64_t angular_plan_size =
      ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(maximum_angular_momentum);
  angular_plan_.resize(static_cast<std::size_t>(angular_plan_size));
  ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<double>(
      maximum_angular_momentum, angular_plan_.data(), angular_plan_size,
      std::sqrt(4.0 * std::acos(-1.0)));
  // The recurrence uses the output arrays as its working storage.
  source_tile_policy_ = make_cpu_source_tile_policy(
      model_->maximum_radial_base_count(), model_->maximum_contracted_width(),
      static_cast<int>(ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum)),
      model_->bond_count(),
      angular_plan_.size() * sizeof(double) +
          angular_workspace_.size() * sizeof(std::complex<double>),
      cpu_reported_l2_cache_bytes());
  std::int64_t maximum_workspace = 0;
  std::int64_t maximum_tiled_workspace = 0;
  std::int64_t maximum_block_output = 0;
  std::int64_t maximum_block_power_storage = 0;
  std::int64_t maximum_block_factors = 0;
  std::int64_t maximum_block_monomials = 0;
  std::int64_t maximum_scalar_values = 0;
  std::int64_t maximum_coupled_product_components = 0;
  for (const auto &species : model_->species()) {
    const auto &plan = species.polynomial;
    if (plan.binary_dag) {
      const std::int64_t value_count = static_cast<std::int64_t>(plan.power_channels.size()) +
          static_cast<std::int64_t>(plan.binary_node_left.size());
      maximum_tiled_workspace =
          std::max(maximum_tiled_workspace, (4 * value_count + 2) * binary_dag_tile_size);
    } else {
      maximum_workspace = std::max(maximum_workspace,
                                   2 * static_cast<std::int64_t>(plan.power_channels.size()) +
                                       2 * static_cast<std::int64_t>(plan.dag_node_parents.size()));
    }
    std::int64_t block_output = 0;
    maximum_block_power_storage =
        std::max(maximum_block_power_storage, species.block_program.power_storage_size);
    for (const auto &block : species.block_program.power_plans) {
      block_output += block.output_dimension;
      if (block.direct_input_plan) continue;
      maximum_block_monomials = std::max(maximum_block_monomials, block.monomial_count);
      for (std::int64_t term = 0; term < block.monomial_count; ++term)
        maximum_block_factors =
            std::max(maximum_block_factors,
                     block.monomial_factor_offsets[static_cast<std::size_t>(term + 1)] -
                         block.monomial_factor_offsets[static_cast<std::size_t>(term)]);
    }
    for (const auto &route : species.block_program.routes)
      for (std::size_t term = 0; term + 1 < route.term_factor_offsets.size(); ++term)
        maximum_block_factors =
            std::max(maximum_block_factors,
                     route.term_factor_offsets[term + 1] - route.term_factor_offsets[term]);
    maximum_block_output = std::max(maximum_block_output, block_output);
    maximum_scalar_values =
        std::max(maximum_scalar_values, species.block_program.scalar_program.value_count);
    for (const auto &coupled : species.block_program.coupled_product_plans)
      maximum_coupled_product_components =
          std::max(maximum_coupled_product_components, coupled.total_node_components);
  }
  monomial_workspace_.resize(static_cast<std::size_t>(maximum_workspace));
  tiled_monomial_workspace_.resize(static_cast<std::size_t>(maximum_tiled_workspace));
  block_outputs_.resize(static_cast<std::size_t>(maximum_block_output));
  block_output_adjoint_.resize(static_cast<std::size_t>(maximum_block_output));
  block_powers_.resize(static_cast<std::size_t>(maximum_block_power_storage));
  block_prefix_.resize(static_cast<std::size_t>(maximum_block_factors + 1));
  block_monomials_.resize(static_cast<std::size_t>(maximum_block_monomials));
  block_monomial_adjoint_.resize(static_cast<std::size_t>(maximum_block_monomials));
  const std::int64_t tiled_block_scalars = 2 * block_tile_size *
      (maximum_block_power_storage + 2 * maximum_block_output + 2 * maximum_block_monomials +
       maximum_block_factors + 3);
  tiled_block_workspace_.resize(static_cast<std::size_t>(tiled_block_scalars));
  tiled_scalar_workspace_.resize(
      static_cast<std::size_t>(4 * scalar_power_tile_size * maximum_scalar_values));
  tiled_coupled_product_workspace_.resize(static_cast<std::size_t>(
      (4 * maximum_coupled_product_components + 2) * coupled_product_tile_size));
}

void YE3TCPUEvaluator::evaluate_block_program(const YACESpecies &species,
                                              const std::complex<double> *input,
                                              std::complex<double> &output,
                                              std::complex<double> *input_adjoint)
{
  const auto &program = species.block_program;
  if (program.routes.empty()) return;
  std::fill(block_output_adjoint_.begin(), block_output_adjoint_.end(),
            std::complex<double>(0.0, 0.0));

  for (std::size_t index = 0; index < program.power_channels.size(); ++index) {
    const std::int64_t channel = program.power_channels[index];
    const std::int64_t maximum = program.power_maximum_exponents[index];
    std::complex<double> *powers = block_powers_.data() + program.power_offsets[index];
    powers[0] = std::complex<double>(1.0, 0.0);
    if (maximum == 0) continue;
    powers[1] = input[static_cast<std::size_t>(channel)];
    for (std::int64_t exponent = 2; exponent <= maximum; ++exponent)
      powers[exponent] = powers[exponent - 1] * powers[1];
  }

  for (const auto &plan : program.power_plans) {
    std::complex<double> *plan_output = block_outputs_.data() + plan.output_storage_offset;
    if (plan.direct_input_plan) {
      for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
        const std::int64_t channel =
            plan.direct_input_channels[static_cast<std::size_t>(component)];
        if (channel < 0) continue;
        plan_output[component] = plan.direct_input_scales[static_cast<std::size_t>(component)] *
            input[static_cast<std::size_t>(channel)];
      }
      if (plan.conjugate_half_output) {
        const std::int64_t magnetic_width = 2 * plan.output_L + 1;
        for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
          for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
            const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
            plan_output[base + plan.output_L - magnetic] =
                phase * std::conj(plan_output[base + plan.output_L + magnetic]);
          }
      }
      continue;
    }
    for (std::int64_t term = 0; term < plan.monomial_count; ++term) {
      const std::int64_t begin = plan.monomial_factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t end = plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
      const std::int64_t first_component =
          plan.monomial_factor_components[static_cast<std::size_t>(begin)];
      const std::int64_t first_exponent =
          plan.monomial_factor_exponents[static_cast<std::size_t>(begin)];
      std::complex<double> value = block_powers_[static_cast<std::size_t>(
          plan.input_power_offsets[static_cast<std::size_t>(first_component)] + first_exponent)];
      for (std::int64_t factor = begin + 1; factor < end; ++factor) {
        const std::int64_t component =
            plan.monomial_factor_components[static_cast<std::size_t>(factor)];
        const std::int64_t exponent =
            plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
        value *= block_powers_[static_cast<std::size_t>(
            plan.input_power_offsets[static_cast<std::size_t>(component)] + exponent)];
      }
      block_monomials_[static_cast<std::size_t>(term)] = value;
    }
    for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
      const std::int64_t magnetic_component = component % (2 * plan.output_L + 1);
      if (plan.conjugate_half_output && magnetic_component < plan.output_L) continue;
      std::complex<double> value(0.0, 0.0);
      const std::int64_t begin = plan.output_offsets[static_cast<std::size_t>(component)];
      const std::int64_t end = plan.output_offsets[static_cast<std::size_t>(component + 1)];
      if (plan.real_coefficients) {
        for (std::int64_t coefficient = begin; coefficient < end; ++coefficient)
          value += plan.coefficient_values[static_cast<std::size_t>(coefficient)].real() *
              block_monomials_[static_cast<std::size_t>(
                  plan.coefficient_terms[static_cast<std::size_t>(coefficient)])];
      } else {
        for (std::int64_t coefficient = begin; coefficient < end; ++coefficient)
          value += plan.coefficient_values[static_cast<std::size_t>(coefficient)] *
              block_monomials_[static_cast<std::size_t>(
                  plan.coefficient_terms[static_cast<std::size_t>(coefficient)])];
      }
      plan_output[component] = value;
    }
    if (plan.conjugate_half_output) {
      const std::int64_t magnetic_width = 2 * plan.output_L + 1;
      for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
        for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
          const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
          plan_output[base + plan.output_L - magnetic] =
              phase * std::conj(plan_output[base + plan.output_L + magnetic]);
        }
    }
  }

  for (const auto &route : program.routes) {
    if (!route.term_factor_offsets.empty()) {
      for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
        const std::int64_t begin = route.term_factor_offsets[term];
        const std::int64_t end = route.term_factor_offsets[term + 1];
        block_prefix_[0] = std::complex<double>(1.0, 0.0);
        for (std::int64_t factor = begin; factor < end; ++factor) {
          const auto &plan = program.power_plans[static_cast<std::size_t>(
              route.term_factor_plans[static_cast<std::size_t>(factor)])];
          const std::complex<double> *values = block_outputs_.data() + plan.output_storage_offset;
          const std::int64_t component =
              route.term_factor_components[static_cast<std::size_t>(factor)];
          block_prefix_[static_cast<std::size_t>(factor - begin + 1)] =
              block_prefix_[static_cast<std::size_t>(factor - begin)] * values[component];
        }
        const std::complex<double> coefficient = route.coefficients[term];
        output += coefficient * block_prefix_[static_cast<std::size_t>(end - begin)];
        std::complex<double> suffix(1.0, 0.0);
        for (std::int64_t factor = end; factor-- > begin;) {
          const auto &plan = program.power_plans[static_cast<std::size_t>(
              route.term_factor_plans[static_cast<std::size_t>(factor)])];
          const std::complex<double> *values = block_outputs_.data() + plan.output_storage_offset;
          std::complex<double> *adjoint = block_output_adjoint_.data() + plan.output_storage_offset;
          const std::int64_t component =
              route.term_factor_components[static_cast<std::size_t>(factor)];
          const std::complex<double> derivative =
              coefficient * block_prefix_[static_cast<std::size_t>(factor - begin)] * suffix;
          adjoint[component] += std::conj(derivative);
          suffix *= values[component];
        }
      }
      continue;
    }
    const auto &left_plan = program.power_plans[static_cast<std::size_t>(route.left_plan)];
    const std::complex<double> *left = block_outputs_.data() + left_plan.output_storage_offset;
    std::complex<double> *left_adjoint =
        block_output_adjoint_.data() + left_plan.output_storage_offset;
    if (route.right_plan < 0) {
      for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
        const std::int64_t left_component = route.left_components[term];
        if (route.real_coefficients) {
          const double coefficient = route.coefficients[term].real();
          output += coefficient * left[left_component];
          left_adjoint[left_component] += coefficient;
        } else {
          const std::complex<double> coefficient = route.coefficients[term];
          output += coefficient * left[left_component];
          left_adjoint[left_component] += std::conj(coefficient);
        }
      }
      continue;
    }
    const auto &right_plan = program.power_plans[static_cast<std::size_t>(route.right_plan)];
    const std::complex<double> *right = block_outputs_.data() + right_plan.output_storage_offset;
    std::complex<double> *right_adjoint =
        block_output_adjoint_.data() + right_plan.output_storage_offset;
    for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
      const std::int64_t left_component = route.left_components[term];
      const std::int64_t right_component = route.right_components[term];
      if (route.real_coefficients) {
        const double coefficient = route.coefficients[term].real();
        output += coefficient * (left[left_component] * right[right_component]);
        left_adjoint[left_component] += coefficient * std::conj(right[right_component]);
        right_adjoint[right_component] += coefficient * std::conj(left[left_component]);
      } else {
        const std::complex<double> coefficient = route.coefficients[term];
        output += coefficient * left[left_component] * right[right_component];
        left_adjoint[left_component] += std::conj(coefficient * right[right_component]);
        right_adjoint[right_component] += std::conj(coefficient * left[left_component]);
      }
    }
  }

  for (const auto &plan : program.power_plans) {
    std::fill(block_monomial_adjoint_.begin(),
              block_monomial_adjoint_.begin() + plan.monomial_count,
              std::complex<double>(0.0, 0.0));
    std::complex<double> *plan_adjoint = block_output_adjoint_.data() + plan.output_storage_offset;
    if (plan.conjugate_half_output) {
      const std::int64_t magnetic_width = 2 * plan.output_L + 1;
      for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
        for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
          const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
          const std::int64_t negative = base + plan.output_L - magnetic;
          const std::int64_t positive = base + plan.output_L + magnetic;
          plan_adjoint[positive] += phase * std::conj(plan_adjoint[negative]);
          plan_adjoint[negative] = std::complex<double>(0.0, 0.0);
        }
    }
    if (plan.direct_input_plan) {
      for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
        const std::int64_t channel =
            plan.direct_input_channels[static_cast<std::size_t>(component)];
        if (channel < 0) continue;
        input_adjoint[static_cast<std::size_t>(channel)] +=
            plan.direct_input_scales[static_cast<std::size_t>(component)] * plan_adjoint[component];
      }
      continue;
    }
    for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
      const std::int64_t magnetic_component = component % (2 * plan.output_L + 1);
      if (plan.conjugate_half_output && magnetic_component < plan.output_L) continue;
      const std::int64_t begin = plan.output_offsets[static_cast<std::size_t>(component)];
      const std::int64_t end = plan.output_offsets[static_cast<std::size_t>(component + 1)];
      if (plan.real_coefficients) {
        for (std::int64_t coefficient = begin; coefficient < end; ++coefficient)
          block_monomial_adjoint_[static_cast<std::size_t>(
              plan.coefficient_terms[static_cast<std::size_t>(coefficient)])] +=
              plan.coefficient_values[static_cast<std::size_t>(coefficient)].real() *
              plan_adjoint[component];
      } else {
        for (std::int64_t coefficient = begin; coefficient < end; ++coefficient)
          block_monomial_adjoint_[static_cast<std::size_t>(
              plan.coefficient_terms[static_cast<std::size_t>(coefficient)])] +=
              plan_adjoint[component] *
              std::conj(plan.coefficient_values[static_cast<std::size_t>(coefficient)]);
      }
    }
    for (std::int64_t term = 0; term < plan.monomial_count; ++term) {
      const std::complex<double> root = block_monomial_adjoint_[static_cast<std::size_t>(term)];
      const std::int64_t begin = plan.monomial_factor_offsets[static_cast<std::size_t>(term)];
      const std::int64_t end = plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
      const std::int64_t support = end - begin;
      block_prefix_[0] = std::complex<double>(1.0, 0.0);
      for (std::int64_t local = 0; local < support; ++local) {
        const std::int64_t factor = begin + local;
        const std::int64_t component =
            plan.monomial_factor_components[static_cast<std::size_t>(factor)];
        const std::int64_t exponent =
            plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
        block_prefix_[static_cast<std::size_t>(local + 1)] =
            block_prefix_[static_cast<std::size_t>(local)] *
            block_powers_[static_cast<std::size_t>(
                plan.input_power_offsets[static_cast<std::size_t>(component)] + exponent)];
      }
      std::complex<double> suffix(1.0, 0.0);
      for (std::int64_t local = support; local-- > 0;) {
        const std::int64_t factor = begin + local;
        const std::int64_t component =
            plan.monomial_factor_components[static_cast<std::size_t>(factor)];
        const std::int64_t exponent =
            plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
        const std::int64_t power_offset =
            plan.input_power_offsets[static_cast<std::size_t>(component)];
        const std::complex<double> derivative = static_cast<double>(exponent) *
            block_prefix_[static_cast<std::size_t>(local)] * suffix *
            block_powers_[static_cast<std::size_t>(power_offset + exponent - 1)];
        input_adjoint[static_cast<std::size_t>(
            plan.input_channels[static_cast<std::size_t>(component)])] +=
            root * std::conj(derivative);
        suffix *= block_powers_[static_cast<std::size_t>(power_offset + exponent)];
      }
    }
  }
}

void YE3TCPUEvaluator::evaluate_block_program_tiled(
    const YACESpecies &species, const std::complex<double> *input, std::int64_t batch_size,
    std::int64_t input_dimension, std::complex<double> *output, std::complex<double> *input_adjoint)
{
  const auto &program = species.block_program;
  if (program.routes.empty() || batch_size == 0) return;
  if (batch_size < 0 || input_dimension <= 0 || input == nullptr || output == nullptr ||
      input_adjoint == nullptr)
    throw std::invalid_argument("invalid tiled block-program batch");

  std::int64_t output_dimension = 0;
  std::int64_t maximum_monomials = 0;
  std::int64_t maximum_factors = 0;
  for (const auto &plan : program.power_plans) {
    output_dimension =
        std::max(output_dimension, plan.output_storage_offset + plan.output_dimension);
    maximum_monomials = std::max(maximum_monomials, plan.monomial_count);
    for (std::int64_t term = 0; term < plan.monomial_count; ++term)
      maximum_factors = std::max(maximum_factors,
                                 plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)] -
                                     plan.monomial_factor_offsets[static_cast<std::size_t>(term)]);
  }
  for (const auto &route : program.routes)
    for (std::size_t term = 0; term + 1 < route.term_factor_offsets.size(); ++term)
      maximum_factors = std::max(
          maximum_factors, route.term_factor_offsets[term + 1] - route.term_factor_offsets[term]);

  const std::int64_t power_values = program.power_storage_size * block_tile_size;
  const std::int64_t output_values = output_dimension * block_tile_size;
  const std::int64_t monomial_values = maximum_monomials * block_tile_size;
  const std::int64_t prefix_values = (maximum_factors + 1) * block_tile_size;
  const std::int64_t required_workspace = 2 * power_values + 4 * output_values +
      4 * monomial_values + 2 * prefix_values + 4 * block_tile_size;
  if (static_cast<std::int64_t>(tiled_block_workspace_.size()) < required_workspace)
    throw std::logic_error("tiled block-program workspace is too small");

  double *power_real = tiled_block_workspace_.data();
  double *power_imaginary = power_real + power_values;
  double *block_real = power_imaginary + power_values;
  double *block_imaginary = block_real + output_values;
  double *block_adjoint_real = block_imaginary + output_values;
  double *block_adjoint_imaginary = block_adjoint_real + output_values;
  double *monomial_real = block_adjoint_imaginary + output_values;
  double *monomial_imaginary = monomial_real + monomial_values;
  double *monomial_adjoint_real = monomial_imaginary + monomial_values;
  double *monomial_adjoint_imaginary = monomial_adjoint_real + monomial_values;
  double *prefix_real = monomial_adjoint_imaginary + monomial_values;
  double *prefix_imaginary = prefix_real + prefix_values;
  double *suffix_real = prefix_imaginary + prefix_values;
  double *suffix_imaginary = suffix_real + block_tile_size;
  double *density_real = suffix_imaginary + block_tile_size;
  double *density_imaginary = density_real + block_tile_size;

  for (std::int64_t tile_begin = 0; tile_begin < batch_size; tile_begin += block_tile_size) {
    const std::int64_t lane_count = std::min(block_tile_size, batch_size - tile_begin);
    std::fill(block_adjoint_real, block_adjoint_real + output_values, 0.0);
    std::fill(block_adjoint_imaginary, block_adjoint_imaginary + output_values, 0.0);
    std::fill(density_real, density_real + lane_count, 0.0);
    std::fill(density_imaginary, density_imaginary + lane_count, 0.0);

    for (std::size_t index = 0; index < program.power_channels.size(); ++index) {
      const std::int64_t channel = program.power_channels[index];
      const std::int64_t maximum = program.power_maximum_exponents[index];
      const std::int64_t base = program.power_offsets[index] * block_tile_size;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        power_real[base + lane] = 1.0;
        power_imaginary[base + lane] = 0.0;
      }
      if (maximum == 0) continue;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const auto value = input[(tile_begin + lane) * input_dimension + channel];
        power_real[base + block_tile_size + lane] = value.real();
        power_imaginary[base + block_tile_size + lane] = value.imag();
      }
      for (std::int64_t exponent = 2; exponent <= maximum; ++exponent) {
        const std::int64_t previous = base + (exponent - 1) * block_tile_size;
        const std::int64_t current = base + exponent * block_tile_size;
        const std::int64_t first = base + block_tile_size;
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          const double left_real = power_real[previous + lane];
          const double left_imaginary = power_imaginary[previous + lane];
          const double right_real = power_real[first + lane];
          const double right_imaginary = power_imaginary[first + lane];
          power_real[current + lane] = left_real * right_real - left_imaginary * right_imaginary;
          power_imaginary[current + lane] =
              left_real * right_imaginary + left_imaginary * right_real;
        }
      }
    }

    for (const auto &plan : program.power_plans) {
      const std::int64_t plan_output = plan.output_storage_offset * block_tile_size;
      if (plan.direct_input_plan) {
        for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
          const std::int64_t channel =
              plan.direct_input_channels[static_cast<std::size_t>(component)];
          if (channel < 0) continue;
          const double scale = plan.direct_input_scales[static_cast<std::size_t>(component)];
          const std::int64_t destination = plan_output + component * block_tile_size;
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const auto value = input[(tile_begin + lane) * input_dimension + channel];
            block_real[destination + lane] = scale * value.real();
            block_imaginary[destination + lane] = scale * value.imag();
          }
        }
        if (plan.conjugate_half_output) {
          const std::int64_t magnetic_width = 2 * plan.output_L + 1;
          for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
            for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
              const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
              const std::int64_t negative =
                  plan_output + (base + plan.output_L - magnetic) * block_tile_size;
              const std::int64_t positive =
                  plan_output + (base + plan.output_L + magnetic) * block_tile_size;
              for (std::int64_t lane = 0; lane < lane_count; ++lane) {
                block_real[negative + lane] = phase * block_real[positive + lane];
                block_imaginary[negative + lane] = -phase * block_imaginary[positive + lane];
              }
            }
        }
        continue;
      }
      for (std::int64_t term = 0; term < plan.monomial_count; ++term) {
        const std::int64_t begin = plan.monomial_factor_offsets[static_cast<std::size_t>(term)];
        const std::int64_t end = plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
        const std::int64_t first_component =
            plan.monomial_factor_components[static_cast<std::size_t>(begin)];
        const std::int64_t first_exponent =
            plan.monomial_factor_exponents[static_cast<std::size_t>(begin)];
        const std::int64_t first =
            (plan.input_power_offsets[static_cast<std::size_t>(first_component)] + first_exponent) *
            block_tile_size;
        const std::int64_t monomial = term * block_tile_size;
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          monomial_real[monomial + lane] = power_real[first + lane];
          monomial_imaginary[monomial + lane] = power_imaginary[first + lane];
        }
        for (std::int64_t factor = begin + 1; factor < end; ++factor) {
          const std::int64_t component =
              plan.monomial_factor_components[static_cast<std::size_t>(factor)];
          const std::int64_t exponent =
              plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
          const std::int64_t source =
              (plan.input_power_offsets[static_cast<std::size_t>(component)] + exponent) *
              block_tile_size;
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const double left_real = monomial_real[monomial + lane];
            const double left_imaginary = monomial_imaginary[monomial + lane];
            const double right_real = power_real[source + lane];
            const double right_imaginary = power_imaginary[source + lane];
            monomial_real[monomial + lane] =
                left_real * right_real - left_imaginary * right_imaginary;
            monomial_imaginary[monomial + lane] =
                left_real * right_imaginary + left_imaginary * right_real;
          }
        }
      }

      for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
        const std::int64_t magnetic_component = component % (2 * plan.output_L + 1);
        if (plan.conjugate_half_output && magnetic_component < plan.output_L) continue;
        const std::int64_t destination = plan_output + component * block_tile_size;
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          block_real[destination + lane] = 0.0;
          block_imaginary[destination + lane] = 0.0;
        }
        const std::int64_t begin = plan.output_offsets[static_cast<std::size_t>(component)];
        const std::int64_t end = plan.output_offsets[static_cast<std::size_t>(component + 1)];
        for (std::int64_t coefficient = begin; coefficient < end; ++coefficient) {
          const auto value = plan.coefficient_values[static_cast<std::size_t>(coefficient)];
          const std::int64_t source =
              plan.coefficient_terms[static_cast<std::size_t>(coefficient)] * block_tile_size;
          if (plan.real_coefficients) {
            const double scale = value.real();
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              block_real[destination + lane] += scale * monomial_real[source + lane];
              block_imaginary[destination + lane] += scale * monomial_imaginary[source + lane];
            }
          } else {
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              const double right_real = monomial_real[source + lane];
              const double right_imaginary = monomial_imaginary[source + lane];
              block_real[destination + lane] +=
                  value.real() * right_real - value.imag() * right_imaginary;
              block_imaginary[destination + lane] +=
                  value.real() * right_imaginary + value.imag() * right_real;
            }
          }
        }
      }
      if (plan.conjugate_half_output) {
        const std::int64_t magnetic_width = 2 * plan.output_L + 1;
        for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
          for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
            const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
            const std::int64_t negative =
                plan_output + (base + plan.output_L - magnetic) * block_tile_size;
            const std::int64_t positive =
                plan_output + (base + plan.output_L + magnetic) * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              block_real[negative + lane] = phase * block_real[positive + lane];
              block_imaginary[negative + lane] = -phase * block_imaginary[positive + lane];
            }
          }
      }
    }

    for (const auto &route : program.routes) {
      if (!route.term_factor_offsets.empty()) {
        for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
          const std::int64_t begin = route.term_factor_offsets[term];
          const std::int64_t end = route.term_factor_offsets[term + 1];
          const std::int64_t support = end - begin;
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            prefix_real[lane] = 1.0;
            prefix_imaginary[lane] = 0.0;
          }
          for (std::int64_t local = 0; local < support; ++local) {
            const std::int64_t factor = begin + local;
            const auto &plan = program.power_plans[static_cast<std::size_t>(
                route.term_factor_plans[static_cast<std::size_t>(factor)])];
            const std::int64_t component =
                route.term_factor_components[static_cast<std::size_t>(factor)];
            const std::int64_t source = (plan.output_storage_offset + component) * block_tile_size;
            const std::int64_t previous = local * block_tile_size;
            const std::int64_t current = (local + 1) * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              const double left_real = prefix_real[previous + lane];
              const double left_imaginary = prefix_imaginary[previous + lane];
              const double right_real = block_real[source + lane];
              const double right_imaginary = block_imaginary[source + lane];
              prefix_real[current + lane] =
                  left_real * right_real - left_imaginary * right_imaginary;
              prefix_imaginary[current + lane] =
                  left_real * right_imaginary + left_imaginary * right_real;
            }
          }
          const auto coefficient = route.coefficients[term];
          const std::int64_t product = support * block_tile_size;
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const double product_real = prefix_real[product + lane];
            const double product_imaginary = prefix_imaginary[product + lane];
            density_real[lane] +=
                coefficient.real() * product_real - coefficient.imag() * product_imaginary;
            density_imaginary[lane] +=
                coefficient.real() * product_imaginary + coefficient.imag() * product_real;
            suffix_real[lane] = 1.0;
            suffix_imaginary[lane] = 0.0;
          }
          for (std::int64_t local = support; local-- > 0;) {
            const std::int64_t factor = begin + local;
            const auto &plan = program.power_plans[static_cast<std::size_t>(
                route.term_factor_plans[static_cast<std::size_t>(factor)])];
            const std::int64_t component =
                route.term_factor_components[static_cast<std::size_t>(factor)];
            const std::int64_t source = (plan.output_storage_offset + component) * block_tile_size;
            const std::int64_t prefix = local * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              const double prefix_suffix_real = prefix_real[prefix + lane] * suffix_real[lane] -
                  prefix_imaginary[prefix + lane] * suffix_imaginary[lane];
              const double prefix_suffix_imaginary =
                  prefix_real[prefix + lane] * suffix_imaginary[lane] +
                  prefix_imaginary[prefix + lane] * suffix_real[lane];
              const double gradient_real = coefficient.real() * prefix_suffix_real -
                  coefficient.imag() * prefix_suffix_imaginary;
              const double gradient_imaginary = coefficient.real() * prefix_suffix_imaginary +
                  coefficient.imag() * prefix_suffix_real;
              block_adjoint_real[source + lane] += gradient_real;
              block_adjoint_imaginary[source + lane] -= gradient_imaginary;
              const double value_real = block_real[source + lane];
              const double value_imaginary = block_imaginary[source + lane];
              const double next_suffix_real =
                  value_real * suffix_real[lane] - value_imaginary * suffix_imaginary[lane];
              suffix_imaginary[lane] =
                  value_real * suffix_imaginary[lane] + value_imaginary * suffix_real[lane];
              suffix_real[lane] = next_suffix_real;
            }
          }
        }
        continue;
      }
      const auto &left_plan = program.power_plans[static_cast<std::size_t>(route.left_plan)];
      const std::int64_t left_base = left_plan.output_storage_offset * block_tile_size;
      if (route.right_plan < 0) {
        for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
          const auto coefficient = route.coefficients[term];
          const std::int64_t left = left_base + route.left_components[term] * block_tile_size;
          if (route.real_coefficients) {
            const double scale = coefficient.real();
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              density_real[lane] += scale * block_real[left + lane];
              density_imaginary[lane] += scale * block_imaginary[left + lane];
              block_adjoint_real[left + lane] += scale;
            }
          } else {
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              density_real[lane] += coefficient.real() * block_real[left + lane] -
                  coefficient.imag() * block_imaginary[left + lane];
              density_imaginary[lane] += coefficient.real() * block_imaginary[left + lane] +
                  coefficient.imag() * block_real[left + lane];
              block_adjoint_real[left + lane] += coefficient.real();
              block_adjoint_imaginary[left + lane] -= coefficient.imag();
            }
          }
        }
        continue;
      }

      const auto &right_plan = program.power_plans[static_cast<std::size_t>(route.right_plan)];
      const std::int64_t right_base = right_plan.output_storage_offset * block_tile_size;
      for (std::size_t term = 0; term < route.coefficients.size(); ++term) {
        const auto coefficient = route.coefficients[term];
        const std::int64_t left = left_base + route.left_components[term] * block_tile_size;
        const std::int64_t right = right_base + route.right_components[term] * block_tile_size;
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          const double left_real = block_real[left + lane];
          const double left_imaginary = block_imaginary[left + lane];
          const double right_real = block_real[right + lane];
          const double right_imaginary = block_imaginary[right + lane];
          const double product_real = left_real * right_real - left_imaginary * right_imaginary;
          const double product_imaginary =
              left_real * right_imaginary + left_imaginary * right_real;
          if (route.real_coefficients) {
            const double scale = coefficient.real();
            density_real[lane] += scale * product_real;
            density_imaginary[lane] += scale * product_imaginary;
            block_adjoint_real[left + lane] += scale * right_real;
            block_adjoint_imaginary[left + lane] -= scale * right_imaginary;
            block_adjoint_real[right + lane] += scale * left_real;
            block_adjoint_imaginary[right + lane] -= scale * left_imaginary;
          } else {
            density_real[lane] +=
                coefficient.real() * product_real - coefficient.imag() * product_imaginary;
            density_imaginary[lane] +=
                coefficient.real() * product_imaginary + coefficient.imag() * product_real;

            const double left_gradient_real =
                coefficient.real() * right_real - coefficient.imag() * right_imaginary;
            const double left_gradient_imaginary =
                coefficient.real() * right_imaginary + coefficient.imag() * right_real;
            block_adjoint_real[left + lane] += left_gradient_real;
            block_adjoint_imaginary[left + lane] -= left_gradient_imaginary;
            const double right_gradient_real =
                coefficient.real() * left_real - coefficient.imag() * left_imaginary;
            const double right_gradient_imaginary =
                coefficient.real() * left_imaginary + coefficient.imag() * left_real;
            block_adjoint_real[right + lane] += right_gradient_real;
            block_adjoint_imaginary[right + lane] -= right_gradient_imaginary;
          }
        }
      }
    }
    for (std::int64_t lane = 0; lane < lane_count; ++lane)
      output[tile_begin + lane] +=
          std::complex<double>(density_real[lane], density_imaginary[lane]);

    for (const auto &plan : program.power_plans) {
      std::fill(monomial_adjoint_real,
                monomial_adjoint_real + plan.monomial_count * block_tile_size, 0.0);
      std::fill(monomial_adjoint_imaginary,
                monomial_adjoint_imaginary + plan.monomial_count * block_tile_size, 0.0);
      const std::int64_t plan_output = plan.output_storage_offset * block_tile_size;
      if (plan.conjugate_half_output) {
        const std::int64_t magnetic_width = 2 * plan.output_L + 1;
        for (std::int64_t base = 0; base < plan.output_dimension; base += magnetic_width)
          for (int magnetic = 1; magnetic <= plan.output_L; ++magnetic) {
            const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
            const std::int64_t negative =
                plan_output + (base + plan.output_L - magnetic) * block_tile_size;
            const std::int64_t positive =
                plan_output + (base + plan.output_L + magnetic) * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              block_adjoint_real[positive + lane] += phase * block_adjoint_real[negative + lane];
              block_adjoint_imaginary[positive + lane] -=
                  phase * block_adjoint_imaginary[negative + lane];
              block_adjoint_real[negative + lane] = 0.0;
              block_adjoint_imaginary[negative + lane] = 0.0;
            }
          }
      }
      if (plan.direct_input_plan) {
        for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
          const std::int64_t channel =
              plan.direct_input_channels[static_cast<std::size_t>(component)];
          if (channel < 0) continue;
          const double scale = plan.direct_input_scales[static_cast<std::size_t>(component)];
          const std::int64_t source = plan_output + component * block_tile_size;
          for (std::int64_t lane = 0; lane < lane_count; ++lane)
            input_adjoint[(tile_begin + lane) * input_dimension + channel] +=
                std::complex<double>(scale * block_adjoint_real[source + lane],
                                     scale * block_adjoint_imaginary[source + lane]);
        }
        continue;
      }
      for (std::int64_t component = 0; component < plan.output_dimension; ++component) {
        const std::int64_t magnetic_component = component % (2 * plan.output_L + 1);
        if (plan.conjugate_half_output && magnetic_component < plan.output_L) continue;
        const std::int64_t source = plan_output + component * block_tile_size;
        const std::int64_t begin = plan.output_offsets[static_cast<std::size_t>(component)];
        const std::int64_t end = plan.output_offsets[static_cast<std::size_t>(component + 1)];
        if (plan.real_coefficients) {
          for (std::int64_t coefficient = begin; coefficient < end; ++coefficient) {
            const double scale =
                plan.coefficient_values[static_cast<std::size_t>(coefficient)].real();
            const std::int64_t destination =
                plan.coefficient_terms[static_cast<std::size_t>(coefficient)] * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              monomial_adjoint_real[destination + lane] +=
                  scale * block_adjoint_real[source + lane];
              monomial_adjoint_imaginary[destination + lane] +=
                  scale * block_adjoint_imaginary[source + lane];
            }
          }
        } else {
          for (std::int64_t coefficient = begin; coefficient < end; ++coefficient) {
            const auto value = plan.coefficient_values[static_cast<std::size_t>(coefficient)];
            const std::int64_t destination =
                plan.coefficient_terms[static_cast<std::size_t>(coefficient)] * block_tile_size;
            for (std::int64_t lane = 0; lane < lane_count; ++lane) {
              const double gradient_real = block_adjoint_real[source + lane];
              const double gradient_imaginary = block_adjoint_imaginary[source + lane];
              monomial_adjoint_real[destination + lane] +=
                  gradient_real * value.real() + gradient_imaginary * value.imag();
              monomial_adjoint_imaginary[destination + lane] +=
                  gradient_imaginary * value.real() - gradient_real * value.imag();
            }
          }
        }
      }

      for (std::int64_t term = 0; term < plan.monomial_count; ++term) {
        const std::int64_t root = term * block_tile_size;
        const std::int64_t begin = plan.monomial_factor_offsets[static_cast<std::size_t>(term)];
        const std::int64_t end = plan.monomial_factor_offsets[static_cast<std::size_t>(term + 1)];
        const std::int64_t support = end - begin;
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          prefix_real[lane] = 1.0;
          prefix_imaginary[lane] = 0.0;
        }
        for (std::int64_t local = 0; local < support; ++local) {
          const std::int64_t factor = begin + local;
          const std::int64_t component =
              plan.monomial_factor_components[static_cast<std::size_t>(factor)];
          const std::int64_t exponent =
              plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
          const std::int64_t source =
              (plan.input_power_offsets[static_cast<std::size_t>(component)] + exponent) *
              block_tile_size;
          const std::int64_t previous = local * block_tile_size;
          const std::int64_t current = (local + 1) * block_tile_size;
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const double left_real = prefix_real[previous + lane];
            const double left_imaginary = prefix_imaginary[previous + lane];
            const double right_real = power_real[source + lane];
            const double right_imaginary = power_imaginary[source + lane];
            prefix_real[current + lane] = left_real * right_real - left_imaginary * right_imaginary;
            prefix_imaginary[current + lane] =
                left_real * right_imaginary + left_imaginary * right_real;
          }
        }
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          suffix_real[lane] = 1.0;
          suffix_imaginary[lane] = 0.0;
        }
        for (std::int64_t local = support; local-- > 0;) {
          const std::int64_t factor = begin + local;
          const std::int64_t component =
              plan.monomial_factor_components[static_cast<std::size_t>(factor)];
          const std::int64_t exponent =
              plan.monomial_factor_exponents[static_cast<std::size_t>(factor)];
          const std::int64_t power_offset =
              plan.input_power_offsets[static_cast<std::size_t>(component)];
          const std::int64_t left = local * block_tile_size;
          const std::int64_t derivative_power = (power_offset + exponent - 1) * block_tile_size;
          const std::int64_t factor_power = (power_offset + exponent) * block_tile_size;
          const std::int64_t channel = plan.input_channels[static_cast<std::size_t>(component)];
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const double prefix_times_suffix_real = prefix_real[left + lane] * suffix_real[lane] -
                prefix_imaginary[left + lane] * suffix_imaginary[lane];
            const double prefix_times_suffix_imaginary =
                prefix_real[left + lane] * suffix_imaginary[lane] +
                prefix_imaginary[left + lane] * suffix_real[lane];
            const double derivative_real = static_cast<double>(exponent) *
                (prefix_times_suffix_real * power_real[derivative_power + lane] -
                 prefix_times_suffix_imaginary * power_imaginary[derivative_power + lane]);
            const double derivative_imaginary = static_cast<double>(exponent) *
                (prefix_times_suffix_real * power_imaginary[derivative_power + lane] +
                 prefix_times_suffix_imaginary * power_real[derivative_power + lane]);
            const double root_real = monomial_adjoint_real[root + lane];
            const double root_imaginary = monomial_adjoint_imaginary[root + lane];
            input_adjoint[(tile_begin + lane) * input_dimension + channel] += std::complex<double>(
                root_real * derivative_real + root_imaginary * derivative_imaginary,
                root_imaginary * derivative_real - root_real * derivative_imaginary);

            const double old_suffix_real = suffix_real[lane];
            const double old_suffix_imaginary = suffix_imaginary[lane];
            suffix_real[lane] = old_suffix_real * power_real[factor_power + lane] -
                old_suffix_imaginary * power_imaginary[factor_power + lane];
            suffix_imaginary[lane] = old_suffix_real * power_imaginary[factor_power + lane] +
                old_suffix_imaginary * power_real[factor_power + lane];
          }
        }
      }
    }
  }
}

void YE3TCPUEvaluator::evaluate_scalar_power_program_tiled(
    const YACESpecies &species, const std::complex<double> *input, std::int64_t batch_size,
    std::int64_t input_dimension, std::complex<double> *output, std::complex<double> *input_adjoint)
{
  const auto &program = species.block_program.scalar_program;
  if (program.routes.empty() || batch_size == 0) return;
  if (batch_size < 0 || input_dimension <= 0 || input == nullptr || output == nullptr ||
      input_adjoint == nullptr || program.value_count <= 0)
    throw std::invalid_argument("invalid tiled scalar-power batch");
  const std::int64_t scalar_count = program.value_count * scalar_power_tile_size;
  if (static_cast<std::int64_t>(tiled_scalar_workspace_.size()) < 4 * scalar_count)
    throw std::logic_error("tiled scalar-power workspace is too small");
  double *value_real = tiled_scalar_workspace_.data();
  double *value_imaginary = value_real + scalar_count;
  double *adjoint_real = value_imaginary + scalar_count;
  double *adjoint_imaginary = adjoint_real + scalar_count;

  for (std::int64_t tile_begin = 0; tile_begin < batch_size; tile_begin += scalar_power_tile_size) {
    const std::int64_t lane_count = std::min(scalar_power_tile_size, batch_size - tile_begin);
    std::fill(adjoint_real, adjoint_real + scalar_count, 0.0);
    std::fill(adjoint_imaginary, adjoint_imaginary + scalar_count, 0.0);
    for (std::size_t base_index = 0; base_index < program.bases.size(); ++base_index) {
      const auto &base = program.bases[base_index];
      const std::int64_t destination =
          static_cast<std::int64_t>(base_index) * scalar_power_tile_size;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        std::complex<double> value(0.0, 0.0);
        const std::int64_t atom_offset = (tile_begin + lane) * input_dimension;
        for (std::size_t term = 0; term < base.coefficients.size(); ++term) {
          const auto left = input[static_cast<std::size_t>(atom_offset + base.left_channels[term])];
          const auto right =
              input[static_cast<std::size_t>(atom_offset + base.right_channels[term])];
          value += base.coefficients[term] * left * right;
        }
        value_real[destination + lane] = value.real();
        value_imaginary[destination + lane] = value.imag();
      }
    }
    for (std::size_t node_index = 0; node_index < program.nodes.size(); ++node_index) {
      const auto &node = program.nodes[node_index];
      const std::int64_t destination =
          static_cast<std::int64_t>(program.bases.size() + node_index) * scalar_power_tile_size;
      const std::int64_t left = node.left_value * scalar_power_tile_size;
      const std::int64_t right = node.right_value * scalar_power_tile_size;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const double left_real = value_real[left + lane];
        const double left_imaginary = value_imaginary[left + lane];
        const double right_real = value_real[right + lane];
        const double right_imaginary = value_imaginary[right + lane];
        value_real[destination + lane] = left_real * right_real - left_imaginary * right_imaginary;
        value_imaginary[destination + lane] =
            left_real * right_imaginary + left_imaginary * right_real;
      }
    }
    for (const auto &route : program.routes) {
      const std::int64_t source = route.value_index * scalar_power_tile_size;
      const std::complex<double> root = std::conj(route.scale);
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const std::complex<double> value(value_real[source + lane], value_imaginary[source + lane]);
        output[tile_begin + lane] += route.scale * value;
        adjoint_real[source + lane] += root.real();
        adjoint_imaginary[source + lane] += root.imag();
      }
    }
    for (std::size_t node_index = program.nodes.size(); node_index-- > 0;) {
      const auto &node = program.nodes[node_index];
      const std::int64_t source =
          static_cast<std::int64_t>(program.bases.size() + node_index) * scalar_power_tile_size;
      const std::int64_t left = node.left_value * scalar_power_tile_size;
      const std::int64_t right = node.right_value * scalar_power_tile_size;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const std::complex<double> root(adjoint_real[source + lane],
                                        adjoint_imaginary[source + lane]);
        const std::complex<double> left_value(value_real[left + lane],
                                              value_imaginary[left + lane]);
        const std::complex<double> right_value(value_real[right + lane],
                                               value_imaginary[right + lane]);
        const auto left_root = root * std::conj(right_value);
        const auto right_root = root * std::conj(left_value);
        adjoint_real[left + lane] += left_root.real();
        adjoint_imaginary[left + lane] += left_root.imag();
        adjoint_real[right + lane] += right_root.real();
        adjoint_imaginary[right + lane] += right_root.imag();
      }
    }
    for (std::size_t base_index = 0; base_index < program.bases.size(); ++base_index) {
      const auto &base = program.bases[base_index];
      const std::int64_t source = static_cast<std::int64_t>(base_index) * scalar_power_tile_size;
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const std::int64_t atom_offset = (tile_begin + lane) * input_dimension;
        const std::complex<double> root(adjoint_real[source + lane],
                                        adjoint_imaginary[source + lane]);
        for (std::size_t term = 0; term < base.coefficients.size(); ++term) {
          const std::int64_t left_channel = base.left_channels[term];
          const std::int64_t right_channel = base.right_channels[term];
          const auto left = input[static_cast<std::size_t>(atom_offset + left_channel)];
          const auto right = input[static_cast<std::size_t>(atom_offset + right_channel)];
          input_adjoint[static_cast<std::size_t>(atom_offset + left_channel)] +=
              root * std::conj(base.coefficients[term] * right);
          input_adjoint[static_cast<std::size_t>(atom_offset + right_channel)] +=
              root * std::conj(base.coefficients[term] * left);
        }
      }
    }
  }
}

void YE3TCPUEvaluator::evaluate_coupled_product_program_tiled(
    const YACESpecies &species, const std::complex<double> *input, std::int64_t batch_size,
    std::int64_t input_dimension, std::complex<double> *output, std::complex<double> *input_adjoint)
{
  for (const auto &plan : species.block_program.coupled_product_plans)
    ye3t::runtime::ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
        input, batch_size, input_dimension, plan.node_offsets.data(), plan.node_dimensions.data(),
        plan.node_leaf_offsets.data(), plan.leaf_input_components.data(),
        static_cast<std::int64_t>(plan.leaf_input_components.size()),
        plan.node_coefficient_offsets.data(), static_cast<std::int64_t>(plan.node_offsets.size()),
        plan.coefficient_left_components.data(), plan.coefficient_right_components.data(),
        plan.coefficient_output_components.data(), plan.coefficient_values.data(),
        static_cast<std::int64_t>(plan.coefficient_values.size()), plan.readout_components.data(),
        plan.readout_coefficients.data(), static_cast<std::int64_t>(plan.readout_components.size()),
        coupled_product_tile_size, tiled_coupled_product_workspace_.data(),
        static_cast<std::int64_t>(tiled_coupled_product_workspace_.size()), output, input_adjoint);
}

void YE3TCPUEvaluator::evaluate_readout_batch(const YACESpecies &species, int atom_count,
                                              const std::complex<double> *input,
                                              std::complex<double> *density,
                                              std::complex<double> *input_adjoint)
{
  const auto &plan = species.polynomial;
  if (plan.monomial_coefficients.empty()) {
    // A sidecar may replace every direct descriptor with compiled blocks.
  } else if (plan.binary_dag) {
    ye3t::runtime::
        symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
            input, atom_count, static_cast<std::int64_t>(species.channels.size()),
            plan.power_channels.data(), plan.power_exponents.data(),
            static_cast<std::int64_t>(plan.power_channels.size()), plan.binary_node_left.data(),
            plan.binary_node_right.data(), static_cast<std::int64_t>(plan.binary_node_left.size()),
            plan.monomial_nodes.data(), plan.monomial_coefficients.data(),
            static_cast<std::int64_t>(plan.monomial_coefficients.size()), binary_dag_tile_size,
            tiled_monomial_workspace_.data(),
            static_cast<std::int64_t>(tiled_monomial_workspace_.size()), density, input_adjoint);
  } else {
    ye3t::runtime::symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated(
        input, atom_count, static_cast<std::int64_t>(species.channels.size()),
        plan.power_channels.data(), plan.power_exponents.data(),
        static_cast<std::int64_t>(plan.power_channels.size()), plan.dag_node_parents.data(),
        plan.dag_node_powers.data(), static_cast<std::int64_t>(plan.dag_node_parents.size()),
        plan.dag_root_node_count, plan.monomial_nodes.data(), plan.monomial_coefficients.data(),
        static_cast<std::int64_t>(plan.monomial_coefficients.size()), monomial_workspace_.data(),
        static_cast<std::int64_t>(monomial_workspace_.size()), density, input_adjoint);
  }
  if (!species.block_program.routes.empty())
    evaluate_block_program_tiled(species, input, atom_count,
                                 static_cast<std::int64_t>(species.channels.size()), density,
                                 input_adjoint);
  if (!species.block_program.scalar_program.routes.empty())
    evaluate_scalar_power_program_tiled(species, input, atom_count,
                                        static_cast<std::int64_t>(species.channels.size()), density,
                                        input_adjoint);
  if (!species.block_program.coupled_product_plans.empty())
    evaluate_coupled_product_program_tiled(species, input, atom_count,
                                           static_cast<std::int64_t>(species.channels.size()),
                                           density, input_adjoint);
}

void YE3TCPUEvaluator::prepare_source_tile(int atom_count, const int *central_species,
                                           int edge_count, const int *edge_centers,
                                           const int *edge_neighbor_species,
                                           const double *edge_vectors, int center_index_base)
{
  last_source_edges_evaluated_ += static_cast<std::uint64_t>(edge_count);
  const int bond_count = model_->bond_count();
  const int radial_base_width = model_->maximum_radial_base_count();
  const int contracted_width = model_->maximum_contracted_width();
  const int angular_width = static_cast<int>(
      ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(active_angular_maximum_));
  const bool single_bond = bond_count == 1;
  resize_source_table(radii_, static_cast<std::size_t>(edge_count));
  resize_source_table(radial_directions_, static_cast<std::size_t>(edge_count) * 3);
  if (!single_bond) {
    resize_source_table(edge_bonds_, static_cast<std::size_t>(edge_count));
    resize_source_table(bond_counts_, static_cast<std::size_t>(bond_count));
    resize_source_table(bond_offsets_, static_cast<std::size_t>(bond_count + 1));
    resize_source_table(bond_cursors_, static_cast<std::size_t>(bond_count));
    resize_source_table(bond_edges_, static_cast<std::size_t>(edge_count));
    std::fill(bond_counts_.begin(), bond_counts_.end(), std::int64_t{0});
  }

  for (int edge = 0; edge < edge_count; ++edge) {
    const int center = edge_centers[edge] - center_index_base;
    if (center < 0 || center >= atom_count)
      throw std::invalid_argument("YE3T edge center is out of range");
    const int central = central_species[center];
    const int neighbor = edge_neighbor_species[edge];
    if (central < 0 || central >= model_->species_count() || neighbor < 0 ||
        neighbor >= model_->species_count())
      throw std::invalid_argument("YE3T edge species is out of range");
    if (!single_bond) {
      const std::int64_t bond =
          static_cast<std::int64_t>(central * model_->species_count() + neighbor);
      edge_bonds_[static_cast<std::size_t>(edge)] = bond;
      ++bond_counts_[static_cast<std::size_t>(bond)];
    }
    const double x = edge_vectors[edge * 3];
    const double y = edge_vectors[edge * 3 + 1];
    const double z = edge_vectors[edge * 3 + 2];
    const double radius = std::sqrt(x * x + y * y + z * z);
    if (!(radius > 0.0) || !std::isfinite(radius))
      throw std::invalid_argument("YE3T edge radius must be finite and positive");
    radii_[static_cast<std::size_t>(edge)] = radius;
    radial_directions_[static_cast<std::size_t>(edge * 3)] = x / radius;
    radial_directions_[static_cast<std::size_t>(edge * 3 + 1)] = y / radius;
    radial_directions_[static_cast<std::size_t>(edge * 3 + 2)] = z / radius;
  }

  if (!single_bond) {
    bond_offsets_[0] = 0;
    for (int bond = 0; bond < bond_count; ++bond)
      bond_offsets_[static_cast<std::size_t>(bond + 1)] =
          bond_offsets_[static_cast<std::size_t>(bond)] +
          bond_counts_[static_cast<std::size_t>(bond)];
    std::copy(bond_offsets_.begin(), bond_offsets_.begin() + bond_count, bond_cursors_.begin());
    for (int edge = 0; edge < edge_count; ++edge) {
      const std::int64_t bond = edge_bonds_[static_cast<std::size_t>(edge)];
      bond_edges_[static_cast<std::size_t>(bond_cursors_[static_cast<std::size_t>(bond)]++)] = edge;
    }
  }

  resize_source_table(radial_base_values_,
                      static_cast<std::size_t>(edge_count) * radial_base_width);
  resize_source_table(radial_base_derivatives_,
                      static_cast<std::size_t>(edge_count) * radial_base_width);
  resize_source_table(contracted_values_, static_cast<std::size_t>(edge_count) * contracted_width);
  resize_source_table(contracted_derivatives_,
                      static_cast<std::size_t>(edge_count) * contracted_width);
  if (single_bond) {
    const auto &bond = model_->bond(0, 0);
    ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
        radii_.data(), bond.radial_base_spline.data(), edge_count, bond.radial_base_count,
        bond.spline_interval_count, bond.cutoff, radial_base_values_.data(),
        radial_base_derivatives_.data());
    ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
        radii_.data(), bond.contracted_spline.data(), edge_count, bond.contracted_width(),
        bond.spline_interval_count, bond.cutoff, contracted_values_.data(),
        contracted_derivatives_.data());
  } else {
    std::fill(radial_base_values_.begin(), radial_base_values_.end(), 0.0);
    std::fill(radial_base_derivatives_.begin(), radial_base_derivatives_.end(), 0.0);
    std::fill(contracted_values_.begin(), contracted_values_.end(), 0.0);
    std::fill(contracted_derivatives_.begin(), contracted_derivatives_.end(), 0.0);
  }

  if (!single_bond) {
    for (int bond_index = 0; bond_index < bond_count; ++bond_index) {
      const std::int64_t begin = bond_offsets_[static_cast<std::size_t>(bond_index)];
      const std::int64_t finish = bond_offsets_[static_cast<std::size_t>(bond_index + 1)];
      const std::int64_t count = finish - begin;
      if (count == 0) continue;
      const int central = bond_index / model_->species_count();
      const int neighbor = bond_index % model_->species_count();
      const auto &bond = model_->bond(central, neighbor);
      resize_source_table(gathered_radii_, static_cast<std::size_t>(count));
      for (std::int64_t position = 0; position < count; ++position) {
        const std::int64_t edge = bond_edges_[static_cast<std::size_t>(begin + position)];
        gathered_radii_[static_cast<std::size_t>(position)] =
            radii_[static_cast<std::size_t>(edge)];
      }

      resize_source_table(gathered_values_,
                          static_cast<std::size_t>(count) * bond.radial_base_count);
      resize_source_table(gathered_derivatives_,
                          static_cast<std::size_t>(count) * bond.radial_base_count);
      ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
          gathered_radii_.data(), bond.radial_base_spline.data(), count, bond.radial_base_count,
          bond.spline_interval_count, bond.cutoff, gathered_values_.data(),
          gathered_derivatives_.data());
      for (std::int64_t position = 0; position < count; ++position) {
        const std::int64_t edge = bond_edges_[static_cast<std::size_t>(begin + position)];
        for (int radial = 0; radial < bond.radial_base_count; ++radial) {
          radial_base_values_[static_cast<std::size_t>(edge * radial_base_width + radial)] =
              gathered_values_[static_cast<std::size_t>(position * bond.radial_base_count +
                                                        radial)];
          radial_base_derivatives_[static_cast<std::size_t>(edge * radial_base_width + radial)] =
              gathered_derivatives_[static_cast<std::size_t>(position * bond.radial_base_count +
                                                             radial)];
        }
      }

      const int local_contracted_width = bond.contracted_width();
      resize_source_table(gathered_values_,
                          static_cast<std::size_t>(count) * local_contracted_width);
      resize_source_table(gathered_derivatives_,
                          static_cast<std::size_t>(count) * local_contracted_width);
      ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
          gathered_radii_.data(), bond.contracted_spline.data(), count, local_contracted_width,
          bond.spline_interval_count, bond.cutoff, gathered_values_.data(),
          gathered_derivatives_.data());
      for (std::int64_t position = 0; position < count; ++position) {
        const std::int64_t edge = bond_edges_[static_cast<std::size_t>(begin + position)];
        for (int radial = 0; radial < local_contracted_width; ++radial) {
          contracted_values_[static_cast<std::size_t>(edge * contracted_width + radial)] =
              gathered_values_[static_cast<std::size_t>(position * local_contracted_width +
                                                        radial)];
          contracted_derivatives_[static_cast<std::size_t>(edge * contracted_width + radial)] =
              gathered_derivatives_[static_cast<std::size_t>(position * local_contracted_width +
                                                             radial)];
        }
      }
    }
  }

  resize_source_table(angular_values_, static_cast<std::size_t>(edge_count) * angular_width);
  resize_source_table(angular_derivatives_,
                      static_cast<std::size_t>(edge_count) * angular_width * 3);
  if (edge_count > 0) {
    ye3t::runtime::
        complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
            double>(radial_directions_.data(), radii_.data(), edge_count, active_angular_maximum_,
                    angular_plan_.data(), static_cast<std::int64_t>(angular_plan_.size()),
                    angular_values_.data(), angular_derivatives_.data());
  }
}

void YE3TCPUEvaluator::evaluate(int atom_count, const int *central_species, int edge_count,
                                const int *edge_centers, const int *edge_neighbor_species,
                                const double *edge_vectors, double *atomic_energies,
                                double *edge_gradients)
{
  if (atom_count < 0 || edge_count < 0)
    throw std::invalid_argument("YE3T evaluator dimensions must be nonnegative");
  if (atom_count > 0 && (!central_species || !atomic_energies))
    throw std::invalid_argument("YE3T evaluator received a null atom array");
  if (edge_count > 0 &&
      (!edge_centers || !edge_neighbor_species || !edge_vectors || !edge_gradients))
    throw std::invalid_argument("YE3T evaluator received a null edge array");
  last_source_edges_evaluated_ = last_complete_batches_ = 0;
  if (reference_edge_tiling_ || atom_count == 0) {
    evaluate_complete_batch(atom_count, central_species, edge_count, edge_centers,
                            edge_neighbor_species, edge_vectors, atomic_energies, edge_gradients, 0,
                            source_tile_policy_.edge_capacity);
    return;
  }
  const int first_species = central_species[0];
  bool homogeneous = true;
  for (int atom = 0; atom < atom_count; ++atom) {
    if (central_species[atom] < 0 || central_species[atom] >= model_->species_count())
      throw std::invalid_argument("YE3T central species is out of range");
    homogeneous = homogeneous && central_species[atom] == first_species;
  }
  resize_growing(center_edge_offsets_, static_cast<std::size_t>(atom_count) + 1);
  std::fill(center_edge_offsets_.begin(), center_edge_offsets_.end(), 0);
  bool ordered = true;
  int previous = -1;
  for (int edge = 0; edge < edge_count; ++edge) {
    const int center = edge_centers[edge];
    if (center < 0 || center >= atom_count)
      throw std::invalid_argument("YE3T edge center is out of range");
    ++center_edge_offsets_[center + 1];
    ordered = ordered && center >= previous;
    previous = center;
  }
  for (int atom = 0; atom < atom_count; ++atom)
    center_edge_offsets_[atom + 1] += center_edge_offsets_[atom];
  if (!ordered) {
    edge_cursors_ = center_edge_offsets_;
    resize_growing(edge_order_, static_cast<std::size_t>(edge_count));
    // Stable grouping: the summation order WITHIN each center is unchanged.
    for (int edge = 0; edge < edge_count; ++edge)
      edge_order_[edge_cursors_[edge_centers[edge]]++] = edge;
  }
  double max_imaginary = 0.0;
  // Native readouts themselves use eight lanes. Keep one or two such tiles
  // live through source -> readout -> force; no edge-table replay is needed.
  // Cache capacity is a SOFT target, never permission to split an environment.
  constexpr int lanes = 8;
  if (homogeneous && ordered) {
    for (int begin = 0; begin < atom_count;) {
      int end = begin + std::min(lanes, atom_count - begin);
      const int candidate = begin + std::min(2 * lanes, atom_count - begin);
      if (center_edge_offsets_[candidate] - center_edge_offsets_[begin] <=
          source_tile_policy_.edge_capacity)
        end = candidate;
      const int edge_begin = center_edge_offsets_[begin];
      const int count = center_edge_offsets_[end] - edge_begin;
      evaluate_complete_batch(
          end - begin, central_species + begin, count,
          edge_centers ? edge_centers + edge_begin : nullptr,
          edge_neighbor_species ? edge_neighbor_species + edge_begin : nullptr,
          edge_vectors ? edge_vectors + static_cast<std::size_t>(edge_begin) * 3 : nullptr,
          atomic_energies + begin,
          edge_gradients ? edge_gradients + static_cast<std::size_t>(edge_begin) * 3 : nullptr,
          begin, std::max(1, count));
      max_imaginary = std::max(max_imaginary, maximum_imaginary_density_);
      begin = end;
    }
  } else {
    // Preserve species batching without reordering LAMMPS atom storage. Only
    // small complete-environment batches are gathered; all outputs are scattered
    // back to their original atom/edge indices. Arbitrary edge ordering is valid.
    center_batches_.build(atom_count, model_->species_count(), central_species);
    for (int species = 0; species < model_->species_count(); ++species) {
      const int finish = center_batches_.offsets[species + 1];
      for (int begin = center_batches_.offsets[species]; begin < finish;) {
        int end = begin + std::min(lanes, finish - begin);
        const int candidate = begin + std::min(2 * lanes, finish - begin);
        int candidate_edges = 0;
        for (int p = begin; p < candidate; ++p) {
          const int atom = center_batches_.atoms[p];
          candidate_edges += center_edge_offsets_[atom + 1] - center_edge_offsets_[atom];
        }
        if (candidate_edges <= source_tile_policy_.edge_capacity) end = candidate;
        int count = 0;
        for (int p = begin; p < end; ++p) {
          const int atom = center_batches_.atoms[p];
          count += center_edge_offsets_[atom + 1] - center_edge_offsets_[atom];
        }
        const int centers = end - begin;
        resize_growing(complete_central_species_, centers);
        std::fill(complete_central_species_.begin(), complete_central_species_.end(), species);
        resize_growing(complete_edge_centers_, count);
        resize_growing(complete_neighbor_species_, count);
        resize_growing(complete_original_edges_, count);
        resize_growing(complete_vectors_, static_cast<std::size_t>(count) * 3);
        resize_growing(complete_gradients_, static_cast<std::size_t>(count) * 3);
        resize_growing(complete_energies_, centers);
        int destination = 0;
        for (int p = begin; p < end; ++p) {
          const int atom = center_batches_.atoms[p];
          for (int pos = center_edge_offsets_[atom]; pos < center_edge_offsets_[atom + 1]; ++pos) {
            const int edge = ordered ? pos : edge_order_[pos];
            complete_edge_centers_[destination] = p - begin;
            complete_neighbor_species_[destination] = edge_neighbor_species[edge];
            complete_original_edges_[destination] = edge;
            std::copy_n(edge_vectors + static_cast<std::size_t>(edge) * 3, 3,
                        complete_vectors_.data() + static_cast<std::size_t>(destination) * 3);
            ++destination;
          }
        }
        evaluate_complete_batch(centers, complete_central_species_.data(), count,
                                complete_edge_centers_.data(), complete_neighbor_species_.data(),
                                complete_vectors_.data(), complete_energies_.data(),
                                complete_gradients_.data(), 0, std::max(1, count));
        max_imaginary = std::max(max_imaginary, maximum_imaginary_density_);
        for (int p = begin; p < end; ++p)
          atomic_energies[center_batches_.atoms[p]] = complete_energies_[p - begin];
        for (int local = 0; local < count; ++local)
          std::copy_n(complete_gradients_.data() + static_cast<std::size_t>(local) * 3, 3,
                      edge_gradients +
                          static_cast<std::size_t>(complete_original_edges_[local]) * 3);
        begin = end;
      }
    }
  }
  maximum_imaginary_density_ = max_imaginary;
}

void YE3TCPUEvaluator::evaluate_complete_batch(int atom_count, const int *central_species,
                                               int edge_count, const int *edge_centers,
                                               const int *edge_neighbor_species,
                                               const double *edge_vectors, double *atomic_energies,
                                               double *edge_gradients, int center_index_base,
                                               int source_edge_capacity)
{
  ++last_complete_batches_;
  if (atom_count < 0 || edge_count < 0)
    throw std::invalid_argument("YE3T evaluator dimensions must be nonnegative");
  if (atom_count > 0 && (central_species == nullptr || atomic_energies == nullptr))
    throw std::invalid_argument("YE3T evaluator received a null atom array");
  if (edge_count > 0 &&
      (edge_centers == nullptr || edge_neighbor_species == nullptr || edge_vectors == nullptr ||
       edge_gradients == nullptr))
    throw std::invalid_argument("YE3T evaluator received a null edge array");

  resize_growing(atomic_offsets_, static_cast<std::size_t>(atom_count + 1));
  resize_growing(source_offsets_, static_cast<std::size_t>(atom_count + 1));
  atomic_offsets_[0] = 0;
  source_offsets_[0] = 0;
  for (int atom = 0; atom < atom_count; ++atom) {
    const int species = central_species[atom];
    if (species < 0 || species >= model_->species_count())
      throw std::invalid_argument("YE3T central species is out of range");
    atomic_offsets_[static_cast<std::size_t>(atom + 1)] =
        atomic_offsets_[static_cast<std::size_t>(atom)] +
        static_cast<std::int64_t>(model_->species(species).channels.size());
    source_offsets_[static_cast<std::size_t>(atom + 1)] =
        source_offsets_[static_cast<std::size_t>(atom)] +
        static_cast<std::int64_t>(model_->species(species).source_channels.size());
  }
  const std::size_t atomic_value_count =
      static_cast<std::size_t>(atomic_offsets_[static_cast<std::size_t>(atom_count)]);
  const std::size_t source_value_count =
      static_cast<std::size_t>(source_offsets_[static_cast<std::size_t>(atom_count)]);
  resize_growing(atomic_values_, atomic_value_count);
  resize_growing(atomic_adjoint_, atomic_value_count);
  resize_growing(source_values_, source_value_count);
  resize_growing(source_adjoint_, source_value_count);
  std::fill(source_values_.begin(), source_values_.end(), std::complex<double>(0.0, 0.0));
  std::fill(atomic_adjoint_.begin(), atomic_adjoint_.end(), std::complex<double>(0.0, 0.0));

  const std::size_t radial_base_width =
      static_cast<std::size_t>(model_->maximum_radial_base_count());
  const std::size_t contracted_width = static_cast<std::size_t>(model_->maximum_contracted_width());
  const std::size_t angular_width = static_cast<std::size_t>(
      ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(active_angular_maximum_));
  cpu_source_tiled_passes(
      edge_count, source_edge_capacity,
      [&](int begin, int end) {
        prepare_source_tile(atom_count, central_species, end - begin, edge_centers + begin,
                            edge_neighbor_species + begin,
                            edge_vectors + static_cast<std::size_t>(begin) * 3, center_index_base);
      },
      [&](int begin, int end) {
        for (int edge = begin; edge < end; ++edge) {
          const std::size_t local = static_cast<std::size_t>(edge - begin);
          const int center = edge_centers[edge] - center_index_base;
          const auto &bond = model_->bond(central_species[center], edge_neighbor_species[edge]);
          if (radii_[local] >= bond.cutoff) continue;
          cpu_accumulate_ace_source_edge(
              bond,
              radial_base_width ? radial_base_values_.data() + local * radial_base_width : nullptr,
              contracted_width ? contracted_values_.data() + local * contracted_width : nullptr,
              angular_values_.data() + local * angular_width,
              source_value_count ? source_values_.data() + source_offsets_[center] : nullptr);
        }
      },
      [&]() {
        for (int atom = 0; atom < atom_count; ++atom) {
          const auto &species = model_->species(central_species[atom]);
          const std::int64_t atomic_offset = atomic_offsets_[static_cast<std::size_t>(atom)];
          const std::int64_t source_offset = source_offsets_[static_cast<std::size_t>(atom)];
          for (std::size_t channel = 0; channel < species.channels.size(); ++channel) {
            std::complex<double> value =
                source_values_[static_cast<std::size_t>(source_offset) +
                               static_cast<std::size_t>(species.full_channel_sources[channel])];
            const int transform = species.full_channel_transforms[channel];
            if (transform != 0) value = static_cast<double>(transform) * std::conj(value);
            atomic_values_[static_cast<std::size_t>(atomic_offset) + channel] = value;
          }
        }

        resize_growing(density_, static_cast<std::size_t>(atom_count));
        std::fill(density_.begin(), density_.end(), std::complex<double>(0.0, 0.0));
        maximum_imaginary_density_ = 0.0;
        bool single_species_batch = atom_count > 0;
        int batch_species = atom_count > 0 ? central_species[0] : -1;
        for (int atom = 1; atom < atom_count; ++atom)
          single_species_batch = single_species_batch && central_species[atom] == batch_species;
        if (single_species_batch) {
          // Preserve the allocation-free, original homogeneous path.
          evaluate_readout_batch(model_->species(batch_species), atom_count, atomic_values_.data(),
                                 density_.data(), atomic_adjoint_.data());
        } else if (atom_count > 0) {
          readout_batches_.build(atom_count, model_->species_count(), central_species);
          for (int species_index = 0; species_index < model_->species_count(); ++species_index) {
            const int begin = readout_batches_.offsets[species_index];
            const int finish = readout_batches_.offsets[species_index + 1];
            const int count = finish - begin;
            if (count == 0) continue;
            const auto &species = model_->species(species_index);
            const std::size_t width = species.channels.size();
            if (width > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(count))
              throw std::overflow_error("YE3T readout batch size overflows");
            resize_growing(batch_atomic_values_, static_cast<std::size_t>(count) * width);
            resize_growing(batch_atomic_adjoint_, static_cast<std::size_t>(count) * width);
            resize_growing(batch_density_, static_cast<std::size_t>(count));
            std::fill(batch_atomic_adjoint_.begin(), batch_atomic_adjoint_.end(),
                      std::complex<double>{});
            std::fill(batch_density_.begin(), batch_density_.end(), std::complex<double>{});
            for (int lane = 0; lane < count; ++lane) {
              const int atom = readout_batches_.atoms[begin + lane];
              if (width > 0)
                std::copy_n(atomic_values_.data() + atomic_offsets_[atom], width,
                            batch_atomic_values_.data() + static_cast<std::size_t>(lane) * width);
            }
            evaluate_readout_batch(species, count, batch_atomic_values_.data(),
                                   batch_density_.data(), batch_atomic_adjoint_.data());
            for (int lane = 0; lane < count; ++lane) {
              const int atom = readout_batches_.atoms[begin + lane];
              density_[atom] = batch_density_[lane];
              if (width > 0)
                std::copy_n(batch_atomic_adjoint_.data() + static_cast<std::size_t>(lane) * width,
                            width, atomic_adjoint_.data() + atomic_offsets_[atom]);
            }
          }
        }

        for (int atom = 0; atom < atom_count; ++atom) {
          const auto &species = model_->species(central_species[atom]);
          const double density = density_[static_cast<std::size_t>(atom)].real();
          maximum_imaginary_density_ =
              std::max(maximum_imaginary_density_,
                       std::abs(density_[static_cast<std::size_t>(atom)].imag()));
          if (density >= species.density_safe_limit)
            throw std::runtime_error(
                "YE3T density entered the unsupported core-smoothing interval");
          atomic_energies[atom] = species.reference_energy + species.embedding_scale * density;
          const std::int64_t atomic_offset = atomic_offsets_[static_cast<std::size_t>(atom)];
          const std::int64_t source_offset = source_offsets_[static_cast<std::size_t>(atom)];
          std::fill(source_adjoint_.begin() + source_offset,
                    source_adjoint_.begin() + source_offsets_[static_cast<std::size_t>(atom + 1)],
                    std::complex<double>(0.0, 0.0));
          for (std::size_t channel = 0; channel < species.channels.size(); ++channel) {
            std::complex<double> root = species.embedding_scale *
                atomic_adjoint_[static_cast<std::size_t>(atomic_offset) + channel];
            const int transform = species.full_channel_transforms[channel];
            if (transform != 0) root = static_cast<double>(transform) * std::conj(root);
            source_adjoint_[static_cast<std::size_t>(source_offset) +
                            static_cast<std::size_t>(species.full_channel_sources[channel])] +=
                root;
          }
        }
      },
      [&](int begin, int end) {
        for (int edge = begin; edge < end; ++edge) {
          const std::size_t local = static_cast<std::size_t>(edge - begin);
          double *gradient = edge_gradients + static_cast<std::size_t>(edge) * 3;
          const int center = edge_centers[edge] - center_index_base;
          const auto &bond = model_->bond(central_species[center], edge_neighbor_species[edge]);
          if (radii_[local] >= bond.cutoff) {
            std::fill_n(gradient, 3, 0.0);
            continue;
          }
          cpu_pullback_ace_source_edge(
              bond, radial_directions_.data() + local * 3,
              radial_base_width ? radial_base_derivatives_.data() + local * radial_base_width
                                : nullptr,
              contracted_width ? contracted_values_.data() + local * contracted_width : nullptr,
              contracted_width ? contracted_derivatives_.data() + local * contracted_width
                               : nullptr,
              angular_values_.data() + local * angular_width,
              angular_derivatives_.data() + local * angular_width * 3,
              source_value_count ? source_adjoint_.data() + source_offsets_[center] : nullptr,
              gradient);
        }
      });
}

double YE3TCPUEvaluator::memory_usage() const
{
  double bytes = 0.0;
  bytes += readout_batches_.memory_usage();
  bytes += (batch_atomic_values_.capacity() + batch_atomic_adjoint_.capacity() +
            batch_density_.capacity()) *
      sizeof(std::complex<double>);
  bytes += radii_.capacity() * sizeof(double);
  bytes += radial_directions_.capacity() * sizeof(double);
  bytes += edge_bonds_.capacity() * sizeof(std::int64_t);
  bytes += bond_counts_.capacity() * sizeof(std::int64_t);
  bytes += bond_offsets_.capacity() * sizeof(std::int64_t);
  bytes += bond_cursors_.capacity() * sizeof(std::int64_t);
  bytes += bond_edges_.capacity() * sizeof(std::int64_t);
  bytes += gathered_radii_.capacity() * sizeof(double);
  bytes += gathered_values_.capacity() * sizeof(double);
  bytes += gathered_derivatives_.capacity() * sizeof(double);
  bytes += radial_base_values_.capacity() * sizeof(double);
  bytes += radial_base_derivatives_.capacity() * sizeof(double);
  bytes += contracted_values_.capacity() * sizeof(double);
  bytes += contracted_derivatives_.capacity() * sizeof(double);
  bytes += angular_plan_.capacity() * sizeof(double);
  bytes += angular_workspace_.capacity() * sizeof(std::complex<double>);
  bytes += angular_values_.capacity() * sizeof(std::complex<double>);
  bytes += angular_derivatives_.capacity() * sizeof(std::complex<double>);
  bytes += atomic_offsets_.capacity() * sizeof(std::int64_t);
  bytes += source_offsets_.capacity() * sizeof(std::int64_t);
  bytes += source_values_.capacity() * sizeof(std::complex<double>);
  bytes += source_adjoint_.capacity() * sizeof(std::complex<double>);
  bytes += atomic_values_.capacity() * sizeof(std::complex<double>);
  bytes += atomic_adjoint_.capacity() * sizeof(std::complex<double>);
  bytes += density_.capacity() * sizeof(std::complex<double>);
  bytes += monomial_workspace_.capacity() * sizeof(std::complex<double>);
  bytes += tiled_monomial_workspace_.capacity() * sizeof(double);
  bytes += block_outputs_.capacity() * sizeof(std::complex<double>);
  bytes += block_output_adjoint_.capacity() * sizeof(std::complex<double>);
  bytes += block_powers_.capacity() * sizeof(std::complex<double>);
  bytes += block_prefix_.capacity() * sizeof(std::complex<double>);
  bytes += block_monomials_.capacity() * sizeof(std::complex<double>);
  bytes += block_monomial_adjoint_.capacity() * sizeof(std::complex<double>);
  bytes += tiled_block_workspace_.capacity() * sizeof(double);
  bytes += tiled_scalar_workspace_.capacity() * sizeof(double);
  bytes += tiled_coupled_product_workspace_.capacity() * sizeof(double);
  bytes += center_batches_.memory_usage();
  bytes += (center_edge_offsets_.capacity() + edge_cursors_.capacity() + edge_order_.capacity() +
            complete_central_species_.capacity() + complete_edge_centers_.capacity() +
            complete_neighbor_species_.capacity() + complete_original_edges_.capacity()) *
      sizeof(int);
  bytes += (complete_vectors_.capacity() + complete_gradients_.capacity() +
            complete_energies_.capacity()) *
      sizeof(double);
  return bytes;
}

}    // namespace YE3T_LAMMPS
