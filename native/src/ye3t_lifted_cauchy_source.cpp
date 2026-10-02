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

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {
namespace {

  double integer_power(double base, int exponent)
  {
    double result = 1.0;
    double factor = base;
    int remaining = exponent;
    while (remaining > 0) {
      if (remaining & 1) result *= factor;
      factor *= factor;
      remaining >>= 1;
    }
    return result;
  }

  struct RadialValue {
    double value = 0.0;
    double derivative = 0.0;
  };

  RadialValue direct_radial(const LiftedCauchyDirectPolynomial &polynomial, double coordinate)
  {
    const auto &coefficients = polynomial.coefficients;
    double value = coefficients.back();
    double derivative = 0.0;
    for (std::size_t reverse = coefficients.size() - 1; reverse > 0; --reverse) {
      derivative = derivative * coordinate + value;
      value = value * coordinate + coefficients[reverse - 1];
    }
    const double envelope = 1.0 - coordinate;
    return {envelope * envelope * value,
            -2.0 * envelope * value + envelope * envelope * derivative};
  }

  RadialValue factorized_radial(const LiftedCauchyFactorizedRadial &radial, double coordinate)
  {
    const double envelope = 1.0 - coordinate;
    const double envelope_power = integer_power(envelope, radial.envelope_power);
    const double coordinate_power = integer_power(coordinate, radial.x_power);
    double derivative = -static_cast<double>(radial.envelope_power) *
        integer_power(envelope, radial.envelope_power - 1) * coordinate_power;
    if (radial.x_power > 0)
      derivative += static_cast<double>(radial.x_power) * envelope_power *
          integer_power(coordinate, radial.x_power - 1);
    return {envelope_power * coordinate_power, derivative};
  }

  struct EdgeGeometry {
    double radius = 0.0;
    double coordinate = 0.0;
    double direction[3]{};
  };

  bool edge_geometry(const LiftedCauchyModel &model, const LiftedCauchyEdge &edge,
                     EdgeGeometry &geometry)
  {
    double squared = 0.0;
    for (double value : edge.displacement) {
      if (!std::isfinite(value))
        throw std::invalid_argument("lifted-Cauchy edge displacement must be finite");
      squared += value * value;
    }
    geometry.radius = std::sqrt(squared);
    if (!(geometry.radius < model.cutoff) ||
        (model.exclude_zero_separation && geometry.radius == 0.0))
      return false;
    geometry.coordinate = geometry.radius / model.cutoff;
    if (geometry.radius > 0.0)
      for (int component = 0; component < 3; ++component)
        geometry.direction[component] = edge.displacement[component] / geometry.radius;
    return true;
  }

  std::vector<double> legendre_coefficients(int angular)
  {
    if (angular < 0) throw std::invalid_argument("angular momentum must be nonnegative");
    if (angular == 0) return {1.0};
    std::vector<double> previous{1.0};
    std::vector<double> current{0.0, 1.0};
    for (int degree = 1; degree < angular; ++degree) {
      std::vector<double> next(static_cast<std::size_t>(degree + 2), 0.0);
      for (std::size_t power = 0; power < current.size(); ++power)
        next[power + 1] += (2.0 * degree + 1.0) / (degree + 1.0) * current[power];
      for (std::size_t power = 0; power < previous.size(); ++power)
        next[power] -= static_cast<double>(degree) / (degree + 1.0) * previous[power];
      previous = std::move(current);
      current = std::move(next);
    }
    return current;
  }

  detail::LiftedCauchyAngularPlan make_angular_plan(int angular)
  {
    detail::LiftedCauchyAngularPlan plan;
    std::vector<double> coefficients = legendre_coefficients(angular);
    for (int magnetic = 0; magnetic <= angular; ++magnetic) {
      if (magnetic > 0) {
        std::vector<double> next;
        next.reserve(coefficients.size() - 1);
        for (std::size_t power = 1; power < coefficients.size(); ++power)
          next.push_back(static_cast<double>(power) * coefficients[power]);
        coefficients = std::move(next);
      }
      plan.derivative_coefficients.push_back(coefficients);
      plan.scales.push_back(magnetic == 0
                                ? 1.0
                                : std::sqrt(2.0 *
                                            std::exp(std::lgamma(angular - magnetic + 1.0) -
                                                     std::lgamma(angular + magnetic + 1.0))));
    }
    return plan;
  }

  void regular_solid(const LiftedCauchyModel &model, const LiftedCauchySourceGroup &group,
                     const detail::LiftedCauchyAngularPlan &plan, const LiftedCauchyEdge &edge,
                     std::vector<double> &values, std::vector<std::array<double, 3>> *derivatives)
  {
    const int angular = group.angular;
    const int width = 2 * angular + 1;
    values.assign(static_cast<std::size_t>(width), 0.0);
    if (derivatives) derivatives->assign(static_cast<std::size_t>(width), {});
    const double x = edge.displacement[0] / model.cutoff;
    const double y = edge.displacement[1] / model.cutoff;
    const double z = edge.displacement[2] / model.cutoff;
    const double radius2 = x * x + y * y + z * z;
    const std::complex<double> xy(x, y);
    std::complex<double> xy_power(1.0, 0.0);
    std::complex<double> xy_previous(1.0, 0.0);
    for (int magnetic = 0; magnetic <= angular; ++magnetic) {
      if (magnetic > 0) {
        xy_previous = xy_power;
        xy_power *= xy;
      }
      const auto &derivative_coefficients = plan.derivative_coefficients[magnetic];
      std::complex<double> value(0.0, 0.0);
      std::array<std::complex<double>, 3> gradient{};
      for (std::size_t power = 0; power < derivative_coefficients.size(); ++power) {
        const double coefficient = derivative_coefficients[power];
        if (coefficient == 0.0) continue;
        const int remaining = angular - magnetic - static_cast<int>(power);
        if (remaining < 0 || remaining % 2) continue;
        const int radial_power = remaining / 2;
        const double radial = integer_power(radius2, radial_power);
        const double z_power = integer_power(z, static_cast<int>(power));
        const std::complex<double> base = coefficient * radial * z_power;
        value += base * xy_power;
        if (derivatives && radial_power > 0) {
          const double radial_derivative =
              2.0 * radial_power * integer_power(radius2, radial_power - 1) * z_power;
          gradient[0] += coefficient * radial_derivative * x * xy_power;
          gradient[1] += coefficient * radial_derivative * y * xy_power;
          gradient[2] += coefficient * radial_derivative * z * xy_power;
        }
        if (derivatives && power > 0)
          gradient[2] += coefficient * radial * static_cast<double>(power) *
              integer_power(z, static_cast<int>(power) - 1) * xy_power;
        if (derivatives && magnetic > 0) {
          const std::complex<double> xy_derivative = static_cast<double>(magnetic) * xy_previous;
          gradient[0] += base * xy_derivative;
          gradient[1] += std::complex<double>(0.0, 1.0) * base * xy_derivative;
        }
      }
      const double scale = plan.scales[magnetic];
      if (magnetic == 0) {
        values[static_cast<std::size_t>(angular)] = value.real();
        if (derivatives) {
          for (int axis = 0; axis < 3; ++axis)
            (*derivatives)[static_cast<std::size_t>(angular)][axis] =
                gradient[axis].real() / model.cutoff;
        }
      } else {
        const int cosine = angular - magnetic;
        const int sine = angular + magnetic;
        values[static_cast<std::size_t>(cosine)] = scale * value.real();
        values[static_cast<std::size_t>(sine)] = -scale * value.imag();
        if (derivatives) {
          for (int axis = 0; axis < 3; ++axis) {
            (*derivatives)[static_cast<std::size_t>(cosine)][axis] =
                scale * gradient[axis].real() / model.cutoff;
            (*derivatives)[static_cast<std::size_t>(sine)][axis] =
                -scale * gradient[axis].imag() / model.cutoff;
          }
        }
      }
    }
  }

  void add_source(const LiftedCauchyModel &model, const LiftedCauchySourceGroup &group,
                  const LiftedCauchyEdge &edge, const EdgeGeometry &geometry, bool direct,
                  const detail::LiftedCauchyAngularPlan &plan, std::vector<double> &solid,
                  std::vector<double> &values)
  {
    regular_solid(model, group, plan, edge, solid, nullptr);
    for (int q = 0; q < group.source_dimension; ++q) {
      const RadialValue radial = direct
          ? direct_radial(group.direct_q_polynomials[q], geometry.coordinate)
          : factorized_radial(group.factorized_radials[q], geometry.coordinate);
      const std::int64_t offset = direct
          ? group.q_source_variable_offsets[q]
          : static_cast<std::int64_t>(q) * group.real_component_count;
      for (int component = 0; component < group.real_component_count; ++component)
        values[static_cast<std::size_t>(offset + component)] +=
            radial.value * solid[static_cast<std::size_t>(component)];
    }
  }

  void add_edge_vjp(const LiftedCauchyModel &model, const LiftedCauchySourceGroup &group,
                    const LiftedCauchyEdge &edge, const EdgeGeometry &geometry, bool direct,
                    const detail::LiftedCauchyAngularPlan &plan, std::vector<double> &solid,
                    std::vector<std::array<double, 3>> &solid_derivative,
                    const std::vector<double> &adjoint, std::array<double, 3> &gradient)
  {
    regular_solid(model, group, plan, edge, solid, &solid_derivative);
    for (int q = 0; q < group.source_dimension; ++q) {
      const RadialValue radial = direct
          ? direct_radial(group.direct_q_polynomials[q], geometry.coordinate)
          : factorized_radial(group.factorized_radials[q], geometry.coordinate);
      const std::int64_t offset = direct
          ? group.q_source_variable_offsets[q]
          : static_cast<std::int64_t>(q) * group.real_component_count;
      for (int component = 0; component < group.real_component_count; ++component) {
        const double seed = adjoint[static_cast<std::size_t>(offset + component)];
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
          const double jacobian =
              radial.value * solid_derivative[static_cast<std::size_t>(component)][cartesian] +
              radial.derivative * solid[static_cast<std::size_t>(component)] *
                  geometry.direction[cartesian] / model.cutoff;
          gradient[cartesian] += seed * jacobian;
        }
      }
    }
  }

  void validate_edges(const LiftedCauchyModel &model, const std::vector<LiftedCauchyEdge> &edges)
  {
    for (const auto &edge : edges)
      if (edge.neighbor_species_index < 0 ||
          edge.neighbor_species_index >= static_cast<int>(model.central_species_order.size()))
        throw std::invalid_argument("lifted-Cauchy edge has an invalid neighbor species index");
  }

}    // namespace

LiftedCauchyCPUSource::LiftedCauchyCPUSource(const LiftedCauchyModel *model) : model_(model)
{
  if (model_ == nullptr) throw std::invalid_argument("lifted-Cauchy CPU source requires a model");
  if (model_->real_component_count < 1 || model_->role_dimension != 2)
    throw std::invalid_argument("unsupported lifted-Cauchy native source shape");
  for (const auto &group : model_->source_groups)
    if (group.real_component_count != 2 * group.angular + 1 ||
        group.real_component_count > model_->real_component_count)
      throw std::invalid_argument("lifted-Cauchy source group has an invalid angular width");
  angular_plans_.reserve(model_->source_groups.size());
  for (const auto &group : model_->source_groups)
    angular_plans_.push_back(make_angular_plan(group.angular));
}

double LiftedCauchyCPUSource::memory_usage() const
{
  double bytes = angular_plans_.capacity() * sizeof(detail::LiftedCauchyAngularPlan);
  for (const auto &plan : angular_plans_) {
    bytes += plan.scales.capacity() * sizeof(double);
    bytes += plan.derivative_coefficients.capacity() * sizeof(std::vector<double>);
    for (const auto &coefficients : plan.derivative_coefficients)
      bytes += coefficients.capacity() * sizeof(double);
  }
  return bytes;
}

void LiftedCauchyCPUSource::accumulate(const std::vector<LiftedCauchyEdge> &edges,
                                       LiftedCauchySourcePolicy policy,
                                       std::vector<double> &source_values) const
{
  validate_edges(*model_, edges);
  source_values.assign(static_cast<std::size_t>(model_->source_variable_count), 0.0);
  std::vector<double> solid;
  solid.reserve(static_cast<std::size_t>(model_->real_component_count));
  if (policy == LiftedCauchySourcePolicy::DIRECT_Q) {
    for (const auto &edge : edges) {
      EdgeGeometry geometry;
      if (!edge_geometry(*model_, edge, geometry)) continue;
      for (std::size_t group_index = 0; group_index < model_->source_groups.size(); ++group_index) {
        const auto &group = model_->source_groups[group_index];
        if (edge.neighbor_species_index == group.neighbor_species_index)
          add_source(*model_, group, edge, geometry, true, angular_plans_[group_index], solid,
                     source_values);
      }
    }
    return;
  }

  std::vector<double> factorized;
  for (std::size_t group_index = 0; group_index < model_->source_groups.size(); ++group_index) {
    const auto &group = model_->source_groups[group_index];
    factorized.assign(static_cast<std::size_t>(group.source_dimension) * group.real_component_count,
                      0.0);
    for (const auto &edge : edges) {
      if (edge.neighbor_species_index != group.neighbor_species_index) continue;
      EdgeGeometry geometry;
      if (edge_geometry(*model_, edge, geometry))
        add_source(*model_, group, edge, geometry, false, angular_plans_[group_index], solid,
                   factorized);
    }
    for (int q = 0; q < group.source_dimension; ++q) {
      const std::int64_t output_offset = group.q_source_variable_offsets[q];
      for (int p = 0; p < group.source_dimension; ++p) {
        const double transform =
            group.transform_q_from_f[static_cast<std::size_t>(q * group.source_dimension + p)];
        const std::int64_t input_offset = static_cast<std::int64_t>(p) * group.real_component_count;
        for (int component = 0; component < group.real_component_count; ++component)
          source_values[static_cast<std::size_t>(output_offset + component)] +=
              transform * factorized[static_cast<std::size_t>(input_offset + component)];
      }
    }
  }
}

void LiftedCauchyCPUSource::vjp(const std::vector<LiftedCauchyEdge> &edges,
                                const std::vector<double> &source_adjoint,
                                LiftedCauchySourcePolicy policy,
                                std::vector<std::array<double, 3>> &edge_gradients) const
{
  validate_edges(*model_, edges);
  if (source_adjoint.size() != static_cast<std::size_t>(model_->source_variable_count))
    throw std::invalid_argument("lifted-Cauchy source adjoint has the wrong size");
  if (!std::all_of(source_adjoint.begin(), source_adjoint.end(), [](double value) {
        return std::isfinite(value);
      }))
    throw std::invalid_argument("lifted-Cauchy source adjoint must be finite");
  edge_gradients.assign(edges.size(), {});
  std::vector<double> solid;
  std::vector<std::array<double, 3>> solid_derivative;
  solid.reserve(static_cast<std::size_t>(model_->real_component_count));
  solid_derivative.reserve(static_cast<std::size_t>(model_->real_component_count));

  if (policy == LiftedCauchySourcePolicy::DIRECT_Q) {
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
      EdgeGeometry geometry;
      if (!edge_geometry(*model_, edges[edge_index], geometry)) continue;
      for (std::size_t group_index = 0; group_index < model_->source_groups.size(); ++group_index) {
        const auto &group = model_->source_groups[group_index];
        if (edges[edge_index].neighbor_species_index == group.neighbor_species_index)
          add_edge_vjp(*model_, group, edges[edge_index], geometry, true,
                       angular_plans_[group_index], solid, solid_derivative, source_adjoint,
                       edge_gradients[edge_index]);
      }
    }
    return;
  }

  std::vector<double> factorized_adjoint;
  for (std::size_t group_index = 0; group_index < model_->source_groups.size(); ++group_index) {
    const auto &group = model_->source_groups[group_index];
    factorized_adjoint.assign(
        static_cast<std::size_t>(group.source_dimension) * group.real_component_count, 0.0);
    for (int p = 0; p < group.source_dimension; ++p) {
      const std::int64_t output_offset = static_cast<std::int64_t>(p) * group.real_component_count;
      for (int q = 0; q < group.source_dimension; ++q) {
        const double transform =
            group.transform_q_from_f[static_cast<std::size_t>(q * group.source_dimension + p)];
        const std::int64_t input_offset = group.q_source_variable_offsets[q];
        for (int component = 0; component < group.real_component_count; ++component)
          factorized_adjoint[static_cast<std::size_t>(output_offset + component)] +=
              transform * source_adjoint[static_cast<std::size_t>(input_offset + component)];
      }
    }
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
      if (edges[edge_index].neighbor_species_index != group.neighbor_species_index) continue;
      EdgeGeometry geometry;
      if (edge_geometry(*model_, edges[edge_index], geometry))
        add_edge_vjp(*model_, group, edges[edge_index], geometry, false,
                     angular_plans_[group_index], solid, solid_derivative, factorized_adjoint,
                     edge_gradients[edge_index]);
    }
  }
}

}    // namespace YE3T_LAMMPS
