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

#include "ye3t_tagged_cauchy_cpu.h"

#include "ye3t_runtime_core.h"
#include "ye3t_shifted_jacobi.h"
#include "ye3t_tagged_cauchy_readout_plan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace YE3T_LAMMPS {
namespace {

  constexpr double kUnitEpsilon = 1.0e-12;
  constexpr std::size_t kMomentNodeCacheBytes = 4u * 1024u * 1024u;
  constexpr std::size_t kMomentEdgeTile = 8;
  constexpr int kSourceCenterBatch = 8;

  constexpr std::array<TaggedCauchyExecutionPolicy, 4> kAutoPolicies = {
      TaggedCauchyExecutionPolicy::COMPILED_DIRECT,
      TaggedCauchyExecutionPolicy::GENERIC_DAG,
      TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
      TaggedCauchyExecutionPolicy::BLOCK,
  };

  // F(z, p) = (z - p)! / (z - k)! for k <= z (exact, via a short integer
  // falling-factorial product; z and p are always small nonnegative counts
  // here), 0 when k > z. Matches the reference `evaluate_real`/`evaluate_
  // complex` free-count formula exactly, including that the whole factor is
  // forced to zero whenever z < tag_count regardless of p (not only when
  // count = tag_count - p would itself go negative).
  double falling_factorial(std::int64_t z, int p, int tag_count)
  {
    if (z < tag_count) return 0.0;
    const int count = tag_count - p;
    double value = 1.0;
    double top = static_cast<double>(z - p);
    for (int i = 0; i < count; ++i) value *= (top - static_cast<double>(i));
    return value;
  }

  void power_series_with_derivative(const std::vector<double> &coefficients, double x,
                                    double &value, double &derivative)
  {
    if (coefficients.empty())
      throw std::runtime_error("tagged-Cauchy shifted-Jacobi source has no coefficients");
    value = coefficients.back();
    derivative = 0.0;
    for (std::size_t reverse = coefficients.size() - 1; reverse > 0; --reverse) {
      derivative = derivative * x + value;
      value = value * x + coefficients[reverse - 1];
    }
  }

}    // namespace

TaggedCauchyCPUEvaluator::TaggedCauchyCPUEvaluator(const TaggedCauchyModel *model,
                                                   TaggedCauchyExecutionPolicy policy) :
    model_(model), requested_policy_(policy), selected_policy_(policy)
{
  if (model_ == nullptr)
    throw std::invalid_argument("tagged-Cauchy CPU evaluator requires a model");
  total_components_ = model_->total_component_count;
  l_list_ = model_->distinct_angular_l;
  angular_maximum_ = l_list_.empty() ? -1 : *std::max_element(l_list_.begin(), l_list_.end());
  l_slot_of_l_.assign(static_cast<std::size_t>(angular_maximum_ + 1), -1);
  for (std::size_t slot = 0; slot < l_list_.size(); ++slot)
    l_slot_of_l_[static_cast<std::size_t>(l_list_[slot])] = static_cast<int>(slot);
  angular_values_by_slot_.resize(l_list_.size());
  angular_derivatives_by_slot_.resize(l_list_.size());
  if (angular_maximum_ >= 0) {
    const std::int64_t plan_size =
        ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(angular_maximum_);
    angular_plan_.resize(static_cast<std::size_t>(plan_size));
    ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<double>(
        angular_maximum_, angular_plan_.data(), plan_size, 1.0);
  }

  channel_angular_slot_.resize(model_->channels.size());
  for (std::size_t c = 0; c < model_->channels.size(); ++c)
    channel_angular_slot_[c] = l_slot_of_l_[static_cast<std::size_t>(model_->channels[c].l)];
  real_form_angular_slot_.assign(model_->real_forms.size(), -1);
  real_form_offsets_.resize(model_->real_forms.size() + 1, 0);
  for (std::size_t form = 0; form < model_->real_forms.size(); ++form) {
    const TaggedCauchyRealForm &record = model_->real_forms[form];
    if (record.l < 0 || record.l > angular_maximum_ || record.width != 2 * record.l + 1)
      throw std::runtime_error("tagged-Cauchy real form has an inconsistent angular width");
    real_form_angular_slot_[form] = l_slot_of_l_[static_cast<std::size_t>(record.l)];
    real_form_offsets_[form + 1] =
        real_form_offsets_[form] + static_cast<std::size_t>(record.width);
  }
  channel_real_form_offsets_.resize(model_->channels.size());
  for (std::size_t c = 0; c < model_->channels.size(); ++c) {
    const int form = model_->channels[c].real_form_index;
    if (form < 0 || static_cast<std::size_t>(form) >= model_->real_forms.size())
      throw std::runtime_error("tagged-Cauchy channel has an invalid real-form index");
    channel_real_form_offsets_[c] = real_form_offsets_[static_cast<std::size_t>(form)];
  }
  jacobi_max_q_by_slot_.assign(l_list_.size(), -1);
  for (std::size_t c = 0; c < model_->channels.size(); ++c) {
    const int slot = channel_angular_slot_[c];
    jacobi_max_q_by_slot_[static_cast<std::size_t>(slot)] = std::max(
        jacobi_max_q_by_slot_[static_cast<std::size_t>(slot)], model_->channels[c].radial_channel);
  }
  jacobi_offsets_by_slot_.resize(l_list_.size() + 1, 0);
  for (std::size_t slot = 0; slot < l_list_.size(); ++slot)
    jacobi_offsets_by_slot_[slot + 1] =
        jacobi_offsets_by_slot_[slot] + static_cast<std::size_t>(jacobi_max_q_by_slot_[slot] + 1);
  jacobi_values_.resize(jacobi_offsets_by_slot_.back());
  jacobi_derivatives_.resize(jacobi_offsets_by_slot_.back());

  TaggedCauchyReadoutPlan readout = compile_tagged_cauchy_readout(*model_);
  density_key_count_ = readout.density_key_count;
  maximum_term_p_ = readout.maximum_term_p;
  species_term_offsets_ = std::move(readout.species_term_offsets);
  term_p_ = std::move(readout.term_p);
  term_coefficient_ = std::move(readout.term_coefficient);
  term_density_offsets_ = std::move(readout.term_density_offsets);
  term_density_flat_ = std::move(readout.term_density_flat);
  term_moment_offsets_ = std::move(readout.term_moment_offsets);
  term_moment_flat_ = std::move(readout.term_moment_flat);
  species_readout_node_offsets_ = std::move(readout.species_readout_node_offsets);
  readout_node_parent_ = std::move(readout.readout_node_parent);
  readout_node_variable_ = std::move(readout.readout_node_variable);
  term_product_node_ = std::move(readout.term_product_node);
  species_live_moment_offsets_ = std::move(readout.species_live_moment_offsets);
  species_live_moments_ = std::move(readout.species_live_moments);
  species_live_moment_nodes_ = std::move(readout.species_live_moment_nodes);
  species_moment_node_offsets_ = std::move(readout.species_moment_node_offsets);
  moment_node_parent_ = std::move(readout.moment_node_parent);
  moment_node_factor_ = std::move(readout.moment_node_factor);
  moment_node_terminal_offsets_ = std::move(readout.moment_node_terminal_offsets);
  moment_node_terminal_moments_ = std::move(readout.moment_node_terminal_moments);
  free_count_by_p_.resize(static_cast<std::size_t>(maximum_term_p_ + 1), 1.0);
  moment_node_values_.resize(moment_node_parent_.size());
  moment_node_adjoint_.resize(moment_node_parent_.size());
  readout_node_values_.resize(readout_node_parent_.size());
  readout_node_adjoint_.resize(readout_node_parent_.size());

  const auto &portfolio = model_->execution_portfolio;
  if (policy != TaggedCauchyExecutionPolicy::COMPILED_DIRECT &&
      policy != TaggedCauchyExecutionPolicy::GENERIC_DAG && !portfolio.present)
    throw std::runtime_error("tagged-Cauchy non-direct policy requires a compiler execution "
                             "portfolio; use compiled-direct compatibility for legacy bundles");
  if (policy == TaggedCauchyExecutionPolicy::SYMMETRIC_POWER && !portfolio.symmetric_power_eligible)
    throw std::runtime_error("tagged-Cauchy symmetric-power policy is not eligible");
  if (policy == TaggedCauchyExecutionPolicy::BLOCK && !portfolio.block_eligible)
    throw std::runtime_error("tagged-Cauchy block policy is not eligible");
  if (policy == TaggedCauchyExecutionPolicy::AUTO) {
    if (!portfolio.present)
      throw std::runtime_error("tagged-Cauchy AUTO requires a compiler execution portfolio");
    // Static multiply counts are useful compiler metadata but are not a
    // portable performance model, so AUTO starts in the exact direct route
    // and is frozen only after the bounded initialization calibration below.
    selected_policy_ = TaggedCauchyExecutionPolicy::COMPILED_DIRECT;
  }

  const std::size_t adjoint_count = model_->adjoint_terms.size();
  adjoint_remaining_offsets_.assign(adjoint_count + 1, 0);
  adjoint_source_index_.resize(adjoint_count);
  for (std::size_t t = 0; t < adjoint_count; ++t) {
    const TaggedCauchyAdjointTerm &term = model_->adjoint_terms[t];
    adjoint_remaining_offsets_[t + 1] = adjoint_remaining_offsets_[t] +
        static_cast<std::int64_t>(term.remaining_source_indices.size());
    adjoint_source_index_[t] = term.source_index;
  }
  adjoint_remaining_flat_.resize(static_cast<std::size_t>(adjoint_remaining_offsets_.back()));
  for (std::size_t t = 0; t < adjoint_count; ++t)
    std::copy(model_->adjoint_terms[t].remaining_source_indices.begin(),
              model_->adjoint_terms[t].remaining_source_indices.end(),
              adjoint_remaining_flat_.begin() +
                  static_cast<std::ptrdiff_t>(adjoint_remaining_offsets_[t]));
  folded_adjoint_coefficient_.assign(model_->beta.size(), std::vector<double>(adjoint_count));
  for (std::size_t species = 0; species < model_->beta.size(); ++species)
    for (std::size_t t = 0; t < adjoint_count; ++t)
      folded_adjoint_coefficient_[species][t] =
          model_->beta[species][static_cast<std::size_t>(model_->adjoint_terms[t].feature_index)] *
          model_->adjoint_terms[t].coefficient;

  // Precompute which neighbor species own at least one channel.
  species_has_channel_.assign(model_->species_order.size(), false);
  for (const TaggedCauchyChannel &channel : model_->channels)
    if (channel.neighbor_species_index >= 0 &&
        static_cast<std::size_t>(channel.neighbor_species_index) < species_has_channel_.size())
      species_has_channel_[static_cast<std::size_t>(channel.neighbor_species_index)] = true;
}

const char *TaggedCauchyCPUEvaluator::selected_evaluator_name() const
{
  switch (selected_policy_) {
    case TaggedCauchyExecutionPolicy::COMPILED_DIRECT:
      return "compiled_direct";
    case TaggedCauchyExecutionPolicy::GENERIC_DAG:
      return "generic_dag";
    case TaggedCauchyExecutionPolicy::SYMMETRIC_POWER:
      return "symmetric_power";
    case TaggedCauchyExecutionPolicy::BLOCK:
      return "block";
    case TaggedCauchyExecutionPolicy::AUTO:
      break;
  }
  return "invalid";
}

bool TaggedCauchyCPUEvaluator::auto_requested() const
{
  return requested_policy_ == TaggedCauchyExecutionPolicy::AUTO;
}

bool TaggedCauchyCPUEvaluator::auto_calibrated() const
{
  return auto_calibrated_;
}

bool TaggedCauchyCPUEvaluator::policy_eligible(TaggedCauchyExecutionPolicy policy) const
{
  const auto &portfolio = model_->execution_portfolio;
  switch (policy) {
    case TaggedCauchyExecutionPolicy::COMPILED_DIRECT:
    case TaggedCauchyExecutionPolicy::GENERIC_DAG:
      return true;
    case TaggedCauchyExecutionPolicy::SYMMETRIC_POWER:
      return portfolio.present && portfolio.symmetric_power_eligible;
    case TaggedCauchyExecutionPolicy::BLOCK:
      return portfolio.present && portfolio.block_eligible;
    case TaggedCauchyExecutionPolicy::AUTO:
      return false;
  }
  return false;
}

std::array<double, 4> TaggedCauchyCPUEvaluator::calibrate_auto(int center_count,
                                                               int neighbors_per_center,
                                                               int measured_repeats)
{
  if (!auto_requested())
    throw std::runtime_error("tagged-Cauchy AUTO calibration requires the AUTO policy");
  if (center_count <= 0 || neighbors_per_center <= 0 || measured_repeats <= 0)
    throw std::invalid_argument("tagged-Cauchy AUTO calibration dimensions must be positive");

  const int species_count = static_cast<int>(model_->species_order.size());
  if (species_count <= 0 || model_->channels.empty())
    throw std::runtime_error("tagged-Cauchy AUTO calibration requires species and channels");

  std::vector<int> central_species(static_cast<std::size_t>(center_count));
  std::vector<std::size_t> edge_offsets(static_cast<std::size_t>(center_count + 1));
  const std::size_t edge_count =
      static_cast<std::size_t>(center_count) * static_cast<std::size_t>(neighbors_per_center);
  std::vector<int> neighbor_species(edge_count);
  std::vector<double> edge_vectors(edge_count * 3);
  for (int center = 0; center < center_count; ++center) {
    central_species[static_cast<std::size_t>(center)] = center % species_count;
    edge_offsets[static_cast<std::size_t>(center)] =
        static_cast<std::size_t>(center) * static_cast<std::size_t>(neighbors_per_center);
  }
  edge_offsets.back() = edge_count;

  // Deterministic golden-angle directions and several radii exercise every
  // source channel without tying AUTO to the current atom ordering.  The
  // calibration is a route microbenchmark, not a model prediction.
  constexpr double golden_angle = 2.39996322972865332223;
  for (std::size_t edge = 0; edge < edge_count; ++edge) {
    const TaggedCauchyChannel &channel = model_->channels[edge % model_->channels.size()];
    neighbor_species[edge] = channel.neighbor_species_index;
    const double u =
        (static_cast<double>(edge % static_cast<std::size_t>(neighbors_per_center)) + 0.5) /
        static_cast<double>(neighbors_per_center);
    const double z = 1.0 - 2.0 * u;
    const double xy = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double phi = golden_angle * static_cast<double>(edge);
    const double radius_fraction = 0.35 + 0.35 * static_cast<double>(edge % 7) / 6.0;
    const double radius = radius_fraction * model_->cutoff;
    edge_vectors[edge * 3] = radius * xy * std::cos(phi);
    edge_vectors[edge * 3 + 1] = radius * xy * std::sin(phi);
    edge_vectors[edge * 3 + 2] = radius * z;
  }

  std::vector<double> energies(static_cast<std::size_t>(center_count));
  std::vector<double> gradients(edge_count * 3);
  std::array<double, 4> elapsed;
  elapsed.fill(std::numeric_limits<double>::infinity());
  auto_calibration_lower_seconds_.fill(std::numeric_limits<double>::infinity());
  auto_calibration_upper_seconds_.fill(std::numeric_limits<double>::infinity());
  std::array<std::vector<double>, 4> samples;
  const TaggedCauchyExecutionPolicy saved_policy = selected_policy_;
  auto_calibration_in_progress_ = true;

  try {
    // One warmup per eligible route grows its scratch before timing.
    for (std::size_t candidate = 0; candidate < kAutoPolicies.size(); ++candidate) {
      if (!policy_eligible(kAutoPolicies[candidate])) continue;
      selected_policy_ = kAutoPolicies[candidate];
      evaluate(center_count, central_species.data(), edge_offsets.data(), neighbor_species.data(),
               edge_vectors.data(), energies.data(), gradients.data());
      samples[candidate].reserve(static_cast<std::size_t>(measured_repeats));
    }

    // Rotate the first candidate so cache warmth and short-term drift do not
    // systematically favor the same route in every repeat.
    for (int repeat = 0; repeat < measured_repeats; ++repeat) {
      for (std::size_t order = 0; order < kAutoPolicies.size(); ++order) {
        const std::size_t candidate =
            (static_cast<std::size_t>(repeat) + order) % kAutoPolicies.size();
        if (!policy_eligible(kAutoPolicies[candidate])) continue;
        selected_policy_ = kAutoPolicies[candidate];
        const auto begin = std::chrono::steady_clock::now();
        evaluate(center_count, central_species.data(), edge_offsets.data(), neighbor_species.data(),
                 edge_vectors.data(), energies.data(), gradients.data());
        const auto end = std::chrono::steady_clock::now();
        samples[candidate].push_back(std::chrono::duration<double>(end - begin).count());
      }
    }
  } catch (...) {
    selected_policy_ = saved_policy;
    auto_calibration_in_progress_ = false;
    throw;
  }
  selected_policy_ = saved_policy;
  auto_calibration_in_progress_ = false;

  for (std::size_t candidate = 0; candidate < samples.size(); ++candidate) {
    auto &values = samples[candidate];
    if (values.empty()) continue;
    std::sort(values.begin(), values.end());
    elapsed[candidate] = values[values.size() / 2];
    auto_calibration_lower_seconds_[candidate] = values[values.size() / 4];
    auto_calibration_upper_seconds_[candidate] = values[(3 * values.size()) / 4];
  }
  return elapsed;
}

bool TaggedCauchyCPUEvaluator::auto_candidate_confidently_faster(TaggedCauchyExecutionPolicy policy,
                                                                 double retention_ratio) const
{
  if (!(retention_ratio > 0.0 && retention_ratio < 1.0))
    throw std::invalid_argument("tagged-Cauchy AUTO retention ratio must be in (0, 1)");
  std::size_t candidate = kAutoPolicies.size();
  for (std::size_t index = 0; index < kAutoPolicies.size(); ++index)
    if (kAutoPolicies[index] == policy) {
      candidate = index;
      break;
    }
  if (candidate == kAutoPolicies.size() || candidate == 0 || !policy_eligible(policy)) return false;
  return auto_calibration_upper_seconds_[candidate] <
      retention_ratio * auto_calibration_lower_seconds_[0];
}

const std::array<double, 4> &TaggedCauchyCPUEvaluator::auto_calibration_lower_seconds() const
{
  return auto_calibration_lower_seconds_;
}

const std::array<double, 4> &TaggedCauchyCPUEvaluator::auto_calibration_upper_seconds() const
{
  return auto_calibration_upper_seconds_;
}

void TaggedCauchyCPUEvaluator::freeze_auto_policy(TaggedCauchyExecutionPolicy policy)
{
  if (!auto_requested())
    throw std::runtime_error("tagged-Cauchy AUTO route can be frozen only for AUTO policy");
  if (!policy_eligible(policy))
    throw std::runtime_error("tagged-Cauchy AUTO selected an ineligible execution policy");
  selected_policy_ = policy;
  auto_calibrated_ = true;
}

void TaggedCauchyCPUEvaluator::compute_edge_components(
    std::int64_t edge_count, const int *edge_neighbor_species, const double *edge_vectors, const double *edge_cutoffs,
    bool need_gradient, std::vector<double> &values, std::vector<double> &gradients)
{
  values.assign(static_cast<std::size_t>(edge_count) * static_cast<std::size_t>(total_components_),
                0.0);
  if (need_gradient)
    gradients.assign(static_cast<std::size_t>(edge_count) *
                         static_cast<std::size_t>(total_components_) * 3,
                     0.0);
  else
    gradients.clear();
  if (edge_count <= 0) return;
  const TaggedCauchyModel &model = *model_;
  const std::size_t n = static_cast<std::size_t>(edge_count);

  radii_.resize(n);
  unit_vectors_.resize(n * 3);
  bool recurrence_prevalidated = true;
  for (std::size_t e = 0; e < n; ++e) {
    const double dx = edge_vectors[e * 3];
    const double dy = edge_vectors[e * 3 + 1];
    const double dz = edge_vectors[e * 3 + 2];
    radii_[e] = std::sqrt(dx * dx + dy * dy + dz * dz);
    recurrence_prevalidated =
        recurrence_prevalidated && radii_[e] > kUnitEpsilon && std::isfinite(radii_[e]);
    const double safe_radius = std::max(radii_[e], kUnitEpsilon);
    unit_vectors_[e * 3] = dx / safe_radius;
    unit_vectors_[e * 3 + 1] = dy / safe_radius;
    unit_vectors_[e * 3 + 2] = dz / safe_radius;
    if (model.is_physical_image() && radii_[e] == 0.0)
      throw std::runtime_error("tagged-Cauchy V3 source rejects exact zero separation before "
                               "direction evaluation");
  }
  if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2) {
    radial_cutoffs_.assign(n, model.radial_rc);
    radial_cutoff_widths_.assign(n, model.radial_cutoff_width);
    radial_lambdas_.assign(n, model.radial_lambda);
    radial_values_.resize(n * static_cast<std::size_t>(model.radial_count));
    radial_derivatives_.resize(n * static_cast<std::size_t>(model.radial_count));
    ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
        radii_.data(), radial_cutoffs_.data(), radial_cutoff_widths_.data(), radial_lambdas_.data(),
        edge_count, model.radial_count, radial_values_.data(), radial_derivatives_.data());
  }

  const bool use_packed_angular = recurrence_prevalidated && angular_maximum_ >= 0;
  std::size_t packed_angular_width = 0;
  if (use_packed_angular) {
    packed_angular_width = static_cast<std::size_t>(
        ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(angular_maximum_));
    angular_values_.resize(n * packed_angular_width);
    angular_derivatives_.resize(n * packed_angular_width * 3);
    ye3t::runtime::
        complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
            double>(unit_vectors_.data(), radii_.data(), edge_count, angular_maximum_,
                    angular_plan_.data(), static_cast<std::int64_t>(angular_plan_.size()),
                    angular_values_.data(), angular_derivatives_.data());
  } else {
    for (std::size_t slot = 0; slot < l_list_.size(); ++slot) {
      const int l = l_list_[slot];
      const std::size_t width = static_cast<std::size_t>(2 * l + 1);
      angular_values_by_slot_[slot].resize(n * width);
      angular_derivatives_by_slot_[slot].resize(n * width * 3);
      ye3t::runtime::complex_spherical_harmonics_with_derivative<double>(
          edge_vectors, edge_count, l, kUnitEpsilon, angular_values_by_slot_[slot].data(),
          angular_derivatives_by_slot_[slot].data());
    }
  }

  const std::size_t real_angular_width = real_form_offsets_.back();
  real_angular_values_.resize(n * real_angular_width);
  if (need_gradient)
    real_angular_derivatives_.resize(n * real_angular_width * 3);
  else
    real_angular_derivatives_.clear();
  for (std::size_t form_index = 0; form_index < model.real_forms.size(); ++form_index) {
    const int slot = real_form_angular_slot_[form_index];
    if (slot < 0) continue;
    const TaggedCauchyRealForm &form = model.real_forms[form_index];
    const std::size_t width = static_cast<std::size_t>(form.width);
    const std::size_t form_offset = real_form_offsets_[form_index];
    for (std::size_t e = 0; e < n; ++e) {
      const std::complex<double> *Yc = use_packed_angular
          ? nullptr
          : &angular_values_by_slot_[static_cast<std::size_t>(slot)][e * width];
      const std::complex<double> *dYc = need_gradient && !use_packed_angular
          ? &angular_derivatives_by_slot_[static_cast<std::size_t>(slot)][e * width * 3]
          : nullptr;
      double *real_values = &real_angular_values_[e * real_angular_width + form_offset];
      double *real_derivatives = need_gradient
          ? &real_angular_derivatives_[(e * real_angular_width + form_offset) * 3]
          : nullptr;
      for (std::size_t a = 0; a < width; ++a) {
        std::complex<double> value_acc(0.0, 0.0);
        std::complex<double> gradient_acc[3] = {};
        const int begin = form.inverse_row_offsets[a];
        const int end = form.inverse_row_offsets[a + 1];
        for (int entry = begin; entry < end; ++entry) {
          const std::size_t matrix_row =
              static_cast<std::size_t>(form.inverse_columns[static_cast<std::size_t>(entry)]);
          const std::complex<double> inverse = form.inverse_values[static_cast<std::size_t>(entry)];
          const int magnetic = form.magnetic_order[static_cast<std::size_t>(matrix_row)];
          const std::size_t canonical_row = static_cast<std::size_t>(magnetic + form.l);
          std::complex<double> angular;
          std::complex<double> angular_gradient[3] = {};
          if (use_packed_angular) {
            const int order = std::abs(magnetic);
            const std::size_t packed = static_cast<std::size_t>(form.l * (form.l + 1) / 2 + order);
            const double sign = order % 2 == 0 ? 1.0 : -1.0;
            const std::complex<double> positive =
                angular_values_[e * packed_angular_width + packed];
            angular = magnetic < 0 ? sign * std::conj(positive) : positive;
            if (need_gradient)
              for (int axis = 0; axis < 3; ++axis) {
                const std::complex<double> positive_gradient =
                    angular_derivatives_[(e * packed_angular_width + packed) * 3 +
                                         static_cast<std::size_t>(axis)];
                angular_gradient[axis] = magnetic < 0 ? sign * std::conj(positive_gradient)
                                                      : positive_gradient;
              }
          } else {
            angular = Yc[canonical_row];
            if (need_gradient)
              for (int axis = 0; axis < 3; ++axis)
                angular_gradient[axis] = dYc[canonical_row * 3 + static_cast<std::size_t>(axis)];
          }
          value_acc += inverse * angular;
          if (need_gradient)
            for (int axis = 0; axis < 3; ++axis)
              gradient_acc[axis] += inverse * angular_gradient[axis];
        }
        real_values[a] = value_acc.real();
        if (need_gradient)
          for (int axis = 0; axis < 3; ++axis)
            real_derivatives[a * 3 + static_cast<std::size_t>(axis)] = gradient_acc[axis].real();
      }
    }
  }

  for (std::size_t e = 0; e < n; ++e) {
    const int species = edge_neighbor_species[e];
    // This edge's neighbor species matches no channel at
    // all -- every channel in the loop below would `continue` immediately,
    // and `value_row`/`gradient_row` are already zero from the `.assign`
    // above, so skip the unit-vector division and the whole channel loop.
    if (species < 0 || static_cast<std::size_t>(species) >= species_has_channel_.size() ||
        !species_has_channel_[static_cast<std::size_t>(species)])
      continue;
    const double *unit = unit_vectors_.data() + e * 3;
    double *value_row = &values[e * static_cast<std::size_t>(total_components_)];
    double *gradient_row =
        need_gradient ? &gradients[e * static_cast<std::size_t>(total_components_) * 3] : nullptr;
    if (model.source_realization == TaggedCauchySourceRealization::ShiftedJacobiThreeTermV1) {
      const double x = radii_[e] / edge_cutoffs[e];
      for (std::size_t slot = 0; slot < l_list_.size(); ++slot) {
        const std::size_t offset = jacobi_offsets_by_slot_[slot];
        shifted_jacobi_ladder_with_derivative(jacobi_max_q_by_slot_[slot], l_list_[slot], x,
                                              jacobi_values_.data() + offset,
                                              jacobi_derivatives_.data() + offset);
      }
    }
    for (std::size_t c = 0; c < model.channels.size(); ++c) {
      const TaggedCauchyChannel &channel = model.channels[c];
      if (channel.neighbor_species_index != species) continue;
      const TaggedCauchyRealForm &form =
          model.real_forms[static_cast<std::size_t>(channel.real_form_index)];
      const int width = form.width;
      const std::size_t form_offset = channel_real_form_offsets_[c];
      const double *real_values = &real_angular_values_[e * real_angular_width + form_offset];
      const double *real_derivatives = need_gradient
          ? &real_angular_derivatives_[(e * real_angular_width + form_offset) * 3]
          : nullptr;
      double R = 0.0;
      double dR = 0.0;
      if (model.is_physical_image()) {
        const double x = radii_[e] / edge_cutoffs[e];
        if (x < 1.0) {
          double polynomial = 0.0;
          double polynomial_dx = 0.0;
          if (model.source_realization == TaggedCauchySourceRealization::ShiftedJacobiThreeTermV1) {
            const std::size_t slot = static_cast<std::size_t>(channel_angular_slot_[c]);
            const std::size_t index =
                jacobi_offsets_by_slot_[slot] + static_cast<std::size_t>(channel.radial_channel);
            polynomial = jacobi_values_[index];
            polynomial_dx = jacobi_derivatives_[index];
          } else {
            power_series_with_derivative(channel.shifted_jacobi_power_coefficients, x, polynomial,
                                         polynomial_dx);
          }
          const double one_minus = 1.0 - x;
          const double envelope = one_minus * one_minus;
          const double envelope_dx = -2.0 * one_minus;
          const double x_l = std::pow(x, channel.l);
          const double x_l_dx =
              channel.l == 0 ? 0.0 : static_cast<double>(channel.l) * std::pow(x, channel.l - 1);
          const double scale = channel.normalization * channel.angular_scale;
          R = scale * envelope * polynomial * x_l;
          dR = scale *
              (envelope_dx * polynomial * x_l + envelope * polynomial_dx * x_l +
               envelope * polynomial * x_l_dx) /
              edge_cutoffs[e];
        }
      } else {
        R = radial_values_[e * static_cast<std::size_t>(model.radial_count) +
                           static_cast<std::size_t>(channel.radial_channel)];
        dR = radial_derivatives_[e * static_cast<std::size_t>(model.radial_count) +
                                 static_cast<std::size_t>(channel.radial_channel)];
      }
      for (int a = 0; a < width; ++a) {
        const double y_real = real_values[static_cast<std::size_t>(a)];
        const int flat = channel.component_offset + a;
        value_row[flat] = R * y_real;
        if (need_gradient)
          for (int axis = 0; axis < 3; ++axis)
            gradient_row[flat * 3 + axis] = dR * unit[axis] * y_real +
                R *
                    real_derivatives[static_cast<std::size_t>(a) * 3 +
                                     static_cast<std::size_t>(axis)];
      }
    }
  }
}

void TaggedCauchyCPUEvaluator::evaluate(int atom_count, const int *central_species_indices,
                                        const std::size_t *edge_offsets,
                                        const int *edge_neighbor_species,
                                        const double *edge_vectors, double *atomic_energies,
                                        double *edge_gradients, double *atomic_features)
{
  if (auto_requested() && !auto_calibrated_ && !auto_calibration_in_progress_)
    throw std::runtime_error("tagged-Cauchy AUTO must be calibrated and frozen before evaluation");
  if (atom_count < 0)
    throw std::invalid_argument("tagged-Cauchy evaluator received a negative atom count");
  const TaggedCauchyModel &model = *model_;
  const int DK = static_cast<int>(model.real_density_keys.size());
  const int MK = static_cast<int>(model.real_moment_keys.size());
  density_.assign(static_cast<std::size_t>(DK), 0.0);
  density_adjoint_.assign(static_cast<std::size_t>(DK), 0.0);
  moment_.assign(static_cast<std::size_t>(MK), 0.0);
  moment_adjoint_.assign(static_cast<std::size_t>(MK), 0.0);
  component_seed_.assign(static_cast<std::size_t>(total_components_), 0.0);

  for (int source_batch_begin = 0; source_batch_begin < atom_count;
       source_batch_begin += kSourceCenterBatch) {
    const int source_batch_end = std::min(atom_count, source_batch_begin + kSourceCenterBatch);
    const std::size_t source_edge_begin =
        edge_offsets[static_cast<std::size_t>(source_batch_begin)];
    const std::size_t source_edge_end = edge_offsets[static_cast<std::size_t>(source_batch_end)];
    const std::int64_t source_edge_count =
        static_cast<std::int64_t>(source_edge_end - source_edge_begin);
    source_batch_cutoffs_.assign(static_cast<std::size_t>(source_edge_count), model.cutoff);
    if (!model.pair_cutoffs.empty())
      for (int center = source_batch_begin; center < source_batch_end; ++center)
        for (std::size_t edge = edge_offsets[center]; edge < edge_offsets[center + 1]; ++edge)
          source_batch_cutoffs_[edge - source_edge_begin] = model.pair_cutoffs[
              static_cast<std::size_t>(central_species_indices[center]) * model.species_order.size() +
              static_cast<std::size_t>(edge_neighbor_species[edge])];
    compute_edge_components(source_edge_count, edge_neighbor_species + source_edge_begin,
                            edge_vectors + source_edge_begin * 3, source_batch_cutoffs_.data(), true,
                            source_batch_component_values_, source_batch_component_gradients_);

    for (int center = source_batch_begin; center < source_batch_end; ++center) {
      const std::size_t begin = edge_offsets[static_cast<std::size_t>(center)];
      const std::size_t end = edge_offsets[static_cast<std::size_t>(center) + 1];
      const std::size_t local_begin = begin - source_edge_begin;
      const std::int64_t edge_count = static_cast<std::int64_t>(end - begin);
      const std::size_t component_count =
          static_cast<std::size_t>(edge_count) * static_cast<std::size_t>(total_components_);
      component_values_.resize(component_count);
      component_gradients_.resize(component_count * 3);
      std::copy_n(source_batch_component_values_.data() +
                      local_begin * static_cast<std::size_t>(total_components_),
                  component_count, component_values_.data());
      std::copy_n(source_batch_component_gradients_.data() +
                      local_begin * static_cast<std::size_t>(total_components_) * 3,
                  component_count * 3, component_gradients_.data());
      const std::size_t species =
          static_cast<std::size_t>(central_species_indices[static_cast<std::size_t>(center)]);
      std::fill(density_.begin(), density_.end(), 0.0);
      const std::int64_t live_begin = species_live_moment_offsets_[species];
      const std::int64_t live_end = species_live_moment_offsets_[species + 1];
      const std::int64_t moment_node_begin = species_moment_node_offsets_[species];
      const std::int64_t moment_node_end = species_moment_node_offsets_[species + 1];
      const std::size_t moment_node_count =
          static_cast<std::size_t>(moment_node_end - moment_node_begin);
      const bool use_binary_moments =
          selected_policy_ == TaggedCauchyExecutionPolicy::SYMMETRIC_POWER ||
          selected_policy_ == TaggedCauchyExecutionPolicy::BLOCK;
      const bool use_binary_outer = selected_policy_ == TaggedCauchyExecutionPolicy::GENERIC_DAG ||
          selected_policy_ == TaggedCauchyExecutionPolicy::BLOCK;
      const std::size_t maximum_cached_values = kMomentNodeCacheBytes / sizeof(double);
      const std::size_t binary_moment_node_count =
          use_binary_moments ? model_->execution_portfolio.moment_plan.nodes.size() : 0;
      const bool cache_binary_moment_nodes = use_binary_moments && edge_count > 0 &&
          binary_moment_node_count <= maximum_cached_values / static_cast<std::size_t>(edge_count);
      const bool moment_workspace_fits = edge_count > 0 && moment_node_count > 0 &&
          moment_node_count <= maximum_cached_values / static_cast<std::size_t>(edge_count);
      const bool use_edge_tiled_moments =
          moment_workspace_fits && static_cast<std::size_t>(edge_count) >= kMomentEdgeTile;
      const bool cache_edge_moment_nodes = moment_workspace_fits && !use_edge_tiled_moments;
      if (cache_edge_moment_nodes)
        edge_moment_node_values_.resize(static_cast<std::size_t>(edge_count) * moment_node_count);
      for (std::int64_t live = live_begin; live < live_end; ++live)
        moment_[static_cast<std::size_t>(species_live_moments_[static_cast<std::size_t>(live)])] =
            0.0;

      if (use_binary_moments) {
        const auto &plan = model_->execution_portfolio.moment_plan;
        const std::size_t edges = static_cast<std::size_t>(edge_count);
        const std::size_t components = static_cast<std::size_t>(total_components_);
        component_values_transposed_.resize(components * edges);
        for (std::size_t component = 0; component < components; ++component)
          for (std::size_t edge = 0; edge < edges; ++edge)
            component_values_transposed_[component * edges + edge] =
                component_values_[edge * components + component];
        for (int idx = 0; idx < DK; ++idx) {
          const int flat = model_->real_density_flat_index[static_cast<std::size_t>(idx)];
          const double *values =
              component_values_transposed_.data() + static_cast<std::size_t>(flat) * edges;
          double sum = 0.0;
          for (std::size_t edge = 0; edge < edges; ++edge) sum += values[edge];
          density_[static_cast<std::size_t>(idx)] = sum;
        }
        std::fill(moment_.begin(), moment_.end(), 0.0);
        for (std::size_t moment = 0; moment < plan.roots.size(); ++moment)
          if (plan.roots[moment] < 0) moment_[moment] = static_cast<double>(edge_count);

        if (cache_binary_moment_nodes) {
          binary_moment_node_values_.resize(plan.nodes.size() * edges);
          for (std::size_t node_index = 0; node_index < plan.nodes.size(); ++node_index) {
            const TaggedCauchyBinaryNode &node = plan.nodes[node_index];
            double *output = binary_moment_node_values_.data() + node_index * edges;
            const double *left = node.left_value < plan.base_count
                ? component_values_transposed_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.left_value)]) *
                        edges
                : binary_moment_node_values_.data() +
                    static_cast<std::size_t>(node.left_value - plan.base_count) * edges;
            const double *right = node.right_value < plan.base_count
                ? component_values_transposed_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.right_value)]) *
                        edges
                : binary_moment_node_values_.data() +
                    static_cast<std::size_t>(node.right_value - plan.base_count) * edges;
            for (std::size_t edge = 0; edge < edges; ++edge)
              output[edge] = left[edge] * right[edge];
          }
          for (std::size_t moment = 0; moment < plan.roots.size(); ++moment) {
            const int root = plan.roots[moment];
            if (root < 0) continue;
            const double *values = root < plan.base_count ? component_values_transposed_.data() +
                    static_cast<std::size_t>(
                        model_->real_density_flat_index[static_cast<std::size_t>(root)]) *
                        edges
                                                          : binary_moment_node_values_.data() +
                    static_cast<std::size_t>(root - plan.base_count) * edges;
            double sum = 0.0;
            for (std::size_t edge = 0; edge < edges; ++edge) sum += values[edge];
            moment_[moment] = sum;
          }
        } else {
          binary_moment_node_values_.resize(plan.nodes.size() * kMomentEdgeTile);
          for (std::size_t tile_begin = 0; tile_begin < edges; tile_begin += kMomentEdgeTile) {
            const std::size_t lanes = std::min(kMomentEdgeTile, edges - tile_begin);
            for (std::size_t node_index = 0; node_index < plan.nodes.size(); ++node_index) {
              const TaggedCauchyBinaryNode &node = plan.nodes[node_index];
              double *output = binary_moment_node_values_.data() + node_index * kMomentEdgeTile;
              const double *left = node.left_value < plan.base_count
                  ? component_values_transposed_.data() +
                      static_cast<std::size_t>(
                          model_->real_density_flat_index[static_cast<std::size_t>(
                              node.left_value)]) *
                          edges +
                      tile_begin
                  : binary_moment_node_values_.data() +
                      static_cast<std::size_t>(node.left_value - plan.base_count) * kMomentEdgeTile;
              const double *right = node.right_value < plan.base_count
                  ? component_values_transposed_.data() +
                      static_cast<std::size_t>(
                          model_->real_density_flat_index[static_cast<std::size_t>(
                              node.right_value)]) *
                          edges +
                      tile_begin
                  : binary_moment_node_values_.data() +
                      static_cast<std::size_t>(node.right_value - plan.base_count) *
                          kMomentEdgeTile;
              for (std::size_t lane = 0; lane < lanes; ++lane)
                output[lane] = left[lane] * right[lane];
            }
            for (std::size_t moment = 0; moment < plan.roots.size(); ++moment) {
              const int root = plan.roots[moment];
              if (root < 0) continue;
              const double *values = root < plan.base_count ? component_values_transposed_.data() +
                      static_cast<std::size_t>(
                          model_->real_density_flat_index[static_cast<std::size_t>(root)]) *
                          edges +
                      tile_begin
                                                            : binary_moment_node_values_.data() +
                      static_cast<std::size_t>(root - plan.base_count) * kMomentEdgeTile;
              double sum = 0.0;
              for (std::size_t lane = 0; lane < lanes; ++lane) sum += values[lane];
              moment_[moment] += sum;
            }
          }
        }
      } else if (use_edge_tiled_moments) {
        const std::size_t edges = static_cast<std::size_t>(edge_count);
        const std::size_t components = static_cast<std::size_t>(total_components_);
        component_values_transposed_.resize(components * edges);
        for (std::size_t component = 0; component < components; ++component)
          for (std::size_t edge = 0; edge < edges; ++edge)
            component_values_transposed_[component * edges + edge] =
                component_values_[edge * components + component];

        for (int idx = 0; idx < DK; ++idx) {
          const int flat = model.real_density_flat_index[static_cast<std::size_t>(idx)];
          const double *values =
              component_values_transposed_.data() + static_cast<std::size_t>(flat) * edges;
          double sum = 0.0;
          for (std::size_t edge = 0; edge < edges; ++edge) sum += values[edge];
          density_[static_cast<std::size_t>(idx)] = sum;
        }
        for (std::int64_t live = live_begin; live < live_end; ++live)
          if (species_live_moment_nodes_[static_cast<std::size_t>(live)] < 0)
            moment_[static_cast<std::size_t>(
                species_live_moments_[static_cast<std::size_t>(live)])] =
                static_cast<double>(edge_count);

        batched_moment_node_values_.resize(moment_node_count * edges);
        for (std::int64_t node = moment_node_begin; node < moment_node_end; ++node) {
          const std::size_t local = static_cast<std::size_t>(node - moment_node_begin);
          const int parent = moment_node_parent_[static_cast<std::size_t>(node)];
          const int factor = moment_node_factor_[static_cast<std::size_t>(node)];
          const double *factor_values =
              component_values_transposed_.data() + static_cast<std::size_t>(factor) * edges;
          const double *parent_values = parent < 0 ? nullptr
                                                   : batched_moment_node_values_.data() +
                  static_cast<std::size_t>(parent - moment_node_begin) * edges;
          double *node_values = batched_moment_node_values_.data() + local * edges;
          if (parent_values == nullptr)
            std::copy(factor_values, factor_values + edges, node_values);
          else
            for (std::size_t edge = 0; edge < edges; ++edge)
              node_values[edge] = parent_values[edge] * factor_values[edge];

          for (std::int64_t terminal =
                   moment_node_terminal_offsets_[static_cast<std::size_t>(node)];
               terminal < moment_node_terminal_offsets_[static_cast<std::size_t>(node) + 1];
               ++terminal) {
            double sum = 0.0;
            for (std::size_t edge = 0; edge < edges; ++edge) sum += node_values[edge];
            moment_[static_cast<std::size_t>(
                moment_node_terminal_moments_[static_cast<std::size_t>(terminal)])] = sum;
          }
        }
      } else {
        // The scalar fallback preserves bounded memory for unusual wide plans
        // and high-coordination environments.
        for (std::int64_t e = 0; e < edge_count; ++e) {
          const double *v = &component_values_[static_cast<std::size_t>(e) *
                                               static_cast<std::size_t>(total_components_)];
          for (int idx = 0; idx < DK; ++idx)
            density_[static_cast<std::size_t>(idx)] +=
                v[model.real_density_flat_index[static_cast<std::size_t>(idx)]];

          for (std::int64_t node = moment_node_begin; node < moment_node_end; ++node) {
            const int parent = moment_node_parent_[static_cast<std::size_t>(node)];
            const int factor = moment_node_factor_[static_cast<std::size_t>(node)];
            const double value =
                (parent < 0 ? 1.0 : moment_node_values_[static_cast<std::size_t>(parent)]) *
                v[static_cast<std::size_t>(factor)];
            moment_node_values_[static_cast<std::size_t>(node)] = value;
            if (cache_edge_moment_nodes)
              edge_moment_node_values_[static_cast<std::size_t>(e) * moment_node_count +
                                       static_cast<std::size_t>(node - moment_node_begin)] = value;
          }
          for (std::int64_t live = live_begin; live < live_end; ++live) {
            const int m = species_live_moments_[static_cast<std::size_t>(live)];
            const int node = species_live_moment_nodes_[static_cast<std::size_t>(live)];
            moment_[static_cast<std::size_t>(m)] += node < 0
                ? 1.0
                : (cache_edge_moment_nodes
                       ? edge_moment_node_values_[static_cast<std::size_t>(e) * moment_node_count +
                                                  static_cast<std::size_t>(node -
                                                                           moment_node_begin)]
                       : moment_node_values_[static_cast<std::size_t>(node)]);
          }
        }
      }

      std::fill(density_adjoint_.begin(), density_adjoint_.end(), 0.0);
      std::fill(moment_adjoint_.begin(), moment_adjoint_.end(), 0.0);
      double energy = model.offsets[species];
      if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2)
        for (int p = 0; p <= maximum_term_p_; ++p)
          free_count_by_p_[static_cast<std::size_t>(p)] =
              falling_factorial(edge_count, p, model.tag_count);

      if (atomic_features != nullptr) {
        double *features = atomic_features +
            static_cast<std::size_t>(center) * static_cast<std::size_t>(model.feature_count);
        std::fill(features, features + model.feature_count, 0.0);
        for (const TaggedCauchyTerm &term : model.terms) {
          double product = term.coefficient;
          for (const int density_index : term.density_factor_indices)
            product *= density_[static_cast<std::size_t>(density_index)];
          for (const int moment_index : term.moment_indices)
            product *= moment_[static_cast<std::size_t>(moment_index)];
          if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2)
            product *= free_count_by_p_[static_cast<std::size_t>(term.p)];
          features[static_cast<std::size_t>(term.feature_index)] += product;
        }
      }

      const std::int64_t term_begin = species_term_offsets_[species];
      const std::int64_t term_end = species_term_offsets_[species + 1];
      if (use_binary_outer) {
        const auto &portfolio = model_->execution_portfolio;
        const auto &plan = portfolio.outer_plan;
        binary_outer_bases_.resize(static_cast<std::size_t>(plan.base_count));
        std::copy(density_.begin(), density_.end(), binary_outer_bases_.begin());
        std::copy(moment_.begin(), moment_.end(), binary_outer_bases_.begin() + density_.size());
        binary_outer_values_.resize(plan.nodes.size());
        for (std::size_t node_index = 0; node_index < plan.nodes.size(); ++node_index) {
          const TaggedCauchyBinaryNode &node = plan.nodes[node_index];
          const double left = node.left_value < plan.base_count
              ? binary_outer_bases_[static_cast<std::size_t>(node.left_value)]
              : binary_outer_values_[static_cast<std::size_t>(node.left_value - plan.base_count)];
          const double right = node.right_value < plan.base_count
              ? binary_outer_bases_[static_cast<std::size_t>(node.right_value)]
              : binary_outer_values_[static_cast<std::size_t>(node.right_value - plan.base_count)];
          binary_outer_values_[node_index] = left * right;
        }
        binary_outer_adjoint_.assign(plan.nodes.size(), 0.0);
        binary_outer_base_adjoint_.assign(static_cast<std::size_t>(plan.base_count), 0.0);
        const auto &routes = portfolio.species_routes[species];
        for (const TaggedCauchyExecutionRoute &route : routes) {
          const double seed =
              route.coefficient * free_count_by_p_[static_cast<std::size_t>(route.p)];
          if (seed == 0.0) continue;
          const double value = route.root_value < 0
              ? 1.0
              : (route.root_value < plan.base_count
                     ? binary_outer_bases_[static_cast<std::size_t>(route.root_value)]
                     : binary_outer_values_[static_cast<std::size_t>(route.root_value -
                                                                     plan.base_count)]);
          energy += seed * value;
          if (route.root_value >= 0) {
            if (route.root_value < plan.base_count)
              binary_outer_base_adjoint_[static_cast<std::size_t>(route.root_value)] += seed;
            else
              binary_outer_adjoint_[static_cast<std::size_t>(route.root_value - plan.base_count)] +=
                  seed;
          }
        }
        for (std::size_t reverse = plan.nodes.size(); reverse > 0;) {
          --reverse;
          const double seed = binary_outer_adjoint_[reverse];
          if (seed == 0.0) continue;
          const TaggedCauchyBinaryNode &node = plan.nodes[reverse];
          const double left = node.left_value < plan.base_count
              ? binary_outer_bases_[static_cast<std::size_t>(node.left_value)]
              : binary_outer_values_[static_cast<std::size_t>(node.left_value - plan.base_count)];
          const double right = node.right_value < plan.base_count
              ? binary_outer_bases_[static_cast<std::size_t>(node.right_value)]
              : binary_outer_values_[static_cast<std::size_t>(node.right_value - plan.base_count)];
          if (node.left_value < plan.base_count)
            binary_outer_base_adjoint_[static_cast<std::size_t>(node.left_value)] += seed * right;
          else
            binary_outer_adjoint_[static_cast<std::size_t>(node.left_value - plan.base_count)] +=
                seed * right;
          if (node.right_value < plan.base_count)
            binary_outer_base_adjoint_[static_cast<std::size_t>(node.right_value)] += seed * left;
          else
            binary_outer_adjoint_[static_cast<std::size_t>(node.right_value - plan.base_count)] +=
                seed * left;
        }
        std::copy(binary_outer_base_adjoint_.begin(),
                  binary_outer_base_adjoint_.begin() + density_.size(), density_adjoint_.begin());
        std::copy(binary_outer_base_adjoint_.begin() + density_.size(),
                  binary_outer_base_adjoint_.end(), moment_adjoint_.begin());
      } else if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2) {
        const std::int64_t node_begin = species_readout_node_offsets_[species];
        const std::int64_t node_end = species_readout_node_offsets_[species + 1];
        std::fill(readout_node_adjoint_.begin() + node_begin,
                  readout_node_adjoint_.begin() + node_end, 0.0);
        for (std::int64_t node = node_begin; node < node_end; ++node) {
          const int parent = readout_node_parent_[static_cast<std::size_t>(node)];
          const int variable = readout_node_variable_[static_cast<std::size_t>(node)];
          const double factor = variable < density_key_count_
              ? density_[static_cast<std::size_t>(variable)]
              : moment_[static_cast<std::size_t>(variable - density_key_count_)];
          readout_node_values_[static_cast<std::size_t>(node)] =
              (parent < 0 ? 1.0 : readout_node_values_[static_cast<std::size_t>(parent)]) * factor;
        }
        for (std::int64_t term = term_begin; term < term_end; ++term) {
          const std::size_t t = static_cast<std::size_t>(term);
          const double free_count = free_count_by_p_[static_cast<std::size_t>(term_p_[t])];
          if (free_count == 0.0) continue;
          const double seed = term_coefficient_[t] * free_count;
          const int node = term_product_node_[t];
          energy += seed * (node < 0 ? 1.0 : readout_node_values_[static_cast<std::size_t>(node)]);
          if (node >= 0) readout_node_adjoint_[static_cast<std::size_t>(node)] += seed;
        }
        for (std::int64_t node = node_end; node > node_begin;) {
          --node;
          const double seed = readout_node_adjoint_[static_cast<std::size_t>(node)];
          if (seed == 0.0) continue;
          const int parent = readout_node_parent_[static_cast<std::size_t>(node)];
          const int variable = readout_node_variable_[static_cast<std::size_t>(node)];
          const double factor = variable < density_key_count_
              ? density_[static_cast<std::size_t>(variable)]
              : moment_[static_cast<std::size_t>(variable - density_key_count_)];
          const double prefix = parent < 0 ? 1.0
                                           : readout_node_values_[static_cast<std::size_t>(parent)];
          if (variable < density_key_count_)
            density_adjoint_[static_cast<std::size_t>(variable)] += seed * prefix;
          else
            moment_adjoint_[static_cast<std::size_t>(variable - density_key_count_)] +=
                seed * prefix;
          if (parent >= 0) readout_node_adjoint_[static_cast<std::size_t>(parent)] += seed * factor;
        }
      } else {
        // V3 retains compiler forward order and uses its separately certified
        // division-free adjoint schedule below.
        for (std::int64_t term = term_begin; term < term_end; ++term) {
          const std::size_t t = static_cast<std::size_t>(term);
          double product = term_coefficient_[t];
          for (std::int64_t index = term_density_offsets_[t]; index < term_density_offsets_[t + 1];
               ++index)
            product *= density_[static_cast<std::size_t>(
                term_density_flat_[static_cast<std::size_t>(index)])];
          for (std::int64_t index = term_moment_offsets_[t]; index < term_moment_offsets_[t + 1];
               ++index)
            product *= moment_[static_cast<std::size_t>(
                term_moment_flat_[static_cast<std::size_t>(index)])];
          energy += product;
        }
        const std::vector<double> &folded_adjoint = folded_adjoint_coefficient_[species];
        for (std::size_t t = 0; t < adjoint_source_index_.size(); ++t) {
          double product = folded_adjoint[t];
          for (std::int64_t index = adjoint_remaining_offsets_[t];
               index < adjoint_remaining_offsets_[t + 1]; ++index)
            product *= density_[static_cast<std::size_t>(
                adjoint_remaining_flat_[static_cast<std::size_t>(index)])];
          density_adjoint_[static_cast<std::size_t>(adjoint_source_index_[t])] += product;
        }
      }
      if (!std::isfinite(energy))
        throw std::runtime_error("tagged-Cauchy evaluator produced a non-finite atomic energy");
      atomic_energies[static_cast<std::size_t>(center)] = energy;

      // Pass 2 (reverse): distribute dE/dA and dE/dM back through the same
      // cached per-edge factor products and contract with the source gradient.
      if (use_binary_moments) {
        const auto &plan = model_->execution_portfolio.moment_plan;
        const std::size_t edges = static_cast<std::size_t>(edge_count);
        const std::size_t components = static_cast<std::size_t>(total_components_);
        if (!cache_binary_moment_nodes)
          binary_moment_node_values_.resize(plan.nodes.size() * kMomentEdgeTile);
        binary_moment_tile_adjoint_.resize(plan.nodes.size() * kMomentEdgeTile);
        component_seed_tile_.resize(components * kMomentEdgeTile);
        for (std::size_t tile_begin = 0; tile_begin < edges; tile_begin += kMomentEdgeTile) {
          const std::size_t lanes = std::min(kMomentEdgeTile, edges - tile_begin);
          if (!cache_binary_moment_nodes) {
            for (std::size_t node_index = 0; node_index < plan.nodes.size(); ++node_index) {
              const TaggedCauchyBinaryNode &node = plan.nodes[node_index];
              double *output = binary_moment_node_values_.data() + node_index * kMomentEdgeTile;
              const double *left = node.left_value < plan.base_count
                  ? component_values_transposed_.data() +
                      static_cast<std::size_t>(
                          model_->real_density_flat_index[static_cast<std::size_t>(
                              node.left_value)]) *
                          edges +
                      tile_begin
                  : binary_moment_node_values_.data() +
                      static_cast<std::size_t>(node.left_value - plan.base_count) * kMomentEdgeTile;
              const double *right = node.right_value < plan.base_count
                  ? component_values_transposed_.data() +
                      static_cast<std::size_t>(
                          model_->real_density_flat_index[static_cast<std::size_t>(
                              node.right_value)]) *
                          edges +
                      tile_begin
                  : binary_moment_node_values_.data() +
                      static_cast<std::size_t>(node.right_value - plan.base_count) *
                          kMomentEdgeTile;
              for (std::size_t lane = 0; lane < lanes; ++lane)
                output[lane] = left[lane] * right[lane];
            }
          }
          std::fill(binary_moment_tile_adjoint_.begin(), binary_moment_tile_adjoint_.end(), 0.0);
          std::fill(component_seed_tile_.begin(), component_seed_tile_.end(), 0.0);
          for (std::size_t moment = 0; moment < plan.roots.size(); ++moment) {
            const double seed = moment_adjoint_[moment];
            const int root = plan.roots[moment];
            if (seed == 0.0 || root < 0) continue;
            if (root < plan.base_count) {
              const int flat = model_->real_density_flat_index[static_cast<std::size_t>(root)];
              double *target =
                  component_seed_tile_.data() + static_cast<std::size_t>(flat) * kMomentEdgeTile;
              for (std::size_t lane = 0; lane < lanes; ++lane) target[lane] += seed;
            } else {
              double *target = binary_moment_tile_adjoint_.data() +
                  static_cast<std::size_t>(root - plan.base_count) * kMomentEdgeTile;
              for (std::size_t lane = 0; lane < lanes; ++lane) target[lane] += seed;
            }
          }
          for (std::size_t reverse = plan.nodes.size(); reverse > 0;) {
            --reverse;
            const TaggedCauchyBinaryNode &node = plan.nodes[reverse];
            const double *left = node.left_value < plan.base_count
                ? component_values_transposed_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.left_value)]) *
                        edges +
                    tile_begin
                : binary_moment_node_values_.data() +
                    static_cast<std::size_t>(node.left_value - plan.base_count) *
                        (cache_binary_moment_nodes ? edges : kMomentEdgeTile) +
                    (cache_binary_moment_nodes ? tile_begin : 0);
            const double *right = node.right_value < plan.base_count
                ? component_values_transposed_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.right_value)]) *
                        edges +
                    tile_begin
                : binary_moment_node_values_.data() +
                    static_cast<std::size_t>(node.right_value - plan.base_count) *
                        (cache_binary_moment_nodes ? edges : kMomentEdgeTile) +
                    (cache_binary_moment_nodes ? tile_begin : 0);
            double *left_seed = node.left_value < plan.base_count
                ? component_seed_tile_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.left_value)]) *
                        kMomentEdgeTile
                : binary_moment_tile_adjoint_.data() +
                    static_cast<std::size_t>(node.left_value - plan.base_count) * kMomentEdgeTile;
            double *right_seed = node.right_value < plan.base_count
                ? component_seed_tile_.data() +
                    static_cast<std::size_t>(
                        model_
                            ->real_density_flat_index[static_cast<std::size_t>(node.right_value)]) *
                        kMomentEdgeTile
                : binary_moment_tile_adjoint_.data() +
                    static_cast<std::size_t>(node.right_value - plan.base_count) * kMomentEdgeTile;
            const double *node_seed =
                binary_moment_tile_adjoint_.data() + reverse * kMomentEdgeTile;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
              const double seed = node_seed[lane];
              if (seed == 0.0) continue;
              left_seed[lane] += seed * right[lane];
              right_seed[lane] += seed * left[lane];
            }
          }
          for (int idx = 0; idx < DK; ++idx) {
            const int flat = model.real_density_flat_index[static_cast<std::size_t>(idx)];
            double *target =
                component_seed_tile_.data() + static_cast<std::size_t>(flat) * kMomentEdgeTile;
            const double seed = density_adjoint_[static_cast<std::size_t>(idx)];
            for (std::size_t lane = 0; lane < lanes; ++lane) target[lane] += seed;
          }
          for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::size_t edge = tile_begin + lane;
            const double *gradient = &component_gradients_[edge * components * 3];
            double gx = 0.0, gy = 0.0, gz = 0.0;
            for (std::size_t component = 0; component < components; ++component) {
              const double seed = component_seed_tile_[component * kMomentEdgeTile + lane];
              if (seed == 0.0) continue;
              gx += seed * gradient[component * 3];
              gy += seed * gradient[component * 3 + 1];
              gz += seed * gradient[component * 3 + 2];
            }
            edge_gradients[(begin + edge) * 3] = gx;
            edge_gradients[(begin + edge) * 3 + 1] = gy;
            edge_gradients[(begin + edge) * 3 + 2] = gz;
          }
        }
      } else if (use_edge_tiled_moments) {
        const std::size_t edges = static_cast<std::size_t>(edge_count);
        const std::size_t components = static_cast<std::size_t>(total_components_);
        moment_node_tile_adjoint_.resize(moment_node_count * kMomentEdgeTile);
        component_seed_tile_.resize(components * kMomentEdgeTile);
        for (std::size_t tile_begin = 0; tile_begin < edges; tile_begin += kMomentEdgeTile) {
          const std::size_t lanes = std::min(kMomentEdgeTile, edges - tile_begin);
          std::fill(component_seed_tile_.begin(), component_seed_tile_.end(), 0.0);
          for (std::int64_t node = moment_node_begin; node < moment_node_end; ++node) {
            double terminal_seed = 0.0;
            for (std::int64_t terminal =
                     moment_node_terminal_offsets_[static_cast<std::size_t>(node)];
                 terminal < moment_node_terminal_offsets_[static_cast<std::size_t>(node) + 1];
                 ++terminal)
              terminal_seed += moment_adjoint_[static_cast<std::size_t>(
                  moment_node_terminal_moments_[static_cast<std::size_t>(terminal)])];
            double *node_adjoint = moment_node_tile_adjoint_.data() +
                static_cast<std::size_t>(node - moment_node_begin) * kMomentEdgeTile;
            std::fill(node_adjoint, node_adjoint + lanes, terminal_seed);
          }

          for (std::int64_t node = moment_node_end; node > moment_node_begin;) {
            --node;
            const std::size_t local = static_cast<std::size_t>(node - moment_node_begin);
            const int parent = moment_node_parent_[static_cast<std::size_t>(node)];
            const int factor = moment_node_factor_[static_cast<std::size_t>(node)];
            const double *factor_values = component_values_transposed_.data() +
                static_cast<std::size_t>(factor) * edges + tile_begin;
            const double *parent_values = parent < 0 ? nullptr
                                                     : batched_moment_node_values_.data() +
                    static_cast<std::size_t>(parent - moment_node_begin) * edges + tile_begin;
            double *node_adjoint = moment_node_tile_adjoint_.data() + local * kMomentEdgeTile;
            double *parent_adjoint = parent < 0 ? nullptr
                                                : moment_node_tile_adjoint_.data() +
                    static_cast<std::size_t>(parent - moment_node_begin) * kMomentEdgeTile;
            double *factor_seed =
                component_seed_tile_.data() + static_cast<std::size_t>(factor) * kMomentEdgeTile;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
              const double seed = node_adjoint[lane];
              if (seed == 0.0) continue;
              factor_seed[lane] += seed * (parent_values == nullptr ? 1.0 : parent_values[lane]);
              if (parent_adjoint != nullptr) parent_adjoint[lane] += seed * factor_values[lane];
            }
          }

          for (int idx = 0; idx < DK; ++idx) {
            const int flat = model.real_density_flat_index[static_cast<std::size_t>(idx)];
            double *seed =
                component_seed_tile_.data() + static_cast<std::size_t>(flat) * kMomentEdgeTile;
            const double value = density_adjoint_[static_cast<std::size_t>(idx)];
            for (std::size_t lane = 0; lane < lanes; ++lane) seed[lane] += value;
          }

          for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::size_t edge = tile_begin + lane;
            const double *gradient = &component_gradients_[edge * components * 3];
            double gx = 0.0, gy = 0.0, gz = 0.0;
            for (std::size_t component = 0; component < components; ++component) {
              const double seed = component_seed_tile_[component * kMomentEdgeTile + lane];
              if (seed == 0.0) continue;
              gx += seed * gradient[component * 3];
              gy += seed * gradient[component * 3 + 1];
              gz += seed * gradient[component * 3 + 2];
            }
            edge_gradients[(begin + edge) * 3] = gx;
            edge_gradients[(begin + edge) * 3 + 1] = gy;
            edge_gradients[(begin + edge) * 3 + 2] = gz;
          }
        }
      } else {
        for (std::int64_t e = 0; e < edge_count; ++e) {
          const double *v = &component_values_[static_cast<std::size_t>(e) *
                                               static_cast<std::size_t>(total_components_)];
          const double *g = &component_gradients_[static_cast<std::size_t>(e) *
                                                  static_cast<std::size_t>(total_components_) * 3];
          std::fill(component_seed_.begin(), component_seed_.end(), 0.0);
          for (int idx = 0; idx < DK; ++idx)
            component_seed_[model.real_density_flat_index[static_cast<std::size_t>(idx)]] +=
                density_adjoint_[static_cast<std::size_t>(idx)];

          if (!cache_edge_moment_nodes)
            for (std::int64_t node = moment_node_begin; node < moment_node_end; ++node) {
              const int parent = moment_node_parent_[static_cast<std::size_t>(node)];
              const int factor = moment_node_factor_[static_cast<std::size_t>(node)];
              moment_node_values_[static_cast<std::size_t>(node)] =
                  (parent < 0 ? 1.0 : moment_node_values_[static_cast<std::size_t>(parent)]) *
                  v[static_cast<std::size_t>(factor)];
            }
          std::fill(moment_node_adjoint_.begin() + moment_node_begin,
                    moment_node_adjoint_.begin() + moment_node_end, 0.0);
          for (std::int64_t live = live_begin; live < live_end; ++live) {
            const int m = species_live_moments_[static_cast<std::size_t>(live)];
            const int node = species_live_moment_nodes_[static_cast<std::size_t>(live)];
            if (node >= 0)
              moment_node_adjoint_[static_cast<std::size_t>(node)] +=
                  moment_adjoint_[static_cast<std::size_t>(m)];
          }
          for (std::int64_t node = moment_node_end; node > moment_node_begin;) {
            --node;
            const double seed = moment_node_adjoint_[static_cast<std::size_t>(node)];
            if (seed == 0.0) continue;
            const int parent = moment_node_parent_[static_cast<std::size_t>(node)];
            const int factor = moment_node_factor_[static_cast<std::size_t>(node)];
            const double prefix = parent < 0
                ? 1.0
                : (cache_edge_moment_nodes
                       ? edge_moment_node_values_[static_cast<std::size_t>(e) * moment_node_count +
                                                  static_cast<std::size_t>(parent -
                                                                           moment_node_begin)]
                       : moment_node_values_[static_cast<std::size_t>(parent)]);
            component_seed_[static_cast<std::size_t>(factor)] += seed * prefix;
            if (parent >= 0)
              moment_node_adjoint_[static_cast<std::size_t>(parent)] +=
                  seed * v[static_cast<std::size_t>(factor)];
          }

          double gx = 0.0, gy = 0.0, gz = 0.0;
          for (int c = 0; c < total_components_; ++c) {
            const double seed = component_seed_[static_cast<std::size_t>(c)];
            if (seed == 0.0) continue;
            gx += seed * g[static_cast<std::size_t>(c) * 3];
            gy += seed * g[static_cast<std::size_t>(c) * 3 + 1];
            gz += seed * g[static_cast<std::size_t>(c) * 3 + 2];
          }
          edge_gradients[(begin + static_cast<std::size_t>(e)) * 3] = gx;
          edge_gradients[(begin + static_cast<std::size_t>(e)) * 3 + 1] = gy;
          edge_gradients[(begin + static_cast<std::size_t>(e)) * 3 + 2] = gz;
        }
      }
      if (!model.zbl_pairs.empty()) {
        const double weights[] = {0.18175, 0.50986, 0.28022, 0.02817};
        const double rates[] = {3.19980, 0.94229, 0.40290, 0.20162};
        for (std::size_t edge = begin; edge < end; ++edge) {
          const auto &pair = model.zbl_pairs[species * model.species_order.size() +
                                            static_cast<std::size_t>(edge_neighbor_species[edge])];
          const double *delta = edge_vectors + 3 * edge;
          const double r = std::sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
          if (r >= pair.outer) continue;
          if (r <= 0.0) throw std::runtime_error("ZBL rejects coincident nuclei");
          double phi = 0.0, first = 0.0;
          for (int term = 0; term < 4; ++term) {
            const double rate = rates[term]/pair.screening_length;
            const double value = weights[term]*std::exp(-rate*r);
            phi += value; first -= rate*value;
          }
          const double shift = std::max(0.0, r-pair.inner);
          const double square = shift*shift;
          const double value = pair.amplitude*phi/r + pair.constant +
              pair.cubic*square*shift/3 + pair.quartic*square*square/4;
          const double derivative = pair.amplitude*(first/r-phi/(r*r)) +
              pair.cubic*square + pair.quartic*square*shift;
          // Each centered full-list edge owns half of this pair reference.
          atomic_energies[center] += 0.5*value;
          for (int axis = 0; axis < 3; ++axis)
            edge_gradients[edge*3+axis] += 0.5*derivative*delta[axis]/r;
        }
      }
    }
  }
}

double TaggedCauchyCPUEvaluator::memory_usage() const
{
  double bytes = sizeof(TaggedCauchyCPUEvaluator);
  bytes += l_list_.capacity() * sizeof(int);
  bytes += l_slot_of_l_.capacity() * sizeof(int);
  bytes += channel_angular_slot_.capacity() * sizeof(int);
  bytes += real_form_angular_slot_.capacity() * sizeof(int);
  bytes += real_form_offsets_.capacity() * sizeof(std::size_t);
  bytes += channel_real_form_offsets_.capacity() * sizeof(std::size_t);
  bytes += jacobi_max_q_by_slot_.capacity() * sizeof(int);
  bytes += jacobi_offsets_by_slot_.capacity() * sizeof(std::size_t);
  bytes += jacobi_values_.capacity() * sizeof(double);
  bytes += jacobi_derivatives_.capacity() * sizeof(double);
  bytes += radii_.capacity() * sizeof(double);
  bytes += radial_cutoffs_.capacity() * sizeof(double);
  bytes += radial_cutoff_widths_.capacity() * sizeof(double);
  bytes += radial_lambdas_.capacity() * sizeof(double);
  bytes += radial_values_.capacity() * sizeof(double);
  bytes += radial_derivatives_.capacity() * sizeof(double);
  bytes += unit_vectors_.capacity() * sizeof(double);
  bytes += angular_plan_.capacity() * sizeof(double);
  bytes += angular_values_.capacity() * sizeof(std::complex<double>);
  bytes += angular_derivatives_.capacity() * sizeof(std::complex<double>);
  bytes += angular_values_by_slot_.capacity() * sizeof(std::vector<std::complex<double>>);
  for (const auto &buffer : angular_values_by_slot_)
    bytes += buffer.capacity() * sizeof(std::complex<double>);
  bytes += angular_derivatives_by_slot_.capacity() * sizeof(std::vector<std::complex<double>>);
  for (const auto &buffer : angular_derivatives_by_slot_)
    bytes += buffer.capacity() * sizeof(std::complex<double>);
  bytes += real_angular_values_.capacity() * sizeof(double);
  bytes += real_angular_derivatives_.capacity() * sizeof(double);
  bytes += component_values_.capacity() * sizeof(double);
  bytes += component_gradients_.capacity() * sizeof(double);
  bytes += source_batch_component_values_.capacity() * sizeof(double);
  bytes += source_batch_component_gradients_.capacity() * sizeof(double);
  bytes += source_batch_cutoffs_.capacity() * sizeof(double);
  bytes += component_values_transposed_.capacity() * sizeof(double);
  bytes += density_.capacity() * sizeof(double);
  bytes += density_adjoint_.capacity() * sizeof(double);
  bytes += moment_.capacity() * sizeof(double);
  bytes += moment_adjoint_.capacity() * sizeof(double);
  bytes += component_seed_.capacity() * sizeof(double);
  bytes += species_moment_node_offsets_.capacity() * sizeof(std::int64_t);
  bytes += moment_node_parent_.capacity() * sizeof(int);
  bytes += moment_node_factor_.capacity() * sizeof(int);
  bytes += moment_node_terminal_offsets_.capacity() * sizeof(std::int64_t);
  bytes += moment_node_terminal_moments_.capacity() * sizeof(int);
  bytes += moment_node_values_.capacity() * sizeof(double);
  bytes += moment_node_adjoint_.capacity() * sizeof(double);
  bytes += edge_moment_node_values_.capacity() * sizeof(double);
  bytes += batched_moment_node_values_.capacity() * sizeof(double);
  bytes += moment_node_tile_adjoint_.capacity() * sizeof(double);
  bytes += component_seed_tile_.capacity() * sizeof(double);
  bytes += binary_moment_node_values_.capacity() * sizeof(double);
  bytes += binary_moment_tile_adjoint_.capacity() * sizeof(double);
  bytes += binary_outer_values_.capacity() * sizeof(double);
  bytes += binary_outer_adjoint_.capacity() * sizeof(double);
  bytes += binary_outer_bases_.capacity() * sizeof(double);
  bytes += binary_outer_base_adjoint_.capacity() * sizeof(double);
  bytes += species_term_offsets_.capacity() * sizeof(std::int64_t);
  bytes += term_density_offsets_.capacity() * sizeof(std::int64_t);
  bytes += term_density_flat_.capacity() * sizeof(int);
  bytes += term_moment_offsets_.capacity() * sizeof(std::int64_t);
  bytes += term_moment_flat_.capacity() * sizeof(int);
  bytes += term_p_.capacity() * sizeof(int);
  bytes += term_coefficient_.capacity() * sizeof(double);
  bytes += term_product_node_.capacity() * sizeof(int);
  bytes += free_count_by_p_.capacity() * sizeof(double);
  bytes += species_readout_node_offsets_.capacity() * sizeof(std::int64_t);
  bytes += readout_node_parent_.capacity() * sizeof(int);
  bytes += readout_node_variable_.capacity() * sizeof(int);
  bytes += readout_node_values_.capacity() * sizeof(double);
  bytes += readout_node_adjoint_.capacity() * sizeof(double);
  bytes += species_live_moment_offsets_.capacity() * sizeof(std::int64_t);
  bytes += species_live_moments_.capacity() * sizeof(int);
  bytes += species_live_moment_nodes_.capacity() * sizeof(int);
  bytes += adjoint_remaining_offsets_.capacity() * sizeof(std::int64_t);
  bytes += adjoint_remaining_flat_.capacity() * sizeof(int);
  bytes += adjoint_source_index_.capacity() * sizeof(int);
  bytes += folded_adjoint_coefficient_.capacity() * sizeof(std::vector<double>);
  for (const auto &row : folded_adjoint_coefficient_) bytes += row.capacity() * sizeof(double);
  bytes += (species_has_channel_.capacity() + 7) / 8;
  return bytes;
}

}    // namespace YE3T_LAMMPS
