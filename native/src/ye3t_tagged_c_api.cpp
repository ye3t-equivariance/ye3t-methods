// C ABI for tagged-Cauchy and tagged-plus-ordinary ACE CPU models.
// Both component evaluators and their sum match pair_style ye3t.

#include "ye3t_tagged_cauchy_cpu.h"
#include "ye3t_cpu_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using YE3T_LAMMPS::TaggedCauchyCPUEvaluator;
using YE3T_LAMMPS::TaggedCauchyExecutionPolicy;
using YE3T_LAMMPS::TaggedCauchyModel;
using YE3T_LAMMPS::YACEBlockPolicy;
using YE3T_LAMMPS::YACEModel;
using YE3T_LAMMPS::YE3TCPUEvaluator;

struct Handle {
  TaggedCauchyModel model;
  TaggedCauchyCPUEvaluator evaluator;
  std::unique_ptr<YACEModel> ordinary_model;
  std::unique_ptr<YE3TCPUEvaluator> ordinary_evaluator;
  std::vector<int> edge_centers;
  std::vector<double> ordinary_atomic_energies;
  std::vector<double> ordinary_edge_gradients;

  Handle(const char *path, bool automatic)
      : model(TaggedCauchyModel::load(path)),
        evaluator(&model, automatic ? TaggedCauchyExecutionPolicy::AUTO
                                    : TaggedCauchyExecutionPolicy::COMPILED_DIRECT) {
    if (model.has_ordinary_backbone()) {
      ordinary_model = std::make_unique<YACEModel>(
          YACEModel::load(model.ordinary_model_path, std::string(), YACEBlockPolicy::DIRECT));
      if (ordinary_model->species_count() != static_cast<int>(model.species_order.size()))
        throw std::runtime_error("ordinary and tagged component species counts differ");
      for (int species = 0; species < ordinary_model->species_count(); ++species)
        if (ordinary_model->species(species).element !=
            model.species_order[static_cast<std::size_t>(species)])
          throw std::runtime_error("ordinary and tagged component species ordering differs");
      const double tolerance = 2.0e-12 * std::max(1.0, std::abs(model.cutoff));
      if (std::abs(ordinary_model->maximum_cutoff() - model.cutoff) > tolerance)
        throw std::runtime_error("ordinary and tagged component cutoffs differ");
      ordinary_evaluator = std::make_unique<YE3TCPUEvaluator>(ordinary_model.get());
    }
    if (!automatic) return;
    const auto times = evaluator.calibrate_auto(128, 16, 11);
    int fastest = 0;
    for (int candidate = 1; candidate < 4; ++candidate)
      if (times[static_cast<std::size_t>(candidate)] < times[static_cast<std::size_t>(fastest)])
        fastest = candidate;
    constexpr std::array<TaggedCauchyExecutionPolicy, 4> policies = {
        TaggedCauchyExecutionPolicy::COMPILED_DIRECT,
        TaggedCauchyExecutionPolicy::GENERIC_DAG,
        TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
        TaggedCauchyExecutionPolicy::BLOCK,
    };
    if (fastest && !evaluator.auto_candidate_confidently_faster(policies[fastest], 0.92))
      fastest = 0;
    evaluator.freeze_auto_policy(policies[fastest]);
  }

  void evaluate(int atom_count, const int *central_species, const std::size_t *edge_offsets,
                const int *edge_neighbor_species, const double *edge_vectors,
                double *atomic_energies, double *edge_gradients,
                double *atomic_features = nullptr) {
    if (atomic_features && ordinary_evaluator)
      throw std::invalid_argument("Composite feature export needs both ordinary ACE and tagged columns.");
    if (edge_offsets[atom_count] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Native edge count exceeds the int range.");
    evaluator.evaluate(atom_count, central_species, edge_offsets, edge_neighbor_species,
                       edge_vectors, atomic_energies, edge_gradients, atomic_features);
    if (!ordinary_evaluator) return;
    const auto edge_count = static_cast<int>(edge_offsets[atom_count]);
    edge_centers.resize(static_cast<std::size_t>(edge_count));
    for (int center = 0; center < atom_count; ++center)
      std::fill(edge_centers.begin() + static_cast<std::ptrdiff_t>(edge_offsets[center]),
                edge_centers.begin() + static_cast<std::ptrdiff_t>(edge_offsets[center + 1]), center);
    ordinary_atomic_energies.resize(static_cast<std::size_t>(atom_count));
    ordinary_edge_gradients.resize(static_cast<std::size_t>(edge_count) * 3);
    ordinary_evaluator->evaluate(atom_count, central_species, edge_count, edge_centers.data(),
                                 edge_neighbor_species, edge_vectors,
                                 ordinary_atomic_energies.data(), ordinary_edge_gradients.data());
    for (int center = 0; center < atom_count; ++center)
      atomic_energies[center] += ordinary_atomic_energies[static_cast<std::size_t>(center)];
    for (std::size_t value = 0; value < static_cast<std::size_t>(edge_count) * 3; ++value)
      edge_gradients[value] += ordinary_edge_gradients[value];
  }
};

struct OrdinaryHandle {
  YACEModel model;
  YE3TCPUEvaluator evaluator;
  std::vector<int> edge_centers;

  explicit OrdinaryHandle(const char *path)
      : model(YACEModel::load(path, std::string(), YACEBlockPolicy::DIRECT)), evaluator(&model) {}

  void evaluate(int atom_count, const int *central_species, const std::size_t *edge_offsets,
                const int *edge_neighbor_species, const double *edge_vectors,
                double *atomic_energies, double *edge_gradients) {
    if (edge_offsets[atom_count] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Native edge count exceeds the int range.");
    const int edge_count = static_cast<int>(edge_offsets[atom_count]);
    edge_centers.resize(static_cast<std::size_t>(edge_count));
    for (int center = 0; center < atom_count; ++center)
      std::fill(edge_centers.begin() + static_cast<std::ptrdiff_t>(edge_offsets[center]),
                edge_centers.begin() + static_cast<std::ptrdiff_t>(edge_offsets[center + 1]), center);
    evaluator.evaluate(atom_count, central_species, edge_count, edge_centers.data(),
                       edge_neighbor_species, edge_vectors, atomic_energies, edge_gradients);
  }
};

void error_text(char *buffer, std::size_t capacity, const char *message) noexcept {
  if (buffer && capacity) std::snprintf(buffer, capacity, "%s", message);
}

}  // namespace

extern "C" {

void *ye3t_yace_open(const char *path, char *error, std::size_t error_capacity) noexcept {
  try {
    if (!path || !*path) throw std::invalid_argument("YACE model path is empty.");
    return new OrdinaryHandle(path);
  } catch (const std::exception &exception) {
    error_text(error, error_capacity, exception.what());
  } catch (...) {
    error_text(error, error_capacity, "Unknown native YACE-model load error.");
  }
  return nullptr;
}

void ye3t_yace_close(void *handle) noexcept { delete static_cast<OrdinaryHandle *>(handle); }

int ye3t_yace_species_count(void *handle) noexcept {
  return handle ? static_cast<OrdinaryHandle *>(handle)->model.species_count() : -1;
}

const char *ye3t_yace_species_name(void *handle, int species_index) noexcept {
  if (!handle) return nullptr;
  const auto &model = static_cast<OrdinaryHandle *>(handle)->model;
  if (species_index < 0 || species_index >= model.species_count()) return nullptr;
  return model.species(species_index).element.c_str();
}

double ye3t_yace_maximum_cutoff(void *handle) noexcept {
  return handle ? static_cast<OrdinaryHandle *>(handle)->model.maximum_cutoff() : 0.0;
}

int ye3t_yace_evaluate(void *handle, int atom_count, const int *central_species,
                       const std::size_t *edge_offsets, const int *edge_neighbor_species,
                       const double *edge_vectors, double *atomic_energies,
                       double *edge_gradients, char *error,
                       std::size_t error_capacity) noexcept {
  try {
    if (!handle || atom_count < 0 || !edge_offsets ||
        (atom_count > 0 && (!central_species || !atomic_energies)))
      throw std::invalid_argument("Invalid YACE evaluator array or atom count.");
    if (edge_offsets[0] != 0)
      throw std::invalid_argument("CSR edge offsets must start at zero.");
    for (int center = 0; center < atom_count; ++center)
      if (edge_offsets[center + 1] < edge_offsets[center])
        throw std::invalid_argument("CSR edge offsets must be nondecreasing.");
    if (edge_offsets[atom_count] &&
        (!edge_neighbor_species || !edge_vectors || !edge_gradients))
      throw std::invalid_argument("Nonempty edge list requires species, vectors, and gradients.");
    static_cast<OrdinaryHandle *>(handle)->evaluate(
        atom_count, central_species, edge_offsets, edge_neighbor_species,
        edge_vectors, atomic_energies, edge_gradients);
    return 0;
  } catch (const std::exception &exception) {
    error_text(error, error_capacity, exception.what());
  } catch (...) {
    error_text(error, error_capacity, "Unknown native YACE-evaluation error.");
  }
  return -1;
}

void *ye3t_tagged_open(const char *path, int automatic, char *error, std::size_t error_capacity) noexcept {
  try {
    if (!path || !*path) throw std::invalid_argument("Tagged model path is empty.");
    return new Handle(path, automatic != 0);
  } catch (const std::exception &exception) {
    error_text(error, error_capacity, exception.what());
  } catch (...) {
    error_text(error, error_capacity, "Unknown native tagged-model load error.");
  }
  return nullptr;
}

void ye3t_tagged_close(void *handle) noexcept { delete static_cast<Handle *>(handle); }

const char *ye3t_tagged_selected_policy(void *handle) noexcept {
  if (!handle) return "invalid";
  return static_cast<Handle *>(handle)->evaluator.selected_evaluator_name();
}

int ye3t_tagged_evaluate(void *handle, int atom_count, const int *central_species,
                         const std::size_t *edge_offsets, const int *edge_neighbor_species,
                         const double *edge_vectors, double *atomic_energies,
                         double *edge_gradients, char *error,
                         std::size_t error_capacity) noexcept {
  try {
    if (!handle || atom_count < 0 || !central_species || !edge_offsets || !atomic_energies)
      throw std::invalid_argument("Invalid tagged evaluator array or atom count.");
    if (edge_offsets[0] != 0) throw std::invalid_argument("CSR edge offsets must start at zero.");
    for (int center = 0; center < atom_count; ++center)
      if (edge_offsets[center + 1] < edge_offsets[center])
        throw std::invalid_argument("CSR edge offsets must be nondecreasing.");
    if (edge_offsets[atom_count] && (!edge_neighbor_species || !edge_vectors || !edge_gradients))
      throw std::invalid_argument("Nonempty edge list requires species, vectors, and gradients.");
    static_cast<Handle *>(handle)->evaluate(
        atom_count, central_species, edge_offsets, edge_neighbor_species,
        edge_vectors, atomic_energies, edge_gradients);
    return 0;
  } catch (const std::exception &exception) {
    error_text(error, error_capacity, exception.what());
  } catch (...) {
    error_text(error, error_capacity, "Unknown native tagged-evaluation error.");
  }
  return -1;
}

int ye3t_tagged_evaluate_with_features(
    void *handle, int atom_count, const int *central_species,
    const std::size_t *edge_offsets, const int *edge_neighbor_species,
    const double *edge_vectors, double *atomic_energies, double *edge_gradients,
    double *atomic_features, char *error, std::size_t error_capacity) noexcept {
  try {
    if (!handle || atom_count < 0 || !central_species || !edge_offsets ||
        !atomic_energies || !atomic_features)
      throw std::invalid_argument("Invalid tagged feature array or atom count.");
    if (edge_offsets[0] != 0)
      throw std::invalid_argument("CSR edge offsets must start at zero.");
    for (int center = 0; center < atom_count; ++center)
      if (edge_offsets[center + 1] < edge_offsets[center])
        throw std::invalid_argument("CSR edge offsets must be nondecreasing.");
    if (edge_offsets[atom_count] &&
        (!edge_neighbor_species || !edge_vectors || !edge_gradients))
      throw std::invalid_argument("Nonempty edge list requires species, vectors, and gradients.");
    static_cast<Handle *>(handle)->evaluate(
        atom_count, central_species, edge_offsets, edge_neighbor_species,
        edge_vectors, atomic_energies, edge_gradients, atomic_features);
    return 0;
  } catch (const std::exception &exception) {
    error_text(error, error_capacity, exception.what());
  } catch (...) {
    error_text(error, error_capacity, "Unknown native tagged-feature error.");
  }
  return -1;
}

}  // extern "C"
