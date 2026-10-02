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

#ifndef LMP_YE3T_TAGGED_CAUCHY_READOUT_PLAN_H
#define LMP_YE3T_TAGGED_CAUCHY_READOUT_PLAN_H

#include "ye3t_tagged_cauchy_model.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace YE3T_LAMMPS {

// Load-time lowering shared by the CPU and Kokkos fixed-readout evaluators.
// V2 monomials are commutative, so their factor multisets may be sorted and
// identical fitted terms coalesced. V3 retains compiler-emitted term order.
struct TaggedCauchyReadoutPlan {
  int density_key_count = 0;
  int maximum_term_p = 0;
  int raw_term_count = 0;

  std::vector<std::int64_t> species_term_offsets;
  std::vector<int> term_p;
  std::vector<double> term_coefficient;
  std::vector<std::int64_t> term_density_offsets;
  std::vector<int> term_density_flat;
  std::vector<std::int64_t> term_moment_offsets;
  std::vector<int> term_moment_flat;

  std::vector<std::int64_t> species_readout_node_offsets;
  std::vector<int> readout_node_parent;
  std::vector<int> readout_node_variable;
  std::vector<int> term_product_node;

  std::vector<std::int64_t> species_live_moment_offsets;
  std::vector<int> species_live_moments;
  std::vector<int> species_live_moment_nodes;
  std::vector<std::int64_t> species_moment_node_offsets;
  std::vector<int> moment_node_parent;
  std::vector<int> moment_node_factor;
  std::vector<std::int64_t> moment_node_terminal_offsets;
  std::vector<int> moment_node_terminal_moments;
};

inline TaggedCauchyReadoutPlan compile_tagged_cauchy_readout(const TaggedCauchyModel &model)
{
  struct CompiledTerm {
    int p = 0;
    std::vector<int> density;
    std::vector<int> moment;
    double coefficient = 0.0;
  };
  using TermKey = std::tuple<int, std::vector<int>, std::vector<int>>;

  TaggedCauchyReadoutPlan plan;
  plan.density_key_count = static_cast<int>(model.real_density_keys.size());
  plan.raw_term_count = static_cast<int>(model.terms.size());
  plan.species_term_offsets.push_back(0);
  plan.species_live_moment_offsets.push_back(0);
  plan.species_moment_node_offsets.push_back(0);
  plan.species_readout_node_offsets.push_back(0);
  plan.term_density_offsets.push_back(0);
  plan.term_moment_offsets.push_back(0);
  std::vector<std::vector<int>> moment_node_terminals;

  for (std::size_t species = 0; species < model.beta.size(); ++species) {
    std::vector<CompiledTerm> compiled;
    std::map<TermKey, std::size_t> positions;
    for (const TaggedCauchyTerm &term : model.terms) {
      const double coefficient =
          model.beta[species][static_cast<std::size_t>(term.feature_index)] * term.coefficient;
      if (coefficient == 0.0) continue;

      CompiledTerm candidate;
      candidate.p = term.p;
      candidate.density = term.density_factor_indices;
      candidate.moment = term.moment_indices;
      candidate.coefficient = coefficient;
      if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2) {
        std::sort(candidate.density.begin(), candidate.density.end());
        std::sort(candidate.moment.begin(), candidate.moment.end());
        const TermKey key(candidate.p, candidate.density, candidate.moment);
        const auto found = positions.find(key);
        if (found != positions.end()) {
          compiled[found->second].coefficient += coefficient;
          continue;
        }
        positions.emplace(key, compiled.size());
      }
      compiled.push_back(std::move(candidate));
    }

    std::vector<bool> live_moments(model.real_moment_keys.size(), false);
    std::map<std::pair<int, int>, int> readout_nodes;
    for (const CompiledTerm &term : compiled) {
      if (term.coefficient == 0.0) continue;
      plan.maximum_term_p = std::max(plan.maximum_term_p, term.p);
      plan.term_p.push_back(term.p);
      plan.term_coefficient.push_back(term.coefficient);
      plan.term_density_flat.insert(plan.term_density_flat.end(), term.density.begin(),
                                    term.density.end());
      plan.term_density_offsets.push_back(static_cast<std::int64_t>(plan.term_density_flat.size()));
      plan.term_moment_flat.insert(plan.term_moment_flat.end(), term.moment.begin(),
                                   term.moment.end());
      plan.term_moment_offsets.push_back(static_cast<std::int64_t>(plan.term_moment_flat.size()));
      for (int index : term.moment) live_moments[static_cast<std::size_t>(index)] = true;

      int product_node = -1;
      if (model.deployment_kind == TaggedCauchyDeploymentKind::LegacyMomentV2) {
        std::vector<int> variables;
        variables.reserve(term.density.size() + term.moment.size());
        variables.insert(variables.end(), term.density.begin(), term.density.end());
        for (int index : term.moment) variables.push_back(plan.density_key_count + index);
        std::sort(variables.begin(), variables.end());
        for (int variable : variables) {
          const std::pair<int, int> key(product_node, variable);
          const auto found = readout_nodes.find(key);
          if (found != readout_nodes.end()) {
            product_node = found->second;
          } else {
            const int node = static_cast<int>(plan.readout_node_parent.size());
            plan.readout_node_parent.push_back(product_node);
            plan.readout_node_variable.push_back(variable);
            readout_nodes.emplace(key, node);
            product_node = node;
          }
        }
      }
      plan.term_product_node.push_back(product_node);
    }
    plan.species_term_offsets.push_back(static_cast<std::int64_t>(plan.term_coefficient.size()));
    plan.species_readout_node_offsets.push_back(
        static_cast<std::int64_t>(plan.readout_node_parent.size()));

    std::map<std::pair<int, int>, int> moment_nodes;
    for (std::size_t moment = 0; moment < live_moments.size(); ++moment) {
      if (!live_moments[moment]) continue;
      plan.species_live_moments.push_back(static_cast<int>(moment));
      std::vector<int> factors = model.real_moment_flat_indices[moment];
      std::sort(factors.begin(), factors.end());
      int product_node = -1;
      for (int factor : factors) {
        const std::pair<int, int> key(product_node, factor);
        const auto found = moment_nodes.find(key);
        if (found != moment_nodes.end()) {
          product_node = found->second;
        } else {
          const int node = static_cast<int>(plan.moment_node_parent.size());
          plan.moment_node_parent.push_back(product_node);
          plan.moment_node_factor.push_back(factor);
          moment_node_terminals.emplace_back();
          moment_nodes.emplace(key, node);
          product_node = node;
        }
      }
      plan.species_live_moment_nodes.push_back(product_node);
      if (product_node >= 0)
        moment_node_terminals[static_cast<std::size_t>(product_node)].push_back(
            static_cast<int>(moment));
    }
    plan.species_live_moment_offsets.push_back(
        static_cast<std::int64_t>(plan.species_live_moments.size()));
    plan.species_moment_node_offsets.push_back(
        static_cast<std::int64_t>(plan.moment_node_parent.size()));
  }
  plan.moment_node_terminal_offsets.reserve(moment_node_terminals.size() + 1);
  plan.moment_node_terminal_offsets.push_back(0);
  for (const std::vector<int> &terminals : moment_node_terminals) {
    plan.moment_node_terminal_moments.insert(plan.moment_node_terminal_moments.end(),
                                             terminals.begin(), terminals.end());
    plan.moment_node_terminal_offsets.push_back(
        static_cast<std::int64_t>(plan.moment_node_terminal_moments.size()));
  }
  return plan;
}

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_TAGGED_CAUCHY_READOUT_PLAN_H
