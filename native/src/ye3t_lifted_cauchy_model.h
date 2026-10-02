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

#ifndef LMP_YE3T_LIFTED_CAUCHY_MODEL_H
#define LMP_YE3T_LIFTED_CAUCHY_MODEL_H

#include <cstdint>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

struct LiftedCauchyDirectPolynomial {
  int q = -1;
  std::vector<double> coefficients;
};

struct LiftedCauchyFactorizedRadial {
  int q = -1;
  int x_power = 0;
  int envelope_power = 0;
};

struct LiftedCauchySourceGroup {
  std::string neighbor_species;
  std::string source_family_id;
  int neighbor_species_index = -1;
  int angular = 1;
  int real_component_count = 3;
  int source_dimension = 0;
  std::vector<int> channel_indices;
  std::vector<int> channel_positions;
  std::vector<std::int64_t> q_source_variable_offsets;
  std::vector<LiftedCauchyDirectPolynomial> direct_q_polynomials;
  std::vector<LiftedCauchyFactorizedRadial> factorized_radials;
  std::vector<double> transform_q_from_f;
};

struct LiftedCauchySparsePolynomial {
  std::vector<std::int64_t> factor_offsets;
  std::vector<std::int64_t> factor_indices;
  std::vector<std::int64_t> factor_exponents;
  std::vector<double> monomial_coefficients;
  double offset = 0.0;
  int maximum_tensor_rank = 0;
  std::int64_t maximum_factor_count = 0;
};

struct LiftedCauchyHead {
  std::string central_species;
  int central_species_index = -1;
  LiftedCauchySparsePolynomial polynomial;
};

struct LiftedCauchyModel {
  static LiftedCauchyModel load(const std::string &path);
  double memory_usage() const;

  std::string bundle_root;
  std::string model_path;
  std::string native_runtime_path;
  std::string compiler_artifact_path;
  std::string compiler_binding_path;
  std::string model_self_hash;
  std::string native_self_hash;
  std::string compiler_artifact_self_hash;
  std::string compiler_binding_self_hash;
  std::string composite_artifact_hash;
  std::string source_plan_hash;
  std::string deployment_identity_hash;
  std::string default_source_realization;
  bool composite_compiler_binding = false;
  bool exclude_zero_separation = false;
  double cutoff = 0.0;
  int role_dimension = 0;
  int real_component_count = 0;
  std::int64_t source_variable_count = 0;
  std::vector<std::string> central_species_order;
  std::vector<int> type_map;
  std::vector<LiftedCauchySourceGroup> source_groups;
  std::vector<LiftedCauchyHead> heads;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_LIFTED_CAUCHY_MODEL_H
