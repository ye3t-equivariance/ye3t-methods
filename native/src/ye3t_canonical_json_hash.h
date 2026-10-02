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

#ifndef LMP_YE3T_CANONICAL_JSON_HASH_H
#define LMP_YE3T_CANONICAL_JSON_HASH_H

#include <string>

namespace YE3T_LAMMPS {

struct CanonicalExecutionPlanHashes {
  std::string coefficient_hash;
  std::string plan_hash;
};

CanonicalExecutionPlanHashes canonical_execution_plan_hashes(const std::string &path);

std::string canonical_json_hash_without_root_member(const std::string &path,
                                                    const std::string &omitted_member);

std::string canonical_json_root_member_value(const std::string &path, const std::string &member);

std::string canonical_json_nested_member_value(const std::string &path,
                                               const std::string &object_member,
                                               const std::string &member);

// Variants for an already extracted canonical JSON object.  These keep
// nested model artifacts hash-verifiable without writing temporary files.
std::string canonical_json_value_hash_without_root_member(const std::string &json_value,
                                                          const std::string &omitted_member);

std::string canonical_json_value_root_member(const std::string &json_value,
                                             const std::string &member);

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_CANONICAL_JSON_HASH_H
