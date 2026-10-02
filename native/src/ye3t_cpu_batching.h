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

#ifndef LMP_YE3T_CPU_BATCHING_H
#define LMP_YE3T_CPU_BATCHING_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace YE3T_LAMMPS {

// Stable buckets within ONE caller-owned spatial chunk. This indexes readout
// rows only: atom arrays, neighbor lists, edge order, and ghost ownership do not
// change. Workspace is retained by the evaluator, not shared between threads.
class CPUSpeciesBatches {
 public:
  void build(int atom_count, int species_count, const int *species)
  {
    if (atom_count < 0 || species_count < 0 || (atom_count > 0 && species == nullptr))
      throw std::invalid_argument("invalid CPU species-batch dimensions");
    offsets.assign(static_cast<std::size_t>(species_count) + 1, 0);
    for (int atom = 0; atom < atom_count; ++atom) {
      if (species[atom] < 0 || species[atom] >= species_count)
        throw std::invalid_argument("CPU batch central species is out of range");
      ++offsets[static_cast<std::size_t>(species[atom]) + 1];
    }
    for (int species_index = 0; species_index < species_count; ++species_index)
      offsets[static_cast<std::size_t>(species_index) + 1] +=
          offsets[static_cast<std::size_t>(species_index)];
    cursors_ = offsets;
    atoms.resize(static_cast<std::size_t>(atom_count));
    for (int atom = 0; atom < atom_count; ++atom)
      atoms[static_cast<std::size_t>(cursors_[species[atom]]++)] = atom;
  }

  double memory_usage() const
  {
    return static_cast<double>((offsets.capacity() + atoms.capacity() + cursors_.capacity()) *
                               sizeof(int));
  }

  std::vector<int> offsets;
  std::vector<int> atoms;

 private:
  std::vector<int> cursors_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_CPU_BATCHING_H
