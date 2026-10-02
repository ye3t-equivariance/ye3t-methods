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

#ifndef LMP_YE3T_SHA256_H
#define LMP_YE3T_SHA256_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace YE3T_LAMMPS {

class SHA256Builder {
 public:
  void update(const void *data, std::size_t size);
  void update(const std::string &value);
  std::string finish();

 private:
  void transform();

  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<unsigned char, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t bit_count_ = 0;
  bool finished_ = false;
};

std::string sha256_file(const std::string &path);
std::string sha256_string(const std::string &value);

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_SHA256_H
