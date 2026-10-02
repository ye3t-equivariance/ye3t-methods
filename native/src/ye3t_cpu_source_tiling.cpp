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

#include "ye3t_cpu_source_tiling.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace YE3T_LAMMPS {
namespace {
  std::size_t checked_add(std::size_t a, std::size_t b)
  {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::overflow_error("YE3T CPU source workspace size overflows");
    return a + b;
  }
  std::size_t checked_mul(std::size_t a, std::size_t b)
  {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
      throw std::overflow_error("YE3T CPU source workspace size overflows");
    return a * b;
  }
}    // namespace

std::size_t cpu_reported_l2_cache_bytes() noexcept
{
#if defined(_SC_LEVEL2_CACHE_SIZE)
  // This optional libc query reports capacity, not private/available cache or
  // the topology of MPI ranks sharing it. Preserve errno in this best-effort
  // hardware probe. Unsupported/nonpositive results use the portable fallback.
  const int saved_errno = errno;
  const long bytes = sysconf(_SC_LEVEL2_CACHE_SIZE);
  errno = saved_errno;
  return bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
#else
  return 0;
#endif
}

CPUSourceTilePolicy make_cpu_source_tile_policy(int radial_base_width, int contracted_width,
                                                int angular_width, int bond_count,
                                                std::size_t angular_plan_bytes,
                                                std::size_t reported_l2_bytes)
{
  if (radial_base_width < 0 || contracted_width < 0 || angular_width <= 0 || bond_count <= 0)
    throw std::invalid_argument("invalid YE3T CPU source workspace dimensions");

  CPUSourceTilePolicy result;
  // Conservative, bounded heuristic, not a measured optimum: leave headroom
  // for spline tables, source rows and other processes/threads sharing caches.
  // Correctness must never depend on whether cache detection is available.
  constexpr std::size_t fallback = 256 * 1024;
  constexpr std::size_t minimum = 64 * 1024;
  constexpr std::size_t maximum = 1024 * 1024;
  result.target_bytes =
      reported_l2_bytes ? std::min(maximum, std::max(minimum, reported_l2_bytes / 2)) : fallback;

  const auto rb = static_cast<std::size_t>(radial_base_width);
  const auto rc = static_cast<std::size_t>(contracted_width);
  const auto h = static_cast<std::size_t>(angular_width);
  // Radius + unit direction, radial values/derivatives, and one complex angular
  // value plus three complex Cartesian derivatives per nonnegative-m component.
  result.bytes_per_edge = checked_add(
      checked_add(4 * sizeof(double), checked_mul(2 * sizeof(double), checked_add(rb, rc))),
      checked_mul(4 * sizeof(std::complex<double>), h));
  result.fixed_bytes = angular_plan_bytes;
  if (bond_count > 1) {
    // Account for the padded radial tables AND retained bond-gather scratch.
    result.bytes_per_edge =
        checked_add(result.bytes_per_edge,
                    checked_add(2 * sizeof(std::int64_t) + sizeof(double),
                                checked_mul(2 * sizeof(double), std::max(rb, rc))));
    result.fixed_bytes = checked_add(
        result.fixed_bytes,
        checked_mul(sizeof(std::int64_t),
                    checked_add(checked_mul(3, static_cast<std::size_t>(bond_count)), 1)));
  }
  const std::size_t available =
      result.target_bytes > result.fixed_bytes ? result.target_bytes - result.fixed_bytes : 0;
  const std::size_t count = std::max<std::size_t>(1, available / result.bytes_per_edge);
  result.edge_capacity = static_cast<int>(
      std::min<std::size_t>(count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  // A single extremely wide edge can exceed the target; always make progress.
  return result;
}

}    // namespace YE3T_LAMMPS
