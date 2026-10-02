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

#ifndef LMP_YE3T_SHIFTED_JACOBI_H
#define LMP_YE3T_SHIFTED_JACOBI_H

namespace YE3T_LAMMPS {

#ifdef KOKKOS_INLINE_FUNCTION
#define YE3T_JACOBI_INLINE KOKKOS_INLINE_FUNCTION
#else
#define YE3T_JACOBI_INLINE inline
#endif

// Evaluate J_q(x)=P_q^(4,2*l+2)(2*x-1) and dJ_q/dx.  The exact expanded
// coefficients remain part of the compiler/source artifact; this recurrence
// is only the stable binary64 realization selected by source-plan V2.
YE3T_JACOBI_INLINE void shifted_jacobi_ladder_with_derivative(int maximum_degree, int angular_l,
                                                              double x, double *values,
                                                              double *derivatives)
{
  const double alpha = 4.0;
  const double beta = 2.0 * static_cast<double>(angular_l) + 2.0;
  values[0] = 1.0;
  derivatives[0] = 0.0;
  if (maximum_degree == 0) return;

  values[1] = (alpha + beta + 2.0) * x - (beta + 1.0);
  derivatives[1] = alpha + beta + 2.0;
  const double shifted = 2.0 * x - 1.0;
  for (int degree = 1; degree < maximum_degree; ++degree) {
    const double total = 2.0 * static_cast<double>(degree) + alpha + beta;
    const double a_n =
        (total + 1.0) * (total + 2.0) / (2.0 * (degree + 1.0) * (degree + alpha + beta + 1.0));
    const double b_n = (alpha * alpha - beta * beta) * (total + 1.0) /
        (2.0 * (degree + 1.0) * (degree + alpha + beta + 1.0) * total);
    const double c_n = (degree + alpha) * (degree + beta) * (total + 2.0) /
        ((degree + 1.0) * (degree + alpha + beta + 1.0) * total);
    const double multiplier = a_n * shifted + b_n;
    values[degree + 1] = multiplier * values[degree] - c_n * values[degree - 1];
    derivatives[degree + 1] = 2.0 * a_n * values[degree] + multiplier * derivatives[degree] -
        c_n * derivatives[degree - 1];
  }
}

YE3T_JACOBI_INLINE void shifted_jacobi_value_with_derivative(int degree, int angular_l, double x,
                                                             double &value, double &derivative)
{
  if (degree == 0) {
    value = 1.0;
    derivative = 0.0;
    return;
  }
  const double alpha = 4.0;
  const double beta = 2.0 * static_cast<double>(angular_l) + 2.0;
  double previous_value = 1.0;
  double previous_derivative = 0.0;
  value = (alpha + beta + 2.0) * x - (beta + 1.0);
  derivative = alpha + beta + 2.0;
  const double shifted = 2.0 * x - 1.0;
  for (int n = 1; n < degree; ++n) {
    const double total = 2.0 * static_cast<double>(n) + alpha + beta;
    const double a_n = (total + 1.0) * (total + 2.0) / (2.0 * (n + 1.0) * (n + alpha + beta + 1.0));
    const double b_n = (alpha * alpha - beta * beta) * (total + 1.0) /
        (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * total);
    const double c_n =
        (n + alpha) * (n + beta) * (total + 2.0) / ((n + 1.0) * (n + alpha + beta + 1.0) * total);
    const double multiplier = a_n * shifted + b_n;
    const double next_value = multiplier * value - c_n * previous_value;
    const double next_derivative =
        2.0 * a_n * value + multiplier * derivative - c_n * previous_derivative;
    previous_value = value;
    previous_derivative = derivative;
    value = next_value;
    derivative = next_derivative;
  }
}

#undef YE3T_JACOBI_INLINE

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_SHIFTED_JACOBI_H
