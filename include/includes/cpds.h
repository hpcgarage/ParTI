/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PARTI_CPDS_H
#define PARTI_CPDS_H


/**
 * CP-ALS
 */
int sptCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptKruskalTensor * ktensor);
int sptOmpCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  sptKruskalTensor * ktensor);
int sptCudaCpdAls(
  sptSparseTensor const * const spten,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptKruskalTensor * ktensor);
int sptCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  sptRankKruskalTensor * ktensor);
int sptOmpCpdAlsHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptIndex const rank,
  sptIndex const niters,
  double const tol,
  const int tk,
  sptRankKruskalTensor * ktensor);

#endif