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

#ifndef PTI_ERRCODE_INCLUDED
#define PTI_ERRCODE_INCLUDED

namespace pti {

enum ErrCode {
    ERR_NO_ERROR       = 0,
    ERR_UNKNOWN        = 1,
    ERR_BUILD_CONFIG   = 2,
    ERR_SHAPE_MISMATCH = 3,
    ERR_VALUE_ERROR    = 4,
    ERR_ZERO_DIVISION  = 5,
    ERR_BLAS_LIBRARY   = 6,
    ERR_LAPACK_LIBRARY = 7,
    ERR_CUDA_LIBRARY   = 8,
};

}

#endif
