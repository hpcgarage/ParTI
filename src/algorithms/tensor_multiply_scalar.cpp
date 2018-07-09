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

#include <ParTI/sptensor.hpp>
#include <cstdio>
#include <memory>
#include <ParTI/error.hpp>

namespace pti {

	SparseTensor tensor_multiply_scalar(SparseTensor& X, Scalar scalar){
//		SparseTensor Y = X;
		size_t nmodes = X.nmodes;
		std::unique_ptr<size_t[]> Y_shape(new size_t [nmodes]);
		for(size_t m = 0; m < nmodes; ++m) {
			Y_shape[m] = X.shape(cpu)[m];
		}
		bool const* X_is_dense = X.is_dense(cpu);
		
		
		
		
		
		
		std::unique_ptr<bool[]> Y_is_dense(new bool [nmodes]);
		for(size_t m = 0; m < nmodes; ++m) {
			Y_is_dense[m] = X_is_dense[m];
		}
		SparseTensor Y(nmodes, Y_shape.get(), Y_is_dense.get());
		Y.num_chunks = X.num_chunks;
		for(size_t m = 0; m < nmodes; ++m){
			Y.indices[m].allocate(cpu, X.num_chunks);
		}
		for(size_t m = 0; m < nmodes; ++m){
			size_t* X_indices_m = X.indices[m](cpu);
			size_t* Y_indices_m = Y.indices[m](cpu);
			for(size_t i = 0; i < Y.num_chunks; ++i){
				Y_indices_m[i] = X_indices_m[i];
			}
		}
		Y.values.allocate(cpu, X.num_chunks);
		
		Scalar* X_values = X.values(cpu);
		Scalar* Y_values = Y.values(cpu);
		for (size_t i = 0; i < Y.num_chunks; ++i){
			Y_values[i *  Y.chunk_size] = scalar * X_values[i * X.chunk_size];
		}

		return Y;
	
	}

}
