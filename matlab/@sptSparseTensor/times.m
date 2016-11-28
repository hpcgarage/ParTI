%{
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
%}

function Z = times(X, Y)
    classX = class(X);
    classY = class(Y);
    Xistensor = strcmp(classX, 'sptSparseTensor');
    Yistensor = strcmp(classY, 'sptSparseTensor');
    if Xistensor & Yistensor
        Z = sptSparseTensorDotMul(X, Y);
    elseif Xistensor & ~Yistensor
        Z = sptCopySparseTensor(X);
        sptSparseTensorMulScalar(Z, Y);
    elseif ~Xistensor & Yistensor
        Z = sptCopySparseTensor(Y);
        sptSparseTensorMulScalar(Z, X);
    else
        error('SpTOL:sptSparseTensor:times', 'Input is not sptSparseTensor.');
    end
end
