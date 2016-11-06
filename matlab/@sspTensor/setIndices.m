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

function [dest, fiberidx] = setIndices(dest, ref)
    lastidx = 0;
    if ref.sortkey ~= dest.mode
        ref = ref.sortAtMode(ref, dest.mode);
    end
    fiberidx = zeros(0, 0, 'uint64');
    dest.nnz = 0;
    for i = 1:ref.nnz
        if lastidx == 0 || compareExceptMode(ref, lastidx, i, dest.mode) ~= 0
            dest.inds(:, dest.nnz+1) = ref.inds(:, i);
            dest.inds(dest.mode, dest.nnz+1) = 0;
            lastidx = i;
            dest.nnz = dest.nnz + 1;
            fiberidx(length(fiberidx)+1, 1) = i;
        end
    end
    fiberidx(length(fiberidx)+1, 1) = ref.nnz+1;
    dest.values = zeros(dest.nnz, dest.stride);
end

function result = compareExceptMode(tsr, ind1, ind2, mode)
    for i = 1:tsr.nmodes
        if i ~= mode
            eleind1 = tsr.inds(i, ind1);
            eleind2 = tsr.inds(i, ind2);
            if eleind1 < eleind2
                result = -1;
                return;
            elseif eleind1 > eleind2
                result = 1;
                return;
            end
        end
    end
    result = 0;
end
