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

function tsr = sort(tsr)
    tsr = quickSortIndex(tsr, 1, tsr.nnz);
    tsr.sortkey = tsr.nmodes;
end

function result = compareIndices(tsr, ind1, ind2)
    for i = 1:tsr.nmodes
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
    result = 0;
end

function tsr = swapValues(tsr, ind1, ind2)
    temp = tsr.inds(:, ind1);
    tsr.inds(:, ind1) = tsr.inds(:, ind2);
    tsr.inds(:, ind2) = temp;

    temp = tsr.values(ind1);
    tsr.values(ind1) = tsr.values(ind2);
    tsr.values(ind2) = temp;
end


function tsr = quickSortIndex(tsr, l, r)
    r = r + 1;
    if r-l < 2
        return;
    end
    p = floor((l+r) / 2);

    i = l;
    j = r-1;
    while true
        while compareIndices(tsr, i, p) < 0
            i = i + 1;
        end
        while compareIndices(tsr, p, j) < 0
            j = j - 1;
        end
        if i >= j
            break;
        end
        tsr = swapValues(tsr, i, j);
        if i == p
            p = j;
        elseif j == p
            p = i;
        end
        i = i + 1;
        j = j - 1;
    end

    tsr = quickSortIndex(tsr, l, i-1);
    tsr = quickSortIndex(tsr, i, r-1);
end
