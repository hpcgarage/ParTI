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
