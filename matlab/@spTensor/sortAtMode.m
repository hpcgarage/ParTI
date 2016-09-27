function tsr = sortAtMode(tsr, mode)
    tsr = quickSortAtMode(tsr, 1, tsr.nnz, mode);
    tsr.sortkey = mode;
end

function result = compareAtMode(tsr, ind1, ind2, mode)
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
    eleind1 = tsr.inds(mode, ind1);
    eleind2 = tsr.inds(mode, ind2);
    if eleind1 < eleind2
        result = -1;
    elseif eleind1 > eleind2
        result = 1;
    else
        result = 0;
    end
end

function tsr = swapValues(tsr, ind1, ind2)
    temp = tsr.inds(:, ind1);
    tsr.inds(:, ind1) = tsr.inds(:, ind2);
    tsr.inds(:, ind2) = temp;

    temp = tsr.values(ind1);
    tsr.values(ind1) = tsr.values(ind2);
    tsr.values(ind2) = temp;
end


function tsr = quickSortAtMode(tsr, l, r, mode)
    r = r + 1;
    if r-l < 2
        return;
    end
    p = floor((l+r) / 2);

    i = l;
    j = r-1;
    while true
        while compareAtMode(tsr, i, p, mode) < 0
            i = i + 1;
        end
        while compareAtMode(tsr, p, j, mode) < 0
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

    tsr = quickSortAtMode(tsr, l, i-1, mode);
    tsr = quickSortAtMode(tsr, i, r-1, mode);
end
