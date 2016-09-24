function nwritten = dump(obj, start_index, fp)
    fprintf(fp, '%u\n', obj.nmodes);
    fprintf(fp, '%u ', obj.ndims);
    fprintf(fp, '\n');
    for i = 1:obj.nnz
        fprintf(fp, '%u ', obj.inds(:, i) + start_index - 1);
        fprintf(fp, '%f\n', obj.values(i, 1));
    end
    nwritten = obj.nnz;
end
