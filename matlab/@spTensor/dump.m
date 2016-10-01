function nwritten = dump(tsr, start_index, fp)
    fprintf(fp, '%u\n', tsr.nmodes);
    fprintf(fp, '%u\t', tsr.ndims);
    fprintf(fp, '\n');
    for i = 1:tsr.nnz
        fprintf(fp, '%u\t', tsr.inds(:, i) + start_index - 1);
        fprintf(fp, '%f\n', tsr.values(i));
    end
    nwritten = tsr.nnz;
end
