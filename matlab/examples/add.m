ifile = '../../tensors/3d_3_8.tns';

tns = sptLoadSparseTensor(1, ifile);

ts = tic;
for niter = 1:5
    Z = sptSparseTensorAdd(tns, tns);
    sptFreeSparseTensor(Z);
end
time = toc(ts);
fprintf('seq add time: %f sec\n', time/5);

sptFreeSparseTensor(tns);
