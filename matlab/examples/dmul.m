ifile = '../../tensors/3d_3_8.tns';

tns = sptLoadSparseTensor(1, ifile);

ts = tic;
for niter = 1:5
    Z = sptSparseTensorDotMulEq(tns, tns);
    sptFreeSparseTensor(Z);
end
time = toc(ts);
fprintf('seq dmul_eq time: %f sec\n', time/5);

ts = tic;
for niter = 1:5
    Z = sptOmpSparseTensorDotMulEq(tns, tns);
    sptFreeSparseTensor(Z);
end
time = toc(ts);
fprintf('omp dmul_eq time: %f sec\n', time/5);

ts = tic;
for niter = 1:5
    Z = sptCudaSparseTensorDotMulEq(tns, tns);
    sptFreeSparseTensor(Z);
end
time = toc(ts);
fprintf('cuda dmul_eq time: %f sec\n', time/5);

sptFreeSparseTensor(tns);
