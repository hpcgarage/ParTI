% ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6.tns'
% ifile = '/home/jli/Work/SpTOL-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell1.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/delicious.tns'

X = sptLoadSparseTensor(1, ifile);
nmodes= X.nmodes
ndims = X.ndims

Y = sptLoadSparseTensor(1, ifile);
nmodes= Y.nmodes
ndims = Y.ndims

ts = tic;
for niter = 1:5
Z = sptSparseTensorDotMulEq(X, Y);
end
time = toc(ts);
fprintf('seq Hadamard product time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
Z = sptOmpSparseTensorDotMulEq(X, Y);
end
time = toc(ts);
fprintf('omp Hadamard product time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
Z = sptCudaSparseTensorDotMulEq(X, Y);
end
time = toc(ts);
fprintf('cuda Hadamard product time: %f sec\n', time/5);


clear

