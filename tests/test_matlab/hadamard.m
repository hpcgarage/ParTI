ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6.tns'
m = 1
R = 4


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
fprintf('Hadamard product time: %f sec\n', time/5);