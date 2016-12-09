ifile = '../../tensors/3d_3_8.tns';
m = 3
R = 16


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes
ndims = tns.ndims

U_data = rand(ndims(m), R);
U = sptNewMatrix(ndims(m), R);
U.setvalues(U_data);

ts = tic;
for niter = 1:5
Y = sptSparseTensorMulMatrix(tns, U, m);
end
time = toc(ts);
fprintf('seq TTM time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
Y = sptOmpSparseTensorMulMatrix(tns, U, m);
end
time = toc(ts);
fprintf('omp TTM time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
Y = sptCudaSparseTensorMulMatrix(tns, U, m);
end
time = toc(ts);
fprintf('cuda TTM time: %f sec\n', time/5);

clear

