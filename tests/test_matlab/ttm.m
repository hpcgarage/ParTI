ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6.tns'
m = 1
R = 4


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
fprintf('Hadamard product time: %f sec\n', time/5);