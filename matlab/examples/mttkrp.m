ifile = '../../tensors/3d_3_8.tns';
m = 3;
R = 16;


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes;
nnz = tns.nnz;
ndims = tns.ndims;
max_ndims = max(ndims);

for i = 1:nmodes
	U(i) = sptNewMatrix(ndims(i), R);
	sptConstantMatrix(U(i), 1);
end
U(nmodes+1) = sptNewMatrix(max_ndims, R);
sptConstantMatrix(U(nmodes+1), 0);
stride = U(i).stride;

mats_order = sptNewSizeVector(nmodes-1, nmodes-1);
mats_order_data = [];
j = 1;
for i=nmodes:-1:1
  if i ~= m
      mats_order_data(j) = i;
      j = j + 1;
  end
end
mats_order.setdata(uint64(mats_order_data));

scratch = sptNewVector(R, R);
sptConstantVector(scratch, 0);

ts = tic;
for niter = 1:5
sptMTTKRP(tns, U, mats_order, m, scratch);
end
time = toc(ts);
fprintf('seq MTTKRP time: %f sec\n', time/5);
clear scratch;

scratch = sptNewVector(nnz * stride, nnz * stride);
sptConstantVector(scratch, 0);

ts = tic;
for niter = 1:5
sptOmpMTTKRP(tns, U, mats_order, m, scratch);
end
time = toc(ts);
fprintf('omp MTTKRP time: %f sec\n', time/5);
clear scratch;


scratch = sptNewVector(0, 0);
ts = tic;
for niter = 1:5
sptCudaMTTKRP(tns, U, mats_order, m);
end
time = toc(ts);
fprintf('cuda MTTKRP time: %f sec\n', time/5);


clear


