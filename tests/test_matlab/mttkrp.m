% ifile = '/home/jli/Work/ParTI-dev/tensors/3d_3_6.tns'
% ifile = '/home/jli/Work/ParTI-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell1.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/delicious.tns'
m = 3
R = 16


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes
ndims = tns.ndims
max_ndims = max(ndims);

% U = cell(1,nmodes+1);
for i = 1:nmodes
	U(i) = sptNewMatrix(ndims(i), R);
	sptConstantMatrix(U(i), 1);
end
U(nmodes+1) = sptNewMatrix(max_ndims, R);
sptConstantMatrix(U(nmodes+1), 0);

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


ts = tic;
for niter = 1:5
sptOmpMTTKRP(tns, U, mats_order, m, scratch);
end
time = toc(ts);
fprintf('omp MTTKRP time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
sptCudaMTTKRP(tns, U, mats_order, m, scratch);
end
time = toc(ts);
fprintf('cuda MTTKRP time: %f sec\n', time/5);


clear


