% ifile = '/home/jli/Work/ParTI-dev/tensors/3d_3_6.tns'
 ifile = '/nethome/jli458/ParTI-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
%ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell1.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/delicious.tns'
m = 3
R = 16
niters = 5;
tol = 1e-4;

sptKruskalTensor ktensor;

tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes
ndims = tns.ndims
max_ndims = max(ndims);

ts = tic;
for niter = 1:5
sptCpdAls(tns, R, niters, tol, ktensor);
end
time = toc(ts);
fprintf('seq CPD-ALS time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
sptOmpCpdAls(tns, R, niters, tol, ktensor);
end
time = toc(ts);
fprintf('omp CPD-ALS time: %f sec\n', time/5);


ts = tic;
for niter = 1:5
sptCudaCpdAls(tns, R, niters, tol, ktensor);
end
time = toc(ts);
fprintf('cuda CPD-ALS time: %f sec\n', time/5);


clear


