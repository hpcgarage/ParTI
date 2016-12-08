% ifile = '/nethome/jli458/ParTI-dev/tensors/3d_3_8.tns'
% ifile = '/nethome/jli458/ParTI-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
% ifile = '/nethome/jli458/BIGTENSORS/choa100k.tns';
ifile = '/nethome/jli458/BIGTENSORS/choa200k.tns';
% ifile = '/nethome/jli458/BIGTENSORS/choa700k.tns'

m = 3;
R = 8;
niters = 5;
tol = 1e-4;


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes;
ndims = tns.ndims;
nnz = tns.nnz;

fprintf('file: %s\n', ifile);
fprintf('ndims: [%d, %d, %d]\n', ndims(1), ndims(2), ndims(3));
fprintf('nnz: %d\n', nnz);
fprintf('R: %d\n', R);


fprintf('\n===== cuda GPU =====\n');
ts = tic;
ktensor = sptNewKruskalTensor(nmodes, ndims, R);
sptCudaCpdAls(tns, R, niters, tol, ktensor);
clear ktensor;
time = toc(ts);
% fprintf('cuda CPD-ALS time: %f sec\n', time);


fprintf('\n===== omp CPU =====\n');
ts = tic;
ktensor = sptNewKruskalTensor(nmodes, ndims, R);
sptOmpCpdAls(tns, R, niters, tol, ktensor);
clear ktensor;
time = toc(ts);
% fprintf('omp CPD-ALS time: %f sec\n', time);



% fprintf('\n===== seq CPU =====\n');
% ts = tic;
% ktensor = sptNewKruskalTensor(nmodes, ndims, R);
% sptCpdAls(tns, R, niters, tol, ktensor);
% clear ktensor;
% time = toc(ts);
% fprintf('seq CPD-ALS time: %f sec\n', time);



clear


