% ifile = '/nethome/jli458/ParTI-dev/tensors/3d_3_8.tns'
% ifile = '/nethome/jli458/ParTI-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
ifile = '/nethome/jli458/BIGTENSORS/choa100k.tns'
% ifile = '/nethome/jli458/BIGTENSORS/choa200k.tns'

m = 3;
R = 16;
niters = 5;
tol = 1e-4;


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes;
ndims = tns.ndims



ts = tic;
ktensor = sptNewKruskalTensor(nmodes, ndims, R);
sptCpdAls(tns, R, niters, tol, ktensor);
clear ktensor;
time = toc(ts);
fprintf('seq CPD-ALS time: %f sec\n', time);


ts = tic;
ktensor = sptNewKruskalTensor(nmodes, ndims, R);
sptOmpCpdAls(tns, R, niters, tol, ktensor);
clear ktensor;
time = toc(ts);
fprintf('omp CPD-ALS time: %f sec\n', time);



ts = tic;
ktensor = sptNewKruskalTensor(nmodes, ndims, R);
sptCudaCpdAls(tns, R, niters, tol, ktensor);
clear ktensor;
time = toc(ts);
fprintf('cuda CPD-ALS time: %f sec\n', time);


clear


