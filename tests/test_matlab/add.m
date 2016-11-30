% ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6.tns'
% ifile = '/home/jli/Work/SpTOL-dev/tensors/3D_12031.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell1.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/delicious.tns'
ifile = '../tensors/3d_3_6.tns';

tns = sptLoadSparseTensor(1, ifile);

ts = tic;
for niter = 1:5
    Z = sptSparseTensorAdd(tns, tns);
    sptFreeSparseTensor(Z);
end
time = toc(ts);
fprintf('seq add time: %f sec\n', time/5);

sptFreeSparseTensor(tns);
