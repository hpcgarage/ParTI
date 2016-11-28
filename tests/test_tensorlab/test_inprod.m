% ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6_init.tns'
% sz = [3 3 3];
% ifile = '/home/jli/Work/SpTOL-dev/tensors/3D_12031_init.tns'
% sz = [100 80 60];
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/brainq_init.tns'
% sz = [60 70365 9];
ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell2_init.tns'
sz = [12092 9184 28818];
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/nell1.tns'
% ifile = '/mnt/BIGDATA/jli/BIGTENSORS/delicious.tns'

A = load(ifile);
nmodes = length(sz);

T.val = A(:, 4);              % The non-zero elements
T.sub = A(:, 1:3);   % Their positions
T.size = sz;
T.sparse = true;                 % The sparse flag

X = fmt(T);
Y = fmt(T);

ts = tic;
for niter = 1:5
Z = inprod(X, Y);
end
time = toc(ts);
fprintf('Tensorlab: Hadamard product (inprod) time: %f sec\n', time/5);


clear

