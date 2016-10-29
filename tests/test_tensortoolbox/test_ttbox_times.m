%A = load('/home/jli/Work/SpTOL/tensors/3D_12031.tns');
%sz = [100 80 60];
% file = '/mnt/BIGDATA/jli/BIGTENSORS/brainq_init.tns';
% sz = [60 70365 9];
% file = '/home/BIGDATA/Collection/SPLATT/brainq.fixed.tns';
% sz = [60 22870 9];
% file = 'rand_1000_0.0001.tns';
% sz = [1000 1000 1000];
file = '/mnt/BIGDATA/jli/BIGTENSORS/nell2_init.tns';
sz = [12092 9184 28818];
%file = '/home/BIGDATA/Collection/SPLATT/nell1.tns';
%sz = [2902330 2143368 25495389];
%file = '/home/BIGDATA/Collection/SPLATT/delicious.tns';
%sz = [532924 17262471 2480308];

fprintf('%s\n', file);

A = load(file);
subs = A(:, 1:3);
vals = A(:, 4);
spA = sptensor(subs, vals, sz);

B = load(file);
subs = B(:, 1:3);
vals = B(:, 4);
spB = sptensor(subs, vals, sz);


ts = tic;
for niter = 1:5
  res = times(spA, spB);
end
time = toc(ts);
fprintf('time: %f sec\n', time/5);
