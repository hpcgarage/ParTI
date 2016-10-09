%A = load('/home/jli/Work/SpTOL/tensors/3D_12031.tns');
%sz = [100 80 60];
% file = '/home/BIGDATA/Collection/SPLATT/brainq.fixed.tns';
% sz = [60 22870 9];
file = 'rand_1000_0.0001.tns';
sz = [1000 1000 1000];
%file = '/home/BIGDATA/Collection/SPLATT/nell2.tns';
%sz = [12092 9184 28818];
%file = '/home/BIGDATA/Collection/SPLATT/nell1.tns';
%sz = [2902330 2143368 25495389];
%file = '/home/BIGDATA/Collection/SPLATT/delicious.tns';
%sz = [532924 17262471 2480308];

mode = 3;
R = 16;
fprintf('R: %d, mode: %d\n', R, mode);

A = load(file);
subs = A(:, 1:3);
vals = A(:, 4);
spA = sptensor(subs, vals, sz);

U = rand(R, sz(mode));

ts = tic;
for niter = 1:5
  res = ttm_timing(spA, U, mode);
end
time = toc(ts);
fprintf('time: %f sec\n', time/5);
