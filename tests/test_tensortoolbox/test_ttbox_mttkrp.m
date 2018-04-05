%{
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
%}

%A = load('/home/jli/Work/ParTI/tensors/3D_12031.tns');
%sz = [100 80 60];
file = '/nethome/jli458/choa_20M_init.tns'
sz = [200000 8719 662];
%file = '/mnt/BIGDATA/jli/BIGTENSORS/brainq_init.tns';
%sz = [60 70365 9];
% file = '/home/BIGDATA/Collection/SPLATT/brainq.fixed.tns';
% sz = [60 22870 9];
% file = 'rand_1000_0.0001.tns';
% sz = [1000 1000 1000];
% file = '/mnt/BIGDATA/jli/BIGTENSORS/nell2_init.tns';
% sz = [12092 9184 28818];
%file = '/home/BIGDATA/Collection/SPLATT/nell1.tns';
%sz = [2902330 2143368 25495389];
%file = '/home/BIGDATA/Collection/SPLATT/delicious.tns';
%sz = [532924 17262471 2480308];

fprintf('%s\n', file);


mode = 3;
R = 16;
fprintf('R: %d, mode: %d\n', R, mode);

nmodes = length(sz);

A = load(file);
subs = A(:, 1:nmodes);
vals = A(:, nmodes+1);
spA = sptensor(subs, vals, sz);

U = cell(nmodes);
for i = 1:nmodes
  U{i} = rand(sz(i), R);
end


ts = tic;
for niter = 1:5
  res = mttkrp(spA, U, mode);
end
time = toc(ts);
fprintf('time: %f sec\n', time/5);
