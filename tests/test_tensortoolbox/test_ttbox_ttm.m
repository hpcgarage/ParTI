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
%file = '/home/BIGDATA/Collection/SPLATT/brainq.tns';
%sz = [60 70365 9];
% file = '/home/BIGDATA/Collection/SPLATT/brainq.fixed.tns';
% sz = [60 22870 9];
% file = 'rand_1000_0.0001.tns';
% sz = [1000 1000 1000];
%file = '/home/BIGDATA/Collection/SPLATT/nell2.tns';
%sz = [12092 9184 28818];
%file = '/home/BIGDATA/Collection/SPLATT/nell1.tns';
%sz = [2902330 2143368 25495389];
%file = '/home/BIGDATA/Collection/SPLATT/delicious.tns';
%sz = [532924 17262471 2480308];
file = '/nethome/jli458/BIGTENSORS/nips-4d_init.tns';
sz = [2482 2862 14036 17];
nmodes = length(sz);

mode = 1;
R = 16;
fprintf('R: %d, mode: %d\n', R, mode);

A = load(file);
subs = A(:, 1:nmodes);
vals = A(:, nmodes+1);
spA = sptensor(subs, vals, sz);

U = rand(R, sz(mode));

res = ttm_timing(spA, U, mode);

niters = 1;
ts = tic;
for i = 1:niters
  res = ttm_timing(spA, U, mode);
end
time = toc(ts);
fprintf('time: %f sec\n', time/niters);
