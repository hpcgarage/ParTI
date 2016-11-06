%{
    This file is part of SpTOL.

    SpTOL is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    SpTOL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with SpTOL.
    If not, see <http://www.gnu.org/licenses/>.
%}

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
