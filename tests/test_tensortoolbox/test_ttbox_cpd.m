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
%ndims = [100 80 60];
% file = '/nethome/jli458/BIGTENSORS/choa100k_init.tns';
% ndims = [99960 7170 598]
file = '/nethome/jli458/BIGTENSORS/choa200k_init.tns';
ndims = [199880 8086 656];
% file = '/nethome/jli458/BIGTENSORS/choa700k_init.tns'
% ndims = [712329 9827 767]
%file = '/mnt/BIGDATA/jli/BIGTENSORS/brainq_init.tns';
%ndims = [60 70365 9];
% file = '/home/BIGDATA/Collection/SPLATT/brainq.fixed.tns';
% ndims = [60 22870 9];
% file = 'rand_1000_0.0001.tns';
% ndims = [1000 1000 1000];
% file = '/mnt/BIGDATA/jli/BIGTENSORS/nell2_init.tns';
% ndims = [12092 9184 28818];

R = 16;


nmodes = length(ndims);

A = load(file);
subs = A(:, 1:nmodes);
vals = A(:, nmodes+1);
spA = sptensor(subs, vals, ndims);
nnz = length(spA.vals);


fprintf('file: %s\n', file);
fprintf('ndims: [%d, %d, %d]\n', ndims(1), ndims(2), ndims(3));
fprintf('nnz: %d\n', nnz);
fprintf('R: %d\n', R);


ts = tic;
res = cp_als_changed(spA, R);
time = toc(ts);
% fprintf('time: %f sec\n', time);
