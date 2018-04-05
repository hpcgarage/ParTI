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
file = '/nethome/jli458/BIGTENSORS/nips-4d_init.tns';
sz = [2482 2862 14036 17];
nmodes = length(sz);

fprintf('%s\n', file);

A = load(file);
subs = A(:, 1:nmodes);
vals = A(:, nmodes+1);
spA = sptensor(subs, vals, sz);

B = load(file);
subs = B(:, 1:nmodes);
vals = B(:, nmodes+1);
spB = sptensor(subs, vals, sz);


ts = tic;
for niter = 1:5
  res = times(spA, spB);
end
time = toc(ts);
fprintf('time: %f sec\n', time/5);
