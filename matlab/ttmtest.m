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

function ttmtest()
    g = gpuDevice(1);

    disp 'Read A from testa.tns';
    fp = fopen('testa.tns', 'r');
    X = spTensor.load(1, fp);
    fclose(fp);

    disp 'Read B from testb.tns';
    fp = fopen('testb.tns', 'r');
    U = spTensor.load(1, fp);
    U = U.toMatrix();
    fclose(fp);

    Y = X.timesMatrix(U, 2);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    disp 'Write A*B to testy.tns (CPU kernel)';
    fp = fopen('testy.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);

    Y = X.timesMatrixCuda(U, 2);
    Y.values = gather(Y.values);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    disp 'Write A*B to testcuda.tns (CUDA kernel)';
    fp = fopen('testcuda.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);
end

