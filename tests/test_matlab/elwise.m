ifile = '../../tensors/3d_3_8.tns';
ofile = '/dev/stdout';

disp 'Load sparse tensor';
tns = sptLoadSparseTensor(1, ifile);

disp 'Print sparse tensor';
sptDumpSparseTensor(tns, 1, ofile);

disp 'Add sparse tensor by itself (double)';
tns2 = tns + tns;
sptDumpSparseTensor(tns2, 1, ofile);
sptFreeSparseTensor(tns2);

disp 'Minus sparse tensor by itself (zero)';
tns0 = tns - tns;
sptDumpSparseTensor(tns0, 1, ofile);
sptFreeSparseTensor(tns0);

disp 'Multiply by itself (square)';
tnss = tns .* tns;
sptDumpSparseTensor(tnss, 1, ofile);
sptFreeSparseTensor(tnss);

disp 'Divide by itself (one)';
tns1 = tns ./ tns;
sptDumpSparseTensor(tns1, 1, ofile);
sptFreeSparseTensor(tns1);

disp 'Multiply 2x';
tns2x = tns * 2;
sptDumpSparseTensor(tns2x, 1, ofile);
sptFreeSparseTensor(tns2x);

disp 'Divide /2';
tns2d = tns / 2;
sptDumpSparseTensor(tns2d, 1, ofile);
sptFreeSparseTensor(tns2d);

disp 'Accelerated multiplication (OpenMP)';
tnsmp = sptOmpSparseTensorDotMulEq(tns, tns);
sptDumpSparseTensor(tnsmp, 1, ofile);
sptFreeSparseTensor(tnsmp);

disp 'Test finished';
sptFreeSparseTensor(tns);
