ifile = '../tensors/3d_3_6.tns';
ofile = '/tmp/3d_3_6.tns';

tns = sptLoadSparseTensor(1, ifile);
sptDumpSparseTensor(tns, 1, ofile);
sptFreeSparseTensor(tns);
