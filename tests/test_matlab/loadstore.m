ifile = '../tensors/3d_3_6.tns';
ofile = '/tmp/3d_3_6.tns';

tns = sptLoadSparseTensor(1, ifile);
tns.sptDumpSparseTensor(1, ofile);
sptFreeSparseTensor(tns);
