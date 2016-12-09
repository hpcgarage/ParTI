ifile = '../../tensors/3d_3_8.tns';
ofile = './dump_3d_3_8.tns';

tns = sptLoadSparseTensor(1, ifile);
sptDumpSparseTensor(tns, 1, ofile);
sptFreeSparseTensor(tns);
