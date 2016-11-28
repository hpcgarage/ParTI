ifile = '/home/jli/Work/SpTOL-dev/tensors/3d_3_6.tns'
m = 1
R = 4


tns = sptLoadSparseTensor(1, ifile);
nmodes= tns.nmodes
ndims = tns.ndims
max_ndims = max(ndims);

U = cell(nmodes+1);
for i = 1:nmodes
	U{i} = sptNewMatrix(ndims(i), R);
	sptConstantMatrix(U{i}, 1);
end
U{nmodes+1} = sptNewMatrix(max_ndims, R);
sptConstantMatrix(U{nmodes+1}, 0);

mats_order = sptNewVector(nmodes-1, nmodes-1);
mats_order_data = [];
j = 1;
for i=nmodes:-1:1
  if i ~= m
      mats_order_data(j) = i;
      j = j + 1;
  end
end
mats_order.setdata(mats_order_data);

scratch = sptNewVector(R, R);
sptConstantVector(scratch, 0);

% sptMTTKRP(tns, U, mats_order, m, scratch);
