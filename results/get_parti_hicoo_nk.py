#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/ParTI-dev/timing_parti/hicoo/uint-fast8-simd-fulltest/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_parti/hicoo/uint16/'
# s3tsrs = ['choa100k', 'choa200k', 'choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'freebase_music', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
test_tsrs = ['freebase_music', 'freebase_sampled']
modes = ['0', '1', '2']
r = 16
tb = 1

sc = 14
# sc = 16

# input parameters
sb = sys.argv[1]
sk = sys.argv[2]
tk = sys.argv[3]

# out_str = 'parti-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
out_str = 'parti-hicoo-uint-fast8-simd-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
print("output file: " + "\"" + out_str + "\"")
# fo = open(out_str, 'w')

for tsr in s3tsrs:
	for m in modes:

		## omp hicoo
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '.txt'

		## sequential hicoo
		# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
		# print(input_str)

		fi = open(input_str, 'r')
		schr = ''
		for line in fi:
			line_array = line.rstrip().split(" ")

			# if(len(line_array) < 1):
			# 	continue;
			# elif(line_array[0] == 'num_kernel_dim:'):
			# 	nLi_tmp = line_array[1]
			# 	nLi = nLi_tmp.rstrip().split(",")[0]
			# 	# fo.write(time_num+'\n')
			# 	print(nLi)

			# if(len(line_array) < 2):
			# 	continue;
			# elif(line_array[0] == 'SCHEDULE'):
			# 	nLj_tmp = line_array[int(m)+2]
			# 	if(m == '2'):
			# 		nLj = nLj_tmp
			# 	else:
			# 		nLj = nLj_tmp.rstrip().split(",")[0]
			# 	print(nLj)

			if(len(line_array) < 1):
				continue;
			# elif(line_array[0] == 'sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled:'):
			# 	schr = 'N'
			elif(line_array[0] == 'sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce:'):
				schr = 'R'

		fi.close()
		print(schr)

# fo.close()





