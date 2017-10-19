#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/ParTI-dev/timing_parti/cpd-hicoo/uint-fast8-simd/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_parti/hicoo/uint16/'
# s3tsrs = ['choa100k', 'choa200k', 'choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
# s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
s3tsrs = ['nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
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
out_str = 'parti-cpd-hicoo-uint-fast8-simd-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:

	## omp hicoo
	input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '.txt'

	## sequential hicoo
	# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
	# print(input_str)

	count = 0
	time_num = (float)(0.0)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print line_array
		if(len(line_array) > 2):
			if(line_array[2] == "its"):
				count += 1
				if(len(line_array) == 17):
					time_num += (float)(line_array[7])
				elif(len(line_array) == 18):
					time_num += (float)(line_array[8])
	
	time_num = time_num / count
	print time_num
	fo.write(str(time_num)+'\n')

	fi.close()

fo.close()





