#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/ParTI-dev/timing_parti/cpd-coo/'
# s3tsrs = ['choa100k', 'choa200k', 'choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
# s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
s3tsrs = ['delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
r = 16

# input parameters
tk = sys.argv[1]

out_str = 'parti-cpd-coo-tk' + str(tk) + '.out'
fo = open(out_str, 'w')


for tsr in s3tsrs:

	## sequential coo
	# input_str = intput_path + tsr + '-r' + str(r) + '-reduce-seq.txt'

	## omp coo
	input_str = intput_path + tsr + '-r' + str(r) + '-t' + str(tk) + '-reduce.txt'
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





