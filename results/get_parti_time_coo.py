#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/ParTI-dev/timing_parti/coo/'
# s3tsrs = ['choa100k', 'choa200k', 'choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
modes = ['0', '1', '2']
sb = 7
sc = 12
r = 16
tb = 1

# input parameters
tk = sys.argv[1]

out_str = 'parti-coo-tk' + str(tk) + '.out'
fo = open(out_str, 'w')

for tsr in s3tsrs:
	for m in modes:

		## sequential coo
		# input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-seq.txt'

		## omp coo
		input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-t' + str(tk) + '.txt'
		# print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 4):
				continue;
			elif(line_array[3] == 'MTTKRP]:'):
				time_num = line_array[4]
				fo.write(time_num+'\n')
		fi.close()

fo.close()





