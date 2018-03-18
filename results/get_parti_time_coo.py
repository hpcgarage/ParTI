#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/Work/ParTI-dev/timing-2018/parti/coo/sortcases/'
# intput_path = '/nethome/jli458/Work/ParTI-dev/timing_parti/coo-simd/'
# s3tsrs = ['choa100k', 'choa200k', 'choa700k', '1998DARPA', 'nell2', 'nell1', 'delicious']
# s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'freebase_music', 'freebase_sampled', 'nell1', 'delicious']
s3tsrs = ['choa700k', 'nell2', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
sl4tsrs = ['uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
modes = ['0', '1', '2']
sb = 7
sc = 12
r = 16
tb = 1

# input parameters
tk = sys.argv[1]
sortcase = sys.argv[2]


out_str = 'parti-coo-tk' + str(tk) + '.out'
input_str = ""
fo = open(out_str, 'w')

for tsr in s3tsrs:
	for m in modes:

		if tk == "1":
			## sequential coo
			input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-s' + str(sortcase) + '-seq.txt'
			# input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
		else:
			## omp coo
			input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-t' + str(tk) + '.txt'
		# print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 4):
				continue;
			elif(line_array[2] == 'MTTKRP]:'):
				time_num = line_array[3]
				fo.write(time_num+'\n')
		fi.close()

fo.close()





