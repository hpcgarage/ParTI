#!/usr/bin/python

import numpy as np
import sys 

# intput_path = '/nethome/jli458/ParTI-dev/timing_splatt/32bit/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/32bit/'
intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/cpd/32bit-allmode/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/64bit-onemode/'
# s3tsrs = ['choa100k_init', 'choa200k_init', 'choa700k_init', '1998DARPA_init', 'nell2_init', 'nell1_init', 'delicious_init']
s3tsrs = ['choa700k_init', '1998DARPA_init', 'nell2_init', 'nell1_init', 'delicious_init']
# s3tsrs = ['1998DARPA_init', 'nell2_init']
l3tsrs = ['amazon-reviews_init', 'patents_init', 'reddit-2015_init']
sl4tsrs = ['delicious-4d_init', 'flickr-4d_init', 'enron-4d_init', 'nips-4d_init']
r = 16
m1_nums = []
m2_nums = []
m3_nums = []

# input parameters
t = sys.argv[1]

out_str = 'splatt-t' + str(t) + '.out'
fo = open(out_str, 'w')

for tsr in s3tsrs:
	input_str = intput_path + tsr + '-r' + str(r) + '-t' + str(t) + '.txt'
	# print(input_str)
	count = 0
	time_num = (float)(0.0)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.strip().split(" ")
		# print(line_array)

		if(line_array[0] == 'its'):
			count += 1
			if(len(line_array) == 13):
				tmp_str = line_array[4]
			elif(len(line_array) == 14):
				tmp_str = line_array[5]

			tmp_list = list(tmp_str)
			time_num += (float)(''.join(tmp_list[1:-2]))

	time_num = time_num / count
	print time_num
	fo.write(str(time_num)+'\n')
		
	fi.close()

fo.close()





