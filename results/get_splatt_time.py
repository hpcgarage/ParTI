#!/usr/bin/python

import numpy as np
import sys 

# intput_path = '/nethome/jli458/ParTI-dev/timing_splatt/32bit/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/32bit/'
intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/32bit-allmode/'
# intput_path = '/nethome/jli458/ParTI-dev/timing_git_splatt/64bit-onemode/'
# s3tsrs = ['choa100k_init', 'choa200k_init', 'choa700k_init', '1998DARPA_init', 'nell2_init', 'nell1_init', 'delicious_init']
s3tsrs = ['choa700k_init', '1998DARPA_init', 'nell2_init', 'nell1_init', 'delicious_init']
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

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.strip().split(" ")
		# print(line_array)

		if(len(line_array) == 3 and line_array[0] == 'mode'):
			if(line_array[1] == '1'):
				m1_nums.append(line_array[2].split("s")[0])
			elif(line_array[1] == '2'):
				m2_nums.append(line_array[2].split("s")[0])
			elif(line_array[1] == '3'):
				m3_nums.append(line_array[2].split("s")[0])
		elif( (line_array[0] == '**' and line_array[1] == 'TTBOX') or line_array[0] == 'thd:' or line_array[0] == 'Timing'):
			m1_nums = map(float, m1_nums)
			m2_nums = map(float, m2_nums)
			m3_nums = map(float, m3_nums)
			fo.write(str(np.average(m1_nums))+'\n')
			fo.write(str(np.average(m2_nums))+'\n')
			fo.write(str(np.average(m3_nums))+'\n')
			break;
		
	fi.close()
	m1_nums = []
	m2_nums = []
	m3_nums = []

fo.close()





