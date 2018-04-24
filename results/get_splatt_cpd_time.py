#!/usr/bin/python

import numpy as np
import sys 


intput_path = '/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/splatt/32bit-single-allmode/'
s3tsrs = ['vast-2015-mc1', 'choa700k', '1998DARPA', 'nell2', 'freebase_music', 'flickr', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
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





