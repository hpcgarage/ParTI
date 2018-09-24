#!/usr/bin/python

import numpy as np
import sys 

intput_path="../timing-results/splatt/32bit-single-allmode-notiling/"
# intput_path="../timing-results/splatt/32bit-single-onemode/"

s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'flickr', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
test_tsrs = ['flickr-4d']
r = 32

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





