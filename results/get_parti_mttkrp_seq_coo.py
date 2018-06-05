#!/usr/bin/python

import sys 

intput_path = '../timing-results/parti/coo/single-renumber-it5/'
s3tsrs = ['vast-2015-mc1', 'choa700k', '1998DARPA', 'nell2', 'freebase_music', 'flickr', 'freebase_sampled', 'nell1', 'delicious']
# s3tsrs = ['nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
# s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']

r = 16

# input parameters
tk = sys.argv[1]
renumber = sys.argv[2]


out_str = 'parti-coo-tk' + str(tk) + '.out'
input_str = ""
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	sum_seq = 0

	if tk == "1":
		## sequential coo
		input_str = intput_path + tsr + '-r' + str(r) + '-e' + str(renumber)+ '-renumber-seq.txt'
		# input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
	else:
		## omp coo
		input_str = intput_path + tsr + '-r' + str(r) + '-t' + str(tk) + '.txt'
	# print(input_str)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print(line_array)
		if(len(line_array) < 4):
			continue;
		elif(line_array[3] == 'MTTKRP'):
			time_num = line_array[6]
			sum_seq = sum_seq + float(time_num)
			fo.write(time_num+'\n')

	fo.write(str(sum_seq)+'\n')
	fi.close()

fo.close()





