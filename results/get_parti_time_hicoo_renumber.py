#!/usr/bin/python

import sys 

intput_path = '../timing-results/parti/hicoo/uint8-single-renumber/'
s3tsrs = ['vast-2015-mc1', 'choa700k', '1998DARPA', 'nell2', 'freebase_music', 'flickr', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
test_tsrs = ['flickr-4d']
niters = 50
iterations = []

for it in range(1, niters+1):
	iterations.append(str(it))

r = 16
sc = 14

# input parameters
sb = 7
sk = 20

renumber = sys.argv[1]

out_str = 'parti-hicoo-renumber.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	sum_seq = 0

	## sequential hicoo
	if (renumber == '0'):
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renumber) + '-seq.txt'
	elif (renumber == '1'):
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renumber) + '-renumber-seq.txt'
	# print(input_str)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		if(len(line_array) < 4):
			continue;
		elif(line_array[3] == 'MTTKRP'):
			time_num = line_array[6]
			fo.write(time_num+'\n')
			sum_seq = sum_seq + float(time_num)

	fo.write(str(sum_seq)+'\n')
	fi.close()

	## sequential hicoo
	# renumber = 1
	# min_num = sys.float_info.max
	# max_num = 0
	# for it in iterations:
	# 	input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-e' + str(renumber)+ '-it' + str(it)  + '-seq.txt'
	

	# 	fi = open(input_str, 'r')
	# 	for line in fi:
	# 		line_array = line.rstrip().split(" ")
	# 		if(len(line_array) < 4):
	# 			continue;
	# 		elif(line_array[3] == 'MTTKRP]:'):
	# 			time_num = line_array[4]
	# 			# print(time_num)
	# 			if( float(time_num) < min_num ):
	# 				min_num = float(time_num)
	# 			if( float(time_num) > min_num ):
	# 				max_num = float(time_num)

	# 	fi.close()
	# print
	# fo.write(str(min_num)+',')
	# fo.write(str(max_num)+'\n')
	
	# print(str(min_num))

fo.close()





