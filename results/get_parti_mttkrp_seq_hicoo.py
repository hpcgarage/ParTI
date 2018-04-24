#!/usr/bin/python

import sys 

intput_path = '../timing-results/parti/hicoo/uint8-single/'
s3tsrs = ['vast-2015-mc1', 'choa700k', '1998DARPA', 'nell2', 'freebase_music', 'flickr', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
test_tsrs = ['choa700k']

r = 16
tb = 1
sc = 14

# input parameters
sb = sys.argv[1]
sk = sys.argv[2]
tk = sys.argv[3]


# out_str = 'parti-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
out_str = 'parti-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	sum_seq = 0

	# Set optimal sk
	if(tsr == 'vast-2015-mc1'):
		sk = 8
	elif(tsr == 'choa700k' or tsr == 'nell2'):
		sk = 10
	elif(tsr == '1998DARPA' or tsr == 'delicious'):
		sk = 14
	elif(tsr == 'freebase_music' or tsr == 'freebase_sampled'):
		sk = 18
	elif(tsr == 'flickr'):
		sk = 11
	elif(tsr == 'nell1'):
		sk = 20

	## omp hicoo
	input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '.txt'

	## sequential hicoo
	# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-seq.txt'
	# print(input_str)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print line_array
		if(len(line_array) < 4):
			continue;
		elif(line_array[3] == 'MTTKRP'):
			if(line_array[5] == '0]:'):
				time_num_m0 = line_array[6]
				# print(time_num_m0)
				sum_seq = sum_seq + float(time_num_m0)
				fo.write(time_num_m0+'\n')
			elif(line_array[5] == '1]:'):
				time_num_m1 = line_array[6]
				# print(time_num_m1)
				sum_seq = sum_seq + float(time_num_m1)
				fo.write(time_num_m1+'\n')
			elif(line_array[5] == '2]:'):
				time_num_m2 = line_array[6]
				# print(time_num_m2)
				sum_seq = sum_seq + float(time_num_m2)
				fo.write(time_num_m2+'\n')
			elif(line_array[5] == '3]:'):
				time_num_m3 = line_array[6]
				# print(time_num_m3)
				sum_seq = sum_seq + float(time_num_m3)
				fo.write(time_num_m3+'\n')
	
	fo.write(str(sum_seq)+'\n')
	fi.close()

fo.close()





