#!/usr/bin/python

import sys 

intput_path = '../timing-results/parti/hicoo/uint8-single-renumber-it5-matrixtiling/'
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
tk = sys.argv[2]

out_str = 'parti-hicoo-renumber.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	sum_seq = 0

	## sequential hicoo
	if (tk == '1'):
		if (renumber == '0'):
			input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renumber) + '-seq.txt'
		elif (renumber == '1'):
			input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renumber) + '-renumber-seq.txt'
	else:
		
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
		# 4-D
		elif(tsr == 'chicago-crime-comm-4d' or tsr == 'uber-4d'):
			sk = 4
		elif(tsr == 'nips-4d'):
			sk = 7
		elif(tsr == 'enron-4d'):
			sk = 8
		elif(tsr == 'flickr-4d'):
			sk = 15
		elif(tsr == 'delicious-4d'):
			sk = 16

		if(sk >= 7):
			sb = 7
		else:
			sb = sk


		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb1' + '-e' + str(renumber) + '-renumber.txt'

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





