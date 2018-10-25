#!/usr/bin/python

import sys 

# KNL
# intput_path = '../timing-results/parti/hicoo/uint8-single-knl/'
intput_path = '../timing-results/parti/hicoo/uint8-single-0924/'
s3tsrs = ['vast-2015-mc1', 'nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'flickr', 'delicious', 'nell1']
# s3tsrs = ['nell2', 'choa700k', '1998DARPA', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
# s4tsrs = ['chicago-crime-comm-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
test_tsrs = ['delicious-4d']

r = 16
tb = 1
sc = 14

# input parameters
tk = sys.argv[1]
renum = int(sys.argv[2])
niters_renum = int(sys.argv[3])


# out_str = 'parti-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
out_str = 'parti-hicoo-uint8.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	sum_seq = 0

	if (tk == '1'):
		# sequential hicoo
		sb = 7
		sk = 20
		# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r)+ '-e' + str(renum) + '-seq.txt'
		# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renum) + '-mattile-seq.txt'
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-e' + str(renum) + '-n' + str(niters_renum) + '-mattile-seq.txt'
	else:
		# Set optimal sk
		if(tsr == 'vast-2015-mc1'):
			sk = 8
		elif(tsr == 'nell2'):
			sk = 9
		elif(tsr == 'choa700k'):
			sk = 10
		elif(tsr == '1998DARPA'):
			sk = 15
		elif(tsr == 'freebase_music' or tsr == 'freebase_sampled'):
			sk = 16
		elif(tsr == 'flickr'):
			sk = 13
		elif(tsr == 'delicious'):
			sk = 16
		elif(tsr == 'nell1'):
			sk = 18
		# 4-D
		elif(tsr == 'chicago-crime-comm-4d' or tsr == 'uber-4d'):
			sk = 4
		elif(tsr == 'nips-4d'):
			sk = 7
		elif(tsr == 'enron-4d'):
			sk = 8
		elif(tsr == 'flickr-4d'):
			sk = 12
		elif(tsr == 'delicious-4d'):
			sk = 15

		if(sk >= 7):
			sb = 7
		else:
			sb = sk
			
		## omp hicoo
		# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '-e' + str(renum) + '.txt'
		# input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '-e' + str(renum) + '-mattile.txt'
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '-e' + str(renum) + '-mattile.txt'
	# print(input_str)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print line_array
		if(len(line_array) < 4):
			continue;
		elif(line_array[3] == 'MTTKRP'):
			time_num = line_array[6]
			# print(time_num_m0)
			sum_seq = sum_seq + float(time_num)
			fo.write(time_num+'\n')
			print(time_num)
	
	fo.write(str(sum_seq)+'\n')
	print(str(sum_seq))
	fi.close()

fo.close()





