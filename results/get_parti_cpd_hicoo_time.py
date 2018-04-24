#!/usr/bin/python

import sys 

intput_path = '../timing-results/parti/hicoo/cpd-uint8-single/'
s3tsrs = ['vast-2015-mc1', 'choa700k', '1998DARPA', 'nell2', 'freebase_music', 'flickr', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
r = 16
tb = 1

sc = 14

# input parameters
sb = sys.argv[1]
sk = sys.argv[2]
tk = sys.argv[3]

# out_str = 'parti-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
out_str = 'parti-cpd-hicoo-uint8-sb' + str(sb) + '-sk' + str(sk) + '-tk' + str(tk) + '.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:

	# Set optimal sk
	if (tk == '1'):
		## sequential hicoo
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-seq.txt'
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
		## omp hicoo
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-r' + str(r) + '-tk' + str(tk) + '-tb' + str(tb) + '.txt'
	# print(input_str)

	count = 0
	time_num = (float)(0.0)

	fi = open(input_str, 'r')
	for line in fi:
		line_array = line.rstrip().split(" ")
		# print line_array
		if(len(line_array) > 2):
			if(line_array[2] == "its"):
				count += 1
				if(len(line_array) == 17):
					time_num += (float)(line_array[7])
				elif(len(line_array) == 18):
					time_num += (float)(line_array[8])
	
	time_num = time_num / count
	print time_num
	fo.write(str(time_num)+'\n')

	fi.close()

fo.close()





