#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/Work/ParTI-dev/timing-2018/parti/hicoo/uint-fast8-single-renumber/'
s3tsrs = ['choa700k', '1998DARPA', 'nell2', 'freebase_music', 'freebase_sampled', 'nell1', 'delicious']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
s4tsrs = ['chicago-crime-comm-4d', 'uber-4d', 'nips-4d', 'enron-4d']
l4tsrs = ['flickr-4d', 'delicious-4d']
test_tsrs = ['flickr-4d']
nmodes = 3
niters = 50
modes = []
iterations = []

for m in range(0, nmodes):
	modes.append(str(m))
for it in range(1, niters+1):
	iterations.append(str(it))

r = 16
sc = 14

# input parameters
sb = 7
sk = 20

out_str = 'parti-hicoo-renumber.out'
print("output file: " + "\"" + out_str + "\"")
fo = open(out_str, 'w')

for tsr in s3tsrs:
	for m in modes:

		## sequential hicoo
		renumber = 0
		input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-e' + str(renumber) + '-seq.txt'
		# print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 4):
				continue;
			elif(line_array[3] == 'MTTKRP]:'):
				time_num = line_array[4]
				fo.write(time_num+',')
				# print(time_num)
		fi.close()

		## sequential hicoo
		renumber = 1
		min_num = sys.float_info.max
		for it in iterations:
			input_str = intput_path + tsr + '-b' + str(sb) + '-k' + str(sk) + '-c' + str(sc) + '-m' + str(m) + '-r' + str(r) + '-e' + str(renumber)+ '-it' + str(it)  + '-seq.txt'
		

			fi = open(input_str, 'r')
			for line in fi:
				line_array = line.rstrip().split(" ")
				if(len(line_array) < 4):
					continue;
				elif(line_array[3] == 'MTTKRP]:'):
					time_num = line_array[4]
					print(time_num)
					if( float(time_num) < min_num ):
						min_num = float(time_num)

			fi.close()
		print
		fo.write(str(min_num)+'\n')
		# print(str(min_num))

	fo.write('\n')
	# print

fo.close()





