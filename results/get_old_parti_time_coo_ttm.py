#!/usr/bin/python

import sys 

intput_path = '/nethome/jli458/Tests/ParTI/timing-ttm/'
# intput_path = '/nethome/jli458/Work/ParTI-dev/timing-ttm/'
s3tsrs = ['choa700k', 'nell2', 'freebase_music', 'freebase_sampled', 'delicious', 'nell1']
l3tsrs = ['amazon-reviews', 'patents', 'reddit-2015']
sl4tsrs = ['uber-4d', 'nips-4d', 'enron-4d', 'flickr-4d', 'delicious-4d']
sm = 12000

# input parameters
impl_num = sys.argv[1]
r = sys.argv[2]

# out_str = 'ttm-gpu-' + str(impl_num) + '.out'
out_str = 'ttm-seq.out'
input_str = ""
fo = open(out_str, 'w')

aver_time_num = 0
count = 0

# modes = ['0', '1', '2']
# for tsr in s3tsrs:
# 	for m in modes:

# 		# GPU
# 		# if(impl_num == "15"):
# 		# 	input_str = intput_path + tsr + '-m' + str(m) + '-i' + str(impl_num) + '-s' + str(sm) + '-r' + str(r) + '-g0.txt'
# 		# else:
# 		# 	input_str = intput_path + tsr + '-m' + str(m) + '-i' + str(impl_num) + '-r' + str(r) + '-g0.txt'
# 		# Seq
# 		input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
# 		print(input_str)

# 		fi = open(input_str, 'r')
# 		for line in fi:
# 			line_array = line.rstrip().split(" ")
# 			if(len(line_array) < 4):
# 				continue;
# 			if(line_array[4] == 'Mtx]:'):
# 				count += 1
# 				if(count > 1):
# 					aver_time_num += float(line_array[5])
# 					# print aver_time_num
					
# 		aver_time_num /= 5
# 		fo.write(str(aver_time_num)+'\n')
# 		aver_time_num = 0
# 		count = 0
# 		fi.close()

# 	fo.write('\n')
	

modes = ['0', '1', '2', '3']
for tsr in sl4tsrs:
	for m in modes:

		# GPU
		# if(impl_num == "15"):
		# 	input_str = intput_path + tsr + '-m' + str(m) + '-i' + str(impl_num) + '-s' + str(sm) + '-r' + str(r) + '-g0.txt'
		# else:
		# 	input_str = intput_path + tsr + '-m' + str(m) + '-i' + str(impl_num) + '-r' + str(r) + '-g0.txt'
		# Seq
		input_str = intput_path + tsr + '-m' + str(m) + '-r' + str(r) + '-seq.txt'
		print(input_str)

		fi = open(input_str, 'r')
		for line in fi:
			line_array = line.rstrip().split(" ")
			if(len(line_array) < 4):
				continue;
			if(line_array[4] == 'Mtx]:'):
				count += 1
				if(count > 1):
					aver_time_num += float(line_array[5])
					# print aver_time_num
					
		aver_time_num /= 5
		fo.write(str(aver_time_num)+'\n')
		aver_time_num = 0
		count = 0
		fi.close()

	fo.write('\n')


fo.close()





