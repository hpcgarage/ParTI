#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("delicious")

tsr_path="/nethome/jli458/BIGTENSORS"
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=4
R=16

for tsr_name in "${s3tsrs[@]}"
do
	# GPU code
	dev_id=2
	for mode in ${modes[@]}
	# for mode in 2
	do
		echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
		./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
	done

	# OpenMP code
	# dev_id=-1
	# for mode in ${modes[@]}
	# # for mode in 2
	# do
	# 	echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-omp.txt"
	# 	./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-omp.txt
	# done

	# Sequetial code
	# dev_id=-2
	# for mode in ${modes[@]}
	# do
	# 	echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-seq.txt"
	# 	./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-seq.txt
	# done

done