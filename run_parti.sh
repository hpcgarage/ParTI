#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")

tsr_path="/nethome/jli458/BIGTENSORS"
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=16
R=16

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
nblocks=32000

for R in 8 16 32 64
do
	for tsr_name in "${s3tsrs[@]}"
	do
		# Single Split GPU code, fine-grain. impl_num=11, 15.
		# dev_id=3
		# for mode in ${modes[@]}
		# do
		# 	echo "./build/tests/mttkrp_one_fine ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-fine-m${mode}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt"
		# 	./build/tests/mttkrp_one_fine ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-fine-m${mode}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt
		# done

		# Single Split GPU code, medium-grain. impl_num=11, 15, 17.
		# dev_id=3
		# for mode in ${modes[@]}
		# # for mode in 0
		# do
		# 	echo "./build/tests/mttkrp_one_medium ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-medium-m${mode}-sm${smem_size}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt"
		# 	./build/tests/mttkrp_one_medium ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-medium-m${mode}-sm${smem_size}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt
		# done


		# Single Split GPU code, coarse-grain. impl_num=11, 15, 16.
		# dev_id=3
		# for mode in ${modes[@]}
		# # for mode in 0
		# do
		# 	echo "./build/tests/mttkrp_one_coarse ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-coarse-m${mode}-sm${smem_size}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt"
		# 	./build/tests/mttkrp_one_coarse ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${nstreams} ${nblocks} ${dev_id} ${R} ${max_nstreams} > timing/${tsr_name}-coarse-m${mode}-sm${smem_size}-ns${nstreams}-nb${nblocks}-i${impl_num}-r${R}-g${dev_id}.txt
		# done



		# Single GPU code
		dev_id=3
		for mode in ${modes[@]}
		do
			echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
			./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
		done



		# OpenMP code
		# dev_id=-1
		# for mode in ${modes[@]}
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
done