#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k")
declare -a threads=("16" "24")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="timing_parti/coo-flops-bw"
nt=32
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for mode in ${modes[@]}
		do

			# # Sequetial code
			dev_id=-2
			echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-m${mode}-r${R}-seq.txt"
			./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-m${mode}-r${R}-seq.txt


			# OpenMP code
			dev_id=-1
			for nt in ${threads[@]}
			do
				echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-m${mode}-r${R}-t${nt}.txt"
				./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-m${mode}-r${R}-t${nt}.txt
			done

		done
	done
done