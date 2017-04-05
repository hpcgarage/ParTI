#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")

tsr_path="/nethome/jli458/BIGTENSORS"
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=11

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8


for R in 8 16 32 64
do
		for tsr_name in "${s3tsrs[@]}"
		do
			# Single GPU code
			dev_id=0
			for mode in ${modes[@]}
			do
				echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
				./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
			done

	done
done