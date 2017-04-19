#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")

tsr_path="/nethome/jli458/BIGTENSORS"
nmodes=4
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=11

# single split
# smem_size=40000 # default
smem_size=14000


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${sl4tsrs[@]}"
	do

		### Single GPU code
		dev_id=3
		for mode in ${modes[@]}
		do
			if [ "${impl_num}" -eq "15" ]
				then
					echo "./build/tests/ttm_new_test ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt"
					./build/tests/ttm_new_test ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt
			else
				echo "./build/tests/ttm_new_test ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
				./build/tests/ttm_new_test ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
			fi
		done

	done
done