#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa700k" "nell2" "freebase_music" "freebase_sampled" "delicious" "nell1")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("nips-4d" "uber-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")

tsr_path="/scratch/jli458/BIGTENSORS"
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=15

# single split
# smem_size=40000 # default
smem_size=12000


for R in 8 16 32 64
# for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

		for impl_num in 11 12 13 14 15
		do
			### Single GPU code
			dev_id=0
			for mode in ${modes[@]}
			do
				if [ "${impl_num}" -eq "15" ]
					then
						echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt"
						./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt
				else
					echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
					./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
				fi
			done
		done




		## OpenMP code
		dev_id=-1
		for mode in ${modes[@]}
		do
			echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-omp.txt"
			./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-omp.txt
		done


		### Sequetial code
		dev_id=-2
		for mode in ${modes[@]}
		do
			echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-seq.txt"
			./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-seq.txt
		done

	done
done


nmodes=4
modes="$(seq -s ' ' 0 $((nmodes-1)))"
for R in 8 16 32 64
# for R in 16
do
	for tsr_name in "${sl4tsrs[@]}"
	do

		for impl_num in 11 12 13 14 15
		do
			### Single GPU code
			dev_id=0
			for mode in ${modes[@]}
			do
				if [ "${impl_num}" -eq "15" ]
					then
						echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt"
						./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-s${smem_size}-r${R}-g${dev_id}.txt
				else
					echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
					./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
				fi
			done
		done




		## OpenMP code
		dev_id=-1
		export OMP_NUM_THREADS=12
		echo ${OMP_NUM_THREADS}
		for mode in ${modes[@]}
		do
			echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-omp.txt"
			./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-omp.txt
		done


		### Sequetial code
		dev_id=-2
		for mode in ${modes[@]}
		do
			echo "./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-seq.txt"
			./build/tests/ttm ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${smem_size} ${dev_id} ${R} > timing-ttm/${tsr_name}-m${mode}-r${R}-seq.txt
		done

	done
done