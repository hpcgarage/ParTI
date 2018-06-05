#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("freebase_music" "freebase_sampled")
declare -a threads=("2" "4" "8" "14" "28" "56")

# Cori
# tsr_path="${SCRATCH}/BIGTENSORS"
# out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/cpd-single"

# wingtip-bigmem1
tsr_path="/dev/shm/jli458/BIGTENSORS"
out_path="/home/jli458/Work/ParTI-dev/timing-results/parti/coo/single"

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
	for tsr_name in "${s3tsrs[@]}" "${s4tsrs[@]}"
	do
		# for mode in ${modes[@]}
		# do

			# # Sequetial code
			# dev_id=-2
			# echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-r${R}-seq.txt"
			# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-r${R}-seq.txt


			# # OpenMP code
			dev_id=-1
			for nt in ${threads[@]}
			do
				# Use reduce
				echo "numactl --interleave=0-3 ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt"
				numactl --interleave=0-3 ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt

				# NOT Use reduce
				# echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} -u 0 > ${out_path}/${tsr_name}-r${R}-t${nt}-noreduce.txt"
				# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} -u 0 > ${out_path}/${tsr_name}-r${R}-t${nt}-noreduce.txt
			done

			# GPU code
			# dev_id=0
			# echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -i ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-i${impl_num}-gpu.txt"
			# ./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -i ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-i${impl_num}-gpu.txt

		# done
	done
done