#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d" "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k")
declare -a sk_range=("8" "10" "12" "14" "16" "18" "20")
declare -a sb_range=("4" "5" "6" "7")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="timing_parti/hicoo/uint_fast8"
nthreads=32
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
sb=7
sc=12
tb=1


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for mode in ${modes[@]}
		do


			# Sequetial code with matrix tiling
			dev_id=-2
			for sk in ${sk_range[@]}
			do
				for sb in ${sb_range[@]}
				do
					echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt"
					./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt
				done
			done

			# OpenMP code
			# dev_id=-1
			# for sk in ${sk_range[@]}
			# do
			# 	for tk in ${threads[@]}
			# 	do
			# 		echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt"
			# 		./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt
			# 	done
			# done
		# for ((tb=1; tb <= ((${nthreads}/${tk})) ; tb=tb*2))

		done
	done
done
