#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d" "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k")
# declare -a threads=("2" "4" "8" "16")
declare -a threads=("16" "24")
declare -a sk_range=("10" "12" "14" "16" "18" "20")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="timing_parti/cpd-hicoo/uint-fast8-simd"
# out_path="timing_parti/hicoo/uint16"
nthreads=32
nmodes=3
# modes="$(seq -s ' ' 0 $((nmodes-1)))"
# impl_num=25

# single split
# smem_size=40000 # default
# smem_size=12000
# max_nstreams=4
# nstreams=8
tb=1

# sb=15
# sc=16
sb=7
sc=14

sk=20
tk=8



# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

			if [ ${tsr_name} = "choa700k" ] || [ ${tsr_name} = "nell2" ]; then
				sk=10
			fi
			if [ ${tsr_name} = "1998DARPA" ] || [ ${tsr_name} = "delicious" ]; then
				sk=14
			fi
			if [ ${tsr_name} = "nell1" ]; then
				sk=20
			fi

# 		# Sequetial code with matrix tiling
# 		# dev_id=-2
# 		# echo "./build/tests/cpd_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt"
# 		# ./build/tests/cpd_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt

		# OpenMP code
		dev_id=-1

		# for sk in ${sk_range[@]}
		# do
			for tk in ${threads[@]}
			do
				echo "./build/tests/cpd_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				./build/tests/cpd_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt
			done
		# done

	done
done
