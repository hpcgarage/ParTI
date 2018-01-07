#!/bin/bash

declare -a array=("one" "two" "three")
declare -a sl4tsrs=("flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("delicious-4d")
declare -a threads=("24")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=("8" "10" "12" "14" "16" "18" "20")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="timing_parti/hicoo/uint-fast8-simd-fulltest"
nthreads=32
nmodes=4
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
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
	for tsr_name in "${sl4tsrs[@]}"
	do

		if [ ${tsr_name} = "enron-4d" ]; then
			sk_range=("8" "9")
		fi
		if [ ${tsr_name} = "flickr-4d" ]; then
			sk_range=("12" "14")
		fi
		if [ ${tsr_name} = "nips-4d" ]; then
			sk_range=("8" "9")
		fi

		for mode in ${modes[@]}
		do

			# # Sequetial code
			# dev_id=-2
			# echo "./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt"
			# ./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt

			# Sequetial code with matrix tiling
			# dev_id=-2
			# echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt"
			# ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt

			# OpenMP code
			dev_id=-1
			for sk in ${sk_range[@]}
			do
				for tk in ${threads[@]}
				do
					echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt"
					./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt
				done
			done

		done
	done
done
