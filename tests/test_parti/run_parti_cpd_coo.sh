#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("delicious")
# declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k")
declare -a threads=("2" "4" "8" "16" "24")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="timing_parti/cpd-coo"
nt=32
nmodes=3
use_reduce=1
# modes="$(seq -s ' ' 0 $((nmodes-1)))"
# impl_num=25

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

			# Sequetial code
			dev_id=-2
			nt=1
			echo "./build/tests/cpd ${tsr_path}/${tsr_name}.tns ${dev_id} ${nt} ${R} ${use_reduce} > ${out_path}/${tsr_name}-r${R}-reduce-seq.txt"
			./build/tests/cpd ${tsr_path}/${tsr_name}.tns ${dev_id} ${nt} ${R} ${use_reduce} > ${out_path}/${tsr_name}-r${R}-reduce-seq.txt


			# OpenMP code
			dev_id=-1
			for nt in ${threads[@]}
			do
				echo "./build/tests/cpd ${tsr_path}/${tsr_name}.tns ${dev_id} ${nt} ${R} ${use_reduce} > ${out_path}/${tsr_name}-r${R}-t${nt}-reduce.txt"
				./build/tests/cpd ${tsr_path}/${tsr_name}.tns ${dev_id} ${nt} ${R} ${use_reduce} > ${out_path}/${tsr_name}-r${R}-t${nt}-reduce.txt
			done

	done
done