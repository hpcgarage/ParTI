#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("freebase_sampled" "freebase_music")
declare -a threads=("32")

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/cpd-single-1009"

# wingtip-bigmem1
# tsr_path="/dev/shm/jli458/BIGTENSORS"
# out_path="/home/jli458/Work/ParTI-dev/timing-results/parti/coo/cpd-single"

nt=32
nmodes=3
use_reduce=0
# modes="$(seq -s ' ' 0 $((nmodes-1)))"
# impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
renum=1
niters_renum=5


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

			# Sequetial code
			# dev_id=-2
			# nt=1
			# echo "./build/tests/cpd -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -t ${nt} -r ${R} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-reduce-seq.txt"
			# ./build/tests/cpd -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -t ${nt} -r ${R} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-reduce-seq.txt

			# OpenMP code
			dev_id=-1
			for nt in ${threads[@]}
			do
				echo "numactl --interleave=0-1 ./build/tests/cpd -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -t ${nt} -r ${R} -u ${use_reduce} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-r${R}-t${nt}-e${renum}-n${niters_renum}.txt"
				numactl --interleave=0-1 ./build/tests/cpd -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -t ${nt} -r ${R} -u ${use_reduce} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-r${R}-t${nt}-e${renum}-n${niters_renum}.txt
			done

	done
done