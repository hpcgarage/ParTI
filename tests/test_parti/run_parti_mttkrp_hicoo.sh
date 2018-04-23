#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("flickr")
declare -a threads=("1")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=("7")
iters="$(seq -s ' ' 1 10)"


tsr_path="${SCRATCH}/BIGTENSORS"
# tsr_path="/scratch/jli458/BIGTENSORS/DenseSynTensors"
out_path="timing-results/parti/hicoo/uint8-single"
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
tb=1

# sb=15
sc=14

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

		# if [ ${tsr_name} = "amazon-reviews" ] || [ ${tsr_name} = "reddit-2015" ]; then
		# 	sk_range=("15" "17")
		# fi
		# if [ ${tsr_name} = "patents" ]; then
		# 	sk_range=("11" "13")
		# fi

		# Sequential code
		dev_id=-2
		sk=20
		sb=7
		renum=0
		echo "./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-seq.txt


	done
done
