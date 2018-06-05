#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("vast-2015-mc1")
declare -a threads=("32")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
# declare -a sk_range=("7")
iters="$(seq -s ' ' 1 50)"

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/single-renumber-it5"

sc=14
tb=1

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do


		# Sequential code
		# dev_id=-2
		# renum=0
		# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt

		# Sequential code with graph renumbering
		dev_id=-2
		renum=0
		echo "./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-renumber-seq.txt"
		./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-renumber-seq.txt

		# OpenMP code
		# dev_id=-1
		# renum=1
		# renum=0
		# for tk in ${threads[@]}
		# do
		# 	echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}-renumber.txt"
		# 	# numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}-renumber.txt
		# done


	done
done
