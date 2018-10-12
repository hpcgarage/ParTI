#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr" "freebase_sampled" "delicious" "nell1")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr")
declare -a threads=("32")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
# declare -a sk_range=("7")
iters="$(seq -s ' ' 1 50)"

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/single-0920"

sc=14
tb=1
sortcase=0
renum=1
niters_renum=5
use_reduce=1

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${test_tsr_names[@]}"
	do
		# Sequential code with graph renumbering
		# dev_id=-2
		# echo "./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-seq.txt

		# echo "./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-n${niters_renum}-seq.txt"
		# ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-r${R}-e${renum}-n${niters_renum}-seq.txt

		# echo "./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -e ${renum} -s ${sortcase} > ${out_path}/${tsr_name}-r${R}-e${renum}-s${sortcase}-seq.txt"
		# ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -e ${renum} -s ${sortcase} > ${out_path}/${tsr_name}-r${R}-e${renum}-s${sortcase}-seq.txt

		# OpenMP code
		dev_id=-1
		for tk in ${threads[@]}
		do
			# echo "numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-reduce.txt"
			# numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-reduce.txt
			
			echo "numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -n ${niters_renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-n${niters_renum}-parsort-reduce.txt"
			numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -n ${niters_renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-n${niters_renum}-parsort-reduce.txt

			# echo "numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -n ${niters_renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-n${niters_renum}-reduce.txt"
			# numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -n ${niters_renum} -u ${use_reduce} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-n${niters_renum}-reduce.txt
			
			# echo "numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -s ${sortcase} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-s${sortcase}.txt"
			# numactl --interleave=0-1 ./build/tests/mttkrp_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -d ${dev_id} -r ${R} -t ${tk} -e ${renum} -s ${sortcase} > ${out_path}/${tsr_name}-r${R}-tk${tk}-e${renum}-s${sortcase}.txt
		done


	done
done
