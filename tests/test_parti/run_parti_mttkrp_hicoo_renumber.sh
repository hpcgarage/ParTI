#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a threads=("1")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=("7")
iters="$(seq -s ' ' 1 50)"

tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single-renumber-it5"

sc=14

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${test_tsr_names[@]}"
	do

		# if [ ${tsr_name} = "amazon-reviews" ] || [ ${tsr_name} = "reddit-2015" ]; then
		# 	sk_range=("15" "17")
		# fi
		# if [ ${tsr_name} = "patents" ]; then
		# 	sk_range=("11" "13")
		# fi


		# Sequential code
		# dev_id=-2
		# sk=20
		# sb=7
		# renum=0
		# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt

		# Sequential code with graph renumbering
		dev_id=-2
		sk=20
		sb=7
		renum=1
		echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-renumber-seq.txt"
		./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-renumber-seq.txt

		# Sequential code with random renumbering
		# dev_id=-2
		# sk=20
		# sb=7
		# renum=1
		# for it in ${iters[@]}
		# do
		# 	echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-it${it}-seq.txt"
		# 	./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-it${it}-seq.txt
		# done


	done
done
