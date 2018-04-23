#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("vast-2015-mc1" "flickr")
declare -a threads=("1")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=("7")
iters="$(seq -s ' ' 1 50)"


tsr_path="/scratch/jli458/BIGTENSORS"
# tsr_path="/scratch/jli458/BIGTENSORS/DenseSynTensors"
out_path="timing-2018/parti/hicoo/uint-fast8-single-renumber"
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
	for tsr_name in "${test_tsr_names[@]}"
	do

		# if [ ${tsr_name} = "amazon-reviews" ] || [ ${tsr_name} = "reddit-2015" ]; then
		# 	sk_range=("15" "17")
		# fi
		# if [ ${tsr_name} = "patents" ]; then
		# 	sk_range=("11" "13")
		# fi

		for mode in ${modes[@]}
		do

			# Sequential code
			# dev_id=-2
			# sk=20
			# sb=7
			# renum=0
			# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-seq.txt"
			# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-seq.txt

			# Sequential code with graph renumbering
			# dev_id=-2
			# sk=20
			# sb=7
			# renum=1
			# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-renumber-seq.txt"
			# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-renumber-seq.txt

			# Sequential code with random renumbering
			dev_id=-2
			sk=20
			sb=7
			renum=1
			for it in ${iters[@]}
			do
				echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-it${it}-seq.txt"
				./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-e${renum}-it${it}-seq.txt
			done


			# Sequential code with matrix tiling
			# dev_id=-2
			# sk=${sb}
			# tk=1
			# echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt"
			# ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt

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
