#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("choa100k" "choa200k")
declare -a threads=("32")

tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/single"

impl_num=15
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

		#### sk set for sortcase = "3" ####
		# if [ ${tsr_name} = "choa700k" ]; then
		# 	sk=14
		# fi
		# if [ ${tsr_name} = "1998DARPA" ]; then
		# 	sk=16
		# fi
		# if [ ${tsr_name} = "nell2" ]; then
		# 	sk=12
		# fi
		# if [ ${tsr_name} = "freebase_music" ] || [ ${tsr_name} = "freebase_sampled" ] || [ ${tsr_name} = "delicious" ]; then
		# 	sk=18
		# fi
		# if [ ${tsr_name} = "nell1" ]; then
		# 	sk=20
		# fi


		#### Sequetial code ####
		dev_id=-2
		echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-m${mode}-r${R}-seq.txt"
		./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-r${R}-seq.txt

		# for sortcase in "1" "2" "4"
		# do
		# 	#### Sequetial code ####
		# 	dev_id=-2
		# 	echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-seq.txt"
		# 	./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-seq.txt
		# done

		# for sortcase in "3"
		# do
		# 	#### Sequetial code ####
		# 	dev_id=-2
		# 	echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} -b ${bs} -k ${sk} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-b${bs}-k${sk}-seq.txt"
		# 	./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} -b ${bs} -k ${sk} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-b${bs}-k${sk}-seq.txt
		# done



		####  OpenMP code ####
		dev_id=-1
		for nt in ${threads[@]}
		do
			echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt"
			./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-r${R}-t${nt}.txt
		done


		#### GPU code ####
		# dev_id=0
		# echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -p ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-p${impl_num}-gpu.txt"
		# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -p ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-p${impl_num}-gpu.txt

	done
done
