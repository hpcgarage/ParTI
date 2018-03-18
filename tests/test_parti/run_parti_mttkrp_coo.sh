#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "freebase_music" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("flickr-4d" "enron-4d" "nips-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("choa100k" "choa200k")
declare -a threads=("16" "24")

tsr_path="/scratch/jli458/BIGTENSORS"
# tsr_path="/scratch/jli458/BIGTENSORS/DenseSynTensors"
out_path="timing-2018/parti/coo/sortcases"
# out_path="timing-2018/parti/coo/gpu"
nt=32
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
bs=7
ks=0


impl_num=15
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

		#### ks set for sortcase = "3" ####
		if [ ${tsr_name} = "choa700k" ]; then
			ks=14
		fi
		if [ ${tsr_name} = "1998DARPA" ]; then
			ks=16
		fi
		if [ ${tsr_name} = "nell2" ]; then
			ks=12
		fi
		if [ ${tsr_name} = "freebase_music" ] || [ ${tsr_name} = "freebase_sampled" ] || [ ${tsr_name} = "delicious" ]; then
			ks=18
		fi
		if [ ${tsr_name} = "nell1" ]; then
			ks=20
		fi


		for mode in ${modes[@]}
		do

			# for sortcase in "1" "2" "4"
			# do
			# 	#### Sequetial code ####
			# 	dev_id=-2
			# 	echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-seq.txt"
			# 	./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-seq.txt
			# done

			for sortcase in "3"
			do
				#### Sequetial code ####
				dev_id=-2
				echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} -b ${bs} -k ${ks} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-b${bs}-k${ks}-seq.txt"
				./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -s ${sortcase} -b ${bs} -k ${ks} > ${out_path}/${tsr_name}-m${mode}-r${R}-s${sortcase}-b${bs}-k${ks}-seq.txt
			done



			####  OpenMP code ####
			# dev_id=-1
			# for nt in ${threads[@]}
			# do
			# 	echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-m${mode}-r${R}-t${nt}.txt"
			# 	./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -t ${nt} > ${out_path}/${tsr_name}-m${mode}-r${R}-t${nt}.txt
			# done


			#### GPU code ####
			# dev_id=0
			# echo "./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -p ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-p${impl_num}-gpu.txt"
			# ./build/tests/mttkrp -i ${tsr_path}/${tsr_name}.tns -m ${mode} -d ${dev_id} -r ${R} -p ${impl_num} > ${out_path}/${tsr_name}-m${mode}-r${R}-p${impl_num}-gpu.txt

		done
	done
done
