#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")
declare -a threads=("1" "2" "4" "8" "16" "32")

tsr_path="/nethome/jli458/BIGTENSORS"
out_path="timing_parti"
nthreads=32
nmodes=3
modes="$(seq -s ' ' 0 $((nmodes-1)))"
impl_num=25

# single split
# smem_size=40000 # default
smem_size=12000
max_nstreams=4
nstreams=8
sb=5
sk=12
sc=8
tk=32
tb=1


# for R in 8 16 32 64
for R in 16
do
		for tsr_name in "${s3tsrs[@]}"
		do
			
				# # Sequetial code
				# dev_id=-2
				# for mode in ${modes[@]}
				# do
				# 	echo "./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt"
				# 	./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-seq.txt
				# done


				# OpenMP code
				for tk in ${threads[@]}
				do
					dev_id=-1
					for ((tb=1; tb <= ((${nthreads}/${tk})) ; tb=tb*2))
					do
						for mode in ${modes[@]}
						do
							echo "./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt"
							./build/tests/mttkrp_hicoo -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -m ${mode} -d ${dev_id} -r ${R} -t ${tk} -h ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-m${mode}-r${R}-tk${tk}-tb${tb}.txt
						done
					done
				done

				# for nblocks in 16000
				# do
					# Single GPU code
					# dev_id=3
					# for mode in ${modes[@]}
					# do
					# 	echo "./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt"
					# 	./build/tests/mttkrp ${tsr_path}/${tsr_name}.tns ${mode} ${impl_num} ${dev_id} ${R} > timing/${tsr_name}-m${mode}-i${impl_num}-r${R}-g${dev_id}.txt
					# done
				# done

	done
done