#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k" "1998DARPA" "nell2" "freebase_music" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("flickr-4d" "delicious-4d")
declare -a threads=("56")
declare -a sk_range=("8" "10" "12" "14" "16" "18")

# Cori
# tsr_path="${SCRATCH}/BIGTENSORS"
# out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single"

# wingtip-bigmem1
tsr_path="/dev/shm/jli458/BIGTENSORS"
out_path="/home/jli458/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single-full"

tb=1
sc=14

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

		# Sequential code with matrix tiling
		# dev_id=-2
		# # sk=${sb}
		# sk=20
		# sb=7
		# tk=1
		# echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt"
		# ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${declareev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt


		# For OpenMP code
		# 3-D tensors
		# if [ ${tsr_name} = "vast-2015-mc1" ]; then
		# 	# sk=8
		# 	# sk_range=("7" "8" "9" "10")
		# 	sk_range=("8")
		# fi
		# if [ ${tsr_name} = "nell2" ]; then
		# 	sk_range=("9")
		# fi
		# if [ ${tsr_name} = "choa700k" ]; then
		# 	sk_range=("10")
		# fi
		# if [ ${tsr_name} = "1998DARPA" ]; then
		# 	sk_range=("15")
		# fi
		# if [ ${tsr_name} = "freebase_music" ]; then
		# 	sk_range=("15")
		# fi
		# if [ ${tsr_name} = "freebase_sampled" ]; then
		# 	sk_range=("15")
		# fi
		# if [ ${tsr_name} = "flickr" ]; then
		# 	sk_range=("11")
		# fi
		# if [ ${tsr_name} = "nell1" ]; then
		# 	sk_range=("16")
		# fi
		# if [ ${tsr_name} = "delicious" ]; then
		# 	sk_range=("16")
		# fi

		# # 4-D tensors
		# if [ ${tsr_name} = "chicago-crime-comm-4d" ]; then
		# 	sk_range=("5")
		# fi
		# if [ ${tsr_name} = "uber-4d" ]; then
		# 	sk_range=("4")
		# fi
		# if [ ${tsr_name} = "nips-4d" ]; then
		# 	sk_range=("9")
		# fi
		# if [ ${tsr_name} = "enron-4d" ]; then
		# 	sk_range=("8")
		# fi
		# if [ ${tsr_name} = "flickr-4d" ]; then
		# 	sk_range=("16")
		# fi
		# if [ ${tsr_name} = "delicious-4d" ]; then
		# 	sk_range=("16")
		# fi

		# if [ ${tsr_name} = "amazon-reviews" ] || [ ${tsr_name} = "reddit-2015" ]; then
		# 	sk_range=("15" "17")
		# fi
		# if [ ${tsr_name} = "patents" ]; then
		# 	sk_range=("11" "13")
		# fi

		# OpenMP code
		dev_id=-1
		for sk in ${sk_range[@]}
		do
			if [ ${sk} -ge "8" ]; then
				sb=7
			else
				sb=${sk}
			fi
			echo "sk=${sk}; sb=${sb}"

			for tk in ${threads[@]}
			do
				# Cori
				# echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				# numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt

				# wingtip-bigmem1
				echo "numactl --interleave=0-3 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				numactl --interleave=0-3 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt
			done
		done


		# for ((tb=1; tb <= ((${nthreads}/${tk})) ; tb=tb*2))

	done
done
