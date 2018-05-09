#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a threads=("32")
# declare -a sk_range=("7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=()


tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single"

tb=1
sc=14

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s4tsrs[@]}"
	do

		# Sequential code with matrix tiling
		dev_id=-2
		# sk=${sb}
		sk=20
		sb=7
		tk=1
		echo "./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt"
		./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt


		# For OpenMP code
		if [ ${tsr_name} = "vast-2015-mc1" ]; then
			# sk=8
			sk_range=("7" "8" "9" "10")
		fi
		if [ ${tsr_name} = "choa700k" ] || [ ${tsr_name} = "nell2" ]; then
			# sk=10
			sk_range=("8" "9" "10" "11" "12")
		fi
		if [ ${tsr_name} = "1998DARPA" ] || [ ${tsr_name} = "delicious" ]; then
			# sk=14
			sk_range=("12" "13" "14" "15" "16")
		fi
		if [ ${tsr_name} = "freebase_music" ] || [ ${tsr_name} = "freebase_sampled" ]; then
			# sk=18
			sk_range=("16" "17" "18" "19" "20")
		fi
		if [ ${tsr_name} = "flickr" ]; then
			# sk=11
			sk_range=("9" "10" "11" "12" "13")
		fi
		if [ ${tsr_name} = "nell1" ]; then
			# sk=20
			sk_range=("18" "19" "20")
		fi

		if [ ${tsr_name} = "chicago-crime-comm-4d" ] || [ ${tsr_name} = "uber-4d" ]; then
			sk_range=("3" "4" "5")
		fi
		if [ ${tsr_name} = "nips-4d" ] || [ ${tsr_name} = "enron-4d" ]; then
			sk_range=("6" "7")
		fi
		if [ ${tsr_name} = "flickr-4d" ]; then
			# sk=11
			sk_range=("13" "14")
		fi
		if [ ${tsr_name} = "delicious-4d" ]; then
			# sk=11
			sk_range=("14" "15")
		fi

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
				echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt
			done
		done


		# for ((tb=1; tb <= ((${nthreads}/${tk})) ; tb=tb*2))

	done
done
