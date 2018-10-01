#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr" "freebase_sampled" "delicious" "nell1")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("1998DARPA")
declare -a threads=("32")
# declare -a sk_range=("8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
# declare -a sk_range=("7")
iters="$(seq -s ' ' 1 50)"

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single-0924"

sc=14
tb=1
renum=1
niters_renum=5

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		# Sequential code
		# dev_id=-2
		# sk=20
		# sb=7
		# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} | tee ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt

		# echo "./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt"
		# ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-seq.txt

		# echo "./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${niters_renum}-mattile-seq.txt"
		# ./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${niters_renum}-mattile-seq.txt


		# For OpenMP code
		# 3-D tensors
		if [ ${tsr_name} = "vast-2015-mc1" ]; then
			sk=8
		fi
		if [ ${tsr_name} = "nell2" ]; then
			sk=9
		fi
		if [ ${tsr_name} = "choa700k" ]; then
			sk=10
		fi
		if [ ${tsr_name} = "1998DARPA" ]; then
			sk=15
		fi
		if [ ${tsr_name} = "freebase_music" ] || [ ${tsr_name} = "freebase_sampled" ]; then
			sk=16
		fi
		if [ ${tsr_name} = "flickr" ]; then
			sk=13
		fi
		if [ ${tsr_name} = "delicious" ]; then
			sk=16
		fi
		if [ ${tsr_name} = "nell1" ]; then
			sk=18
		fi
		# 4-D tensors
		if [ ${tsr_name} = "chicago-crime-comm-4d" ] || [ ${tsr_name} = "uber-4d" ]; then
			sk=4
		fi
		if [ ${tsr_name} = "nips-4d" ]; then
			sk=7
		fi
		if [ ${tsr_name} = "enron-4d" ]; then
			sk=8
		fi
		if [ ${tsr_name} = "flickr-4d" ]; then
			sk=12
		fi
		if [ ${tsr_name} = "delicious-4d" ]; then
			sk=15
		fi


		# OpenMP code
		dev_id=-1
		# for sk in ${sk_range[@]}
		# do
			if [ ${sk} -ge "8" ]; then
				sb=7
			else
				sb=${sk}
			fi
			echo "sk=${sk}; sb=${sb}"

			for tk in ${threads[@]}
			do

				# echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}.txt"
				# numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}.txt

				# echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}.txt"
				# numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}.txt

				echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}-n${niters_renum}-mattile-parsort.txt"
				numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -e ${renum} -n ${niters_renum} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-e${renum}-n${niters_renum}-mattile-parsort.txt

			done
		# done


	done
done
