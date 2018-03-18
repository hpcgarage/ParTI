#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init" "freebase_music_init" "freebase_sampled_init")
declare -a l3tsrs=("amazon-reviews_init" "patents_init" "reddit-2015_init")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d")
declare -a l4tsrs=("flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=()
# declare -a threads=("1" "2" "4" "8" "16" "24" "32")
declare -a threads=("1")

# tsr_path="/scratch/jli458/BIGTENSORS"
tsr_path="/scratch/jli458/BIGTENSORS/DenseSynTensors"
splatt_path="/nethome/jli458/Software/Install/splatt/int32-single"
# out_path="./timing-2018/splatt/mttkrp/32bit-single-onemode"
out_path="./timing-2018/splatt/mttkrp/32bit-single-allmode"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${dense3dtsrs[@]}" 
	# for tsr_name in "${s3tsrs[@]}" "${l4tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			cmd="${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}_init.tns -a csf -r ${R} -t ${th} --tile > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			echo ${cmd}
			# ${cmd}
			${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}_init.tns -a csf -r ${R} -t ${th} --tile > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done