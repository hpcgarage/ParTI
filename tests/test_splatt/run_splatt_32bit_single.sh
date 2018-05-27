#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=()
# declare -a threads=("1" "2" "4" "8" "16" "24" "32")
declare -a threads=("32")

tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/splatt/32bit-single-allmode"
splatt_path="/global/homes/j/jiajiali/Software/Install/splatt/int32-single-allmode"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s4tsrs[@]}" 
	# for tsr_name in "${s3tsrs[@]}" "${l4tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			cmd="${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}_init.tns -a csf -r ${R} -t ${th} --tile > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			echo ${cmd}
			# ${cmd}
			# ${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}_init.tns -a csf -r ${R} -t ${th} --tile > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done