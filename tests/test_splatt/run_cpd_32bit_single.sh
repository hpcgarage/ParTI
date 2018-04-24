#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=()
declare -a threads=("1" "32")

tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/splatt/32bit-single-allmode"
splatt_path="/global/homes/j/jiajiali/Software/Install/splatt/int32-single-allmode"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for th in ${threads[@]}
		do
			# splatt-1.1.1
			echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns -r ${R} -t ${th} --tile --nowrite -i 5 -v > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns -r ${R} -t ${th} --tile --nowrite -i 5 -v > ${out_path}/${tsr_name}-r${R}-t${th}.txt

			# splatt-git
			# echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			# ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done