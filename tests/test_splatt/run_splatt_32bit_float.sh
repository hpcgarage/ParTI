#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init" "freebase_music_init" "freebase_sampled_init")
declare -a l3tsrs=("amazon-reviews_init" "patents_init" "reddit-2015_init")
declare -a sl4tsrs=("delicious-4d_init" "flickr-4d_init" "enron-4d_init" "nips-4d_init")
declare -a test_tsr_names=()
# declare -a threads=("1" "2" "4" "8" "16" "24" "32")
declare -a threads=("24")

tsr_path="/scratch/jli458/BIGTENSORS"
# splatt_path="/nethome/jli458/Software/Install/splatt-git-twomode-32bit"
splatt_path="/nethome/jli458/Software/splatt/build/Linux-x86_64/"
out_path="./timing_git_splatt/mttkrp/32bit-float-onemode-storage"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	# for tsr_name in "${s3tsrs[@]}" "${sl4tsrs[@]}"
	for tsr_name in "${s3tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			echo "${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			# ${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done