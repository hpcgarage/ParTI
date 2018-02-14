#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init" "freebase_music_init" "freebase_sampled_init")
declare -a l3tsrs=("amazon-reviews_init" "patents_init" "reddit-2015_init")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d")
declare -a l4tsrs=("flickr-4d" "delicious-4d")
declare -a test_tsr_names=()
# declare -a threads=("1" "2" "4" "8" "16" "24" "32")
declare -a threads=("24")

tsr_path="/scratch/jli458/BIGTENSORS"
# splatt_path="/nethome/jli458/Software/Install/splatt-git-twomode-32bit"
splatt_path="/nethome/jli458/Software/splatt/build/Linux-x86_64/"
out_path="./timing_git_splatt/mttkrp/32bit-allmode-newtest"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s4tsrs[@]}" "${l4tsrs[@]}"
	# for tsr_name in "${s3tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			echo "${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			# ${splatt_path}/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done