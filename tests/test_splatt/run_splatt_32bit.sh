#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init")
declare -a l3tsrs=("amazon-reviews_init" "patents_init" "reddit-2015_init")
declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
declare -a test_tsr_names=("choa100k_init" "choa200k_init")
declare -a threads=("1" "2" "4" "8" "16" "24" "32")

tsr_path="/scratch/jli458/BIGTENSORS"
out_path="./timing_git_splatt/32bit-allmode"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			echo "/nethome/jli458/Software/Install/splatt-git-allmode-32bit/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			/nethome/jli458/Software/Install/splatt-git-allmode-32bit/bin/splatt bench ${tsr_path}/${tsr_name}.tns -a csf -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done