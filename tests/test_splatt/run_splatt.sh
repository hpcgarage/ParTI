#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa100k_init" "choa200k_init" "choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init")
# declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
# declare -a sl4tsrs=("delicious-4d", "flickr-4d" "enron-4d" "nips-4d")
# declare -a test_tsr_names=("choa100k" "choa200k" "choa700k" "1998DARPA" "nell2")
declare -a threads=("1" "2" "4" "8" "16" "32")

tsr_path="/nethome/jli458/BIGTENSORS"
out_path="./timing_splatt"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for th in ${threads[@]}
		do

			echo "splatt bench ${tsr_path}/${tsr_name}.tns -a splatt -a ttbox -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			splatt bench ${tsr_path}/${tsr_name}.tns -a splatt -a ttbox -r ${R} -t ${th} > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done