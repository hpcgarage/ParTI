#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("choa700k_init" "1998DARPA_init" "nell2_init" "nell1_init" "delicious_init")
declare -a l3tsrs=("amazon-reviews_init" "patents_init" "reddit-2015_init")
declare -a sl4tsrs=("delicious-4d_init" "flickr-4d_init" "enron-4d_init" "nips-4d_init")
declare -a test_tsr_names=("freebase_sampled_init" "freebase_music_init")
declare -a threads=("24")

tsr_path="/scratch/jli458/BIGTENSORS"
splatt_path="/nethome/jli458/Software/Install/splatt-git-allmode-32bit"
out_path="./timing_git_splatt/cpd/32bit-allmode"
nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"


# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${test_tsr_names[@]}"
	do
		for th in ${threads[@]}
		do

			echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt

		done
	done
done