#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr" "freebase_sampled" "delicious" "nell1")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a test_tsr_names=("delicious-4d")
# declare -a threads=("2" "4" "8" "16" "32")
declare -a threads=("32")

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
splatt_path="/global/homes/j/jiajiali/Software/Install/splatt-ipdps19/int32-single-allmode-reorder-bfs"
# splatt_path="/global/homes/j/jiajiali/Software/Install/splatt-ipdps19/int32-single-allmode"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/splatt/32bit-single-1002"

# wingtip-bigmem1
# tsr_path="/dev/shm/jli458/BIGTENSORS"
# out_path="/home/jli458/Work/ParTI-dev/timing-results/splatt/32bit-single-onemode"
# splatt_path="/home/jli458/Software/Install/splatt/int32-single-onemode"
# out_path="/home/jli458/Work/ParTI-dev/timing-results/splatt/32bit-single-allmode"
# splatt_path="/home/jli458/Software/Install/splatt/int32-single-allmode"

# for R in 8 16 32 64
for R in 16
do
	# for tsr_name in "${test_tsr_names[@]}"
	# do

	# 	# Seq Notiling
	# 	th=1
	# 	echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-lexi-it5-tile.txt"
	# 	${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-lexi-it5-tile.txt

	# 	# Seq Tiling
	# 	# echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand-tile.txt"

	# 	# for th in ${threads[@]}
	# 	# do
	# 		# splatt-1.1.1
	# 		# Cori
	# 		# Notiling
	# 		# echo "numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand.txt"
	# 		# numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand.txt

	# 		# Tiling
	# 		# echo "numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs-tile.txt"
	# 		# numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init_nnz.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs-tile.txt
			
	# 	# done
	# done

	# echo ""

	for tsr_name in "${test_tsr_names[@]}"
	do
		# Seq Notiling
		th=1
		echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs.txt"
		${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs.txt

		# Seq Tiling
		# echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand-tile.txt"

		# for th in ${threads[@]}
		# do
			# splatt-1.1.1
			# Cori
			# Notiling
			# echo "numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand.txt"
			# numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-rand.txt

			# echo "numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs-tile.txt"
			# numactl --interleave=0-1 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --nowrite -i 5 -r ${R} -t ${th} -v --tile > ${out_path}/${tsr_name}-r${R}-t${th}-allmode-bfs-tile.txt
			
		# done
	done

			# wingtip-bigmem1
			# echo "numactl --interleave=0-3 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns -r ${R} -t ${th} --tile --nowrite -i 5 -v > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			# numactl --interleave=0-3 ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns -r ${R} -t ${th} --tile --nowrite -i 5 -v > ${out_path}/${tsr_name}-r${R}-t${th}.txt

			# splatt-git
			# echo "${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt"
			# ${splatt_path}/bin/splatt cpd ${tsr_path}/${tsr_name}_init.tns --csf=all -r ${R} -t ${th} --tile --nowrite > ${out_path}/${tsr_name}-r${R}-t${th}.txt

done