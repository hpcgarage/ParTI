#!/bin/bash

do_test() {
    echo "File: $1, mode $2"
    echo "do_test $1 $2 $3"
    PARTI_TTM_NTHREADS=$3 ~/Work/ParTI/tests/ttm_new "$1" "$2" 0 "$((512/$3))" 2>&1
#    for mode in `seq 0 "$(($2 - 1))"`
#    do
#        echo "File: $1, mode $mode"
#        PARTI_TTM_NTHX=64 ~/Work/ParTI/tests/ttm_new "$1" "$mode" 0 2>&1
#        PARTI_TTM_NTHX=32 ~/Work/ParTI/tests/ttm_new "$1" "$mode" 0 2>&1
#        PARTI_TTM_NTHX=16 ~/Work/ParTI/tests/ttm_new "$1" "$mode" 0 2>&1
#        PARTI_TTM_NTHX=8 ~/Work/ParTI/tests/ttm_new "$1" "$mode" 0 2>&1
#    done
}

do_test1() {
    for mode in `seq 0 "$(($2 - 1))"`
    do
        for dev in -2 -1
        do
            echo
            echo "File: $1, mode $mode"
            ~/Work/ParTI/tests/ttm_new "$1" "$mode" "$dev" 2>&1
        done
        for dev in 0 1
        do
            echo
            echo "File: $1, mode $mode, dev $dev, normal kernel"
            ~/Work/ParTI/tests/ttm_new "$1" "$mode" "$dev" 2>&1
            echo "File: $1, mode $mode, dev $dev, naïve kernel"
            PARTI_TTM_KERNEL=naive ~/Work/ParTI/tests/ttm_new "$1" "$mode" "$dev" 2>&1
        done
    done
}

main() {
    OMP_NUM_THREADS="`nproc`"
    echo "Number of CPU cores (including hyperthreads): $OMP_NUM_THREADS"
    lscpu
    /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_99883309.dat 0 8
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_99883309.dat 1 8
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_99883309.dat 2 8
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 0 16
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 1 16
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 2 16
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 0 8
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 1 8
    do_test /mnt/BIGDATA/jli/BIGTENSORS/3D_124996124.dat 2 8
}

echo "Log to test_log.txt"
main | tee test_log.txt
