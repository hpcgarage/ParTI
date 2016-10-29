#!/bin/bash

do_test() {
    for dev in -2 0 1
    do
        ~/Work/SpTOL/tests/dmul "$1" "$1" "$dev" 2>&1
    done
}


main() {
    lscpu
    /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
    do_test /mnt/BIGDATA/jli/BIGTENSORS/brainq.tns
    do_test /mnt/BIGDATA/jli/BIGTENSORS/nell2.tns
    do_test /mnt/BIGDATA/jli/BIGTENSORS/delicious.tns
    do_test /mnt/BIGDATA/jli/BIGTENSORS/nell1.tns
}

echo "Log to test_dmul.txt"
main | tee test_dmul.txt
