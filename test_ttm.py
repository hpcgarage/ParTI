#!/usr/bin/env python3

import logging
import re
import subprocess
import os
import sys

tensors_to_test = [
    '/home/sbliss/tensors/20^3.tns',                    # 8002
    '/home/sbliss/tensors/choa100k.tns',                # 4142859
    '/opt/data/jli/BIGTENSORS/choa700k.tns',            # 26953734
    '/opt/data/jli/BIGTENSORS/nell2.tns',               # 76879421
    '/opt/data/jli/BIGTENSORS/freebase_music.tns',      # 99546553
    '/opt/data/jli/BIGTENSORS/freebase_sampled.tns',    # 139920773
    '/opt/data/jli/BIGTENSORS/delicious.tns',           # 140126183
    '/opt/data/jli/BIGTENSORS/nell1.tns',               # 143599554
]

dense_tensor_generator = '/home/sbliss/ParTImm/tools/generate_dense_tensor.py'
temp_tensor = '/tmp/temp_dense_tensor.tns'
old_test_program = '/home/sbliss/ParTI/build/tests/test_ttm_speed'
new_test_program = '/home/sbliss/ParTImm/build/tests/test_ttm_speed'

R_value = 16
old_cuda_device = 0
new_cuda_device = 8

def main():
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
    hlog = logging.FileHandler('test_ttm.log', encoding='utf-8')
    hlog.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logging.getLogger().addHandler(hlog)
    logging.info(':: Start test')

    report = open('test_ttm.csv', 'w', encoding='utf-8')
    report.write(','.join(['tensor', 'mode'] + ['old' + str(i) for i in range(-2, 5)] + ['new' + str(i) for i in range(-2, 10)] + ['old_avg', 'new_avg']) + '\n')

    row = 2
    for tensor in tensors_to_test:
        f = open(tensor, 'r')
        num_modes = int(f.readline())
        shape = list(map(int, f.readline().split()))
        f.close()
        for mode in range(num_modes):
            logging.info('Tensor: {}, mode {}'.format(tensor, mode))
            report.write('"{}",{}'.format(tensor, mode))

            cmdline = [dense_tensor_generator, temp_tensor, str(R_value), str(shape[mode])]
            logging.info(str(cmdline))
            subprocess.run(cmdline)

            cmdline = [old_test_program, tensor, temp_tensor, str(mode), str(old_cuda_device)]
            logging.info(str(cmdline))
            with subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
                for line in proc.stdout:
                    logging.info(line.rstrip())
                    match = re.match(r'\[CUDA SpTns \* Mtx\]: (.*) s$', line)
                    if match:
                        report.write(',' + match.group(1))
                        report.flush()

            cmdline = [new_test_program, tensor, temp_tensor, '--mode', str(mode), '--dev', str(new_cuda_device)]
            logging.info(str(cmdline))
            with subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
                for line in proc.stdout:
                    logging.info(line.rstrip())
                    match = re.match(r'\[CUDA TTM Kernel\]: (.*) s spent on device ', line)
                    if match:
                        report.write(',' + match.group(1))
                        report.flush()

            os.unlink(temp_tensor)

            report.write(',=AVERAGE(E{}:I{}),=AVERAGE(L{}:U{})\n'.format(row, row, row, row))
            report.flush()
            row += 1

    report.close()
    logging.info(':: Finish test')


if __name__ == '__main__':
    main()
