#!/usr/bin/env python3

import logging
import re
import subprocess
import os

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

test_program = '/home/sbliss/ParTImm/build/tests/test_tucker'

R_value = 16
cuda_device = 8

def main():
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
    hlog = logging.FileHandler('test_tucker.log', encoding='utf-8')
    hlog.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logging.getLogger().addHandler(hlog)
    logging.info(':: Start test')

    os.putenv("PTI_MEMCPY_PROFILE", "1")

    report = open('test_tucker.csv', 'w', encoding='utf-8')
    report.write(','.join(['tensor', 'pre_memcpy', 'memcpy', 'setidx', 'ttmkernel', 'ttmchain (setidx+ttmkernel)', 'svd', 'loop (ttmchain+svd)', 'calc_core (setidx+ttmkernel)', 'calc_norm', 'full_iter (loop+core+norm)']) + '\n')

    row = 2
    for tensor in tensors_to_test:
        f = open(tensor, 'r')
        num_modes = int(f.readline())
        shape = list(map(int, f.readline().split()))
        f.close()
        logging.info('Tensor: {}'.format(tensor))

        cols = ['pre_memcpy', 'memcpy', 'setidx', 'ttmkernel', 'ttmchain', 'svd', 'loop', 'calc_core', 'calc_norm', 'full_iter']
        timing = {col: list() for col in cols}
        current_iteration = 0

        cmdline = [test_program, tensor] + [str(R_value)]*num_modes + list(map(str, range(num_modes))) + ["--dev", str(cuda_device), "-l", "1"]
        logging.info(str(cmdline))
        with subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            for line in proc.stdout:
                logging.info(line.rstrip())
                match = re.match(r'\[CudaMemNode .to.\]: (.*) s spent', line)
                if match:
                    timing['pre_memcpy'].append(match.group(1))
                    # fall-through
                match = re.match(r'\[Tucker Decomp\]: Iter (\d*),', line)
                if match:
                    current_iteration = int(match.group(1))
                    continue
                if current_iteration != 1:
                    continue
                match = re.match(r'\[CudaMemNode .to.\]: (.*) s spent', line)
                if match:
                    timing['memcpy'].append(match.group(1))
                    continue
                match = re.match(r'\[.* TTM SetIdx\]: (.*) s spent', line)
                if match:
                    timing['setidx'].append(match.group(1))
                    continue
                match = re.match(r'\[.* TTM Kernel\]: (.*) s spent', line)
                if match:
                    timing['ttmkernel'].append(match.group(1))
                    continue
                match = re.match(r'\[TTM Chain\]: (.*) s spent', line)
                if match:
                    timing['ttmchain'].append(match.group(1))
                    continue
                match = re.match(r'\[SVD\]: (.*) s spent', line)
                if match:
                    timing['svd'].append(match.group(1))
                    continue
                match = re.match(r'\[Tucker Decomp Loop\]: (.*) s spent', line)
                if match:
                    timing['loop'].append(match.group(1))
                    continue
                match = re.match(r'\[Tucker Decomp Core\]: (.*) s spent', line)
                if match:
                    timing['calc_core'].append(match.group(1))
                    continue
                match = re.match(r'\[Tucker Decomp Norm\]: (.*) s spent', line)
                if match:
                    timing['calc_norm'].append(match.group(1))
                    continue
                match = re.match(r'\[Tucker Decomp Iter\]: (.*) s spent', line)
                if match:
                    timing['full_iter'].append(match.group(1))
                    continue

        timing['pre_memcpy'] = timing['pre_memcpy'][:num_modes]
        report.write('"{}"'.format(tensor))
        for col in cols:
            report.write(',"=SUM({})"'.format(','.join(timing[col])))
        report.write('\n')
        report.flush()
        row += 1

    report.close()
    logging.info(':: Finish test')


if __name__ == '__main__':
    main()
