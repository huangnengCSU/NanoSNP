import numpy as np 
import tables 
import os
import sys

TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
shuffle_bin_size = 50000

no_flanking_bases = 16
no_of_positions = 2 * no_flanking_bases + 1
channel = ('A', 'C', 'G', 'T', 'I', 'I1', 'D', 'D1', '*', 'a', 'c', 'g','t', 'i', 'i1','d', 'd1','#')
channel_size = len(channel)
ont_input_shape = input_shape = [no_of_positions, channel_size]
label_shape = [21, 3, no_of_positions, no_of_positions]
label_size = sum(label_shape)

GT21_LABLES = [
    "AA",
    "AC",
    "AG",
    "AT",
    "CC",
    "CG",
    "CT",
    "GG",
    "GT",
    "TT",
    "--",
    "A-",
    "C-",
    "G-",
    "T-",
    "++",
    "A+",
    "C+",
    "G+",
    "T+",
    "+-"
]

def dump_usage(program):
    print('')
    print('USAGE:')
    print('%s binary_tensor' % (program))

def main(argv):
    n_param = len(argv)
    if n_param != 2:
        dump_usage(argv[0])
        sys.exit(1)
    binary_tensor_input = argv[1]

    '''
    tables.set_blosc_max_threads(64)
    table_file = tables.open_file(binary_tensor_input, 'r')
    N = 20
    tensor_shape = ont_input_shape
    position_matrix = np.empty([N] + tensor_shape, np.int32)
    label = np.empty((N, label_size), np.float32)
    position_matrix[0:N] = table_file.root.position_matrix[0:N]
    label[0:N] = table_file.root.label[0:N]
    position = np.empty((N, 1), np.ndarray)
    alt_info = np.empty((N, 1), np.ndarray)
    position[0:N] = table_file.root.position[0:N]
    alt_info[0:N] = table_file.root.alt_info[0:N]

    for i in range(N):
        print(i)
        print(position_matrix[i])
        print(label[i])
        print(position[i])
        print(alt_info[i])
        print()
        break 
    '''

    table_file = tables.open_file(binary_tensor_input, 'r')
    n_samples = len(table_file.root.position_matrix)
    print('number of samples: %d' % (n_samples))
    
    table_file.close()


if __name__ == "__main__":
    main(sys.argv)