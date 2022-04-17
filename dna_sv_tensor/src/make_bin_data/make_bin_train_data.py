import sys
import gc
import shlex
import os
import tables
import numpy as np 
import time

TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
shuffle_bin_size = 50000

no_flanking_bases = 16
no_of_positions = 2 * no_flanking_bases + 1
channel = ('A', 'C', 'G', 'T', 'I', 'I1', 'D', 'D1', '*', 'a', 'c', 'g','t', 'i', 'i1','d', 'd1','#')
channel_size = len(channel)
ont_input_shape = input_shape = [no_of_positions, channel_size]
label_shape = [21, 3, no_of_positions, no_of_positions]
label_size = sum(label_shape)

def get_time_string():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return now

def dump_usage(program):
    print('')
    print('USAGE:')
    print('%s output tensor1 [...]' % (program))

def check_existence_of_file(path):
    if not os.path.isfile(path):
        now = get_time_string()
        print('[%s] <ERROR>: File \'%s\' does not exist' % (now, path))
        sys.exit(1)

def write_table_dict(table_dict, position_matrix, pos, label, alt_info):
    position_matrix = position_matrix.split()
    table_dict['position_matrix'].append(position_matrix)
    table_dict['position'].append(pos)
    label = label.split()
    table_dict['label'].append(label)
    table_dict['alt_info'].append(alt_info)

def init_table_dict():
    table_dict = {}
    table_dict['position_matrix'] = []
    table_dict['alt_info'] = []
    table_dict['position'] = []
    table_dict['label'] = []
    return table_dict

def write_table_file(table_file, table_dict, tensor_shape, label_size, float_type):
    position_matrix = np.array(table_dict['position_matrix'], np.dtype(float_type)).reshape([-1] + tensor_shape)
    table_file.root.position_matrix.append(position_matrix)
    table_file.root.alt_info.append(np.array(table_dict['alt_info']).reshape(-1, 1))
    table_file.root.position.append(np.array(table_dict['position']).reshape(-1, 1))
    table_file.root.label.append(np.array(table_dict['label'], np.dtype(float_type)).reshape(-1, label_size))

    table_dict = init_table_dict()
    return table_dict

def transform_one_input(input, tensor_shape, label_size, float_type, table_dict, table_file, total_transformed_tensor):
    check_existence_of_file(input)
    for line in open(input, "r"):
        line = line.strip()
        columns = line.split('\t')
        if len(columns) < 4:
            now = get_time_string()
            print('[%s] <ERROR> Invalid tensor input \'%s\' (%d columns, %d expected)\n' % (now, line, len(columns), 4), end = "", file = sys.stderr)
            sys.exit(1)
        position_matrix, label, position, alt_info = columns[0], columns[1], columns[2], columns[3]
        #print("label = %s, len = %d" % (label, len(label)))
        write_table_dict(table_dict, position_matrix, position, label, alt_info)
        total_transformed_tensor += 1
        if total_transformed_tensor % 1000 == 0:
            table_dict = write_table_file(table_file, table_dict, tensor_shape, label_size, float_type)
        #if total_transformed_tensor % 10000 == 0:
        #    now = get_time_string()
        #    print("[%s] <INFO> %10d tensors transformed" % (now, total_transformed_tensor))
    if len(table_dict) > 0:
        table_dict = write_table_file(table_file, table_dict, tensor_shape, label_size, float_type)
    return total_transformed_tensor

def main(argv):
    n_param = len(argv)
    if n_param < 3:
        dump_usage(argv[0])
        sys.exit(1)

    print('\tOutput: %s' % (argv[1]))
    output = argv[1]
    for i in range(2, n_param):
        print('\tInput Tensor %d: %s' % (i - 1, argv[i]))

    float_type = 'int32'
    tensor_shape = ont_input_shape
    tables.set_blosc_max_threads(64)
    int_atom = tables.Atom.from_dtype(np.dtype(float_type))
    string_atom = tables.StringAtom(itemsize = no_of_positions + 50)
    long_string_atom = tables.StringAtom(itemsize = 5000) # max alt_info length

    table_file = tables.open_file(output, mode = 'w', filters = TABLE_FILTERS)
    table_file.create_earray(where = '/', name = 'position_matrix', atom = int_atom, shape = [0] + tensor_shape)
    table_file.create_earray(where = '/', name = 'position', atom = string_atom, shape = (0, 1), filters = TABLE_FILTERS)
    table_file.create_earray(where = '/', name = 'label', atom = int_atom, shape = (0, label_size), filters = TABLE_FILTERS)
    table_file.create_earray(where = '/', name = 'alt_info', atom = long_string_atom, shape = (0, 1), filters = TABLE_FILTERS)
    table_dict = init_table_dict()

    total_transformed_tensor = 0
    for i in range(2, n_param):
        print('Transform input %s' % (argv[i]))
        total_transformed_tensor = transform_one_input(argv[i], tensor_shape, label_size, float_type, table_dict, table_file, total_transformed_tensor)
        print('Done.')
    
    table_file.close()

if __name__ == "__main__":
    main(sys.argv)