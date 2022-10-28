import numpy as np
import tables

def write_to_bins(args, contig_name, adjacent_size, pileup_flanking_size, out_candidate_positions, out_haplotype_positions, out_haplotype_sequences, out_haplotype_hap, out_haplotype_baseq, out_haplotype_mapq, out_pileup_sequences, out_pileup_hap, out_pileup_baseq, out_pileup_mapq, max_haplotype_depth, max_pileup_depth):
    out_candidate_positions = np.array(out_candidate_positions)
    new_candidate_positions = [int(v.split(':')[1]) for v in out_candidate_positions]
    new_index = np.argsort(new_candidate_positions)
    out_candidate_positions = out_candidate_positions[new_index]
    out_haplotype_positions = np.array(out_haplotype_positions)[new_index]
    
    # TODO: fix error "need at least one array to concatenate" temporarily
    if len(out_haplotype_sequences) == 0:
        print("need at least one array to concatenate, continue the loop")
        return
    out_haplotype_sequences = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_sequences]
    out_haplotype_sequences = np.concatenate(out_haplotype_sequences)[new_index]
    out_haplotype_hap = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_hap]
    out_haplotype_hap = np.concatenate(out_haplotype_hap)[new_index]
    out_haplotype_baseq = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_baseq]
    out_haplotype_baseq = np.concatenate(out_haplotype_baseq)[new_index]
    out_haplotype_mapq = [np.expand_dims(np.pad(a, ((0, max_haplotype_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_haplotype_mapq]
    out_haplotype_mapq = np.concatenate(out_haplotype_mapq)[new_index]
    out_pileup_sequences = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_sequences]
    out_pileup_sequences = np.concatenate(out_pileup_sequences)[new_index]
    out_pileup_hap = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_hap]
    out_pileup_hap = np.concatenate(out_pileup_hap)[new_index]
    out_pileup_baseq = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_baseq]
    out_pileup_baseq = np.concatenate(out_pileup_baseq)[new_index]
    out_pileup_mapq = [np.expand_dims(np.pad(a, ((0, max_pileup_depth - a.shape[0]), (0, 0)), 'constant', constant_values=-2), 0) for a in out_pileup_mapq]
    out_pileup_mapq = np.concatenate(out_pileup_mapq)[new_index]
    TABLE_FILTERS = tables.Filters(complib='blosc:lz4hc', complevel=5)
    group_start = out_candidate_positions[0].split(':')[1]
    group_end = out_candidate_positions[-1].split(':')[1]
    output = "{}/{}_{}_{}.bin".format(args.output, contig_name, str(group_start), str(group_end))
    table_file = tables.open_file(output, mode='w')
    int_atom = tables.Atom.from_dtype(np.dtype('int32'))
    string_atom = tables.StringAtom(itemsize=30 * (2 * adjacent_size))

    if args.max_pileup_depth is not None and args.max_pileup_depth < max_pileup_depth:
        max_pileup_depth = args.max_pileup_depth
    if args.max_haplotype_depth is not None and args.max_haplotype_depth < max_haplotype_depth:
        max_haplotype_depth = args.max_haplotype_depth

    table_file.create_earray(where='/', name='haplotype_sequences', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
    table_file.create_earray(where='/', name='haplotype_hap', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
    table_file.create_earray(where='/', name='haplotype_baseq', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
    table_file.create_earray(where='/', name='haplotype_mapq', atom=int_atom, shape=[0, max_haplotype_depth, 2 * adjacent_size + 1])
    table_file.create_earray(where='/', name='pileup_sequences', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
    table_file.create_earray(where='/', name='pileup_hap', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
    table_file.create_earray(where='/', name='pileup_baseq', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
    table_file.create_earray(where='/', name='pileup_mapq', atom=int_atom, shape=[0, max_pileup_depth, 2 * pileup_flanking_size + 1])
    table_file.create_earray(where='/', name='candidate_positions', atom=string_atom, shape=(0, 1), filters=TABLE_FILTERS)
    table_file.create_earray(where='/', name='haplotype_positions', atom=string_atom, shape=(0, adjacent_size * 2 + 1),filters=TABLE_FILTERS)
    table_file.root.haplotype_sequences.append(out_haplotype_sequences[:, :max_haplotype_depth, :])
    table_file.root.haplotype_hap.append(out_haplotype_hap[:, :max_haplotype_depth, :])
    table_file.root.haplotype_baseq.append(out_haplotype_baseq[:, :max_haplotype_depth, :])
    table_file.root.haplotype_mapq.append(out_haplotype_mapq[:, :max_haplotype_depth, :])
    table_file.root.pileup_sequences.append(out_pileup_sequences[:, :max_pileup_depth, :])
    table_file.root.pileup_hap.append(out_pileup_hap[:, :max_pileup_depth, :])
    table_file.root.pileup_baseq.append(out_pileup_baseq[:, :max_pileup_depth, :])
    table_file.root.pileup_mapq.append(out_pileup_mapq[:, :max_pileup_depth, :])
    table_file.root.candidate_positions.append(np.array(out_candidate_positions).reshape(-1, 1))
    table_file.root.haplotype_positions.append(np.array(out_haplotype_positions).reshape(-1, adjacent_size * 2 + 1))
    table_file.close()