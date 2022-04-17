import argparse
from collections import defaultdict

major_contigs_order = ["chr" + str(a) for a in list(range(1, 23)) + ["X", "Y"]] + [
    str(a) for a in list(range(1, 23)) + ["X", "Y"]]


def sort_vcf_file(vcf_file, output_file):
    header = []
    contig_dict = defaultdict(defaultdict)
    row_count = 0
    no_vcf_output = True
    with open(vcf_file,'r') as fin:
        for row in fin:
            row_count += 1
            if row[0] == '#':
                if row not in header:
                    header.append(row)
                continue
            columns = row.strip().split(maxsplit=3)
            ctg_name, pos = columns[0], columns[1]
            contig_dict[ctg_name][int(pos)] = row
            no_vcf_output = False
        if row_count == 0:
            print("[WARNING] No vcf file found, please check the setting")
        if no_vcf_output:
            print("[WARNING] No variant found, please check the setting")
        contigs_order = major_contigs_order + list(contig_dict.keys())
        contigs_order_list = sorted(
            contig_dict.keys(), key=lambda x: contigs_order.index(x))
        with open(output_file, 'w') as output:
            output.write(''.join(header))
            for contig in contigs_order_list:
                all_pos = sorted(contig_dict[contig].keys())
                for pos in all_pos:
                    output.write(contig_dict[contig][pos])


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--i', '-i', help='input vcf file', required=True)
    parse.add_argument('--o', '-o', help='sorted vcf file', required=True)
    args = parse.parse_args()
    sort_vcf_file(vcf_file=args.i, output_file=args.o)


if __name__ == '__main__':
    main()
