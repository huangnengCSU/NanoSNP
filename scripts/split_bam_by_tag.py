import pysam
import argparse

def Run(args):
    samfile = pysam.AlignmentFile(args.in_bam, "rb")
    h1 = pysam.AlignmentFile(args.h1, 'wb', template=samfile)
    h2 = pysam.AlignmentFile(args.h2, 'wb', template=samfile)
    for read in samfile.fetch():
        try:
            if read.get_tag(args.tag) == 1:
                h1.write(read)
            elif read.get_tag(args.tag) == 2:
                h2.write(read)
        except:
            continue
            # print(read.query_name, ' has no tag HP.')
    samfile.close()
    h1.close()
    h2.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-in_bam', help='Input haplotaged bam file', required=True)
    parse.add_argument('-h1', help='output splited bam file h1', required=True)
    parse.add_argument('-h2', help='output splited bam file h2', required=True)
    parse.add_argument('-tag', help='tag field used for split bam file', required=True)
    args = parse.parse_args()
    Run(args)