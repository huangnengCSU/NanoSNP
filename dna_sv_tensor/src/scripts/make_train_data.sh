#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_PATH=`dirname "$0"`
Usage="Usage: ./${SCRIPT_NAME} --bam_fn=BAM --ref_fn=REF --vcf_fn=VCF --bed_fn=BED --output=OUTPUT_DIR --threads=THREADS "

set -e
#./make_train_data.sh -b tmp.bam -f ref.fasta -t 32 -o tmp
print_help_messages()
{
    echo $''
    echo ${Usage}
    echo $''
    echo $'Required parameters:'
    echo $'  -b, --bam_fn=FILE             BAM file input. The input file must be samtools indexed.'
    echo $'  -f, --ref_fn=FILE             FASTA reference file input. The input file must be samtools indexed.'
    echo $'  -v, --vcf_fn=FILE             Truth variants.'
    echo $'  -r, --bed_fn=FILE             High confident regions.'
    echo $'  -t, --threads=INT             Max #threads to be used. The full genome will be divided into small chunks for parallel processing. Each chunk will use 4 threads. The #chunks being processed simultaneously is ceil(#threads/4)*3. 3 is the overloading factor.'
    echo $'  -o, --output=PATH             output directory.'
    echo $''
    echo $''
}

# ERROR="\\033[31m[ERROR]"
# WARNING="\\033[33m[WARNING]"
# NC="\\033[0m"

ARGS=`getopt -o b:f:v:r:t:o: -l bam_fn:,ref_fn:,vcf_fn:,bed_fn:,threads:,output:,help -- "$@"`

[ $? -ne 0 ] && ${Usage}
eval set -- "${ARGS}"

while true; do
   case "$1" in
    -b|--bam_fn )
            BAM_FILE_PATH="$2"
            shift
            ;;
    -f|--ref_fn )
            REFERENCE_FILE_PATH="$2"
            shift
            ;;
    -v|--vcf_fn )
            VCF_FILE_PATH="$2"
            shift
            ;;
    -r|--bed_fn )
            BED_FILE_PATH="$2"
            shift
            ;;
    -t|--threads )
            THREADS="$2"
            shift
            ;;
    -o|--output )
            OUTPUT_FOLDER="$2"
            shift
            ;;
    -h|--help )
            print_help_messages
            ;;
    -- )
            shift
            break
            ;;
   esac
shift
done

if [ -z ${BAM_FILE_PATH} ] || [ -z ${REFERENCE_FILE_PATH} ] || [ -z ${THREADS} ] || [ -z ${OUTPUT_FOLDER} ] || [ -z ${VCF_FILE_PATH} ] || [ -z ${BED_FILE_PATH} ]; then
      if [ -z ${BAM_FILE_PATH} ] && [ -z ${REFERENCE_FILE_PATH} ] && [ -z ${THREADS} ] && [ -z ${OUTPUT_FOLDER} ] && [ -z ${VCF_FILE_PATH} ] && [ -z ${BED_FILE_PATH} ]; then print_help_messages; exit 0; fi
      if [ -z ${BAM_FILE_PATH} ]; then echo -e "${ERROR} Require to define index BAM input by --bam_fn=BAM${NC}"; fi
      if [ -z ${REFERENCE_FILE_PATH} ]; then echo -e "${ERROR} Require to define FASTA reference file input by --ref_fn=REF${NC}"; fi
      if [ -z ${VCF_FILE_PATH} ]; then echo -e "${ERROR} Require to define truth variants by --vcf_fn${NC}"; fi
      if [ -z ${BED_FILE_PATH} ]; then echo -e "${ERROR} Require to define high confident regions by --bed_fn${NC}"; fi
      if [ -z ${THREADS} ]; then echo -e "${ERROR} Require to define max threads to be used by --threads=THREADS${NC}"; fi
      if [ -z ${OUTPUT_FOLDER} ]; then echo -e "${ERROR} Require to define output folder by --output=OUTPUT_DIR${NC}"; fi

      print_help_messages;
      exit 1;
fi


############## The following options must be provided

ALL_CHR_LIST=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)

# BIN_EXEC_PATH=/data1/cy/data/porec/dna_sv_tensor/Linux-amd64/bin
BIN_EXEC_PATH=${SCRIPT_PATH}/../../Linux-amd64/bin/

# MAKE_BIN_TRAIN_DATA_PY=/data1/cy/data/porec/dna_sv_tensor/src/make_bin_data/make_bin_train_data.py
MAKE_BIN_TRAIN_DATA_PY=${SCRIPT_PATH}/../make_bin_data/make_bin_train_data.py

# REFERENCE=/data1/cy/data/rna/grch38/reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna
REFERENCE=${REFERENCE_FILE_PATH}

# VCF_PATH=/data1/cy/data/rna/grch38/hg001_grch38/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf
VCF_PATH=${VCF_FILE_PATH}

# SORTED_BAM_PATH=/data1/cy/data/porec/sorted_hg001_grch38.bam
SORTED_BAM_PATH=${BAM_FILE_PATH}

# CONFIDENT_BED_PATH=/data1/cy/data/rna/grch38/hg001_grch38/hg001_grch38.clean.bed
CONFIDENT_BED_PATH=${BED_FILE_PATH}

# WORK_DIR=/data1/cy/data/porec/train_data
WORK_DIR=${OUTPUT_FOLDER}

# CPU_THREADS=${THREADS}
CPU_THREADS=${THREADS}

############# workding directory

PILEUP_DATA_DIR=${WORK_DIR}/pileup_data

CHR_PILEUP_DATA_DIR=${WORK_DIR}/chr_pileup_data

TRUE_VAR_DIR=${WORK_DIR}/true_var

EXTENDED_BED_DIR=${WORK_DIR}/extended_bed 

CANDIDATE_SNP_TENSOR_DIR=${WORK_DIR}/candidate_snp_tensor

TXT_TRAIN_DATA_DIR=${WORK_DIR}/txt_train_data

BIN_TRAIN_DATA_DIR=${WORK_DIR}/bin_train_data

### options

SAMTOOS_MPILEUP_OPTIONS="--min-MQ 20 --min-BQ 0 --reverse-del --excl-flags 2316 --max-depth 144"

FLANKING_BASES=16
BED_EXTENDED_BASES=31

MIN_AF=0.12
SNP_MIN_AF=0.12
INDEL_MIN_AF=0.12
MIN_DEPTH=6

SHUFFLE_TENSORS=1
MAXINUM_NON_VAR_RATIO=5.0

### dump config info

echo "========== config info"

echo "reference chr list"
echo "  ${ALL_CHR_LIST[@]}"

echo "binary executable path"
echo "  ${BIN_EXEC_PATH}"

echo "vcf path"
echo "  ${VCF_PATH}"

echo "bam path"
echo "  ${SORTED_BAM_PATH}"

echo "confident bed path"
echo "  ${CONFIDENT_BED_PATH}"

echo "working directory"
echo "  ${WORK_DIR}"

echo ""

###

mkdir -p ${PILEUP_DATA_DIR}
pileup_data_done=${PILEUP_DATA_DIR}/job_make_pileup_data.done 
if [ ! -f ${pileup_data_done} ]; then
    echo "[$(date)] ============> Step 1/7: making pileup data from ${SORTED_BAM_PATH}"
    CMD="samtools mpileup ${SAMTOOS_MPILEUP_OPTIONS} -o ${PILEUP_DATA_DIR}/pileup_data ${SORTED_BAM_PATH}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${pileup_data_done}
fi

mkdir -p ${CHR_PILEUP_DATA_DIR}
chr_pileup_data_done=${CHR_PILEUP_DATA_DIR}/job_chr_pileup_data.done 
if [ ! -f ${chr_pileup_data_done} ]; then
    echo "[$(date)] ============> Step 2/7: split chr pileup data"
    CMD="${BIN_EXEC_PATH}/DNA_ExtractChrPileupData ${PILEUP_DATA_DIR}/pileup_data ${CHR_PILEUP_DATA_DIR} ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${chr_pileup_data_done}
fi    

mkdir -p ${EXTENDED_BED_DIR}
extend_bed_done=${EXTENDED_BED_DIR}/job_extend_bed.done
if [ ! -f ${extend_bed_done} ]; then
    echo "[$(date)] ============> Step 3/7: extend each bed interval in ${CONFIDENT_BED_PATH} by ${BED_EXTENDED_BASES} bp"
    CMD="${BIN_EXEC_PATH}/DNA_ExtendBed ${CONFIDENT_BED_PATH} ${BED_EXTENDED_BASES} ${EXTENDED_BED_DIR}/extended_confident.bed"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${extend_bed_done}    
fi 

mkdir -p ${TRUE_VAR_DIR}
make_true_var_done=${TRUE_VAR_DIR}/job_make_true_var.done
if [ ! -f ${make_true_var_done} ]; then
    echo "[$(date)] ============> Step 4/7: split chr vcf in ${VCF_PATH}"
    CMD="${BIN_EXEC_PATH}/DNA_SplitVcf ${VCF_PATH} ${TRUE_VAR_DIR}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${make_true_var_done}    
fi 

mkdir -p ${CANDIDATE_SNP_TENSOR_DIR}
make_can_snp_tensor_done=${CANDIDATE_SNP_TENSOR_DIR}/job_make_snp_tensor.done 
if [ ! -f ${make_can_snp_tensor_done} ]; then
    echo "[$(date)] ============> Step 5/7: Create tensor from pileup data"
    CMD="${BIN_EXEC_PATH}/DNA_CreateCanSnpTensor \
        -reference ${REFERENCE} \
        -chr_pileup_dir ${CHR_PILEUP_DATA_DIR} \
        -output_dir ${CANDIDATE_SNP_TENSOR_DIR} \
        -extended_confident_bed ${EXTENDED_BED_DIR}/extended_confident.bed \
        -confident_bed ${CONFIDENT_BED_PATH} \
        -min_af ${MIN_AF} \
        -snp_min_af ${SNP_MIN_AF} \
        -indel_min_af ${INDEL_MIN_AF} \
        -min_coverage ${MIN_DEPTH} \
        -flanking_base ${FLANKING_BASES} \
        -num_threads ${CPU_THREADS} \
        ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${make_can_snp_tensor_done}
fi

mkdir -p ${TXT_TRAIN_DATA_DIR}
make_train_data_done=${TXT_TRAIN_DATA_DIR}/job_make_train_data.done 
if [ ! -f ${make_train_data_done} ]; then
    echo "[$(date)] ============> Step 6/7: add label to tensor"
    CMD="${BIN_EXEC_PATH}/DNA_CreateTrainData \
        -chr_tensor_dir ${CANDIDATE_SNP_TENSOR_DIR} \
        -chr_true_var_dir ${TRUE_VAR_DIR} \
        -reference ${REFERENCE} \
        -confident_bed ${CONFIDENT_BED_PATH} \
        -output_dir ${TXT_TRAIN_DATA_DIR} \
        -shuffle_tensors ${SHUFFLE_TENSORS} \
        -maxinum_non_variant_ratio ${MAXINUM_NON_VAR_RATIO} \
        -num_threads ${CPU_THREADS} \
        ${ALL_CHR_LIST[@]}"
    echo "[$(date)] Running Command"
    echo "  ${CMD}"
    ${CMD}
    if [ $? -ne 0 ]; then
        echo "[$(date)] Fail at Command"
        echo "  ${CMD}"
        exit 1
    fi
    touch ${make_train_data_done}
fi   

mkdir -p ${BIN_TRAIN_DATA_DIR}
make_bin_train_data_done=${BIN_TRAIN_DATA_DIR}/job_make_bin_train_data.done
if [ ! -f ${make_bin_train_data_done} ]; then
    echo "[$(date)] ============> Step 7/7: making binary train data"
    for chr in ${ALL_CHR_LIST[@]}
    do
        echo "make binary train data for ${chr}"
        python ${MAKE_BIN_TRAIN_DATA_PY} ${BIN_TRAIN_DATA_DIR}/${chr}.td.bin ${TXT_TRAIN_DATA_DIR}/${chr}.td 
        if [ $? -ne 0 ]; then
            echo "[$(date)] Fail at making binary train data for ${chr}"
            exit 1
        fi
    done 
    touch ${make_bin_train_data_done}
fi
