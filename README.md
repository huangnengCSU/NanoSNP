# NanoSNP: A progressive and haplotype-aware SNP caller on low coverage Nanopore sequencing data

## Installation

NanoSNP can be installed using Docker or Singularity:  
* docker installation with gpu device:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP/
docker pull huangnengcsu/nanosnp:v1.1-gpu
```
* singularity installation with gpu devices:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP/
singularity pull docker://huangnengcsu/nanosnp:v1.1-gpu
```


## Usage

For whole genome SNP calling on each chromosome including chr1-chr22,chrX,chrY,chrM, use `run_caller.sh` to call SNPs.

Singularity:
```
INPUT_DIR="path to input directory, which store the input bam and reference genome."    ## Absolute path
OUTPUT_DIR="path to output directory."  ## Absolute path, make sure the output directory is existing.
THREADS="40"  ## number of threads used for computing.

singularity exec --nv --containall -B "${INPUT_DIR}":"${INPUT_DIR}","${OUTPUT_DIR}":"${OUTPUT_DIR}" \
nanosnp_v1.1-gpu.sif run_caller.sh \
-b "${INPUT_DIR}/input.bam" \
-f "${INPUT_DIR}/reference.fa" \
-t "${THREADS}" \
-o "${OUTPUT_DIR}"
```

Docker:
```
INPUT_DIR="path to input directory, which store the input bam and reference genome."    ## Absolute path
OUTPUT_DIR="path to output directory."  ## Absolute path, make sure the output directory is existing.
THREADS="40"  ## number of threads used for computing.

docker run \
-v "${INPUT_DIR}":"${INPUT_DIR}" \
-v "${OUTPUT_DIR}":"${OUTPUT_DIR}" \
--gpus all \
huangnengcsu/nanosnp:v1.1-gpu \
run_caller.sh \
-b "${INPUT_DIR}/input.bam" \
-f "${INPUT_DIR}/reference.fa" \
-t "${THREADS}" \
-o "${OUTPUT_DIR}"
```