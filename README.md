# NanoSNP: A progressive and haplotype-aware SNP caller on low coverage Nanopore sequencing data

## Installation

NanoSNP can be installed using Singularity or Docker:  
with gpu device:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP
singularity pull huangnengcsu/nanosnp:v1-gpu
```
with cpu device only:
```
git clone https://github.com/huangnengCSU/NanoSNP.git
cd NanoSNP
singularity pull huangnengcsu/nanosnp:v1-cpu
```

## Usage

For whole genome SNP calling on each chromosome including chr1-chr22,chrX,chrY,chrM, use `run_caller.sh` to call SNPs.
```
cd NanoSNP
singularity exec -B "Directory on the host system":"Directory inside the container" \
nanosnp_v1-gpu.sif bash run_caller.sh \
-b "[BAM_FILE]" \
-f "[REFERENCE_FILE]" \
-t "[THREADS]" \
-o "[OUTPUT_DIR]"
```