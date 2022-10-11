FROM ubuntu:18.04
RUN ulimit -n 20000
RUN mkdir -p /tools/
WORKDIR /tools/
RUN apt-get update
RUN apt-get install -y wget git vim make gcc g++ parallel libncurses5-dev zlib1g-dev libbz2-dev liblzma-dev tabix
RUN wget https://github.com/samtools/samtools/releases/download/1.15.1/samtools-1.15.1.tar.bz2
RUN tar -jxvf samtools-1.15.1.tar.bz2 && cd samtools-1.15.1 && ./configure && make && make install
RUN cd /tools/
RUN wget https://github.com/samtools/bcftools/releases/download/1.15.1/bcftools-1.15.1.tar.bz2
RUN tar -jxvf bcftools-1.15.1.tar.bz2 && cd bcftools-1.15.1 && ./configure && make && make install
RUN cd /tools/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /tools/miniconda3/
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/tools/miniconda3/bin:${PATH}
RUN conda config --add channels bioconda
RUN conda config --add channels conda-forge
# RUN conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
# RUN conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
# RUN conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
RUN conda install cudatoolkit=10.2
RUN conda install pytorch-gpu
# RUN conda install pytorch-cpu
RUN conda install whatshap=1.0
RUN pip install torchnet torchmetrics pyyaml pandas tqdm tensorboardx matplotlib tables pysam
RUN python -m pip install git+https://github.com/lessw2020/Ranger21.git
RUN git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
RUN cd Ranger-Deep-Learning-Optimizer && pip install -e .

RUN mkdir -p /tools/dna_sv_tensor
COPY dna_sv_tensor /tools/dna_sv_tensor
RUN cd /tools/dna_sv_tensor/src/ && make
RUN cd /tools/dna_sv_tensor/src/scripts/ && chmod +x make_predict_data.sh make_train_data.sh
ENV PATH=/tools/dna_sv_tensor/src/scripts:${PATH}

RUN mkdir -p /tools/HaplotypeModel
COPY HaplotypeModel /tools/HaplotypeModel
RUN mkdir -p PileupModel 
COPY PileupModel /tools/PileupModel
RUN mkdir -p scripts
COPY scripts /tools/scripts
RUN chmod +x /tools/scripts/*.sh
ENV PATH=/tools/scripts:${PATH}
COPY run_caller.sh /tools/
RUN chmod +x run_caller.sh
ENV PATH=/tools/:${PATH}
