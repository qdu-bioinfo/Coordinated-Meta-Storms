# Coordinated Meta-Storms

![Version](https://img.shields.io/badge/Version-1.01-brightgreen)
![Release date](https://img.shields.io/badge/Release%20date-Jan.%2017%2C%202025-brightgreen)



## Contents

- [Introduction](#introduction)
- [System Requirement and dependency](#system-requirement-and-dependency)
- [Installation guide](#installation-guide)
- [Usage](#usage)
- [Example dataset](#example-dataset)
- [Supplementary](#supplementary)
- [Contact](#contact)

# Introduction

Coordianted Meta-Storms (CMS) algorithm not only optimizes Meta-Storms computing kernel, but also introduces a self-adaptive data decomposition strategy and a multi-GPU coordinate architecturea method for large-scale microbiome distance calculation that utilizes multiple GPUs in a coordinated manner. Compared to original Meta-Storms, it can enpower hundreds of times faster computation, greatly accelerating the research on microbial big data.

## System requirement and dependency

### Hardware requirements

Coordianted Meta-Storms requirs Nvidia GPU or AMD GPU support and a standard computer with sufficient RAM to support the operations defined by a user. A server equipped with multiple GPUs is a better configuration for usage. We recommend a computer with the following specs:

RAM: 32+ GB

CPU: 8+ cores

GPU: 1+ Nvidia or AMD GPU

### Software requirements

This software package integrates Meta-Storms. Meta-Storms requires the C/C++ parallel computing library of OpenMP. Most Linux releases have OpenMP already been installed in the system. 

In addition, CMS also requires support from CUDA or HIP. CUDA installation can refer to the next section. HIP installation can refer to the [this address](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).

CMS doesn't provide support for the Mac.

## Installation guide

#### CUDA Download and Install

a. Before installation, please check the current NVIDIA driver version (using the `nvidia-smi` command to see the maximum CUDA Toolkit version supported by the driver) and the Linux server version to ensure they support the required CUDA Toolkit version.

b. Visit the CUDA official website (<https://developer.nvidia.com/cuda-toolkit-archive>) to download the CUDA Toolkit version that matches your system environment.

E.g. The following uses **CUDA Version 12.2** and **Ubuntu 22.04.3** as an example (** **Please choose the appropriate download link based on your system** **) 

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

**Note:** If multiple CUDA versions are installed on the system, you can switch between them using the following commands ( taking CUDA Toolkit 12.0 as an example ):

```
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-12.0 /usr/local/cuda
```

#### CUDA version CMS Download and Install

**a. Download the package**

```
git clone https://github.com/qdu-bioinfo/Coordinated-Meta-Storms.git
```

**b. Extract the package （use unzip command as an example）**

```shell
unzip Coordinated-Meta-Storms.zip
```

**c. Install by installer**

````shell
cd Coordinated-Meta-Storms
bash cms_install.sh
````

**Warnings may appear during compilation, but they can be ignored.**

**d. Verify Compilation Success**

```
source ~/.bashrc

cd cms/bin
cuda-comp -h
```

If a software usage prompt is displayed, the software has been successfully installed.

Examples of program operation and simple usage instructions are in the "example" folder

#### HIP version CMS Download and manully Install

**a. Download the package**

```
git clone https://github.com/qdu-bioinfo/Coordinated-Meta-Storms.git
```

**b. Extract the package （use unzip command as an example）**

```shell
unzip Coordinated-Meta-Storms.zip
```

**c. manully Install**

````shell
cd Coordinated-Meta-Storms
````

Replace the compilation statement in **Makefile**

````shell
nvcc -w -o $(EXE_CMP_CUDA) src/cms_cuda.cu -lgomp -Xcompiler -w
````

to 

````shell
hipcc -w -o bin/hip-comp src/cms_hip.cpp -lgomp 
````

then 

````shell
bash cms_install.sh
````

**Warnings may appear during compilation, but they can be ignored.**

**d. Verify Compilation Success**

```
source ~/.bashrc

cd cms/bin
hip-comp -h
```

If a software usage prompt is displayed, the software has been successfully installed.

Examples of program operation and simple usage instructions are in the "example" folder

## Usage

#### **Input data format**

CMS requires Microbial abundance table (e.g. OTU table) to calculate the distances among microbiomes. Currently CMS supports OTUs of Greengenes (v13-8), Greengenes2, SILVA, and RefSeq. More reference database will be released soon. The input example is as follows:

```
            OTU_1   OTU_2   OTU_3   ...     OTU_M
Sample_1    0.1     0        0.1    ...     0.2
Sample_2    0.2     0.1      0      ...     0.1
...         ...     ...      ...    ...     ...
Sample_N    0       0.3      0.2    ...     0.3
```

This input can be generated by Parallel-Meta suite(PMS). URL for PMS is <https://github.com/qdu-bioinfo/parallel-meta-suite/tree/main>

#### Using NVIDIA GPU for Computation

In this version, CMS assumes that **all GPUs on a single server have identical specifications** and **will utilize all GPU resources on single node**. Therefore, before running the program, please ensure that no other critical tasks are being executed on the server node to avoid disrupting other operations. Command of using NVIDIA GPU for computation is as follows:

```
cuda-comp [option] value
Option:
-D (upper) ref database default is G (GreenGenes-13-8 (16S rRNA, 97% level)), or S (SILVA (16S rRNA, 97% level)), or O (Oral_Core (16S rRNA, 97% level)), or C (GreenGenes-13-8 (16S rRNA, 99% level)), or R (GreenGenes-2 (16S rRNA)), or Q (Refseq (16S rRNA, 100% level)), or E (SILVA (18S rRNA, 97% level)), or T (ITS (ITS1, 97% level))
-T (upper) Input OTU count table (*.OTU.Count) for multi-sample comparison
-o Output file, default is to output on screen
-h Help
```

E.g. Calculate the similarity matrix of the **"taxa.OTU.Count" **file in the /home directory and output the result to "result.dist" using Greengenes 13-8 referance database.

```
cuda-comp -T /home/taxa.OTU.Count -o result.dist -D G
```

#### Using original CPU version Meta-Storms for Computation

The CMS integrates the original version of Meta-Storms, and the commands used are as follows:

```
comp [option] value
-D (upper) ref database, default is G (GreenGenes-13-8 (16S rRNA, 97% level)), or S (SILVA (16S rRNA, 97% level)), or O (Oral_Core (16S rRNA, 97% level)), or C (GreenGenes-13-8 (16S rRNA, 99% level)), or R (GreenGenes-2 (16S rRNA)), or Q (Refseq (16S rRNA, 100% level)), or E (SILVA (18S rRNA, 97% level)), or T (ITS (ITS1, 97% level))
        [Input options, required]
          -i Two samples path for single sample comparison
        or
          -l Input files list for multi-sample comparison
          -p List files path prefix [Optional for -l]
        or
          -T (upper) Input OTU count table (*.OTU.Count) for multi-sample comparison
        [Output options]
          -o Output file, default is to output on screen
          -d Output format, distance (T) or similarity (F), default is T
          -P (upper) Print heatmap and clusters, T(rue) or F(alse), default is F
        [Other options]
          -M (upper) Distance Metric, 0: Meta-Storms; 1: Meta-Storms-unweighted; 2: Cosine; 3: Euclidean; 4: Jensen-Shannon; 5: Bray-Curtis, default is 0
          -r rRNA copy number correction, T(rue) or F(alse), default is T
          -c Cluster number, default is 2 [Optional for -P]
          -t Number of thread, default is auto
          -h Help
```

E.g. Calculate the similarity matrix of the **"taxa.OTU.Count" **file in the /home directory and output the result to "result.dist" using Greengenes 13-8 referance database.

```
comp -T /home/taxa.OTU.Count -o result.dist -D G
```

#### Using AMD GPU for Computation

The current version of the program assumes that **all GPUs on a single server have identical specifications** and **will utilize all GPU resources on single node**. Therefore, before running the program, please ensure that no other critical tasks are being executed on the server node to avoid disrupting other operations. Command of using NVIDIA GPU for computation is as follows:

```
hip-comp [option] value
Option:
-D (upper) ref database default is G (GreenGenes-13-8 (16S rRNA, 97% level)), or S (SILVA (16S rRNA, 97% level)), or O (Oral_Core (16S rRNA, 97% level)), or C (GreenGenes-13-8 (16S rRNA, 99% level)), or R (GreenGenes-2 (16S rRNA)), or Q (Refseq (16S rRNA, 100% level)), or E (SILVA (18S rRNA, 97% level)), or T (ITS (ITS1, 97% level))
-T (upper) Input OTU count table (*.OTU.Count) for multi-sample comparison
-o Output file, default is to output on screen
-h Help
```

E.g. Calculate the similarity matrix of the **"taxa.OTU.Count" **file in the /home directory and output the result to "result.dist" using Greengenes 13-8 referance database.

```
hip-comp -T /home/taxa.OTU.Count -o result.dist -D G
```

# Experiment dataset

MSE dataset contains 200,000 randomly selected microbiomes annotated by Greengenes (v13-8).

NCBI dataset contains 10,000 randomly selected microbiomes annotated by Greengenes (v13-8), Greengenes2, SILVA, and RefSeq respectively.

ENV dataset contains 344 randomly selected microbiomes in gut, oral, soil, marine, plant and river environment  annotated by Greengenes (v13-8).

These three datasets can be downloaded from [this address](http://bioinfo-ai.cn/downloads/Released_Software/Coordinated_Meta_Storms/data/).

Due to the large file size, all dataset files have been compressed. Please refer to the **readme files** in each dataset folder for decompression operations. **Fully decompressing the dataset requires 80GB **of free space on the hard drive. If you want to **reproduce all the experiments, it is recommended to leave 1TB** of free space on the hard drive

# Contact

Any problem please contact Coordinated Meta-Storms development team 

```
Su Xiaoquan	E-mail: suxq@qdu.edu.cn
```

