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

**Coordinated Meta-Storms (CMS)**, a method for large-scale microbiome distance calculation that utilizes multiple GPUs in a coordinated manner. Building on the Meta-Storms algorithm, CMS introduces a dynamic data decomposition strategy and a multi-GPU architecture. Additionally, CMS has been optimized for various computing platforms, enabling the analysis of million-level microbiomes. 

## System requirement and dependency

We have successfully completed the compilation and testing of the code. The NVIDIA GPU code runs in a CUDA 12.2 environment, while the CPU code was compiled and tested using G++ 9.4.0. The GPU tests were conducted on Ubuntu 22.04, and the CPU tests on CentOS 7.6, both demonstrating stable and reliable performance.

## Installation guide

### Download and Install

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

#### CMS Download and Install

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

### Data Format

CMS now accepts OTU count table as input,  the output of the **PM-select-taxa** command from the Parallel Meta suite(PMS) is highly recommend to generate this file. URL for PMS is <https://github.com/qdu-bioinfo/parallel-meta-suite/tree/main>

## Usage

### Using NVIDIA GPU for Computation

The current version of the program assumes that all GPUs on a single server node have identical specifications and **will utilize all GPU resources on single node**. Therefore, before running the program, please ensure that no other critical tasks are being executed on the server node to avoid disrupting other operations.

```
cuda-comp [option] value
Option:
-D (upper) ref database default is G (GreenGenes-13-8 (16S rRNA, 97% level)), or S (SILVA (16S rRNA, 97% level)), or O (Oral_Core (16S rRNA, 97% level)), or C (GreenGenes-13-8 (16S rRNA, 99% level)), or R (GreenGenes-2 (16S rRNA)), or Q (Refseq (16S rRNA, 100% level)), or E (SILVA (18S rRNA, 97% level)), or T (ITS (ITS1, 97% level))
-T (upper) Input OTU count table (*.OTU.Count) for multi-sample comparison
-o Output file, default is to output on screen
-h Help
```

E.g. Calculate the similarity matrix of the **"taxa.OTU.Count" **file in the /home directory and output the result to "result.dist"

```
cuda-comp -T /home/taxa.OTU.Count -o result.dist -D G
```

#### Using CPU for Computation

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

E.g.

```
comp -T /home/taxa.OTU.Count -o result.dist -D G
```

# Contact

Any problem please contact Coordinated Meta-Storms development team 

```
Su Xiaoquan	E-mail: suxq@qdu.edu.cn
```

