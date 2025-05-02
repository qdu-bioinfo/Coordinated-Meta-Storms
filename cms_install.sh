#!/bin/bash

### Coordinated Meta-Storms Installer
### Updated: January 2025
### Bioinformatics Group, College of Computer Science and Technology, Qingdao University
### Code by: Minan Wang, Xiaoquan Su

echo "**Installation Start**"

##Users can change the default environment variables configuration file here
if [[ $SHELL = '/bin/zsh' ]];
then
        PATH_File=~/.zshrc
        if [ ! -f "$PATH_File" ]
        then
                PATH_File=~/.zsh_profile
                if [ ! -f "$PATH_File" ]
                then
                        touch $PATH_File
                fi
        fi
else
        PATH_File=~/.bashrc
        if [ ! -f "$PATH_File" ]
        then
                PATH_File=~/.bash_profile
                if [ ! -f "$PATH_File" ]
                then
                        touch $PATH_File
                fi
        fi
fi

PM_PATH=$(pwd)
Sys_ver=$(uname)

# doesn't support macOS 
if [ "$Sys_ver" = "Darwin" ]; then
    echo "This installer does not support macOS. Please run it on a Linux system."
    exit 1
fi

### check old environment variable
Check_old_pm=$(grep "export ParallelMETA" "$PATH_File" | awk -F '=' '{print $1}')
Check_old_path=$(grep "ParallelMETA/bin" "$PATH_File" | sed 's/\(.\).*/\1/' | awk '{if($1!="#"){print "True";}}')
Add_Part="####DisabledbyParallelMeta3####"

### code complie
BUILD_MODE=$1  # optional parameter：hip
echo "\n**CMS Source Build**"

if [ -f "Makefile" ]; then
    if [ "$BUILD_MODE" = "hip" ]; then
        echo "**Building GCC + HIP version**"
        make clean
        make MODE=hip
    else
        echo "**Building GCC + CUDA version**"
        make clean
        make
    fi
    echo "\n**Build Complete**"
else
    echo "**Binary package detected, skipping compilation**"
fi

### write and update environment variable
if [ "$Check_old_pm" != "" ]; then
    Checking=$(grep ^export\ ParallelMETA "$PATH_File" | awk -F '=' '{print $2}')
    if [ "$Checking" != "$PM_PATH" ]; then
        sed -i "s/^export ParallelMETA/$Add_Part &/g" "$PATH_File"
        sed -i "/$Add_Part export ParallelMETA/a export ParallelMETA=$PM_PATH" "$PATH_File"
    fi
else
    echo "export ParallelMETA=$PM_PATH" >> "$PATH_File"
fi

if [ "$Check_old_path" = "" ]; then
    echo "export PATH=\$PATH:\$ParallelMETA/bin" >> "$PATH_File"
fi

source $PATH_File
echo "\n**Environment Variables Configuration Complete**"

### end
echo "\n**CMS Installation Complete**"
echo "**Example dataset with demo script is available in 'example/' directory**"

