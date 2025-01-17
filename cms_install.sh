###Coordinated Meta-Storms installer
###Updated at January, 2025
###Bioinformatics Group, College of Computer Science and Technology, Qingdao University
###Code by: Minan Wang, Xiaoquan Su, Honglei Wang, Gongchao Jing
#!/bin/bash

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

PM_PATH=`pwd`
Sys_ver=`uname`
### Check if the system is macOS (Darwin) and exit if true ###
if [ "$Sys_ver" = "Darwin" ]; then
    echo "This installer does not support macOS. Please run it on a Linux system."
    exit 1
fi

###Checking that environment variable of Parallel-META exists###
Check_old_pm=`grep "export ParallelMETA"  $PATH_File|awk -F '=' '{print $1}'`
Check_old_path=`grep "ParallelMETA/bin"  $PATH_File |sed 's/\(.\).*/\1/' |awk '{if($1!="#"){print "Ture";}}'`
Add_Part="####DisabledbyParallelMeta3####"
echo "**Installation**"

###Build source code for src package###
if [ -f "Makefile" ]
   then
       echo -e "\n**CMS src package**"
       make
       echo -e "\n**Build Complete**"
else
   echo -e "\n**CMS bin package**"
fi

###Configure environment variables###
if [ "$Check_old_pm" != "" ]
   then
      Checking=`grep ^export\ ParallelMETA  $PATH_File|awk -F '=' '{print $2}'`
      if [ "$Checking" != "$PM_PATH" ]
         then
             sed -i "s/^export\ ParallelMETA/$Add_Part\ &/g" $PATH_File
             sed -i "/$Add_Part\ export\ ParallelMETA/a export\ ParallelMETA=$PM_PATH" $PATH_File
         fi    
elif [ "$Check_old_pm" = "" ]
    then
      echo "export ParallelMETA="${PM_PATH} >> $PATH_File
fi
if [ "$Check_old_path" = "" ]
    then
      echo "export PATH=\$PATH:\$ParallelMETA/bin" >> $PATH_File
	  
###Source the environment variable file###	  
source $PATH_File
echo -e "\n**Environment Variables Configuration Complete**"
fi


###End
echo -e "\n**CMS Installation Complete**"
echo -e "\n**An example dataset with demo script is available in \"example\"**"
