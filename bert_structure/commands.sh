#!/bin/bash

FILE=$1 #script .py que vai rodar
CONFIG=$2 # file com informações pra rodar o script
DATE=`date +"%Y-%m-%d_%T"`
# ARCH=$3
# AUG=$4
echo file "$FILE"
echo congif "$CONFIG"
#echo "DS" "$DS"
# echo "ARCH" "$ARCH"
# echo "AUG" "$AUG"

pwd
echo "getting in folder"
pwd 


CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_LAUNCH_BLOCKING=1
echo Running script $FILE
python3 -u BERT/$FILE.py $CONFIG | tee  "log_$DATE"   #$DS $ARCH $AUG

# echo Finished prep_data
# python3 -u main.py
# | tee results/log.txt
# pwd
# ls
# cd results
# pwd
# ls
