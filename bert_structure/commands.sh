#!/bin/bash

FILE=$1 #script .py que vai rodar
CONFIG=$2 # file com informações pra rodar o script
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
ls

echo Running script $FILE
python3 -u BERT/$FILE.py $CONFIG    #$DS $ARCH $AUG

# echo Finished prep_data
# python3 -u main.py
# | tee results/log.txt
# pwd
# ls
# cd results
# pwd
# ls
