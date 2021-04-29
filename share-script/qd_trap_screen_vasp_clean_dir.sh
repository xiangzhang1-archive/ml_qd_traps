#!/bin/bash

   str1=`pwd`
   sysname_full=`basename $str1`
   sysname=`echo $str2 | cut -d'.' -f 1`

   echo 'cleaning dir */relax...'
   declare -a del_list=('CHG' 'CHGCAR')
   echo 'delete files:'
   echo ${del_list[@]}
   for ((cid=1; cid<=1000; cid++)); do
      file1="./data-vasp/struct/$sysname_full.cf-$cid.vasp"
      if [[ -s $file1 ]] ; then
         echo $cid
         for file2 in ${del_list[@]}; do
            file3="./cf-$cid/relax/$file2"
            if [[ -s $file3 ]]; then
               rm $file3
            fi
         done
      fi
   done
   echo ' '
         
   echo 'cleaning dir */en...'
   declare -a del_list1=('LOCPOT' 'CHG' 'vasprun.xml')
   declare -a del_list2=('CHG' 'CHGCAR' 'LOCPOT' 'WAVECAR' 'vasprun.xml')
   echo 'delete files:'
   echo ${del_list2[@]}
   for ((cid=1; cid<=1000; cid++)); do
      file1="./data-vasp/eigval/$sysname_full.cf-$cid.eigval_near_ef.dat"
      file4="./data-vasp/pdos/$sysname_full.cf-$cid.doscar.dat"
      file5="./data-vasp/pdos/$sysname_full.cf-$cid.procar.dat"
      if [[ -s $file1 ]] ; then
         echo $cid
         for file2 in ${del_list1[@]}; do
            file3="./cf-$cid/en/$file2"
            if [[ -s $file3 ]]; then
               rm $file3
            fi
         done
         if [[ ( -s $file4 ) && ( -s $file5 ) ]] ; then
            for file6 in ${del_list2[@]}; do
            file7="./cf-$cid/en/$file6"
            if [[ -s $file7 ]]; then
               rm $file7
            fi
            done
         fi
      fi
   done
   echo ' '

   echo 'cleaning dir */pdos...'
   declare -a del_list=('CHG' 'CHGCAR' 'LOCPOT' 'WAVECAR' 'vasprun.xml')
   echo 'delete files:'
   echo ${del_list[@]}
   for ((cid=1; cid<=1000; cid++)); do
      file1="./data-vasp/pdos/$sysname_full.cf-$cid.doscar.dat"
      file2="./data-vasp/pdos/$sysname_full.cf-$cid.procar.dat"
      if [[ ( -s $file1 ) && ( -s $file2 ) ]] ; then
         echo $cid
         for file3 in ${del_list[@]}; do
            file4="./cf-$cid/pdos/$file3"
            if [[ -e $file4 ]]; then
               rm $file4
            fi
         done
      fi
   done
   echo ' '
