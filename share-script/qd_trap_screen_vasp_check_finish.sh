#!/bin/bash

   cid_start=1
   cid_end=100

   str1=`pwd`
   sysname_full=`basename $str1`
   sysname=`echo $str2 | cut -d'.' -f 1`

   echo 'checking structure file...'
   istat=0
   for ((cid=$cid_start; cid<=$cid_end; cid++)); do
      file1="./data-vasp/struct/$sysname_full.cf-$cid.vasp"
      if [[ ! -s $file1 ]] ; then
         echo "$cid : missing"
         istat=1
      fi
   done
   if [[ $istat == 0 ]] ; then
      echo 'pass test!'
   fi
   echo ' '
         
   echo 'checking eigenvalue file...'
   istat=0
   for ((cid=$cid_start; cid<=$cid_end; cid++)); do
      file1="./data-vasp/eigval/$sysname_full.cf-$cid.eigval_near_ef.dat"
      if [[ ! -s $file1 ]] ; then
         echo "$cid : missing"
         istat=1
      fi
   done
   if [[ $istat == 0 ]] ; then
      echo 'pass test!'
   fi
   echo ' '
         
   echo 'checking doscar file...'
   istat=0
   for ((cid=$cid_start; cid<=$cid_end; cid++)); do
      file1="./data-vasp/pdos/$sysname_full.cf-$cid.doscar.dat"
      if [[ ! -s $file1 ]] ; then
         echo "$cid : missing"
         istat=1
      fi
   done
   if [[ $istat == 0 ]] ; then
      echo 'pass test!'
   fi
   echo ' '
         
   echo 'checking procar file...'
   istat=0
   for ((cid=$cid_start; cid<=$cid_end; cid++)); do
      file1="./data-vasp/pdos/$sysname_full.cf-$cid.procar.dat"
      if [[ ! -s $file1 ]] ; then
         echo "$cid : missing"
         istat=1
      fi
   done
   if [[ $istat == 0 ]] ; then
      echo 'pass test!'
   fi
   echo ' '
         
