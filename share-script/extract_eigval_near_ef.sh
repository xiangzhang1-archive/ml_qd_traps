#!/bin/bash

   nvb=5
   ncb=nvb

   line=`grep 'NELECT' OUTCAR`
   str1=`echo $line | cut -d '=' -f 2`
   str2=`echo $str1 | cut -d ' ' -f 1`
   nelec=`echo $str2 | cut -d '.' -f 1`
   homo_id=$(($nelec / 2))
   echo "HOMO id = $homo_id"
   start_id=$(($homo_id - $nvb))
   neig=$(($nvb + $ncb))

   grep -A $neig " $start_id " EIGENVAL > eigval_near_ef.dat
   
