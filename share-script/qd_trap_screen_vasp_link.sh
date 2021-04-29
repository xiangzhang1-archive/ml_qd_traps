#!/bin/bash

   share_psp_folder='/home/huashli/Vasp/qd_trap_ml/share-psp'
   share_file_folder='/home/huashli/Vasp/qd_trap_ml/share-script'
   cluster=nanaimo

   str1=`pwd`
   str2=`basename $str1`
   sysname=`echo $str2 | cut -d'.' -f 1`
   srname='qd_trap_screen_vasp'
   for (( i=1;i<=100;i++ )) ; do
      file="$srname"'_'"step$i.sh.template"
      if [[ -e "$file" ]] ; then
         echo $file
         sed "s@faaa@$share_psp_folder@g" $file > temp1
         sed "s@fbbb@$share_file_folder@g" temp1 > temp2
         sed "s/fccc/$cluster/g" temp2 > temp3
         sed "s/fggg/$sysname/g" temp3 > "$srname"'_'"step$i.sh"
         rm temp1 temp2 temp3
      fi
      chmod 777 *.sh
   done

   echo " "
   echo "step1: relax structure in folder "rel"."
   echo "step2: calculate eigenvalues in folder "en"."
   echo "step3: extract information in folder "en"."
   echo "step4: calculate pdos in folder "pdos"."
   echo "step5: extract information in folder "pdos"."
   echo "step6: calculate wavefunction step1 in folder "wfn"."
   echo "step7: calculate wavefunction step2 in folder "wfn"."




