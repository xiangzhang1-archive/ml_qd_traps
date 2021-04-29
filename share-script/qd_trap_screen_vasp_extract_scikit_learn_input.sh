#!/bin/bash

   infodir='data-scikit'
   imax=1000
   if [[ ! -d $infodir ]] ; then
      mkdir $infodir
   fi
   str1=`pwd`
   sysname_full=`basename $str1`
   sysname=`echo $sysname_full | cut -d'.' -f 1`
   cd ./$infodir/
   echo "prepare data file for scikit-learn"
   
   flag='struct-relax-poscar'
   echo " "
   echo "extracting relaxed structures..."
   nsample=0
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/struct/$sysname_full.cf-$i.vasp"
      if [[ -e $file1 ]] ; then
         nsample=$(($nsample + 1))
         id_max=$i
      fi
   done
   echo "$nsample   samples"
   file2="./$sysname_full.$flag.dat"
   echo "$nsample   samples" > $file2
   echo " " >> $file2
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/struct/$sysname_full.cf-$i.vasp"
      if [[ -e $file1 ]] ; then
         echo "------------------------------------------------------------------------" >> $file2
         echo "id  $i " >> $file2
         echo " " >> $file2
         elem_natom=`head -n 7 $file1 | tail -n 1`
         nelem=`echo $elem_natom | wc -w`
         natom=0
         for ((j=1; j<=$nelem; j++)) ; do
            tempint=`echo $elem_natom | cut -d' ' -f $j`
            natom=$(($natom + $tempint))
         done
         nline=$(($natom + 8))
         head -n $nline $file1 >> $file2
      fi
      echo " " >> $file2
   done

   flag='struct-origin-poscar'
   echo " "
   echo "extracting original structures..."
   nsample=0
   for ((i=1; i<=$imax; i++)) ; do
      file1="../struct-gen/struct-poscar/$sysname.cf-$i.vasp"
      if [[ -e $file1 ]] ; then
         nsample=$(($nsample + 1))
         id_max=$i
      fi
   done
   echo "$nsample   samples"
   file2="./$sysname_full.$flag.dat"
   echo "$nsample   samples" > $file2
   echo " " >> $file2
   for ((i=1; i<=$imax; i++)) ; do
      file1="../struct-gen/struct-poscar/$sysname.cf-$i.vasp"
      if [[ -e $file1 ]] ; then
         echo "------------------------------------------------------------------------" >> $file2
         echo "id  $i " >> $file2
         echo " " >> $file2
         elem_natom=`head -n 7 $file1 | tail -n 1`
         nelem=`echo $elem_natom | wc -w`
         natom=0
         for ((j=1; j<=$nelem; j++)) ; do
            tempint=`echo $elem_natom | cut -d' ' -f $j`
            natom=$(($natom + $tempint))
         done
         nline=$(($natom + 8))
         head -n $nline $file1 >> $file2
      fi
      echo " " >> $file2
   done

   flag='en-eigval-vasp'
   echo " "
   echo "extracting energies and eigenvalues..."
   nsample=0
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/eigval/$sysname_full.cf-$i.eigval_near_ef.dat"
      if [[ -e $file1 ]] ; then
         nsample=$(($nsample + 1))
         id_max=$i
      fi
   done
   echo "$nsample   samples"
   file2="./$sysname_full.$flag.dat"
   echo "$nsample   samples" > $file2
   echo " " >> $file2
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/eigval/$sysname_full.cf-$i.eigval_near_ef.dat"
      if [[ -e $file1 ]] ; then
         echo "------------------------------------------------------------------------" >> $file2
         echo "id  $i " >> $file2
         echo " " >> $file2
         echo "eigenvalues (eV), occupation" >> $file2
         cat $file1 >> $file2
         str1=`head -n 1 $file1`
         band_id_start=`echo $str1 | cut -d ' ' -f 1`
         str3=`grep -n 'vacuum level' $file1`
         lid=`echo $str3 | cut -d ':' -f 1`
         lid=$(($lid - 2))
         str2=`head -n $lid $file1 | tail -n 1`
         band_id_end=`echo $str2 | cut -d ' ' -f 1`
      fi
      echo " " >> $file2
   done

   flag='procar-vasp'
   echo " "
   echo "extracting PDOS..."
   nsample=0
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/pdos/$sysname_full.cf-$i.procar.dat"
      if [[ -e $file1 ]] ; then
         nsample=$(($nsample + 1))
         id_max=$i
      fi
   done
   echo "$nsample   samples"
   file2="./$sysname_full.$flag.dat"
   echo "$nsample   samples" > $file2
   echo " " >> $file2
   for ((i=1; i<=$imax; i++)) ; do
      file1="../data-vasp/pdos/$sysname_full.cf-$i.procar.dat"
      if [[ -e $file1 ]] ; then
         echo "------------------------------------------------------------------------" >> $file2
         echo "id  $i " >> $file2
         echo " " >> $file2
         lid_end=$(( 5 + ($natom + 5) * $band_id_end ))
         nline=$(( ($natom + 5) * ($band_id_end - $band_id_start + 1) ))
         head -n $lid_end $file1 | tail -n $nline >> $file2
      fi
      echo " " >> $file2
   done

   flag='system-info'
   echo " "
   echo "extracting system information..."
   nsample=0
   for ((i=1; i<=$imax; i++)) ; do
      file1="../struct-gen/struct-info/$sysname.cf-$i.info.dat"
      if [[ -e $file1 ]] ; then
         nsample=$(($nsample + 1))
         id_max=$i
      fi
   done
   echo "$nsample   samples"
   file2="./$sysname_full.$flag.dat"
   echo "$nsample   samples" > $file2
   echo " " >> $file2
   for ((i=1; i<=$imax; i++)) ; do
      file1="../struct-gen/struct-info/$sysname.cf-$i.info.dat"
      if [[ -e $file1 ]] ; then
         echo "------------------------------------------------------------------------" >> $file2
         echo "id  $i " >> $file2
         echo " " >> $file2
         cat $file1 >> $file2
      fi
      echo " " >> $file2
   done




