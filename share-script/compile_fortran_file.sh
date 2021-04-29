#!/bin/bash

for file in ./* ; do
   if [[ "$file" == *.f90 ]] ; then
      tempstr=`basename $file`
      name=`echo $tempstr | cut -d'.' -f 1`
      echo $name
#      gfortran-4.6 $name.f90 -o $name.x
      ifort $name.f90 -o $name.x
   fi
done
rm *.mod
