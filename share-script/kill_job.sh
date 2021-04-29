#!/bin/bash

id_start=4669
id_end=4694

for (( id=$id_start; id<=$id_end; id++ )) ; do
    scontrol hold $id
#   scancel $id
done
