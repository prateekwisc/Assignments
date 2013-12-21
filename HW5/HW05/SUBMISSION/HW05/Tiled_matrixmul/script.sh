#!/bin/bash
#PBS -n HW05/Tiled_matrixmul
#PBS -l nodes=1:gpus=1,walltime=10:00:00
#PBS -d /home/pkgupta3/prateek/HW05/Tiled_matrixmul
for i in 11344
do
./Testmul $i $i > $i.log
done
