#!/bin/bash

for((i=1;i<=10;i++));  
do   
  srun -n 1 ./SAXPY i 
done  
