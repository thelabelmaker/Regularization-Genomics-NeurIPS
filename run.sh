#!/bin/bash

mkdir ./results
mkdir ./results
for i in 1 2 3
do
    for m in 'mixup' 'no_mixup'
    do
        mkdir ./results/$m
        for bs in 64 32 128
        do
            mkdir ./results/$m/batch_size=$bs
            for f in 1
            do
                mkdir ./results/$m/batch_size=$bs/deepnet_x$f
                for d in .4
                do
                    mkdir ./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d
                    mkdir ./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d/batchnorm=true
                    echo "./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d/batchnorm=true/trial-$i"
                    mkdir ./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d/batchnorm=true/trial-$i
                    python3 ./code/main.py --$m --bs $bs --f $f --dropout --d $d --batchnorm --filepath ./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d/batchnorm=true/trial-$i > ./results/$m/batch_size=$bs/deepnet_x$f/dropout=$d/batchnorm=true/trial-$i/log.out
                done
            done
        done
    done
done
