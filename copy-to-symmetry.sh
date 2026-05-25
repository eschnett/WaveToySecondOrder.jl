#!/bin/sh

rsync \
    --archive \
    --compress \
    --partial \
    --progress \
    --exclude .git \
    --exclude '*~' \
    --exclude '*.ncu-rep' \
    --exclude '*.nsys-rep' \
    --exclude '*.o' \
    --exclude '*.ptx' \
    --exclude '*.sass' \
    --exclude Manifest\*.toml \
    --exclude wavetoy \
    ~/src/jl/WaveToySecondOrder/ \
    symmetry:src/jl/WaveToySecondOrder/
