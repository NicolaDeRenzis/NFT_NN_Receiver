### SCRIPT FOR SUBSMISSION OF SWEEP JOBS TO CPU

#!/bin/sh
#BSUB -q hpc
#BSUB -J NLIN[1-46]
#BSUB -n 1
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o results/logs-%J-%I.out
#BSUB -e results/logs_%J-%I.err
# -- end of LSF options --

mkdir -p results

module load python3/3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2
module load tensorflow/1.12-cpu-python-3.6.2

python3 hyper_AutoEncGeoShapDenseWDM.py -n $LSB_JOBINDEX