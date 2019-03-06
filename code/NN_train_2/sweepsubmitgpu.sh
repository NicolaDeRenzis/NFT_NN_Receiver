### SCRIPT FOR SUBSMISSION OF SWEEP JOBS TO GPU

#!/bin/sh
#BSUB -q gpuk80
#BSUB -J NLIN[1-33]  $ [i-k]: numbers saved in local var LSB_JOBINDEX
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o results/logs-%J-%I.out
#BSUB -e results/logs_%J-%I.err
# -- end of LSF options --

mkdir -p results

nvidia-smi

module load python3/3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2
module load tensorflow/1.12-gpu-python-3.6.2

# $ python3 <mySweepScript.py> -n $LSB_JOBINDEX
# The last variable is the sweep variable that will be loaded in the script
python3 sweep_inft.py -n $LSB_JOBINDEX 