### SCRIPT FOR SUBSMISSION TO SINGLE JOB TO CPU

#!/bin/sh
#BSUB -q hpc
#BSUB -J NLIN
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
# -- end of LSF options --

#nvidia-smi

#module load python3/3.6.2
#module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
#module load scipy/0.19.1-python-3.6.2
#module load matplotlib/2.0.2-python-3.6.2
#module load tensorflow/1.12-cpu-python-3.6.2

