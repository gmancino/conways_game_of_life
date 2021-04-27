# CUDA and CUDA + MPI
---

## Repo summary

CUDA and MPI are the backbone of high performance computing on the [AiMOS](https://cci.rpi.edu/aimos) supercomputer at RPI. This repository contains the homework solutions for Dr. Carother's _Parallel Computing_ course taught in the Spring of 2021 at RPI (verbal permission for posting received on 04-27-2021). All code is simulating a parallel version of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), where increasing levels of parallelism is introduced via CUDA and then MPI.

## What's in the repo

+ `CUDA`: contains the `highlife.cu` file which is a CUDA implementation of Conway's Game of Life (truly, it is an implementation of [Highlife](https://en.wikipedia.org/wiki/Highlife_(cellular_automaton)))
+ `CUDA_MPI`: contains a modified version of the `highlife.cu`, called `highlifeCuda.cu`, that is linked with the `highlifeMpi.c` via `extern` calls. This folder utilizes MPI by communicating "ghost" rows between GPUs

Both folders contain respective `Makefile`'s that compile the code once a user is on AiMOS via the `make all` command. Prompts describing the initial problems to be solved can be found in `CUDA/assignment2.pdf` and `CUDA_MPI/assignment3.pdf` for the CUDA and the CUDA + MPI code, respectively.

## How to run on AiMOS

To run on multiple computing nodes, each with multiple GPUs, make a file called `slurmSpectrum.sh` with the following code (this would be for the `CUDA_MPI` example):

```
#!/bin/bash -x

# ----- GATHER SLURM INFORMATION FOR RUNNING ON MULTIPLE COMPUTE NODES ----- #
if [ "x$SLURM_NPROCS" = "x" ]
then
    if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
    then
        SLURM_NTASKS_PER_NODE=1
    fi
    SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
    else
        if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
        then
            SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
        fi
fi

# ----- SET UP TEMPORARY ENVIRONMENT TO DO COMPUTATIONS IN ----- #
srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID > /tmp/tmp.$SLURM_JOB_ID mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

# ----- LOAD MODULES ----- #
module load xl_r spectrum-mpi cuda/10.2

# ----- RUN COMMAND ----- #
mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS /gpfs/u/home/<Project>/<Project_user>/<barn or scratch>/<path to file> <params>

rm /tmp/hosts.$SLURM_JOB_ID
```

Make sure to specify:
+ `/gpfs/u/home/<Project>/<Project_user>/<barn or scratch>/<path to file>`: as the appropriate file location
+ `<params>`: as the input to the specific programs (e.g. `world_pattern`, `world_size`, `iterations `, and `thread_count`)

Then a batch command such as:

```
sbatch -N <num_nodes> --ntasks-per-node=<num_gpus> --gres=gpu:<num_gpus> -t <hours:minutes:seconds> ./slurmSpectrum.sh
```

will run the programs according to the `<params>` specified in the `slurmSpectrum.sh` file with `<num_nodes>` computing nodes and `<num_gpus>` GPUs per compute node.
