# CSCE435 Project
# Build instructions

## For MPI Algorithms:
    Enter the MPI Folder
    $ cd MPI/

    Initialize cmake build:
    $ . build.sh

    Re-build code:
    $ . make

    Run the batch file, giving array size, number of processors, 
    array fill type (0 for random, 1 for sorted, 2 for reverse sorted),
    and the sorting algorithm to run (0 for sample)
    $ sbatch project.grace_job <a> <p> <t> <s>

## For CUDA Algorithms:
    Enter the CUDA Folder
    $ cd CUDA/

    Initialize cmake build:
    $ . build.sh

    Re-build code:
    $ . make

    Run the batch file, giving array size, number of threads, 
    array fill type (0 for random, 1 for sorted, 2 for reverse sorted),
    and the sorting algorithm to run (0 for bitonic)
    $ sbatch project.grace_job <a> <p> <t> <s>