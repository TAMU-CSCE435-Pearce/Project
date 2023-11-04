# CSCE435 Project
# Build instructions
    Initialize cmake build:
    $ . build.sh

    Re-build code:
    $ . make

    Run the batch file, giving array size, number of processors, and array fill type (0 for random, 1 for sorted, 2 for reverse sorted)
    $ sbatch project.grace_job <a> <p> <t>