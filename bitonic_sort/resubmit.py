import os
import subprocess

# Configuration settings (same as in your job submission script)
input_types = ["s", "r", "rs", "p"]
input_sizes = [16, 18, 20, 22, 24, 26, 28]
mpi_procs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cuda_threads = [64, 128, 256, 512, 1024]
cali_dir = "./cali"
procs_per_node = 64

mpi_job_script = "./mpi/bitonic.grace_job"
cuda_job_script = "./cuda/bitonic.grace_job"


# Function to check if file exists and is not empty
def is_missing_or_empty(filename):
    return not os.path.exists(filename) or os.path.getsize(filename) == 0


# Function to resubmit MPI job
def resubmit_mpi_job(size, procs, input_type):
    nodes = (procs + procs_per_node - 1) // procs_per_node
    mpi_command = [
        "sbatch",
        "--nodes={}".format(nodes),
        mpi_job_script,
        str(size),
        str(procs),
        input_type,
        cali_dir,
    ]
    subprocess.call(mpi_command)


# Function to resubmit CUDA job
def resubmit_cuda_job(size, threads, input_type):
    cuda_command = [
        "sbatch",
        cuda_job_script,
        str(size),
        str(threads),
        input_type,
        cali_dir,
    ]
    subprocess.call(cuda_command)


# Check and resubmit missing or empty MPI and CUDA files
for input_type in input_types:
    for size_power in input_sizes:
        size = 2**size_power
        for procs in mpi_procs:
            mpi_filename = os.path.join(
                cali_dir, "bitonic-mpi-p{}-a{}-i{}.cali".format(procs, size, input_type)
            )
            if is_missing_or_empty(mpi_filename):
                resubmit_mpi_job(size, procs, input_type)

        for threads in cuda_threads:
            cuda_filename = os.path.join(
                cali_dir,
                "bitonic-cuda-t{}-v{}-i{}.cali".format(threads, size, input_type),
            )
            if is_missing_or_empty(cuda_filename):
                resubmit_cuda_job(size, threads, input_type)

print("All missing or empty files have been resubmitted.")
