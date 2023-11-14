#!/bin/bash

# Check if a root directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <root_directory>"
    exit 1
fi

# Base directory for the project
root_dir=$1

# Directories for job files and where to put Caliper files
mpi_job_dir="${root_dir}/mpi"
cuda_job_dir="${root_dir}/cuda"
cali_dir="${root_dir}/cali"

# Ensure the Caliper output directory exists
mkdir -p ${cali_dir}

# Input types and sizes
input_types=("Sorted" "Random" "ReverseSorted" "1%perturbed")
input_sizes=(16 18 20 22 24 26 28)

# MPI processors and CUDA threads
mpi_procs=(2 4 8 16 32 64 128 256 512 1024)
cuda_threads=(64 128 256 512 1024 2048 4096)

# Loop through configurations and submit MPI jobs
for input_type in "${input_types[@]}"; do
    for size_power in "${input_sizes[@]}"; do
        size=$((2**size_power))
        
        # Submit MPI jobs
        for procs in "${mpi_procs[@]}"; do
            sbatch "${mpi_job_dir}/mpi.grace_job" $input_type $size $procs "${cali_dir}"
        done
        
        # Submit CUDA jobs
        for threads in "${cuda_threads[@]}"; do
            sbatch "${cuda_job_dir}/cuda.grace_job" $input_type $size $threads "${cali_dir}"
        done
    done
done

echo "All jobs have been submitted."
