import subprocess

# Configuration settings
mpi_job_script = "./mpi/mpi.grace_job"
cuda_job_script = "./cuda/cuda.grace_job"
cali_dir = "./cali"
input_types = ["s", "r", "rs", "p"]
input_sizes = [16, 18, 20, 22, 24, 26, 28]
mpi_procs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cuda_threads = [64, 128, 256, 512, 1024]
procs_per_node = 64

# Ensure the Caliper output directory exists
subprocess.call(["mkdir", "-p", cali_dir])

# Submit MPI and CUDA jobs
for input_type in input_types:
    for size_power in input_sizes:
        size = 2**size_power

        # MPI jobs
        for procs in mpi_procs:
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

        # CUDA jobs
        for threads in cuda_threads:
            cuda_command = [
                "sbatch",
                cuda_job_script,
                str(size),
                str(threads),
                input_type,
                cali_dir,
            ]
            subprocess.call(cuda_command)

print("All jobs have been submitted.")
