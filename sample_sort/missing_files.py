import os

# Configuration settings (same as in your job submission script)
input_types = ["s", "r", "rs", "p"]
input_sizes = [16, 18, 20, 22, 24, 26, 28]
mpi_procs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cuda_threads = [64, 128, 256, 512, 1024]
cali_dir = "./cali"

# Function to check if file exists and is not empty
def is_missing_or_empty(filename):
    return not os.path.exists(filename) or os.path.getsize(filename) == 0

# Lists to track missing or empty files
missing_mpi_files = []
missing_cuda_files = []

# Check MPI job output files
for input_type in input_types:
    for size_power in input_sizes:
        size = 2**size_power
        for procs in mpi_procs:
            filename = os.path.join(cali_dir, "sample-mpi-p{}-a{}-i{}.cali".format(procs, size, input_type))
            if is_missing_or_empty(filename):
                missing_mpi_files.append("p{}-a{}-i{}".format(procs, size, input_type))

# Check CUDA job output files
for input_type in input_types:
    for size_power in input_sizes:
        size = 2**size_power
        for threads in cuda_threads:
            filename = os.path.join(cali_dir, "sample-cuda-t{}-v{}-i{}.cali".format(threads, size, input_type))
            if is_missing_or_empty(filename):
                missing_cuda_files.append("t{}-v{}-i{}".format(threads, size, input_type))

# Print results
print "Summary of Missing or Empty Output Files:"

print "\nMissing/Empty MPI Files:"
if missing_mpi_files:
    print "\n".join(missing_mpi_files)
    print "\nCount: {}".format(len(missing_mpi_files))
else:
    print "None"

print "\nMissing/Empty CUDA Files:"
if missing_cuda_files:
    print "\n".join(missing_cuda_files)
    print "\nCount: {}".format(len(missing_cuda_files))
else:
    print "None"
