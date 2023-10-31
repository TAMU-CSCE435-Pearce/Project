#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* array_fill = "array_fill";

void parallel_array_fill(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(array_fill);
    
    // Calculate local size based on rank and array size
    int local_size = NUM_VALS / num_procs;
    int start = rank * local_size;
    int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

    local_size = end - start;

    // Print process segment of array
    printf("start: %d, end: %d, local_size:%d\n", start, end, local_size);

    float *local_values = (float *)malloc(local_size * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < local_size; ++i) 
    {
        local_values[i] = (float)rand() / (float)RAND_MAX;
    }

    // Gather local portions into global array
    MPI_Gather(local_values, local_size, MPI_FLOAT, values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(local_values);

    CALI_MARK_END(array_fill);
}

int main(int argc, char* argv[]) 
{
    CALI_CXX_MARK_FUNCTION;

    int NUM_VALS = atoi(argv[1]);

    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize array to be sorted 
    float *values = (float *)malloc(NUM_VALS * sizeof(float));

    // Fill the local portions of the array then gather into values (NUM_VALS MUST BE DIVISIBLE BY num_procs)
    parallel_array_fill(NUM_VALS, values, num_procs, rank);

    if (rank == 0) {
        // Use values array as process 0
    }

    free(values);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}