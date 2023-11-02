#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* array_fill_name = "array_fill";
const char* sort_check_name = "sort_check";

void parallel_array_fill(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(array_fill_name);
    
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

    CALI_MARK_END(array_fill_name);
}

bool sort_check(float *local_values, int local_size)
{
    for (int i = 1; i < local_size; i++)
    {
        if (local_values[i - 1] > local_values[i]) 
        {
            return false;
        }
    }
    return true;
}

void parallel_sort_check(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(sort_check_name);

    // Calculate local size based on rank and array size
    int local_size = NUM_VALS / num_procs;
    int start = rank * local_size;
    int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

    local_size = end - start;

    float* local_values = (float*)malloc(local_size * sizeof(float));

    // Scatter the array among processes
    MPI_Scatter(values, local_size, MPI_FLOAT, local_values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Print process segment of array
    printf("start: %d, end: %d, local_size:%d\n", start, end, local_size);

    bool local_sorted = sort_check(local_values, local_size);

    // Gather local portions into global array
    bool all_sorted;
    MPI_Allreduce(&local_sorted, &all_sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        if (all_sorted) 
        {
            // Check if each segment of values is sorted
            float cur_largest = values[local_size - 1];
            for (int i = 1; i < NUM_VALS/local_size; i++)
            {
                if (values[i*local_size] > cur_largest)
                {
                    cur_largest = values[(i+1)*local_size - 1];
                }
                else
                {
                    all_sorted = false;
                    printf("The entire array is not sorted.");
                    break;
                }
            }
            printf("The entire array is sorted.");
        }
        else
        {
            printf("The entire array is not sorted.");
        }
    }

    free(local_values);

    CALI_MARK_END(sort_check_name);
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

    // Check if values is sorted
    parallel_sort_check(NUM_VALS, values, num_procs, rank);

    free(values);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}