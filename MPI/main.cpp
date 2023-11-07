#include "common.h"

void parallel_array_fill(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank, int array_fill_type)
{
    CALI_MARK_BEGIN(ARRAY_FILL_NAME);
    int start = rank * local_size;
    int end = start + local_size - 1;

    // Print process segment of array
    // printf("rank: %d, start: %d, end: %d, local_size:%d\n", rank, start, end, local_size);

    if (array_fill_type == 0)
    {
        srand(time(NULL) + rank);
        for (int i = 0; i < local_size; ++i) 
        {
            local_values->push_back((float)rand() / (float)RAND_MAX);
        }
    }
    else if (array_fill_type == 1)
    {
        for (int i = 0; i < local_size; ++i) 
        {
            local_values->push_back(start + i);
        }
    }
    else if (array_fill_type == 2)
    {
        for (int i = 0; i < local_size; ++i) 
        {
            local_values->push_back(NUM_VALS - end - i);
        }
    }
    CALI_MARK_END(ARRAY_FILL_NAME);
}

bool sort_check(vector<float> local_values, int local_size)
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

void parallel_sort_check_merged(int NUM_VALS, float *values, vector<float> local_values, int local_size, int num_procs, int rank)
{
    CALI_MARK_BEGIN(SORT_CHECK_NAME);
    int start = rank * local_size;
    int end = start + local_size - 1;

    // Print process segment of array
    printf("rank: %d, start: %d, end: %d, local_size:%d\n", rank, start, end, local_size);

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
                    return;
                }
            }
            printf("The entire array is sorted.");
        }
        else
        {
            printf("The entire array is not sorted.");
        }
    }
    CALI_MARK_END(SORT_CHECK_NAME);
}

void parallel_sort_check_unmerged(int NUM_VALS, vector<float> local_values, int local_size, int num_procs, int rank)
{
    CALI_MARK_BEGIN(SORT_CHECK_NAME);
    int start = rank * local_size;
    int end = start + local_size - 1;

    // Print process segment of array
    // printf("rank: %d, start: %d, end: %d, local_size:%d\n", rank, start, end, local_size);

    MPI_Request request;
    if (rank != 0)
    {
        float to_send;
        if (!local_values.empty())
        {
            to_send = local_values[0];
        }
        else
        {
            to_send = FLT_MAX;
        }
        MPI_Isend(&to_send, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &request);
    }

    float recv_val;
    bool local_sorted = true;
    if (rank != num_procs - 1)
    {
        MPI_Recv(&recv_val, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (!local_values.empty())
        {
            if (local_values[local_values.size()-1] > recv_val)
            {
                local_sorted = false;
            }
        }
    }

    bool all_sorted;
    MPI_Allreduce(&local_sorted, &all_sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (all_sorted) 
        {
            printf("The entire array is sorted.");
        }
        else
        {
            printf("The entire array is not sorted.");
        }
    }
    CALI_MARK_END(SORT_CHECK_NAME);
}

void printArray(int NUM_VALS, float *values, vector<float> local_values, int local_size, int num_procs, int rank)
{
    // Gather local portions into global array
    MPI_Gather(local_values.data(), local_size, MPI_FLOAT, values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Printing from rank: %d\n", rank);
        int i;
        for (i = 0; i < NUM_VALS; i++)
            printf("%f, ", values[i]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) 
{
    CALI_CXX_MARK_FUNCTION;

    int NUM_VALS = atoi(argv[1]);
    int array_fill_type = atoi(argv[2]);
    int sort_alg = atoi(argv[3]);

    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize values to be sorted 
    float *values = (float *)malloc(NUM_VALS * sizeof(float));

    // Initialize local arrays for each process
    int local_size = NUM_VALS / num_procs;
    vector<float> local_values;

    // Fill the local portions of the values then gather into values (NUM_VALS MUST BE DIVISIBLE BY num_procs)
    parallel_array_fill(NUM_VALS, &local_values, local_size, num_procs, rank, array_fill_type);

    // Wait until generation completes on all processes
    MPI_Barrier(MPI_COMM_WORLD);

    if (PRINT_DEBUG)
    {
        printArray(NUM_VALS, values, local_values, local_size, num_procs, rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // SORT
    if (sort_alg == 0) {
        sample_sort(NUM_VALS, &local_values, local_size, num_procs, rank, 10);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    parallel_sort_check_unmerged(NUM_VALS, local_values, local_size, num_procs, rank);

    free(values);

    MPI_Finalize();
    return 0;
}