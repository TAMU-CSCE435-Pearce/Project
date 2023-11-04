#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>
#include <utility>

#include <bits/stdc++.h> 

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

#define PRINT_DEBUG 0

const char* array_fill_name = "array_fill";
const char* sort_check_name = "sort_check";
const char* sample_sort_name = "sample_sort";

void parallel_array_fill(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank, int array_fill_type)
{
    CALI_MARK_BEGIN(array_fill_name);
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
    CALI_MARK_END(array_fill_name);
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
    CALI_MARK_BEGIN(sort_check_name);
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
    CALI_MARK_END(sort_check_name);
}

void parallel_sort_check_unmerged(int NUM_VALS, vector<float> local_values, int local_size, int num_procs, int rank)
{
    CALI_MARK_BEGIN(sort_check_name);
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
    CALI_MARK_END(sort_check_name);
}

int compare_float(const void* a, const void* b) {
    const float* float_a = static_cast<const float*>(a);
    const float* float_b = static_cast<const float*>(b);

    if (*float_a < *float_b) {
        return -1;
    } else if (*float_a > *float_b) {
        return 1;
    } else {
        return 0;
    }
}

void sample_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank, int sample_size)
{
    CALI_MARK_BEGIN(sample_sort_name);

    int start = rank * local_size;
    int end = start + local_size - 1;

    float* sample = (float *)malloc(sample_size * sizeof(float));
    float cutoff;

    if (rank != 0)
    {
        srand(time(NULL) + rank);
        for (int i = 0; i < sample_size; i++)
        {
            int sample_index = rand() % local_size;
            sample[i] = local_values->at(sample_index);
        }

        // Sort samples
        qsort(sample, sample_size, sizeof(float), compare_float);

        // Determine splitters
        cutoff = sample[(int)(sample_size/2)];
    }
    else
    {
        cutoff = 0.0;
    }

    free(sample);

    // Send cutoffs to other processes and recieve cutoffs from other processes
    vector<float> cutoffs;
    for (int i = 0; i < num_procs; i++)
    {
        if (i != rank)
        {
            MPI_Send(&cutoff, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);

            float recv_cutoff;
            MPI_Recv(&recv_cutoff, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cutoffs.push_back(recv_cutoff);
        }
        else
        {
            cutoffs.push_back(cutoff);
        }
    }

    std::sort(cutoffs.begin(), cutoffs.end());

    // Populate cutoff pairs with corresponding values and ranks
    vector<pair<float, int>> cutoff_pairs;
    for (int i = 0; i < num_procs; i++)
    {
        pair<float, int> cutoff_pair;
        cutoff_pair.first = cutoffs[i];
        cutoff_pair.second = i;
        cutoff_pairs.push_back(cutoff_pair);
    }

    // Print received cutoffs
    if (PRINT_DEBUG)
    {
        for (int i = 0; i < num_procs; i++)
        {
            if (i == rank)
            {
                for (const std::pair<float, int>& element : cutoff_pairs) 
                {
                    printf("Rank: %d, Cutoff: %f, From Rank: %d\n", rank, element.first, element.second);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Initialize num_procs send buffers and receive buffers
    vector<vector<float>> send_bufs;
    vector<vector<float>> receive_bufs;
    for (int i = 0; i < num_procs; i++)
    {
        vector<float> send_buf;
        send_bufs.push_back(send_buf);

        vector<float> receive_buf;
        receive_bufs.push_back(receive_buf);
    }

    // Print local_values
    if (PRINT_DEBUG)
    {
        for (int i = 0; i < num_procs; i++)
        {
            if (i == rank)
            {
                printf("Rank: %d local values: ", rank);
                for (int i = 0; i < local_size; i ++) 
                {
                    printf("%f, ", local_values[i]);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Populate send buffers and local buffers
    vector<float> local_buf;
    for (int i = 0; i < local_size; i++)
    {
        for (int j = num_procs - 1; j >= 0; j--)
        {
            if (local_values->at(i) >= cutoff_pairs[j].first)
            {
                if (j != rank)
                {
                    send_bufs[j].push_back(local_values->at(i));
                }
                else
                {
                    local_buf.push_back(local_values->at(i));
                }
                break;
            }
        }
    }

    local_values->clear();
    local_values->insert(local_values->end(), local_buf.begin(), local_buf.end());

    if (PRINT_DEBUG)
    {
        for (int i = 0; i < num_procs; i++)
        {
            if (i == rank)
            {
                printf("Rank: %d buffers\n", rank);
                for (int j = 0; j < send_bufs.size(); j++)
                {
                    printf("\tRank: %d Send Buffer: ", j);
                    for (const float element : send_bufs[j]) {
                        printf("%f, ", element);
                    }
                    printf("\n");
                }
                printf("\tlocal buffer: ", rank);
                for (const float element : local_buf) {
                    printf("%f, ", element);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Send send buffs to other processes and recieve from other processes
    for (int i = 0; i < num_procs; i++)
    {
        int send_vector_size = send_bufs[i].size();
        int recv_vector_size;
        MPI_Request request;

        if (i != rank)
        {
            // if (send_vector_size != 0) printf("Rank %d sending %d values to process %d\n", rank, send_vector_size, i);

            MPI_Isend(&send_vector_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);

            MPI_Recv(&recv_vector_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            receive_bufs[i].resize(recv_vector_size);

            MPI_Isend(send_bufs[i].data(), send_vector_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);

            MPI_Recv(receive_bufs[i].data(), recv_vector_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_values->insert(local_values->end(), receive_bufs[i].begin(), receive_bufs[i].end());

            // if (vector_size != 0) 
            // {
            //     printf("Rank %d received %d values from process %d\n\t", rank, vector_size, i);
            //     for (const float element : receive_bufs[i]) 
            //     {
            //         printf("%f, ", element);
            //     }
            //     printf("\n");
            // }
        }
    }

    std::sort(local_values->begin(), local_values->end());

    // local_values = local_buf;

    MPI_Barrier(MPI_COMM_WORLD);

    // Print sorted local buffs
    if (PRINT_DEBUG)
    {
        for (int i = 0; i < num_procs; i++)
        {
            if (i == rank)
            {
                printf("Rank: %d sorted %d local values\n", rank, local_values->size());
                printf("Rank: %d sorted local values: ", rank);
                for (const float element : *local_values) {
                    printf("%f, ", element);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    CALI_MARK_END(sample_sort_name);
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

    // if (PRINT_DEBUG)
    // {
        printArray(NUM_VALS, values, local_values, local_size, num_procs, rank);
        MPI_Barrier(MPI_COMM_WORLD);
    // }

    // SORT
    sample_sort(NUM_VALS, &local_values, local_size, num_procs, rank, 10);
    
    MPI_Barrier(MPI_COMM_WORLD);

    parallel_sort_check_unmerged(NUM_VALS, local_values, local_size, num_procs, rank);

    free(values);

    MPI_Finalize();
    return 0;
}