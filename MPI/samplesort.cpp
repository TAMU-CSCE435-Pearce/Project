#include "common.h"

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
    CALI_MARK_BEGIN(SAMPLE_SORT_NAME);

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

    CALI_MARK_END(SAMPLE_SORT_NAME);
}
