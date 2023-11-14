#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <random>
#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function to generate sorted data
std::vector<int> generate_sorted_data(size_t size)
{
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(i);
    }
    return data;
}

// Function to generate reverse sorted data
std::vector<int> generate_reverse_sorted_data(size_t size)
{
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(size - i - 1);
    }
    return data;
}

// Function to generate random data
std::vector<int> generate_random_data(size_t size)
{
    std::vector<int> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(size) * 10);

    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int>(dis(gen));
    }
    return data;
}

// Function to generate 1% perturbed data
std::vector<int> generate_perturbed_data(size_t size)
{
    std::vector<int> data = generate_sorted_data(size);
    size_t perturb_count = std::max(1UL, size / 100);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(size) * 10);
    std::uniform_int_distribution<size_t> index_dis(0, size - 1);

    for (size_t i = 0; i < perturb_count; ++i)
    {
        data[index_dis(gen)] = static_cast<int>(dis(gen));
    }
    return data;
}

// Check if the data is correctly sorted
bool is_correct(const std::vector<int> &data)
{
    for (size_t i = 1; i < data.size(); i++)
    {
        if (data[i - 1] > data[i])
        {
            return false;
        }
    }
    return true;
}

// Sample sort
void sample_sort(std::vector<int> &data, int rank, int num_procs)
{
    // Locally sort the data on each process
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(data.begin(), data.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Sample local data to find potential splitters
    size_t size_per_proc = (data.size() + num_procs - 1) / num_procs;
    std::vector<int> local_samples(num_procs - 1);
    for (int i = 0; i < num_procs - 1; i++)
    {
        local_samples[i] = data[(i + 1) * size_per_proc];
    }

    // Gather local samples at rank 0
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    std::vector<int> global_samples;
    if (rank == 0)
    {
        global_samples.resize((num_procs - 1) * num_procs);
    }
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_samples.data(), num_procs - 1, MPI_INT, global_samples.data(), num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Rank 0 sorts global samples and picks splitters
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    std::vector<int> splitters(num_procs - 1);
    if (rank == 0)
    {
        std::sort(global_samples.begin(), global_samples.end());
        for (int i = 0; i < num_procs - 1; i++)
        {
            splitters[i] = global_samples[(i + 1) * (num_procs - 1)];
        }
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Broadcast splitters to all processes
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(splitters.data(), num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Bucket data based on splitters
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    std::vector<std::vector<int>> buckets(num_procs);
    size_t idx = 0;
    for (int i = 0; i < num_procs; i++)
    {
        while (idx < data.size() && (i == num_procs - 1 || data[idx] < splitters[i]))
        {
            buckets[i].push_back(data[idx]);
            idx++;
        }
    }

    // Flatten buckets into a send buffer
    std::vector<int> send_buffer;
    for (int i = 0; i < num_procs; i++)
    {
        send_buffer.insert(send_buffer.end(), buckets[i].begin(), buckets[i].end());
    }

    // Set up send counts based on bucket sizes
    std::vector<int> send_counts(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        send_counts[i] = buckets[i].size();
    }

    // Initialize receive counts to zeros (will be filled in by MPI_Alltoall)
    std::vector<int> receive_counts(num_procs, 0);

    // Compute send and receive displacements
    std::vector<int> send_displacements(num_procs);
    std::vector<int> receive_displacements(num_procs);
    send_displacements[0] = 0;
    receive_displacements[0] = 0;
    for (int i = 1; i < num_procs; i++)
    {
        send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
        receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
    }

    // Allocate space for the sorted data
    int total_receive = data.size();
    std::vector<int> sorted_data(total_receive);
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Communicate data between processes
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Alltoall");
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), MPI_INT,
                  sorted_data.data(), receive_counts.data(), receive_displacements.data(), MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Alltoall");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Resize the sorted_data to actual received size
    total_receive = receive_displacements[num_procs - 1] + receive_counts[num_procs - 1];
    sorted_data.resize(total_receive);

    // Sort the merged data locally
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(sorted_data.begin(), sorted_data.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Replace original data with sorted data
    data = sorted_data;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <inputType> <size>\n";
        return 1;
    }

    std::string inputType = argv[1];
    size_t size = std::stoul(argv[2]);

    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "SampleSort";
    std::string programmingModel = "MPI";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    int groupNumber = 13;
    std::string implementationSource = "AI";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", size);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", num_procs);
    adiak::value("group_num", groupNumber);
    adiak::value("implementationSource", implementationSource);

    CALI_MARK_BEGIN("main");

    CALI_MARK_BEGIN("data_init");
    std::vector<int> data;
    if (inputType == "Sorted")
    {
        data = generate_sorted_data(size);
    }
    else if (inputType == "ReverseSorted")
    {
        data = generate_reverse_sorted_data(size);
    }
    else if (inputType == "Random")
    {
        data = generate_random_data(size);
    }
    else if (inputType == "1%perturbed")
    {
        data = generate_perturbed_data(size);
    }
    else
    {
        std::cerr << "Invalid input type. Use 'Sorted', 'ReverseSorted', 'Random', or '1%perturbed'.\n";
        return 1;
    }
    CALI_MARK_END("data_init");

    sample_sort(data, rank, num_procs);

    CALI_MARK_BEGIN("correctness_check");
    bool correct = is_correct(data);
    if (!correct)
    {
        std::cerr << "Error: The algorithm did not sort the data correctly." << std::endl;
    }
    CALI_MARK_END("correctness_check");

    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}
