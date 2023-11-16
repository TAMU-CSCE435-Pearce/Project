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

// Helper function to compare integers
int int_compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

// Sample sort function
void sample_sort(std::vector<int> &local_data, int rank, int num_procs, int total_size, std::vector<int> &all_sorted_data)
{
    // Local sorting
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(local_data.begin(), local_data.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Selecting samples for splitters
    std::vector<int> local_samples;
    for (int i = 1; i < num_procs; ++i)
    {
        int index = (local_data.size() / num_procs) * i;
        local_samples.push_back(local_data[index]);
    }

    // Gather samples at root and pick splitters
    std::vector<int> all_samples;
    if (rank == 0)
    {
        all_samples.resize(num_procs * (num_procs - 1));
    }
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_samples.data(), local_samples.size(), MPI_INT,
               all_samples.data(), local_samples.size(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    std::vector<int> splitters(num_procs - 1);
    if (rank == 0)
    {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_small");
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 0; i < num_procs - 1; ++i)
        {
            splitters[i] = all_samples[i * num_procs + num_procs / 2];
        }
        CALI_MARK_END("comp_small");
        CALI_MARK_END("comp");
    }

    // Broadcast splitters to all processes
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(splitters.data(), splitters.size(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Bucketing based on splitters
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::vector<std::vector<int>> buckets(num_procs);
    for (int val : local_data)
    {
        int bucket_index = std::lower_bound(splitters.begin(), splitters.end(), val) - splitters.begin();
        buckets[bucket_index].push_back(val);
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Prepare data for sending
    std::vector<int> send_data;
    std::vector<int> send_counts(num_procs, 0);
    for (int i = 0; i < num_procs; ++i)
    {
        send_data.insert(send_data.end(), buckets[i].begin(), buckets[i].end());
        send_counts[i] = buckets[i].size();
    }

    // Communicate the sizes of the buckets
    std::vector<int> receive_counts(num_procs);
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Calculate displacements
    std::vector<int> send_displacements(num_procs, 0);
    std::vector<int> receive_displacements(num_procs, 0);
    for (int i = 1; i < num_procs; ++i)
    {
        send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
        receive_displacements[i] = receive_displacements[i - 1] + receive_counts[i - 1];
    }

    // Gather the sorted data
    int total_receive = std::accumulate(receive_counts.begin(), receive_counts.end(), 0);
    std::vector<int> sorted_data(total_receive);
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Alltoallv");
    MPI_Alltoallv(send_data.data(), send_counts.data(), send_displacements.data(), MPI_INT,
                  sorted_data.data(), receive_counts.data(), receive_displacements.data(), MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Alltoallv");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Final local sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    qsort(sorted_data.data(), sorted_data.size(), sizeof(int), int_compare);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Replace original data with sorted data
    local_data = sorted_data;

    // Prepare for gathering sorted data at the root
    int local_sorted_size = local_data.size();
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(&local_sorted_size, 1, MPI_INT, receive_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Calculate displacements for gathering
    std::vector<int> displacements(num_procs, 0);
    if (rank == 0)
    {
        for (int i = 1; i < num_procs; ++i)
        {
            displacements[i] = displacements[i - 1] + receive_counts[i - 1];
        }
        all_sorted_data.resize(total_size);
    }

    // Gather sorted data at root
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large"); // is this large?
    CALI_MARK_BEGIN("MPI_Gatherv");
    MPI_Gatherv(local_data.data(), local_data.size(), MPI_INT,
                all_sorted_data.data(), receive_counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gatherv");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN("main");

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <size> <inputType>\n";
        return 1;
    }

    size_t size = std::stoul(argv[1]);
    std::string inputTypeShort = argv[2];

    std::string inputType;
    if (inputTypeShort == "r")
    {
        inputType = "Random";
    }
    else if (inputTypeShort == "s")
    {
        inputType = "Sorted";
    }
    else if (inputTypeShort == "rs")
    {
        inputType = "Reverse Sorted";
    }
    else if (inputTypeShort == "p")
    {
        inputType = "1% Perturbed";
    }
    else
    {
        std::cerr << "Invalid input type. Use 'r', 's', 'rs', or 'p'.\n";
        return 1;
    }

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
    adiak::value("implementation_source", implementationSource);

    std::vector<int> data;
    if (rank == 0)
    {
        CALI_MARK_BEGIN("data_init");
        if (inputType == "Sorted")
        {
            data = generate_sorted_data(size);
        }
        else if (inputType == "Reverse Sorted")
        {
            data = generate_reverse_sorted_data(size);
        }
        else if (inputType == "Random")
        {
            data = generate_random_data(size);
        }
        else if (inputType == "1% Perturbed")
        {
            data = generate_perturbed_data(size);
        }
        else
        {
            std::cerr << "Invalid input type. Use 'Sorted', 'Reverse Sorted', 'Random', or '1% Perturbed'.\n";
            return 1;
        }
        CALI_MARK_END("data_init");
    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    size_t local_size = size / num_procs;
    std::vector<int> local_data(local_size);
    MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    std::vector<int> all_sorted_data;
    sample_sort(local_data, rank, num_procs, size, all_sorted_data);

    if (rank == 0)
    {
        CALI_MARK_BEGIN("correctness_check");
        bool correct = is_correct(all_sorted_data);
        if (!correct)
        {
            std::cerr << "Error: The algorithm did not sort the data correctly." << std::endl;
        }
        CALI_MARK_END("correctness_check");
    }

    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}
