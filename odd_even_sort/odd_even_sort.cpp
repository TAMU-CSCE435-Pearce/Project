// disclaimer: majority of code from https://github.com/ashantanu/Odd-Even-Sort-using-MPI/blob/master/oddEven.cpp

#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <caliper/cali.h>
#include <adiak.hpp>

using namespace std;

void initializeMPI(int argc, char *argv[], int &nump, int &rank, int &n, int &localn, int root_process) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nump);
    if (argc != 2) {
        if (rank == root_process) {
            cerr << "Usage: " << argv[0] << " <number_of_values_to_sort>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    n = atoi(argv[1]);
    localn = n / nump;
}

void generateData(int *&data, const int n, const int localn) {
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    data = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 100;
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    printf("array data is:");
    for (int i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

void broadcastLocaln(int &localn) {
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Bcast(&localn, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");
}

void scatterData(const int *data, int *recdata, const int localn, const int rank) {
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Scatter(data, localn, MPI_INT, recdata, 100, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    printf("%d: received data:", rank);
    for (int i = 0; i < localn; i++) {
        printf("%d ", recdata[i]);
    }
    printf("\n");
}

void sortLocalData(int *recdata, const int localn) {
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    sort(recdata, recdata + localn);
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");
}

void oddEvenSort(int *recdata, int *recdata2, int localn, int rank, int nump) {
    // Begin the odd-even sort
    CALI_MARK_BEGIN("comp");
    int oddrank, evenrank;
    MPI_Status status;

    if (rank % 2 == 0) {
        oddrank = rank - 1;
        evenrank = rank + 1;
    } else {
        oddrank = rank + 1;
        evenrank = rank - 1;
    }
    
    // Set the ranks of the processors at the end of the linear
    if (oddrank == -1 || oddrank == nump)
        oddrank = MPI_PROC_NULL;
    if (evenrank == -1 || evenrank == nump)
        evenrank = MPI_PROC_NULL;
    
    CALI_MARK_BEGIN("comp_large");
    for (int p = 0; p < nump - 1; p++) {
        CALI_MARK_BEGIN("comm_small");
        if (p % 2 == 1) { // Odd phase
            MPI_Sendrecv(recdata, localn, MPI_INT, oddrank, 1, recdata2, localn, MPI_INT, oddrank, 1, MPI_COMM_WORLD, &status);
        } else { // Even phase
            MPI_Sendrecv(recdata, localn, MPI_INT, evenrank, 1, recdata2, localn, MPI_INT, evenrank, 1, MPI_COMM_WORLD, &status);
        }
        CALI_MARK_END("comm_small");

        // Small computation for setup before merging
        CALI_MARK_BEGIN("comp_small");
        // Extract localn after sorting the two
        int *temp = (int*)malloc(localn * sizeof(int));
        for (int i = 0; i < localn; i++) {
            temp[i] = recdata[i];
        }
        CALI_MARK_END("comp_small");

        // Begin the large computation for merging
        CALI_MARK_BEGIN("comp_large");
        if (status.MPI_SOURCE == MPI_PROC_NULL) continue;
        else if (rank < status.MPI_SOURCE) {
            // Store the smaller of the two
            int i, j, k;
            for (i = j = k = 0; k < localn; k++) {
                if (j == localn || (i < localn && temp[i] < recdata2[j]))
                    recdata[k] = temp[i++];
                else
                    recdata[k] = recdata2[j++];
            }
        } else {
            // Store the larger of the two
            int i, j, k;
            for (i = j = k = localn - 1; k >= 0; k--) {
                if (j == -1 || (i >= 0 && temp[i] >= recdata2[j]))
                    recdata[k] = temp[i--];
                else
                    recdata[k] = recdata2[j--];
            }
        } // else
        CALI_MARK_END("comp_large");
        free(temp);
    } // for
    CALI_MARK_END("comp_large");

    // End the overarching computation region before the final gather
    CALI_MARK_END("comp");
}

void gatherData(int *recdata, int *&data, const int localn, const int rank, const int root_process) {
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather(recdata, localn, MPI_INT, data, localn, MPI_INT, root_process, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
}

void isCorrect(const int *data, const int n) {
        // Check if the array is sorted
        bool isSorted = true;
        for (int i = 1; i < n; i++) {
            if (data[i - 1] > data[i]) {
                isSorted = false;
                break;
            }
        }

        if (isSorted) {
            printf("The array is sorted.\n");
            printf("final sorted data:");
            for (int i = 0; i < n; i++) {
                printf("%d ", data[i]);
            }
            printf("\n");
        } else {
            printf("The array is NOT sorted.\n");
        }

}

int main(int argc, char *argv[]) {
    int nump, rank, n, localn;
    int *data = NULL, recdata[100], recdata2[100];
    int root_process = 0;
    
    initializeMPI(argc, argv, nump, rank, n, localn, root_process);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "OddEven";
    std::string programmingModel = "MPI";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    std::string inputType = "Random";
    int group_number = 13;
    std::string implementation_source = "GitHub"; // https://github.com/ashantanu/Odd-Even-Sort-using-MPI/blob/master/oddEven.cpp

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", n);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", nump);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("main");

    CALI_MARK_BEGIN("data_init");
    if (rank == root_process) {
        generateData(data, n, localn);
    }
    CALI_MARK_END("data_init");

    broadcastLocaln(localn);

    scatterData(data, recdata, localn, rank);

    sortLocalData(recdata, localn);

    oddEvenSort(recdata, recdata2, localn, rank, nump);

    gatherData(recdata, data, localn, rank, root_process);
    
    CALI_MARK_BEGIN("correctness_check");
    if (rank == root_process) {
        isCorrect(data, n);
    }
    CALI_MARK_END("correctness_check");

    CALI_MARK_END("main");
    MPI_Finalize();

    return 0;
}
