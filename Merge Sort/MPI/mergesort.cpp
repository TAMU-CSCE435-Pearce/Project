#include <iostream>
#include <stdlib.h>
#include "mpi.h"

#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r)
{

    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;

    while ((h <= m) && (j <= r))
    {

        if (a[h] <= a[j])
        {

            b[i] = a[h];
            h++;
        }

        else
        {

            b[i] = a[j];
            j++;
        }

        i++;
    }

    if (m < h)
    {

        for (k = j; k <= r; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    else
    {

        for (k = h; k <= m; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    for (k = l; k <= r; k++)
    {

        a[k] = b[k];
    }
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r)
{

    int m;

    if (l < r)
    {

        m = (l + r) / 2;

        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}

void generateData(int *localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank)
{
    switch (startingSortChoice)
    {
    case 0:
    { // Random
        srand((my_rank + 5) * (my_rank + 12) * 1235);
        for (int i = 0; i < amountToGenerate; i++)
        {
            localData[i] = rand() % amountToGenerate; // Changed inputSize to amountToGenerate
        }
        break;
    }

    case 1:
    { // Sorted
        int endValue = startingPosition + amountToGenerate;
        for (int i = startingPosition; i < endValue; i++)
        {
            localData[i - startingPosition] = i; // Changed push_back to direct assignment
        }
        break;
    }
    case 2:
    {                                                             // Reverse sorted
        int startValue = amountToGenerate - 1 - startingPosition; // Changed inputSize to amountToGenerate
        int endValue = amountToGenerate - startingPosition;
        for (int i = startValue; i >= endValue; i--)
        {
            localData[startValue - i] = i; // Changed push_back to direct assignment
        }
        break;
    }
    }
}

int main(int argc, char **argv)
{

    /********** Create and populate the array using generateData **********/
    int n = atoi(argv[1]);
    int *original_array = new int[n];

    int c;
    srand(time(NULL));
    std::cout << "This is the unsorted array: ";
    generateData(original_array, 0, n, 0, 0); // Generate data directly into original_array

    for (c = 0; c < n; c++)
    {
        std::cout << original_array[c] << " ";
    }

    std::cout << "\n\n";

    /********** Initialize MPI **********/
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /********** Divide the array in equal-sized chunks **********/
    int size = n / world_size;

    /********** Send each subarray to each process **********/
    int *sub_array = new int[size];
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);

    /********** Perform the mergesort on each process **********/
    int *tmp_array = new int[size];
    mergeSort(sub_array, tmp_array, 0, (size - 1));

    /********** Gather the sorted subarrays into one **********/
    int *sorted = nullptr;
    if (world_rank == 0)
    {

        sorted = new int[n];
    }

    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);

    /********** Make the final mergeSort call **********/
    if (world_rank == 0)
    {

        int *other_array = new int[n];
        mergeSort(sorted, other_array, 0, (n - 1));

        /********** Display the sorted array **********/
        std::cout << "This is the sorted array: ";
        for (c = 0; c < n; c++)
        {

            std::cout << sorted[c] << " ";
        }

        std::cout << "\n\n";

        /********** Clean up root **********/
        delete[] sorted;
        delete[] other_array;
    }

    /********** Clean up rest **********/
    delete[] original_array;
    delete[] sub_array;
    delete[] tmp_array;

    /********** Finalize MPI **********/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
