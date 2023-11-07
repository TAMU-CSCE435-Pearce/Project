#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void merge(float *d_array, int l, int m, int r)
{
    // Get the thread ID.
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Merge the two sub-arrays.
    for (int i = l; i < m; i++)
    {
        if (d_array[i] > d_array[m + 1])
        {
            float temp = d_array[i];
            d_array[i] = d_array[m + 1];
            d_array[m + 1] = temp;
        }
    }
}

__global__ void mergesort(float *d_array, int n)
{
    // Check if the array is empty.
    if (n <= 1)
    {
        return;
    }

    // Divide the array into two halves.
    int m = n / 2;

    // Recursively sort the two halves.
    mergesort(d_array, m);
    mergesort(d_array + m, n - m);

    // Merge the two sorted halves.
    merge(d_array, 0, m - 1, n - 1);
}

void main()
{
    // Get the number of elements in the array.
    int n = 10;

    // Allocate memory for the array on the GPU.
    float *d_array = (float *)malloc(sizeof(float) * n);

    // Initialize the array.
    for (int i = 0; i < n; i++)
    {
        d_array[i] = rand() / (float)RAND_MAX;
    }

    // Launch the mergesort kernel.
    mergesort<<<1, 1024>>>(d_array, n);

    // Copy the results back to the CPU.
    float *h_array = (float *)malloc(sizeof(float) * n);
    cudaMemcpy(h_array, d_array, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Print the sorted array.
    for (int i = 0; i < n; i++)
    {
        printf("%f ", h_array[i]);
    }

    // Free the memory on the GPU.
    cudaFree(d_array);

    // Free the memory on the CPU.
    free(h_array);
}