#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void enum_sort(int *a, int *c, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int count = 0;

    for (int d = 0; d < N; d++)
    {
        if (a[d] < a[tid])
        {
            count++;
        }
    }

    c[count] = a[tid];
}

void populateArray(int *arr, int size)
{
    srand(3);

    for (int i = 0; i < size; i++)
    {
        arr[i] = (int)rand();
    }
}

void sequentialEnumSort(int *a, int *b, int N)
{
    for (int k = 0; k < N; k++)
    {
        int count = 0;

        for (int d = 0; d < N; d++)
        {
            if (a[d] < a[k])
            {
                count++;
            }
        }

        b[count] = a[k];
    }
}

int main()
{
    int N, B, T;

    printf("Enter the value for N: ");
    scanf("%d", &N);

    int valid = 0;
    while (valid == 0)
    {
        printf("Enter the number of blocks: ");
        scanf("%d", &B);

        printf("Enter the number of threads: ");
        scanf("%d", &T);

        if (B > 1024 || T > 1024 || B * T < N)
        {
            printf("Invalid input entered.\n");
        }
        else
        {
            valid = 1;
        }
    }

    dim3 Grid(B, B);
    dim3 Block(T, 1);

    int size = N * N * sizeof(int);
    int *a, *b, *c;

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    populateArray(a, N);

    int *dev_a, *dev_c;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    enum_sort<<<Grid, Block>>>(dev_a, dev_c, N);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);
    double gpuTime = elapsed_time_ms;

    cudaEventRecord(start, 0);
    sequentialEnumSort(a, b, N);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);
    double cpuTime = elapsed_time_ms;

    printf("Initial Array:\n");
    for (int h = 0; h < N; h++)
    {
        printf("%d ", a[h]);
    }
    printf("\n");

    printf("Sequential Enum Sort:\n");
    for (int h = 0; h < N; h++)
    {
        printf("%d ", b[h]);
    }

    int error = 0;
    for (int r = 0; r < N; r++)
    {
        if (b[r] != c[r])
        {
            error = 1;
            break;
        }
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
