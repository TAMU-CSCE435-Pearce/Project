#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

//these functions are for a true in-place mergesort
// imsort(), swap(), wmerge(), wsort(), and imsort() are copyrighted under GNU GPL and modified from Liu Xinyu's work (github.com/liuxinyu95/AlgoXY/blob/algoxy/sorting/merge-sort/src/mergesort.c)
__device__ __host__ void imsort(float* xs, int l, int u);

__device__ __host__ void swap(float* xs, int i, int j) {
    float tmp = xs[i]; xs[i] = xs[j]; xs[j] = tmp;
}

__device__ __host__ void wmerge(float* xs, int i, int m, int j, int n, int w) {
    while (i < m && j < n)
        swap(xs, w++, xs[i] < xs[j] ? i++ : j++);
    while (i < m)
        swap(xs, w++, i++);
    while (j < n)
        swap(xs, w++, j++);
} 

__device__ __host__ void wsort(float* xs, int l, int u, int w) {
    int m;
    if (u - l > 1) {
        m = l + (u - l) / 2;
        imsort(xs, l, m);
        imsort(xs, m, u);
        wmerge(xs, l, m, m, u, w);
    }
    else
        while (l < u)
            swap(xs, l++, w++);
}

__device__ __host__ void imsort(float* xs, int l, int u) {
    int m, n, w;
    if (u - l > 1) {
        m = l + (u - l) / 2;
        w = l + u - m;
        wsort(xs, l, m, w); /* the last half contains sorted elements */
        while (w - l > 2) {
            n = w;
            w = l + (n - l + 1) / 2;
            wsort(xs, w, n, l);  /* the first half of the previous working area contains sorted elements */
            wmerge(xs, l, l + n - w, n, u, w);
        }
        for (n = w; n > l; --n) /*switch to insertion sort*/
            for (m = n; m < u && xs[m] < xs[m-1]; ++m)
                swap(xs, m, m - 1);
    }
}


float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f \n",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void mergesortStep(float *values, int num_vals, int sectionWidth)
{
  int id = threadIdx.x; 

  //derive our section of the array based on thread id
  int left = sectionWidth * id;
  int right = left + sectionWidth;
  imsort(values, left, right);
}

void mergesortGPU(float *values, int num_vals)
{

  //pass values to device memory
  float* d_values;
  cudaMalloc((void**)&d_values, sizeof(float)* num_vals);
  cudaMemcpy(d_values, values, sizeof(float) * num_vals, cudaMemcpyHostToDevice);

  int sliceWidth = 2;
  int threadsToUse = THREADS;

  if(threadsToUse > num_vals / sliceWidth) { threadsToUse = num_vals/sliceWidth;}

  if(threadsToUse < num_vals / sliceWidth) {sliceWidth = num_vals / threadsToUse;}

  for(;;) {
    mergesortStep<<<1024, threadsToUse>>>(d_values, num_vals, sliceWidth);
    if(threadsToUse == 1) { break; }
    sliceWidth *= 2;
    threadsToUse /= 2;
  }

  //copy the values back to the host
  cudaMemcpy(values, d_values, sizeof(float) * num_vals, cudaMemcpyDeviceToHost);
  cudaFree(d_values);
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

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[2]);
  NUM_VALS = atoi(argv[1]);

  {
    // End execution if either:
    // - num_procs is not a power of two
    // - NUM_VALS is not divisble by num_procs
    int a = NUM_VALS;
    while(a % 2 ==0) {
      a = a / 2;
    }

    if(a == 1) {
      //good
    } else {
      printf("Error: Number of processes isn't a power of two.\n");
      printf("Values: %d", NUM_VALS);
      printf("Procs: %d", THREADS);
      return 1;
    }

    if(NUM_VALS % THREADS == 0) {
      //good
    } else {
      printf("Error: Number of values isn't divisible by number of processes.\n");
      printf("Values: %d", NUM_VALS);
      printf("Procs: %d", THREADS);
      return 1;
    }
  }

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  printf("Sorting begins.\n");

  mergesortGPU(values, NUM_VALS);

  bool result = sort_check(values, NUM_VALS);
  if(result) {
    printf("The array is sorted.\n");
  } else {
    printf("The array is NOT sorted.\n");
  }

  free(values);

}
