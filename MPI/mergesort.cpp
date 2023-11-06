#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

const char* array_fill_name = "array_fill";
const char* sort_check_name = "sort_check";
const char* MPI_mergesort = "MPI_mergesort";

using namespace std;

void parallel_array_fill(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(array_fill_name);
    
    // Calculate local size based on rank and array size
    int local_size = NUM_VALS / num_procs;
    int start = rank * local_size;
    int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

    local_size = end - start;

    // Print process segment of array
    //printf("start: %d, end: %d, local_size:%d\n", start, end, local_size);

    float *local_values = (float *)malloc(local_size * sizeof(float));

    for (int i = 0; i < local_size; ++i) 
    {
        local_values[i] = (float)rand() / (float)RAND_MAX;
        //printf("Check value for %d : %.6f\n",i, local_values[i]);
    }

    // Gather local portions into global array
    MPI_Gather(local_values, local_size, MPI_FLOAT, values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(local_values);

    CALI_MARK_END(array_fill_name);
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

//merge two arrays so that if the two arrays were sorted, the returning array will be sorted
void merge(float *values, int left, int mid, int right)
{
  //printf("merge()%d%d%d", left, mid, right);
  int leftArrayLen = mid - left + 1;
  int rightArrayLen = right - mid;

  float leftArray[leftArrayLen];
  float rightArray[rightArrayLen];

  for(int i = 0; i < leftArrayLen; i++) {
    leftArray[i] = values[left+i];
  }
  for(int i = 0; i < rightArrayLen; i++) {
    rightArray[i] = values[mid + 1 + i];
  }

  int i, j, k;
  i = 0; j = 0;
  k = left;

  while( i < leftArrayLen && j < rightArrayLen) {
    if(leftArray[i] <= rightArray[j]) {
      values[k] = leftArray[i];
      i++;
    } else {
      values[k] = rightArray[j];
      j++;
    }
    k++;
  }

  //finish with whatever's left in either array
  while(i < leftArrayLen) {
    values[k] = leftArray[i];
    i++;
    k++;
  }

  while(j < rightArrayLen) {
    values[k] = rightArray[j];
    j++;
    k++;
  }
}

void mergesort(float *values, int left, int right)
{
  if(left < right) {
    int mid = left + (right - left)/2;
    mergesort(values, left, mid);
    mergesort(values, mid + 1, right);
    merge(values, left, mid, right);
  }
}

void combineArrays(float *a, int lenA, float *b, int lenB, float *c, int &lenC)
{
  lenC = lenA + lenB;
  c = (float *)malloc(lenC * sizeof(float));
  for(int i = 0; i < lenA; i++) {
    c[i] = a[i];
    printf("Adding to c... %.6f\n", a[i]);
    printf("c[i]... %.6f\n", c[i]);
  }
  for(int i = 0; i < lenB; i++) {
    c[i+lenA] = b[i];
    printf("Adding to c... %.6f\n", b[i]);
    printf("c[i+lenA]... %.6f\n", c[i+lenA]);
  }
}

struct node
{
  int value;
  struct node *left = nullptr;
  struct node *right = nullptr;
  struct node *parent = nullptr;
};

void printTree(std::string prefix, node* n, bool isLeft) {
  if(n != nullptr)
  {
    printf(prefix.c_str());
    if(isLeft) {
      printf("├──");
    } else {
      printf("└──");
    }
    printf("%d\n",n->value,n);
    if(isLeft) {
      printTree(prefix + "|   ", n->left, true);
      printTree(prefix + "|   ", n->right, false);
    } else {
      printTree(prefix + "    ", n->left, true);
      printTree(prefix + "    ", n->right, false);
    }
  }
}

void populateNode(node* a, int height, int maxheight, vector<int>* nodebank)
{
  if(height == maxheight) {return;}
  node* left = new node;
  left->value = a->value;
  left->parent = a;
  a->left = left;
  if(height < maxheight) {
    populateNode(left, height+1, maxheight, nodebank);
  }

  node* right = new node;
  right->value = nodebank->back();
  right->parent = a;
  nodebank->pop_back();
  a->right = right;

  if(height < maxheight) {
    populateNode(right, height+1, maxheight, nodebank);
  }
}

node* searchForLeaf(node* n, int value) {
  if(n == nullptr) {return nullptr;}
  if(n->left != nullptr) {
    node* leftSearchResult = searchForLeaf(n->left, value);
    node* rightSearchResult = searchForLeaf(n->right, value);
    if(leftSearchResult != nullptr) {
      return leftSearchResult;
    } else {
      return rightSearchResult;
    }
  } else {
    if(n->value == value) {
      return n;
    }
  }
  return nullptr;
}

string stepUpTree(node* n, string ret) {
  
  ret += to_string(n->value);
  if(n->parent == nullptr) { 
    return ret; 
  } else {
    return stepUpTree(n->parent, ret);
  }

}

void parallelMergesort(int NUM_VALS, float *values, int num_procs)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Calculate local size based on rank and array size
  int local_size = NUM_VALS / num_procs;
  int start = rank * local_size;
  int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

  local_size = end - start;

  float* local_values = (float*)malloc(local_size * sizeof(float));

  // Scatter the array among processors
  MPI_Scatter(values, local_size, MPI_FLOAT, local_values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  CALI_MARK_BEGIN(MPI_mergesort);

  //generate a binary tree with num_procs leaves
  //it's height is equal to log_2(num_procs)
  node* root = new node;
  root->value = 0;
  int treesHeight = log2(num_procs) + 1;
  
  vector<int> *nodebank = new vector<int>(num_procs); //a vector of integers of nodes which we can add
  for(int i = 1; i < num_procs; i++) {
    nodebank->push_back(i);
  }

  populateNode(root, 1, treesHeight, nodebank);

  //check tree
  if(rank == 0) {
    printTree("", root, false);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //the tree is a representation of how the threads will pass data.
  
  //search tree for leaf with value == rank
  node* myLeaf = searchForLeaf(root, rank);

  //derive a list of destinations that this thread's data will go to
  string destList = stepUpTree(myLeaf, "");
  //printf("My rank is %d and my dest list is %s.\n", rank, destList.c_str());
  
  destList.erase(0,1);

  int i = 1; 
  bool work = 1; //some processors pass through the loop and don't work
  int current_size = local_size;
  for(;;) {
    
    if(work) {
      mergesort(local_values, 0, current_size-1);
  
      string thisDest = "";
      thisDest += destList[0];
      destList.erase(0,1);
      int dest = stoi(thisDest);
  
      if(dest == rank) {
        //we need to listen for a message from the other processor
        float* foreign_values = (float*)malloc(current_size * sizeof(float));
        MPI_Recv(foreign_values, current_size, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float* combined_values = new float[current_size * 2];
        std::copy(local_values, local_values + current_size, combined_values);
        std::copy(foreign_values, foreign_values + current_size, combined_values + current_size);
        merge(combined_values, 0, current_size-1, (current_size * 2)-1);

        free(local_values);
        free(foreign_values);
        local_values = combined_values;
        
      } else {
        //send it to dest
        MPI_Send(local_values, current_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        work = 0; //this processor now rests
      }
    }

    current_size = current_size * 2;

    MPI_Barrier(MPI_COMM_WORLD);

    if(destList.size() == 0) {break;}
  }
  CALI_MARK_END(MPI_mergesort);

  CALI_MARK_BEGIN(sort_check_name);
  //process 0 now has the final result
  if(rank == 0) {
    for(int i = 0; i < NUM_VALS; i++) {
      printf("0: Value: %.6f\n", local_values[i]);
    }
    bool result = sort_check(local_values, NUM_VALS);
    if(result) {
      printf("The array is sorted.\n");
    } else {
      printf("The array is NOT sorted.\n");
    }
  }
  CALI_MARK_END(sort_check_name);
  
}

int main(int argc, char* argv[]) 
{
    srand(time(NULL));
    CALI_CXX_MARK_FUNCTION;

    int NUM_VALS = atoi(argv[1]);

    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    {
      // End execution if either:
      // - num_procs is not a power of two
      // - NUM_VALS is not divisble by num_procs
      int a = num_procs;
      while(a % 2 ==0) {
        a = a / 2;
      }

      if(a == 1) {
        //good
      } else {
        printf("Error: Number of processes isn't a power of two.\n");
        printf("Values: %d", NUM_VALS);
        printf("Procs: %d", num_procs);
        return 1;
      }

      if(NUM_VALS % num_procs == 0) {
        //good
      } else {
        printf("Error: Number of values isn't divisible by number of processes.\n");
        printf("Values: %d", NUM_VALS);
        printf("Procs: %d", num_procs);
        return 1;
      }
    }

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize array to be sorted 
    float *values = (float *)malloc(NUM_VALS * sizeof(float));

    // Fill the local portions of the array then gather into values (NUM_VALS MUST BE DIVISIBLE BY num_procs)
    parallel_array_fill(NUM_VALS, values, num_procs, rank);

    
    //messagePassingCheck(NUM_VALS, values, num_procs);
    parallelMergesort(NUM_VALS, values, num_procs);


    // Check if values is sorted

    free(values);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}
