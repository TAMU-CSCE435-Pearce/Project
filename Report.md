# CSCE 435 Group 16 project

## 1. Group members:

1. Anjali Kumar
2. Shawn Mathen
3. Ashrita Vadlapatla
4. Robert Eads

---

## 2a. Communication Method

Our team's primary method of communication will be GroupMe with Slack as a secondary method.

## 2b. Project topic

Sorting

## 2c. Brief project description (what algorithms will you be comparing and on what architectures)
Each of the following three algorithms will be run in parallel using MPI and CUDA. 

### Algorithm 1: Bubble Sort

```
begin BubbleSort(list)

   for all elements of list
      if list[i] > list[i+1]
         swap(list[i], list[i+1])
      end if
   end for

   return list

end BubbleSort
```


### Algorithm 2: Merge Sort
Sequential:
```python
function merge(left, right):
  result = []
  i, j = 0, 0
  while i < length(left) and j < length(right):
    if left[i] < right[j]:
      result.append(left[i])
      i += 1
    else:
      result.append(right[j])
      j += 1

  result += left[i:]
  result += right[j:]
  return result

function mergeSort(x):
  if length(x) <= 1:
    return x

  
   middle = len(x)//2
   left = mergeSort(x[:middle])
   right = mergeSort(x[middle:])
   return merge(left, right)
```

Parallel:
The following parallel Merge Sort algorithms will be implemented using MPI.
```python
function parallel_merge_sort(arr, num_threads)
    if length(arr) <= 1
        return arr

    mid = length(arr) // 2
    left = arr[0...mid-1]
    right = arr[mid...]

    if num_threads > 1
        left_thread = start_thread(parallel_merge_sort(left, num_threads/2))
        right_thread = start_thread(parallel_merge_sort(right, num_threads/2))
        join_thread(left_thread)
        join_thread(right_thread)

    else
        left = merge_sort(left)
        right = merge_sort(right)

    return parallel_merge(left, right)
```
### Algorithm 3: Quick Sort

```
int partition(int arr[], int low, int high) {

   int pivot = arr[high];
   int i = (low-1);

   for (int j = low; j <= high; j++) {
    if (arr[j] < pivot) {
      i++;
      swap(arr[i], arr[j]);
    }
   }
   swap(arr[i+1], arr[high]);
   return (i+1);
}

void quickSort(int arr[], int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);

    quickSort(arr, low, pi-1);
    quickSort(arr, pi+1, high);
  }
}
```

## 2d. Citations

- https://www.tutorialspoint.com/data_structures_algorithms/bubble_sort_algorithm.htm
- https://compucademy.net/algorithmic-thinking-with-python-part-3-divide-and-conquer-strategy/#:~:text=There%20is%20a%20really%20clever%20trick%20that,the%20same%20type%20as%20the%20original%20problem.
- https://teivah.medium.com/parallel-merge-sort-in-java-e3213ae9fa2c
- https://www.geeksforgeeks.org/quick-sort/

## 3. _due 11/08_ Pseudocode for each algorithm and implementation

## 3. _due 11/08_ Evaluation plan - what and how will you measure and compare

For example:
- Effective use of a GPU (play with problem size and number of threads)
- Strong scaling to more nodes (same problem size, increase number of processors)
- Weak scaling (increase problem size, increase number of processors)
