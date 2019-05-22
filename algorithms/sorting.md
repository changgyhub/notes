---
description: 以左开右闭为例
---

# Sorting

compile and run with:

```cpp
// g++ -std=c++11 "test.cpp" -o "test" && "./test"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
```

## Quick Sort

```cpp
void quick_sort(vector<int> &a, int l, int r) {
    if (l + 1 >= r) return;
    int first = l, last = r-1, key = a[first];
    while (first < last){
        while(first < last && a[last] >= key) --last;
        a[first] = a[last];

        while (first < last && a[first] <= key) ++first;
        a[last] = a[first];
    }
    a[first] = key;
    quick_sort(a, l, first);
    quick_sort(a, first + 1, r);
}
```

## Merge Sort

```cpp
void merge_sort(vector<int> &A, int l, int r, vector<int> &T) {
    if (l + 1 >= r) return;
    // divide
    int m = l + (r - l) / 2;
    merge_sort(A, l, m, T);
    merge_sort(A, m, r, T);
    // conquer
    int p = l, q = m, i = l;
    while (p < m || q < r) {
        if (q >= r || (p < m && A[p] <= A[q])) T[i++] = A[p++];
        else T[i++] = A[q++];
    }
    for (i = l; i < r; ++i) A[i] = T[i];
}
```

## Insertion Sort

```cpp
// insert to correct position
void insertion_sort(vector<int> &arr, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j > 0 && arr[j] < arr[j-1]; --j) swap(arr[j], arr[j-1]);
    }
}
```

## Bubble Sort

```cpp
// bubble largests to the right end
void bubble_sort(vector<int> &arr, int n) {
    bool swapped;
    for (int i = 1; i < n; ++i) {
        swapped = false;
        for (int j = 1; j < n - i + 1; ++j) {
            if (arr[j] < arr[j-1]) {
                swap(arr[j], arr[j-1]);
                   swapped = true;
            }
         }
         if (!swapped) break;
   }
}
```

## Selection Sort

```cpp
// select min then swap
void selection_sort(vector<int> &arr, int n) {
    int min_idx;
    for (int i = 0; i < n - 1; ++i) {
        min_idx = i;
        for (int j = i + 1; j < n; ++j) if (arr[j] < arr[min_idx]) min_idx = j;
        swap(arr[min_idx], arr[i]);
    }
}
```

## Test Code

```cpp
int main() {
    vector<int> arr = {1,3,5,7,2,6,4,8,9,2,8,7,6,0,3,5,9,4,1,0};
    vector<int> arr1(arr), arr2(arr), arr3(arr), arr4(arr), arr5(arr), temp(arr.size());
    sort(arr.begin(), arr.end());
    quick_sort(arr1, 0, arr1.size());
    merge_sort(arr2, 0, arr2.size(), temp);
    insertion_sort(arr3, arr3.size());
    bubble_sort(arr4, arr4.size());
    selection_sort(arr5, arr5.size());
    for (int i: arr) cout << i << ' '; cout << endl;
    for (int i: arr1) cout << i << ' '; cout << endl;
    for (int i: arr2) cout << i << ' '; cout << endl;
    for (int i: arr3) cout << i << ' '; cout << endl;
    for (int i: arr4) cout << i << ' '; cout << endl;
    for (int i: arr5) cout << i << ' '; 
}
```

