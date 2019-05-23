# LeetCode 651 - 700

### 658. Find K Closest Elements

Given a sorted array, two integers `k` and `x`, find the `k` closest elements to `x` in the array. The result should also be sorted in ascending order. If there is a tie, the smaller elements are always preferred.

Example:

```
Input: [1,2,3,4,5], k=4, x=3
Output: [1,2,3,4]
```

Solution: 二分法，十分巧妙，一定要背

```cpp
vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    int l = 0, r = arr.size() - k;
    while(l < r) {
      	int m = l + (r - l) / 2;
      	if (arr[m] + arr[m+k] >= 2*x) r = m;
      	else l = m + 1;
    }
  	return vector<int>(arr.begin() + l, arr.begin() + l + k);
}
```

### 674. Longest Continuous Increasing Subsequence

Given an unsorted array of integers, find the length of longest `continuous` increasing subsequence (subarray).

Example:

```
Input: [1,3,5,4,7]
Output: 3
```

Solution: 遍历一遍即可

```cpp
int findLengthOfLCIS(vector<int>& nums) {
    int n = nums.size();
    if (n < 2) return n;
    int ret = 1, i = 1, start = 0, prev = nums[0];
    while (i < n) {
        if (nums[i] <= prev) {
            ret = max(ret, i - start);
            start = i;
        }
        prev = nums[i++];
    }
    return max(ret, i - start);
}
```

### 681. Next Closest Time

Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

Example:

```
Input: "19:34"
Output: "19:39" (The next closest time choosing from digits 1, 9, 3, 4, is 19:39)
```

Solution: 用一个set储存数字，然后从后往前遍历，注意细节

```cpp
string nextClosestTime(string time) {
    set<char> sorted;
    for(auto c: time) if (c != ':') sorted.insert(c);

    string res = time;
    for (int i = time.size() - 1; i >= 0; i--){
        if (time[i] == ':' ) continue;
        auto it = sorted.find(time[i]);
         if (*it != *sorted.rbegin()) {
            res[i] = *(++it);
            if ((i >= 3 && stoi(res.substr(3,2)) < 60) || (i<2 && stoi(res.substr(0,2)) < 24)) return res;      
         } 
         res[i] = *sorted.begin(); // take the smallest number anyway
    }
    return res;   
}
```

### 683. K Empty Slots

You have `N` bulbs in a row numbered from `1` to `N`. Initially, all the bulbs are turned off. We turn on exactly one bulb everyday until all bulbs are on after `N` days.

You are given an array `bulbs` of length `N` where `bulbs[i] = x` means that on the `(i+1)th` day, we will turn on the bulb at position `x` where `i` is `0-indexed` and `x` is `1-indexed.`

Given an integer `K`, find out the **minimum day number** such that there exists two **turned on** bulbs that have **exactly** `K` bulbs between them that are **all turned off**. If there isn't such day, return `-1`.

Example:

```
Input: bulbs: [1,3,2], K: 1
Output: 2
(On the first day: bulbs[0] = 1, first bulb is turned on: [1,0,0]
On the second day: bulbs[1] = 3, third bulb is turned on: [1,0,1]
On the third day: bulbs[2] = 2, second bulb is turned on: [1,1,1]
We return 2 because on the second day, there were two on bulbs with one off bulb between them.)
```

Solution: 用一个days数组，其中days[i] = t表示在i+1位置上会在第t天开灯；初始化天数指针left为0，right为k+1，然后i从0开始遍历。如果days[i] < days[left]或者days[i] < days[right]，说明窗口中有数字小于边界数字，不符合要求；另一种是days[i]==days[right]，说明中间的数字都是大于左右边界数的，此时应用左右边界中较大的那个数字更新结果res

```cpp
int kEmptySlots(vector<int>& bulbs, int k) {
    vector<int> days(bulbs.size());
    for(int i = 0; i < bulbs.size(); ++i) days[bulbs[i] - 1] = i + 1;
    int left = 0, right = k + 1, res = INT_MAX;
    for(int i = 0; right < days.size(); i++){
        if (days[i] < days[left] || days[i] <= days[right]) {   
            if (i == right) res = min(res, max(days[left], days[right]));
            left = i, right = k + 1 + i;
        }
    }
    return (res == INT_MAX)? -1: res;
}
```

### 684. Redundant Connection

In this problem, a tree is an **undirected** graph that is connected and has no cycles.

The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

The resulting graph is given as a 2D-array of `edges`. Each element of `edges` is a pair `[u, v]` with `u < v`, that represents an **undirected**edge connecting nodes `u` and `v`.

Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge `[u, v]` should be in the same format, with `u < v`.

Example:

```
Input: [[1,2], [2,3], [3,4], [1,4], [1,5]]
Output: [1,4] (
5 - 1 - 2
    |   |
    4 - 3
)
```

Solution: 并查集，一定要背

```cpp
class UF {
public:
    vector<int> id, sz;

    UF(int n) {
        id = vector<int>(n);
        sz = vector<int>(n, 1);
        for (int i = 0; i < n; ++i) id[i] = i;
    }

    int find(int p) {
        while (p != id[p]) {
            id[p] = id[id[p]];
            p = id[p];
        }
        return p;
    }

    void connect(int p, int q) {
        int i = find(p), j = find(q);
        if (i == j) return;
        if (sz[i] < sz[j]) {
            id[i] = j;
            sz[j] += sz[i];
        } else {
            id[j] = i;
            sz[i] += sz[j];
        }
    }
    
    bool isConnected(int p, int q) {
        return find(p) == find(q);
    }
};

class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        UF uf(n + 1);
        for (auto e: edges) {
            int u = e[0], v = e[1];
            if (uf.isConnected(u, v)) return e;
            uf.connect(u, v);
        }
        return vector<int>{-1, -1};
    }
};
```

### 689. Maximum Sum of 3 Non-Overlapping Subarrays

In a given array `nums` of positive integers, find three non-overlapping subarrays with maximum sum. Each subarray will be of size `k`, and we want to maximize the sum of all `3*k` entries. Return the result as a list of indices representing the starting position of each interval (0-indexed). If there are multiple answers, return the lexicographically smallest one.

Example:

```
Input: [1,2,1,2,6,7,5,1], 2
Output: [0, 3, 5] ([1, 2], [2, 6], [7, 5])
```

Solution: 左边dp一次，右边dp一次，然后检测中间区间，一定要理解

```cpp
vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
    int n = nums.size(), maxsum = 0;
    vector<int> sum = {0}, posLeft(n, 0), posRight(n, n-k), ans(3, 0);
    for (int i:nums) sum.push_back(sum.back()+i);
   // DP for starting index of the left max sum interval
    for (int i = k, tot = sum[k]-sum[0]; i <= n - 2 * k; i++) {
        if (sum[i+1]-sum[i+1-k] > tot) {
            posLeft[i] = i+1-k;
            tot = sum[i+1]-sum[i+1-k];
        }
        else 
            posLeft[i] = posLeft[i-1];
    }
    // DP for starting index of the right max sum interval
    // caution: the condition is ">= tot" for right interval, and "> tot" for left interval
    for (int i = n-k-1, tot = sum[n]-sum[n-k]; i >= 2*k - 1; i--) {
        if (sum[i+k]-sum[i] >= tot) {
            posRight[i] = i;
            tot = sum[i+k]-sum[i];
        }
        else
            posRight[i] = posRight[i+1];
    }
    // test all possible middle interval
    for (int i = k; i <= n-2*k; i++) {
        int l = posLeft[i-1], r = posRight[i+k];
        int tot = (sum[i+k]-sum[i]) + (sum[l+k]-sum[l]) + (sum[r+k]-sum[r]);
        if (tot > maxsum) {
            maxsum = tot;
            ans = {l, i, r};
        }
    }
    return ans;
}
```

