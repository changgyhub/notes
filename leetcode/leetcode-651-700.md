# LeetCode 651 - 700

### 653. Two Sum IV - Input is a BST

Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

Example:

```
Input: Target = 9, Tree =
    5
   / \
  3   6
 / \   \
2   4   7
Output: True
```

Solution: 中序遍历存到一个数组后，在进行二分查找；这一题不能用分别在左右子树两部分来处理这种思想，因为两个待求的节点可能分别在左右子树中

```cpp
bool findTarget(TreeNode* root, int k) {
    vector<int> nums;
    helper(root, nums);
    int i = 0, j = nums.size() - 1;
    while (i < j) {
        int sum = nums[i] + nums[j];
        if (sum == k) return true;
        if (sum < k) ++i;
        else --j;
    }
    return false;
}

void helper(TreeNode* root, vector<int>& nums) {
    if (!root) return;
    helper(root->left, nums);
    nums.push_back(root->val);
    helper(root->right, nums);
}
```

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

### 665. Non-decreasing Array

Given an array with `n` integers, your task is to check if it could become non-decreasing by modifying **at most** `1` element.

We define an array is non-decreasing if `array[i] <= array[i + 1]` holds for every `i` (1 <= i < n).

Example:

```
Input: [4,2,3]
Output: True (You could modify the first 4 to 1 to get a non-decreasing array)
```

Solution: 在出现 nums[i] < nums[i - 1] 时，需要考虑的是应该修改数组的哪个数，使得本次修改能使 i 之前的数组成为非递减数组，并且不影响后续的操作。优先考虑令 nums[i - 1] = nums[i]，因为如果修改 nums[i] = nums[i - 1] 的话，那么 nums[i] 这个数会变大，就有可能比 nums[i + 1] 大，从而影响了后续操作。还有一个比较特别的情况就是 nums[i] < nums[i - 2]，只修改 nums[i - 1] = nums[i] 不能使数组成为非递减数组，只能修改 nums[i] = nums[i - 1]

```cpp
bool checkPossibility(vector<int>& nums) {
    int cnt = 0;
    for (int i = 1; i < nums.size() && cnt < 2; ++i) {
        if (nums[i] >= nums[i-1]) continue;
        ++cnt;
        if (i - 2 >= 0 && nums[i-2] > nums[i]) nums[i] = nums[i-1];
        else nums[i-1] = nums[i];
    }
    return cnt <= 1;
}
```

### 667. Beautiful Arrangement II

Given two integers `n` and `k`, you need to construct a list which contains `n`different positive integers ranging from `1` to `n` and obeys the following requirement: Suppose this list is [a1, a2, a3, ... , an], then the list [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] has exactly `k` distinct integers.

If there are multiple answers, print any of them.

Example:

```
Input: n = 3, k = 2
Output: [1, 3, 2]
```

Solution: 让前 k+1 个元素构建出 k 个不相同的差值，序列为：1 k+1 2 k 3 k-1 ... k/2 k/2+1

```cpp
vector<int> constructArray(int n, int k) {
    vector<int> ret(n);
    ret[0] = 1;
    for (int i = 1, interval = k; i <= k; ++i, --interval) {
        ret[i] = i % 2 == 1 ? ret[i - 1] + interval : ret[i - 1] - interval;
    }
    for (int i = k + 1; i < n; ++i) {
        ret[i] = i + 1;
    }
    return ret;
}
```

### 669. Trim a Binary Search Tree

Given a binary search tree and the lowest and highest boundaries as `L` and `R`, trim the tree so that all its elements lies in `[L, R]` (R >= L). You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

Example:

```
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
```

Solution: dfs

```cpp
TreeNode* trimBST(TreeNode* root, int L, int R) {
    if (!root) return root;
    if (root->val > R) return trimBST(root->left, L, R);
    if (root->val < L) return trimBST(root->right, L, R);
    root->left = trimBST(root->left, L, R);
    root->right = trimBST(root->right, L, R);
    return root;
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

### 680. Valid Palindrome II

Given a non-empty string `s`, you may delete **at most** one character. Judge whether you can make it a palindrome.

Example:

```
Input: "abca"
Output: True (You could delete the character 'c'.)
```

Solution: 先找到出现mismatch的位置，然后尝试删掉其中一个字符

```cpp
bool validPalindrome(string s) {
    int i = -1, j = s.length();
    while (++i < --j) if (s[i] != s[j]) return isValid(s, i, j - 1) || isValid(s, i + 1, j);
    return true;
}
bool isValid(string s, int i, int j) {
    while (i < j) if (s[i++] != s[j--]) return false;
    return true;
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

### 687. Longest Univalue Path

Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may or may not pass through the root.

The length of path between two nodes is represented by the number of edges between them.

Example:

```
Input:
       5
      / \
     4   5
    / \   \
   1   1   5
Output: 2
```

Solution: dfs，十分巧妙，一定要背

```cpp
int longestUnivaluePath(TreeNode* root) {
    int path = 0;
    helper(root, path);
    return path;
}
int helper(TreeNode* root, int& path){
    if (!root) return 0;
    int left = helper(root->left, path);
    int right = helper(root->right, path);
    int left_path = root->left && root->left->val == root->val? left + 1: 0;
    int right_path = root->right && root->right->val == root->val? right + 1: 0;
    path = max(path, left_path + right_path);
    return max(left_path, right_path);
}
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
        } else posLeft[i] = posLeft[i-1];
    }
    // DP for starting index of the right max sum interval
    // caution: the condition is ">= tot" for right interval, and "> tot" for left interval
    for (int i = n-k-1, tot = sum[n]-sum[n-k]; i >= 2*k - 1; i--) {
        if (sum[i+k]-sum[i] >= tot) {
            posRight[i] = i;
            tot = sum[i+k]-sum[i];
        } else posRight[i] = posRight[i+1];
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

### 693. Binary Number with Alternating Bits

Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have different values.

Example:

```
Input: 5
Output: True (101)
```

Solution: 对于 1010 这种位级表示的数，把它向右移动 1 位得到 101，这两个数每个位都不同，因此异或得到的结果为 1111。

```cpp
bool hasAlternatingBits(int n) {
    long long a = (n ^ (n >> 1));
    return !(a & (a + 1));
}
```

### 695. Max Area of Island

Given a non-empty 2D array `grid` of 0's and 1's, an **island** is a group of `1`'s (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

Example:

```
Input:
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
 Output: 6
```

Solution: dfs

```cpp
int maxAreaOfIsland(vector<vector<int>>& grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int max_area = 0;
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[0].size(); ++j) {
            max_area = max(max_area, dfs(grid, i, j));
        }
    }
    return max_area;
}

int dfs(vector<vector<int>>& grid, int r, int c) {
    if (r < 0 || r >= grid.size() || c < 0 || c >= grid[0].size() || grid[r][c] == 0) return 0;
    grid[r][c] = 0;
    return 1 + dfs(grid, r + 1, c) + dfs(grid, r - 1, c) + dfs(grid, r, c + 1) + dfs(grid, r, c - 1);
}
```

### 696. Count Binary Substrings

Give a string `s`, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

Example:

```
Input: "00110011"
Output: 6 ("0011", "01", "1100", "10", "0011", and "01")
```

Solution: 维护一个pre一个cur，遍历一遍即可，每当prev大于等于cur则说明出现一个长度为2 * cur的合法子串

```cpp
int countBinarySubstrings(string s) {
    int pre = 0, cur = 1, cnt = 0;
    for (int i = 1; i < s.length(); ++i) {
        if (s[i] == s[i-1]) {
            ++cur;
        } else {
            pre = cur;
            cur = 1;
        }
        if (pre >= cur) ++cnt;
    }
    return cnt;
}
```

### 697. Degree of an Array

Given a non-empty array of non-negative integers `nums`, the **degree** of this array is defined as the maximum frequency of any one of its elements.

Your task is to find the smallest possible length of a (contiguous) subarray of `nums`, that has the same degree as `nums`.

Example:

```
Input: [1, 2, 2, 3, 1]
Output: (The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2)
```

Solution: 记录每个字符的总个数，出现的第一个位置，和出现的最后一个位置

```cpp
int findShortestSubArray(vector<int>& nums) {
    unordered_map<int, int> cnts, first_indices, last_indices;
    int max_cnt = 0;
    for (int i = 0; i < nums.size(); ++i) {
        int num = nums[i];
        ++cnts[num];
        max_cnt = max(max_cnt, cnts[num]);
        last_indices[num] = i;
        if (!first_indices.count(num)) first_indices[num] = i;
    }
    int ret = nums.size();
    for (const auto& [num, cnt]: cnts) {
        if (cnt != max_cnt) continue;
        ret = min(ret, last_indices[num] - first_indices[num] + 1);
    }
    return ret;
}
```