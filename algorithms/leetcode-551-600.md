# LeetCode 551 - 600

### 559. Maximum Depth of N-ary Tree

Given a n-ary tree, find its maximum depth. The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Example: given a `3-ary` tree

![img](../.gitbook/assets/narytreeexample.png)

We should return its max depth, which is 3.

Solution: dfs

```cpp
int maxDepth(Node* root) {
    if (!root) return 0;
    int local_max = 0;
    for (auto * node: root->children) {
        if (node)
            local_max = max(local_max, maxDepth(node));
    }
    return local_max + 1;
}
```

### 560. Subarray Sum Equals K

Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

Example:

```text
Input: nums = [1,1,1], k = 2
Output: 2
```

Solution: 两层loop（慢）或hashmap（快），一定一定要背

```cpp
// method 1: double loop
int subarraySum(vector<int>& nums, int k) {
    int sum, count = 0;
    for (int i = 0; i < nums.size(); ++i) {
        sum = 0;
        for (int j = i; j < nums.size(); ++j) {
            sum += nums[j];
            if (sum == k) ++count;
        }
    }
    return count;
}

// method 2: hashmap
int subarraySum(vector<int>& nums, int k) {
    int count = 0, sum = 0;
    unordered_map<int, int> hash;
    hash[0] = 1;
    for (int i: nums) {
        sum += i;
        count += hash[sum-k];
        ++hash[sum];
    }
    return count;
}
```

### 572. Subtree of Another Tree

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example:

```text
Input: s =       t =
          3          4
         / \        / \
        4   5      1   2
       / \
      1   2
Output: true
```

Solution: 两种方法 \(1\) 传一个是否开始的参数; \(2\) 先递归再compare，起到隐式判定的效果，一定要背和理解

```cpp
// method 1: pass value
bool isSubtree(TreeNode* s, TreeNode* t, bool started=false) {
    if (!s && !t) return true;
    if (!s || !t) return false;
    if (started) return s->val == t->val && isSubtree(s->left, t->left, true) && isSubtree(s->right, t->right, true);
    if (s->val == t->val && isSubtree(s->left, t->left, true) && isSubtree(s->right, t->right, true)) return true;
    return isSubtree(s->left, t) || isSubtree(s->right, t);
}

// method 2: recursion then compare
bool isSubtree(TreeNode* s, TreeNode* t) {
    if (!s) return false;
    return (!isSubtree(s->left, t) && !isSubtree(s->right, t))? compare(s, t): true;
}
bool compare(TreeNode* a, TreeNode* b){
    if (!a && !b) return true;
    if (!a || !b) return false;
    return (a->val == b->val && compare(a->left, b->left) && compare(a->right, b->right));
}
```

### 581. Shortest Unsorted Continuous Subarray

Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too. You need to find the shortest such subarray and output its length.

Example:

```text
Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5 ([6, 4, 8, 10, 9])
```

Solution: 先双指针推进找递增序列，然后计算中间剩余部分的min和max，再把在这个范围的双指针外的部分加入进来，一定要背

```cpp
int findUnsortedSubarray(vector<int>& nums) {
    int shortest = 0, l = 0, r = nums.size() - 1;

    // step 1: find increasing ends
    while (l < r && nums[l] <= nums[l+1]) ++l;
    while (l < r && nums[r] >= nums[r-1]) --r;
    if (l == r) return 0;

    // step 2: find min and max in the middle part
    int vmin = INT_MAX, vmax = INT_MIN;
    for (int i = l; i <= r; ++i) {
        if (nums[i] > vmax) vmax = nums[i];
        if (nums[i] < vmin) vmin = nums[i];
    }

    // step 3: include those in range
    while (l >= 0 && nums[l] > vmin) --l;
    while (r < nums.size() && nums[r] < vmax) ++r;

    return r - l - 1;
}
```

### 593. Valid Square

Given the coordinates of four points in 2D space, return whether the four points could construct a square. The coordinate (x,y) of a point is represented by an integer array with two integers.

Example:

```
Input: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
Output: True
```

Solution: 对角线垂直平分且相等，可以先sort一下顺序

```cpp
bool validSquare(vector<int>& p1, vector<int>& p2, vector<int>& p3, vector<int>& p4) {
    vector<pair<int, int>> square{{p1[0], p1[1]}, {p2[0], p2[1]}, {p3[0], p3[1]}, {p4[0], p4[1]}};
    sort(square.begin(), square.end(), [](pair<int, int>& a, pair<int, int>& b){return a.first < b.first || (a.first == b.first && a.second < b.second);});
    return dist(square[0], square[3]) == dist(square[1], square[2]) && bisect(square[3], square[0], square[1], square[2]);
}

double dist(pair<int, int>& a, pair<int, int>& b){
    double c = b.second - a.second, d = b.first - a.first;
    return c*c + d*d;
}

bool perpendicular(pair<int, int>& a, pair<int, int>& b, pair<int, int>& c, pair<int, int>& d) {
    double l, r;
    if (a.first == b.first) {
        if (a.second == b.second) return false;
        return c.second == d.second && c.first != d.first;
    } l = (a.second - b.second) / (double)(a.first - b.first);
    if (c.first == d.first) {
        if (c.second == d.second) return false;
        return a.second == b.second;
    } r = (c.second - d.second) / (double)(c.first - d.first);
    return abs(l * r + 1) < 1e-8;
}

bool bisect(pair<int, int>& a, pair<int, int>& b, pair<int, int>& c, pair<int, int>& d) {
    if (!perpendicular(a, b, c, d)) return false;
    return a.second + b.second == c.second + d.second && a.first + b.first == c.first + d.first;
}
```