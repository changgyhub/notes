# LeetCode 551 - 600

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

