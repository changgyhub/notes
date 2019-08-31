# LeetCode 501 - 550

### 503. Next Greater Element II

Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element. The Next Greater Number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.

Example:

```
Input: [1,2,1]
Output: [2,-1,2]
```

Solution: 单调栈，循环两次

```cpp
vector<int> nextGreaterElements(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, -1);
    stack<int> indices;
    for (int i = 0; i < n * 2; ++i) {
        int num = nums[i % n];
        while (!indices.empty()) {
            int pre_index = indices.top();
            if (num <= nums[pre_index]) break;
            indices.pop();
            res[pre_index] = num;
        }
        if (i < n) indices.push(i);
    }
    return res;
}
```

### 504. Base 7

Given an integer, return its base 7 string representation.

Example:

```
Input: 100
Output: "202"
```

Solution: 按mod不断处理即可，注意细节

```cpp
string convertToBase7(int num) {
    if (num == 0) return "0";
    bool is_negative = num < 0;
    if (is_negative) num = -num;
    string res;
    while (num) {
        int a = num / 7;
        int b = num % 7;
        res = to_string(b) + res;
        num = a;
    }
    return is_negative? "-" + res: res;
}
```

### 508. Most Frequent Subtree Sum

Given the root of a tree, you are asked to find the most frequent subtree sum. The subtree sum of a node is defined as the sum of all the node values formed by the subtree rooted at that node (including the node itself). So what is the most frequent subtree sum value? If there is a tie, return all the values with the highest frequency in any order.

Example:

```
Input:
  5
 /  \
2   -5
Output: [2] (since 2 happens twice, while -5 only occur once)
```

Solution: hastmap+dfs

```cpp
vector<int> findFrequentTreeSum(TreeNode* root) {
		if (!root) return vector<int>();
    int global = 0;
    unordered_map<int, int> sumfreq;
    subTreeSum(sumfreq, root);
    for (auto i: sumfreq) global = max(global, i.second);
    vector<int> ret;
    for (auto i: sumfreq) if (i.second == global) ret.push_back(i.first);
    return ret;
}

int subTreeSum(unordered_map<int, int> &sumfreq, TreeNode* root) {
    if (!root) return 0;
    int sum = subTreeSum(sumfreq, root->left) + subTreeSum(sumfreq, root->right) + root->val;
    ++sumfreq[sum];
    return sum;
}
```

### 513. Find Bottom Left Tree Value

Given a binary tree, find the leftmost value in the last row of the tree.

Example:

```
Input:
        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

Output: 7
```

Solution: bfs

```cpp
int findBottomLeftValue(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        root = q.front();
        q.pop();
        if (root->right) q.push(root->right);
        if (root->left) q.push(root->left);
    }
    return root->val;
}
```

### 515. Find Largest Value in Each Tree Row

Find the largest value in each row of a binary tree.

Example:

```
Input: 
          1
         / \
        3   2
       / \   \  
      5   3   9 
Output: [1, 3, 9]
```

Solution: dfs or bfs

```cpp
vector<int> largestValues(TreeNode* root) {
    vector<int> ret;
    if (!root) return ret;
    queue<TreeNode*> q;
    q.push(root);
    int cap = 1, localmax;
    TreeNode *cur;
    while (cap) {
        localmax = INT_MIN;
        while (cap--) {
            cur = q.front();
            q.pop();
            localmax = max(localmax, cur->val);
            if (cur->left) q.push(cur->left);
            if (cur->right) q.push(cur->right);
        }
        cap = q.size();
        ret.push_back(localmax);
    }
    return ret;
}
```

### 516. Longest Palindromic Subsequence

Given a string s, find the longest palindromic subsequence's length in s.

Example:

```
Input: "bbbab"
Output: 4 ("bbbb")
```

Solution: dp，不同于5. Longest Palindromic Substring，因为是subsequence，dp\[i][j]不是表示起止位置是不是回文，而是表示起止位置内是否包含可以变成回文的子串，一定要理解

```cpp
int longestPalindromeSubseq(string s) {
    int n = s.length();
    if (n < 2) return n;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    dp[n-1][n-1] = 1;
    for (int i = n - 2; i >= 0; --i) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) dp[i][j] = max(dp[i][j-1], max(dp[i+1][j], dp[i+1][j-1] + 2));
            else dp[i][j] = max(dp[i][j-1], dp[i+1][j]);
        }
    }
    int global = dp[0][n-1];
    return global;
}
```

### 521. Longest Uncommon Subsequence I

Given a group of two strings, you need to find the longest uncommon subsequence of this group of two strings. The longest uncommon subsequence is defined as the longest subsequence of one of these strings and this subsequence should not be **any **subsequence of the other strings.

A **subsequence** is a sequence that can be derived from one sequence by deleting some characters without changing the order of the remaining elements. Trivially, any string is a subsequence of itself and an empty string is a subsequence of any string.

The input will be two strings, and the output needs to be the length of the longest uncommon subsequence. If the longest uncommon subsequence doesn't exist, return -1.

Example:

```text
Input: "aba", "cdc"
Output: 3 ("aba" or "cdc")
```

Solution: 如果相同则不存在，否则结果为长度长的那一个

```cpp
int findLUSlength(string a, string b) {
    return a == b? -1: max(a.length(), b.length());
}
```

### 522. Longest Uncommon Subsequence II

Given a list of strings, you need to find the longest uncommon subsequence among them. The longest uncommon subsequence is defined as the longest subsequence of one of these strings and this subsequence should not be **any** subsequence of the other strings.

A **subsequence** is a sequence that can be derived from one sequence by deleting some characters without changing the order of the remaining elements. Trivially, any string is a subsequence of itself and an empty string is a subsequence of any string.

The input will be a list of strings, and the output needs to be the length of the longest uncommon subsequence. If the longest uncommon subsequence doesn't exist, return -1.

Example:

```text
Input: "aba", "cdc", "eae"
Output: 3
```

Solution: 如果有长度最长的，返回这个最长的；次长的如果有重复字符串，则不考虑；对无重复的字符串则判断是否是长度比它长的字符串的子串，否则返回长度；如果不存在则按照这个思路考虑长度更短的。实际操作为hashmap存是否重复的记录，vector + sort/prioirity queue按照长度存储，一定要背

```cpp
int findLUSlength(vector<string>& strs) {
    unordered_map<string, int> mp;
    for (auto str : strs) ++mp[str];
    vector<pair<string, int>> vec;
    for (auto m : mp) vec.push_back(m);
    sort(vec.begin(), vec.end(), [](pair<string,int> &a, pair<string,int> &b){return a.first.size() > b.first.size();});

    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i].second == 1) {
            int j = 0;
            for (; j < i; ++j) if (isSubStringOf(vec[i].first, vec[j].first)) break;
            if (j == i) return vec[i].first.size();        
        }
    }
    return -1;
}

bool isSubStringOf(string &s1, string &s2){
    int i = 0, j = 0;
    while (i < s1.size()) {
        while (j < s2.size() && s1[i] != s2[j]) ++j;
        if (j == s2.size()) return false;
        ++i;
        ++j;
    }
    return true;
}
```

### 523. Continuous Subarray Sum

Given a list of **non-negative** numbers and a target **integer** k, write a function to check if the array has a continuous subarray of size at least 2 that sums up to the multiple of **k**, that is, sums up to n*k where n is also an **integer**.

Example:

```
Input: [23, 2, 4, 6, 7],  k=6
Output: True ([2, 4] sums up to 6)
```

Solution: hashmap+mod

```cpp
bool checkSubarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> map;
    map[0] = -1;
    int running_sum = 0, prev;
    for (int i = 0; i < nums.size(); ++i) {
        running_sum += nums[i];
        if (k) running_sum %= k;
        if (map.find(running_sum) != map.end()) {
            if (i - map[running_sum] > 1)
                return true;
        } else {
            map[running_sum] = i;
        }
    }
    return false;
}
```

### 524. Longest Word in Dictionary through Deleting

Given a string and a string dictionary, find the longest string in the dictionary that can be formed by deleting some characters of the given string. If there are more than one possible results, return the longest word with the smallest lexicographical order. If there is no possible result, return the empty string.

Example:

```
Input: s = "abpcplea", d = ["ale","apple","monkey","plea"]
Output:  "apple"
```

Solution: 正常处理即可

```cpp
string findLongestWord(string s, vector<string>& d) {
    string longest_word = "";
    for (const string & target : d) {
        int l1 = longest_word.length(), l2 = target.length();
        if (l1 > l2 || (l1 == l2 && longest_word < target)) continue;
        if (isValid(s, target)) longest_word = target;
    }
    return longest_word;
}

bool isValid(string s, string target) {
    int i = 0, j = 0;
    while (i < s.length() && j < target.length()) {
        if (s[i++] == target[j]) ++j;
    }
    return j == target.length();
}
```

### 525. Contiguous Array

Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.

Example:

```
Input: [0,1]
Output: 2 ([0, 1])
```

Solution: hashmap记录running sum，遇0减1，遇1加1

```cpp
int findMaxLength(vector<int>& nums) {
    int n = nums.size();
    if (n < 2) return 0;
    unordered_map<int, int> sumpos;
    sumpos[0] = -1;
    int running = 0, cur, maxlen = 0;
    for (int i = 0; i < n; ++i) {
        cur = nums[i];
        if (cur) ++running;
        else --running;
        if (sumpos.find(running) != sumpos.end()) maxlen = max(maxlen, i - sumpos[running]);
        else sumpos[running] = i;
    }
    return maxlen;
}
```

### 528. Random Pick with Weight

Given an array `w` of positive integers, where `w[i]` describes the weight of index `i`, write a function `pickIndex` which randomly picks an index in proportion to its weight.

Explanation of Input Syntax:

The input is two lists: the subroutines called and their arguments. `Solution`'s constructor has one argument, the array `w`. `pickIndex` has no arguments. Arguments are always wrapped with a list, even if there aren't any.

Example:

```
Input:
actions = ["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
arguments = [[[1,3]],[],[],[],[],[]]
Output: [null,0,1,1,1,0]
```

Solution: 先用partial_sum求积分和，然后直接lower_bound获得随机位置

```cpp
class Solution {
public:
    Solution(vector<int> w): sums(w) {
        partial_sum(sums.begin(), sums.end(), sums.begin());
    }
    int pickIndex() {
        return lower_bound(sums.begin(), sums.end(), (rand() % sums.back()) + 1) - sums.begin();
    }
private:
    vector<int> sums;
};
```

### 530. Minimum Absolute Difference in BST

Given a binary search tree with non-negative values, find the minimum [absolute difference](https://en.wikipedia.org/wiki/Absolute_difference) between values of any two nodes.

Example:

```
Input:
   1
    \
     3
    /
   2

Output: 1
```

Solution: 中序遍历

```cpp
int getMinimumDifference(TreeNode* root) {
    int res = INT_MAX, prev = INT_MIN;
    helper(root, prev, res);
    return res;
}
void helper(TreeNode* node, int& prev, int& res) {
    if (!node) return;
    helper(node->left, prev, res);
    if (prev != INT_MIN) res = min(res, node->val - prev);
    prev = node->val;
    helper(node->right, prev, res);
}
```

### 538. Convert BST to Greater Tree

Given a Binary Search Tree \(BST\), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

Example:

```text
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13
Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

Solution: 从右往左dfs，按引用传一个prev变量，一定要背

```cpp
TreeNode* convertBST(TreeNode *root) {
    int prev = 0;
    helper(root, prev);
    return root;
}

void helper(TreeNode *root, int& prev) {
    if (!root) return;
    if (root->right) helper(root->right, prev);
    root->val += prev;
    prev = root->val;
    if (root->left) helper(root->left, prev);
}
```

### 540. Single Element in a Sorted Array

Given a sorted array consisting of only integers where every element appears exactly twice except for one element which appears exactly once. Find this single element that appears only once.

Example:

```
Input: [1,1,2,3,3,4,4,8,8]
Output: 2
```

Solution: bit manipulation

```cpp
int singleNonDuplicate(vector<int>& nums) {
    int ret = 0;
    for (auto n: nums) ret ^= n;
    return ret;
}
```

### 542. 01 Matrix

Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.

Example:

```
Input:
[[0,0,0],
 [0,1,0],
 [0,0,0]]
Output:
[[0,0,0],
 [0,1,0],
 [0,0,0]]
```

Solution: two-pass dp

```cpp
vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
    if (matrix.empty()) return {};
    int n = matrix.size(), m = matrix[0].size();
    vector<vector<int>> dp(n,vector<int> (m, INT_MAX-1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (!matrix[i][j]) dp[i][j] = 0;
            else {
                if (j > 0) dp[i][j] = min(dp[i][j], dp[i][j-1]+1);
                if (i > 0) dp[i][j] = min(dp[i][j], dp[i-1][j]+1);
            }
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            if (!matrix[i][j]) continue;
            else {
                if (j < m - 1) dp[i][j] = min(dp[i][j], dp[i][j+1]+1);
                if (i < n - 1) dp[i][j] = min(dp[i][j], dp[i+1][j]+1);
            }
        }
    }
    return dp;
}
```

### 543. Diameter of Binary Tree

Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:

```text
Input: 
      1
     / \
    2   3
   / \     
  4   5    
Output: 3 ([4,2,1,3] or [5,2,1,3])
```

Solution: 两种方法：\(1\) 递归max\(左树最大半径，右树最大半径，左深度+右深度\); \(2\) 递归测深度+记录全局最大值。一定一定要背

```cpp
// method 1: depth and local max
int diameterOfBinaryTree(TreeNode* root) {
    return root? max(depth(root->left) + depth(root->right), max(diameterOfBinaryTree(root->left), diameterOfBinaryTree(root->right))): 0;
}
int depth(TreeNode* root){
    return root? 1 + max(depth(root->left), depth(root->right)): 0;
}

// method 2: depth and global max
int diameterOfBinaryTree(TreeNode* root) {
    int diameter = 0;
    helper(root, diameter);
    return diameter;
}
int helper(TreeNode* node, int& diameter) {
    if (!node) return 0;
    int l = helper(node->left, diameter)；
    int r = helper(node->right, diameter);
    diameter = max(l + r, diameter);
    return max(l, r) + 1;
}
```

### 547. Friend Circles

There are **N** students in a class. Some of them are friends, while some are not. Their friendship is transitive in nature. For example, if A is a **direct** friend of B, and B is a **direct** friend of C, then A is an **indirect** friend of C. And we defined a friend circle is a group of students who are direct or indirect friends.

Given a **N\*N** matrix **M** representing the friend relationship between students in the class. If M[i][j] = 1, then the ith and jth students are **direct** friends with each other, otherwise not. And you have to output the total number of friend circles among all the students.

Example:

```
Input: [
   [1,1,0],
   [1,1,0],
   [0,0,1]
]
Output: 2 (cicles: [0, 1], [2])
```

Solution: dfs

```cpp
int findCircleNum(vector<vector<int>>& M) {
    int n = M.size(), cnt = 0;
    vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfs(M, i, visited);
            ++cnt;
        }
    }
    return cnt;
}

void dfs(vector<vector<int>>& M, int i, vector<bool>& visited) {
    visited[i] = true;
    for (int k = 0; k < M.size(); ++k) {
        if (M[i][k] == 1 && !visited[k]) {
            dfs(M, k, visited);
        }
    }
}
```