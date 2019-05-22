# LeetCode 501 - 550

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
int diameterOfBinaryTree(TreeNode* node, int& diameter) {
    if (!node) return 0;
    int l = diameterOfBinaryTree(node->left, diameter), r = diameterOfBinaryTree(node->right, diameter);
    diameter = max(l + r, diameter);
    return max(l, r) + 1;
}
int diameterOfBinaryTree(TreeNode* root) {
    int diameter = 0;
    int d = diameterOfBinaryTree(root, diameter);
    return diameter;
}
```

