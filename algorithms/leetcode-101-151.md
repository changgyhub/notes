# LeetCode 101 - 150

### 101. Symmetric Tree

Given a binary tree, check whether it is a mirror of itself \(ie, symmetric around its center\).

Example:

```text
Input:
    1
   / \
  2   2
 / \ / \
3  4 4  3

Output: true
```

Solution: 递归，加helper，左边的右边等于右边的左边

```cpp
bool mySymmetric(TreeNode* left, TreeNode* right) {
    if (left && right && left->val == right->val)
        return mySymmetric(left->left, right->right) && mySymmetric(left->right, right->left);
    return !left && !right;
}
bool isSymmetric(TreeNode *root) {
    return root? mySymmetric(root->left, root->right): true;
}
```

### 102. Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. \(ie, from left to right, level by level\).

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: [
  [3],
  [9,20],
  [15,7]
]
```

Solution: BFS，可以写成递归或者queue，一定要背

```cpp
// recursion
vector<vector<int>> result;  
void buildVector(TreeNode* root, int depth)  {  
    if (!root) return;  
    if (result.size() == depth) result.push_back(vector<int>());
    result[depth].push_back(root->val);  
    buildVector(root->left, depth + 1);  
    buildVector(root->right, depth + 1);  
}  
vector<vector<int>> levelOrder(TreeNode* root) {  
    buildVector(root, 0);  
    return result;  
}

// queue
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return vector<vector<int>>{};
    vector<vector<int>> res;
    vector<int> row;
    TreeNode *last, *pre_last = root, *cur = root;
    queue<TreeNode*> nodes;
    nodes.push(cur);
    while (!nodes.empty()) {
        cur = nodes.front();
        row.push_back(cur->val);
        if (cur->left) {
            nodes.push(cur->left);
            last = cur->left;
        }
        if (cur->right) {
            nodes.push(cur->right);
            last = cur->right;
        }
        if (cur == pre_last) {
            res.push_back(row);
            row.clear();
            pre_last = last;
        }
        nodes.pop();
    }
    return res;
}
```

### 103. Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. \(ie, from left to right, then right to left for the next level and alternate between\).

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: [
  [3],
  [20,9],
  [15,7]
]
```

Solution: 同Q102, 用BFS，可以写成递归或者queue; 对奇数层可以事前翻转也可以事后翻转

```cpp
// recursion, pre-process (might be slow with vector since elements after need moving)
vector<vector<int>> result;  
void buildVector(TreeNode* root, int depth)  {  
    if (!root) return;  
    if (result.size() == depth) result.push_back(vector<int>());
    if (depth % 2) result[depth].insert(result[depth].begin(), root->val);
    else result[depth].push_back(root->val);  
    buildVector(root->left, depth + 1);  
    buildVector(root->right, depth + 1);  
}  
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {  
    buildVector(root, 0);  
    return result;  
}

// queue, post-process
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    if (!root) return vector<vector<int>>{};
    vector<vector<int>> res;
    vector<int> row;
    TreeNode *last, *pre_last = root, *cur = root;
    queue<TreeNode*> nodes;
    nodes.push(cur);
    while (!nodes.empty()) {
        cur = nodes.front();
        row.push_back(cur->val);
        if (cur->left) {
            nodes.push(cur->left);
            last = cur->left;
        }
        if (cur->right) {
            nodes.push(cur->right);
            last = cur->right;
        }
        if (cur == pre_last) {
            res.push_back(row);
            row.clear();
            pre_last = last;
        }
        nodes.pop();
    }
    for (int i = 1; i < res.size(); i += 2) {  
        reverse(res[i].begin(), res[i].end());  
    }
    return res;
}
```

### 104. Maximum Depth of Binary Tree

Given a binary tree, find its maximum depth.

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: 3
```

Solution: 递归

```cpp
int maxDepth(TreeNode* root) {
    return root? 1 + max(maxDepth(root->left), maxDepth(root->right)): 0;
}
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

Example:

```text
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output:
    3
   / \
  9  20
    /  \
   15   7
```

Solution: 多种解法\(stack, dfs, map\)，核心是通过preorder找到root，然后分左右递归/迭代，一定要仔细体会

```cpp
// map
unordered_map<int, int> map;
TreeNode* buildTreeHelper(vector<int>& preorder, int s0, int e0, int s1) {
    if (s0 > e0) return NULL;
    int mid = preorder[s1], index = map[mid], leftLen = index - s0 - 1;
    TreeNode* node = new TreeNode(mid);
    node->left  = buildTreeHelper(preorder, s0, index-1, s1+1);
    node->right = buildTreeHelper(preorder, index+1, e0, s1+2+leftLen);
    return node;
}        

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    if (preorder.empty()) return NULL;
    for (int i = 0; i < preorder.size(); ++i) map[inorder[i]] = i;
    return buildTreeHelper(preorder, 0, preorder.size()-1, 0);
}

// dfs
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int cur = 0, i = 0, p = preorder.size();
    return dfs(preorder, inorder, cur, i, p);
}

TreeNode* dfs(vector<int>& preorder, vector<int>& inorder, int& cur, int& i, int p) {
    if (i < preorder.size() && (p == preorder.size() || inorder[i] != preorder[p])) {
        TreeNode* ret = new TreeNode(preorder[cur++]);
        ret->left = dfs(preorder, inorder, cur, i, cur-1);
        ++i;
        ret->right = dfs(preorder, inorder, cur, i, p);
        return ret;
    }
    return NULL;
}
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

Given inorder and postorder traversal of a tree, construct the binary tree.

Example:

```text
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output:
    3
   / \
  9  20
    /  \
   15   7
```

Solution: 多种解法\(stack, dfs, map\)，核心是通过postorder找到root，然后分左右递归/迭代，一定要仔细体会

```cpp
// map
unordered_map<int,int> postoin;
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
    if (inorder.empty()) return NULL;
    for (int i = 0; i < inorder.size(); ++i) postoin[inorder[i]] = i;
    return build(postorder, 0, inorder.size()-1, 0, postorder.size()-1);
}

TreeNode* build(vector<int>& postorder, int b0, int e0, int b1, int e1) {
    if (b1 > e1 || b0 > e0) return NULL;
    TreeNode* root = new TreeNode(postorder[e1]);
    int middle = postoin[postorder[e1]];
    root -> left  = build(postorder, b0, middle-1, b1, middle-1-b0+b1);
    root -> right = build(postorder, middle+1, e0, e1-e0+middle, e1-1);
    return root;
}
```

### 107. Binary Tree Level Order Traversal II

Given a binary tree, return the bottom-up level order traversal of its nodes' values. \(ie, from left to right, level by level from leaf to root\).

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: [
  [15,7],
  [3],
  [9,20]
]
```

Solution: 同Q102, reverse结果即可

```cpp
// recursion
vector<vector<int>> result;  
void buildVector(TreeNode* root, int depth)  {  
    if (!root) return;  
    if (result.size() == depth) result.push_back(vector<int>());
    result[depth].push_back(root->val);  
    buildVector(root->left, depth + 1);  
    buildVector(root->right, depth + 1);  
}  
vector<vector<int>> levelOrder(TreeNode* root) {  
    buildVector(root, 0);
    reverse(result.begin(), result.end());
    return result;  
}

// queue
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return vector<vector<int>>{};
    vector<vector<int>> res;
    vector<int> row;
    TreeNode *last, *pre_last = root, *cur = root;
    queue<TreeNode*> nodes;
    nodes.push(cur);
    while (!nodes.empty()) {
        cur = nodes.front();
        row.push_back(cur->val);
        if (cur->left) {
            nodes.push(cur->left);
            last = cur->left;
        }
        if (cur->right) {
            nodes.push(cur->right);
            last = cur->right;
        }
        if (cur == pre_last) {
            res.push_back(row);
            row.clear();
            pre_last = last;
        }
        nodes.pop();
    }
    reverse(res.begin(), res.end());
    return res;
}
```

### 108. Convert Sorted Array to Binary Search Tree

Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

Example:

```text
Input: [-10,-3,0,5,9]

Output: one solution is
      0
     / \
   -3   9
   /   /
 -10  5
```

Solution: 中分递归

```cpp
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return helper(nums, 0, nums.size());
}

TreeNode* helper(vector<int>& nums, int start, int end) {
    if (start == end) return NULL;
    int mid = (start + end) / 2;
    TreeNode *root = new TreeNode(nums[mid]);
    root->left  = helper(nums, start, mid);
    root->right = helper(nums, mid+1, end);
    return root;
}
```

### 109. Convert Sorted List to Binary Search Tree

Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

Example:

```text
Input: [-10,-3,0,5,9]

Output: one solution is
      0
     / \
   -3   9
   /   /
 -10  5
```

Solution: 快慢指针，一定要背

```cpp
TreeNode* sortedListToBST(ListNode* head) {
    if (!head) return NULL;
    if (!head->next) return new TreeNode(head->val);
    ListNode *slow = head, *fast = head, *prevslow = head;
    while (fast->next && fast->next->next) {
        prevslow = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    TreeNode* root = new TreeNode(slow->val);
    if (prevslow != slow) {
        root->right = sortedListToBST(slow->next);
        prevslow->next = NULL;
        root->left = sortedListToBST(head);
    } else {
        root->right = new TreeNode(slow->next->val);;
    }
    return root;
}
```

### 110. Balanced Binary Tree

Given a binary tree, determine if it is height-balanced.

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: true
```

Solution: 对高度dfs\(递归\)，如果balance返回height，否则返回-1，递归回的时候发现-1则直接变成-1。一定要背

```cpp
bool isBalanced(TreeNode* root) {
    return helper(root) != -1;
}
int helper(TreeNode* root) {
    if (!root) return 0;
    int left = helper(root->left), right = helper(root->right);
    if (left == -1 || right == -1 || abs(left - right) > 1) return -1;
    return 1 + max(left, right);
}
```

### 111. Minimum Depth of Binary Tree

Given a binary tree, find its minimum depth.

Example:

```text
Input:
    3
   / \
  9  20
    /  \
   15   7

Output: 2
```

Solution: 直接dfs\(递归\)，注意一侧有一侧没有时，长度为有的那一侧的最短长度而非0

```cpp
int minDepth(TreeNode* root) {
    return root? (
        root->left? (
            root->right? 1 + min(minDepth(root->left), minDepth(root->right)): 1 + minDepth(root->left)
        ): 1 + minDepth(root->right)
    ): 0;
}
```

### 112. Path Sum

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

Example:

```text
Input: sum = 22,
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1

Output: true
```

Solution: 直接dfs\(递归\)，注意一侧有一侧没有时，只递归长度为有的那一侧，另一侧不算leaf

```cpp
bool helper(TreeNode *root, int sum) {
    int val = root->val;
    if (!root->left && !root->right) return sum == val;
    return (root->left && helper(root->left, sum - val)) || (root->right && helper(root->right, sum - val));
}
bool hasPathSum(TreeNode* root, int sum) {
    return root? helper(root, sum): false;
}
```

### 113. Path Sum II

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Example:

```text
Input: sum = 22,
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1

Output: [
   [5,4,11,2],
   [5,8,4,5]
]
```

Solution: backtrack，一定要背

```cpp
vector<vector<int>> ret;
vector<int> buffer;
void helper(TreeNode* root, int sum) {
    int val = root->val;
    buffer.push_back(val);
    if (!root->left && !root->right) {
        if (sum == val) ret.push_back(buffer);
    } else {
        if (root->left) helper(root->left, sum - val);
        if (root->right) helper(root->right, sum - val);
    }
    buffer.pop_back();
}
vector<vector<int>> pathSum(TreeNode* root, int sum) {
    if (root) helper(root, sum);
    return ret;
}
```

### 114. Flatten Binary Tree to Linked List

Given a binary tree, flatten it to a linked list in-place.

Example:

```text
Input:
    1
   / \
  2   5
 / \   \
3   4   6

Output:
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

Solution: 递归，别忘了改完之后把左边赋值NULL

```cpp
void flatten(TreeNode* root) {
    if (!root) return;
    flatten(root->left);
    flatten(root->right);
    TreeNode* ori_right = root->right;
    root->right = root->left;
    root->left = NULL;
    while (root->right) root = root->right;
    root->right = ori_right;
}
```

### 115. Distinct Subsequences

Given a string S and a string T, count the number of distinct subsequences of S which equals T.

Example:

```text
Input: S = "rabbbit", T = "rabbit"
Output: 3
```

Solution: dp, i和j分别表示母子字符串的位数，如果dp\[i\]\[j\] = S\[i-1\] == T\[j-1\]? dp\[i-1\]\[j-1\] + dp\[i-1\]\[j\]: dp\[i-1\]\[j\]，可以反向缩成一维

```cpp
// 1D
int numDistinct(string s, string t) {
    int m = s.size(), n = t.size();
    if (m < n) return 0;
    vector<int> dp(n+1, 0);
    dp[0] = 1;
    for (int i = 1; i <= m; ++i) {
        int lo = max(i-m+n, 1), hi = min(i, n);  // j no higher than i, no less than i-m+n
        for (int j = hi; j >= lo; --j) {
            if (s[i-1] == t[j-1]) dp[j] += dp[j-1];
        }
    }
    return dp.back();
}

// 2D
int numDistinct(string S, string T) {
    vector<vector<int> > dp(T.length() + 1, vector<int>(S.length() + 1, 0));
    for (int i = 0; i < S.length() + 1; ++i) dp[0][i] = 1;
    for (int i = 1; i < T.length() + 1; ++i) dp[i][0] = 0;
    for (int i = 1; i < T.length() + 1; ++i)
        for (int j = 1; j < S.length() + 1; ++j)
            dp[i][j] = S[j-1] == T[i-1]? dp[i-1][j-1] + dp[i][j-1]: dp[i][j-1];
    return dp.back().back();
}
```

### 116. Populating Next Right Pointers in Each Node

Given a binary tree.

```text
struct TreeLinkNode {
  TreeLinkNode *left;
  TreeLinkNode *right;
  TreeLinkNode *next;
}
```

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL. Initially, all next pointers are set to NULL.

Note: You may only use constant extra space; Recursive approach is fine, implicit stack space does not count as extra space for this problem; You may assume that it is a perfect binary tree \(ie, all leaves are at the same level, and every parent has two children\).

Example:

```text
Input:
     1
   /  \
  2    3
 / \  / \
4  5  6  7
Output:
     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \  / \
4->5->6->7 -> NULL
```

Solution: 递归、while loop或者bfs; 除了要把当前node的left的next连到当前node的right之外, \(如果合法\)还要把当前node的right的next连到当前node的next的left

```cpp
// recursive
void connect(TreeLinkNode *root) {
    if (!root) return;
    if (root->left) {
        root->left->next = root->right;
        if (root->next) root->right->next = root->next->left;
    }
    connect(root->left);
    connect(root->right);
}

// while loop
void connect(TreeLinkNode *root) {
    if (!root) return;
    while (root->left) {
        TreeLinkNode *p = root;
        while (p) {
            p->left->next = p->right;
            if (p->next) p->right->next = p->next->left;
            p = p->next;
        }
        root = root->left;
    }
}
```

### 117. Populating Next Right Pointers in Each Node II

Given a binary tree.

```text
struct TreeLinkNode {
  TreeLinkNode *left;
  TreeLinkNode *right;
  TreeLinkNode *next;
}
```

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL. Initially, all next pointers are set to NULL.

Note: You may only use constant extra space; Recursive approach is fine, implicit stack space does not count as extra space for this problem.

Example:

```text
Input:
     1
   /  \
  2    3
 / \    \
4   5    7
Output:
     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \    \
4-> 5 -> 7 -> NULL
```

Solution: 相比Q116, 直接用bfs能比较方便解决树不满叶的问题, 注意处理每层最后一个节点

```cpp
void connect(TreeLinkNode *root) {
    if (!root) return;
    queue<TreeLinkNode*> queue;
    int nodesPerLayer = 1;
    queue.push(root);
    TreeLinkNode* temp;

    while (!queue.empty()) {
        --nodesPerLayer;
        temp = queue.front();
        queue.pop();
        if (temp->left) queue.push(temp->left);
        if (temp->right) queue.push(temp->right);

        if (nodesPerLayer) {
            temp->next = queue.front();
        }
        else {
            temp->next = nullptr;
            nodesPerLayer = queue.size();
        }
    }
}
```

### 118. Pascal Triangle

Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.

Example:

```text
Input: 5
Output: [
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

Solution: dp, row\[j\] = ret\[i-1\]\[j-1\] + ret\[i-1\]\[j\]

```cpp
vector<vector<int>> generate(int numRows) {
    vector<vector<int>> ret;
    for (int i = 0; i < numRows; ++i) {
        vector<int> row(i+1, 1);
        for (int j = 1; j < i ; ++j) {
            row[j] = ret[i-1][j-1] +  ret[i-1][j];
        }
        ret.push_back(row);
    }
    return ret;
}
```

### 119. Pascal Triangle II

Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle. Note that the row index starts from 0.

Example:

```text
Input: 3
Output: [1,3,3,1]
```

Solution: dp, 同Q118，可压成反向一维

```cpp
vector<int> getRow(int rowIndex) {
    vector<int> row(rowIndex+1, 1);
    for (int i = 0; i < rowIndex; ++i)
        for (int j = i; j > 0 ; --j)
            row[j] += row[j-1];
    return row;
}
```

### 120. Triangle

Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

Example:

```text
Input: [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
Output: 11
```

Solution: 从下往上dp, triangle\[i\]\[j\] += min\(triangle\[i+1\]\[j+1\],triangle\[i+1\]\[j\]\)

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    for (int i = triangle.size() - 2; i >= 0; --i)
        for (int j = 0; j <= i; ++j)
            triangle[i][j] += min(triangle[i+1][j+1],triangle[i+1][j]);
    return triangle[0][0];
}
```

### 121. Best Time to Buy and Sell Stock

Say you have an array for which the ith element is the price of a given stock on day i. If you were only permitted to complete at most one transaction \(i.e., buy one and sell one share of the stock\), design an algorithm to find the maximum profit.

Example:

```text
Input: [7,1,5,3,6,4]
Output: 5
```

Solution: 记录historical min price，遍历一遍求max diff。所有此类问题见[此链接](http://liangjiabin.com/blog/2015/04/leetcode-best-time-to-buy-and-sell-stock.html)

```cpp
int maxProfit(vector<int>& prices) {
    if (prices.empty()) return 0;
    int minpirce = prices[0], maxprofit = 0;
    for (int& price: prices) {
        maxprofit = max(maxprofit, price - minpirce);
        minpirce = min(minpirce, price);
    }
    return maxprofit;
}
```

### 122. Best Time to Buy and Sell Stock II

Say you have an array for which the ith element is the price of a given stock on day i. Design an algorithm to find the maximum profit. You may complete as many transactions as you like \(i.e., buy one and sell one share of the stock multiple times\). Note: You must sell the stock before you buy again.

Example:

```text
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```

Solution: 贪心, maxprofit += max\(prices\[i\] - prices\[i-1\], 0\)

```cpp
int maxProfit(vector<int>& prices) {
    int maxprofit = 0;
    for (int i = 1; i < prices.size(); ++i)
        if (prices[i] > prices[i-1])
            maxprofit += prices[i] - prices[i-1];
    return maxprofit;
}
```

### 123. Best Time to Buy and Sell Stock III

Say you have an array for which the ith element is the price of a given stock on day i. Design an algorithm to find the maximum profit. You may complete at most two transactions. Note: You must sell the stock before you buy again.

Example:

```text
Input: [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
             Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
```

Solution: 对每个i分左右，跑Q121\(只能交易一次\)即可; 也可以缩短成一个loop的写法, 维护两个buy和两个sell的变量

```cpp
// original
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n < 2) return 0;
    vector<int> preProfit(n, 0), postProfit(n, 0);

    int curMin = prices[0];
    for (int i = 1; i < n; ++i) {
        curMin = min(curMin, prices[i]);
        preProfit[i] = max(preProfit[i - 1], prices[i] - curMin);
    }

    int curMax = prices[n-1];
    for (int i = n - 2; i >= 0; --i) {
        curMax = max(curMax, prices[i]);
        postProfit[i] = max(postProfit[i + 1], curMax - prices[i]);
    }

    int maxProfit = 0;
    for (int i = 0; i < n; ++i) {
        maxProfit = max(maxProfit, preProfit[i] + postProfit[i]);
    }

    return maxProfit;
}

// simplify
int maxProfit(vector<int>& prices) {
    int sell1 = 0, sell2 = 0, buy1 = INT_MIN, buy2 = INT_MIN;
    for (int i = 0; i < prices.size(); ++i) {
        buy1 = max(buy1, -prices[i]);
        sell1 = max(sell1, buy1 + prices[i]);
        buy2 = max(buy2, sell1 - prices[i]);
        sell2 = max(sell2, buy2 + prices[i]);
    }
    return sell2;
}
```

### 124. Binary Tree Maximum Path Sum

Given a non-empty binary tree, find the maximum path sum. For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example:

```text
Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42 (15->20->7)
```

Solution: 递归, 更新左右和, 但是只返回一条以自己为根的, 到左侧或右侧的最长路径，这样才可以递归求任意路径; 一定要背和理解

```cpp
int maxPathSum(TreeNode* root) {
    int maxValue = INT_MIN;
    maxPathDown(root, maxValue);
    return maxValue;
}
int maxPathDown(TreeNode* node, int &maxValue) {
    if (!node) return 0;
    int left = max(0, maxPathDown(node->left, maxValue));
    int right = max(0, maxPathDown(node->right, maxValue));
    maxValue = max(maxValue, left + right + node->val);
    return max(left, right) + node->val;
}
```

### 125. Valid Palindrome

Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases. We define empty string as valid palindrome.

Example:

```text
Input: "A man, a plan, a canal: Panama"
Output: true
```

Solution: 递归的话长串会栈溢出，先处理再循环的话会超时，正确做法是直接跳过非数字字母然后比对，一定要背

```cpp
bool isPalindrome(string s) {
    int i = 0, j = s.size() - 1;
    while (i < j) {
        while (!isalnum(s[i])) i++;
        while (i < j && !isalnum(s[j])) j--;
        if (i < j && toupper(s[i]) != toupper(s[j])) return false;
        ++i; --j;
    }
    return true;
}
```

### 126. Word Ladder II

Given two words \(beginWord and endWord\), and a dictionary's word list, find all shortest transformation sequence\(s\) from beginWord to endWord, such that \(1\) only one letter can be changed at a time, and \(2\) each transformed word must exist in the word list. Note that beginWord is not a transformed word; Return 0 if there is no such transformation sequence; All words have the same length; All words contain only lowercase alphabetic characters; You may assume no duplicates in the word list; You may assume beginWord and endWord are non-empty and are not the same.

Example:

```text
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
```

Solution: 类似Q127，但是可以采取各种剪枝叶措施，暂时略

### 127. Word Ladder

Given two words \(beginWord and endWord\), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that \(1\) only one letter can be changed at a time, and \(2\) each transformed word must exist in the word list. Note that beginWord is not a transformed word; Return 0 if there is no such transformation sequence; All words have the same length; All words contain only lowercase alphabetic characters; You may assume no duplicates in the word list; You may assume beginWord and endWord are non-empty and are not the same.

Example:

```text
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5 ("hit" -> "hot" -> "dot" -> "dog" -> "cog")
```

Solution: 用hashset辅助完成dfs查找：设置一个begin和一个end的hashset，从begin开始，变一个字符并放入可达到的string，知道率先能在end找到；如果begin大于end，则交换二者提速。一定要背

```cpp
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> begin, end, words;
    int ret = 1;
    if (wordList.empty()) return 0;
    for (auto i : wordList) words.insert(i);
    if (!words.count(endWord)) return 0;
    begin.insert(beginWord);
    end.insert(endWord);

    while (!begin.empty() && !end.empty()) {
        unordered_set<string> *head, *tail;
        if (begin.size() > end.size()) {
            head = &end;
            tail = &begin;
        } else {
            head = &begin;
            tail = &end;
        }
        unordered_set<string> tmp;
        for (auto j : *head) {
            for (int k = 0; k < j.size(); ++k) {
                char ch = j[k];
                for (char m = 'a'; m < 'z'; ++m) {
                    if (m == ch) continue;
                    j[k] = m;
                    if ((*tail).count(j)) return ret + 1;
                    if (words.count(j)) tmp.insert(j);
                }
                j[k] = ch;
            }
            words.erase(j);
        }
        ++ret;
        swap(*head, tmp);
    }
    return 0;
}
```

### 128. Longest Consecutive Sequence

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Example:

```text
Input: [100, 4, 200, 1, 3, 2]
Output: 4 ([1, 2, 3, 4])
```

Solution: 先建hashset，不断取它的head，向前加向后减查找是否有相连数并删除，并记录最长累计长度

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> u_set;
    for (const int & a: nums) u_set.insert(a);
    int res = 0;
    while (!u_set.empty()) {
        int cur = *(u_set.begin());
        u_set.erase(cur);
        int next = cur+1, prev = cur-1;
        while (u_set.count(next)) u_set.erase(next++);
        while (u_set.count(prev)) u_set.erase(prev--);
        res = max(res, next-prev-1);
    }
    return res;
}
```

### 129. Sum Root to Leaf Numbers

Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number. Find the total sum of all root-to-leaf numbers.

Example:

```text
Input: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
Output: 495 + 491 + 40 = 1026.
```

Solution: dfs递归，参数传高位数累计结果，return传分支的和，一定要背

```cpp
int helper(TreeNode *root, int sum) {
    if (!root) return 0;
    sum = sum * 10 + root->val;
    if (!root->left && !root->right) return sum;
    return helper(root->left,sum) + helper(root->right,sum);
}

int sumNumbers(TreeNode *root) {
    return helper(root, 0);
}
```

### 130. Surrounded Regions

Given a 2D board containing 'X' and 'O' \(the letter O\), capture all regions surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:

```text
Input:
X X X X
X O O X
X X O X
X O X X

Output:
X X X X
X X X X
X X X X
X O X X
```

Solution: dfs边界（因为只有边界的水域不会被覆盖），一定要背

```cpp
struct POS {
    int x;
    int y;
    POS(int newx, int newy): x(newx), y(newy) {}
};

void solve(vector<vector<char>> &board) {
    if (board.empty() || board[0].empty()) return;
    int m = board.size(),  n = board[0].size();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (board[i][j] == 'O')
                if (i == 0 || i == m-1 || j == 0 || j == n-1) dfs(board, i, j, m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O') board[i][j] = 'X';
            else if (board[i][j] == '*') board[i][j] = 'O';
        }
    }

}
void dfs(vector<vector<char>> &board, int i, int j, int m, int n) {
    stack<POS*> stk;
    POS* pos = new POS(i, j);
    stk.push(pos);
    board[i][j] = '*';
    while (!stk.empty()) {
        POS* top = stk.top();
        if (top->x > 0 && board[top->x-1][top->y] == 'O') {
            POS* up = new POS(top->x-1, top->y);
            stk.push(up);
            board[up->x][up->y] = '*';
            continue;
        }
        if (top->x < m-1 && board[top->x+1][top->y] == 'O') {
            POS* down = new POS(top->x+1, top->y);
            stk.push(down);
            board[down->x][down->y] = '*';
            continue;
        }
        if (top->y > 0 && board[top->x][top->y-1] == 'O') {
            POS* left = new POS(top->x, top->y-1);
            stk.push(left);
            board[left->x][left->y] = '*';
            continue;
        }
        if (top->y < n-1 && board[top->x][top->y+1] == 'O') {
            POS* right = new POS(top->x, top->y+1);
            stk.push(right);
            board[right->x][right->y] = '*';
            continue;
        }
        stk.pop();
    }
}
```

### 131. Palindrome Partitioning

Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

Example:

```text
Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
```

Solution: backtrack

```cpp
vector<vector<string>> partition(string s) {
    vector<vector<string>> result;
    vector<string> path;
    dfs(s, path, result, 0, 1);
    return result;
}

void dfs(string &s, vector<string> &path, vector<vector<string>> &result, int prev, int start) {
    if (start == s.size()) {
        if (ispalindrome(s, prev, start - 1)) {
            path.push_back(s.substr(prev, start - prev));
            result.push_back(path);
            path.pop_back();
        }
        return;
    }

    dfs(s, path, result, prev, start + 1);
    if (ispalindrome(s, prev, start - 1)) {
        path.push_back(s.substr(prev, start - prev));
        dfs(s, path, result, start, start + 1);
        path.pop_back();
    }
}

bool ispalindrome(const string &s, int start, int end) {
    while (start < end && s[start] == s[end]) {
        ++start;
        --end;
    }
    return start >= end;
}
```

### 132. Palindrome Partitioning II

Given a string s, partition s such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning of s.

Example:

```text
Input: "aab"
Output: 1 (["aa","b"])
```

Solution: 不同于Q131，用backtrack会超时，只能用bp。关键步骤：\(1\) min\[i\]表示从i到n切的最少次数，min\[i\]初始化为min\[i+1\]+1，即初始化s\[i\]与s\[i+1\]之间需要切一刀。这里考虑边界问题，因此min数组设为n+1长度 \(2\) 从i到n-1中间如果存在位置j，同时满足s\[i..j\]为回文串且1 + min\[j+1\] &lt; min\[i\], 那么min\[i\] = 1 + min\[j+1\]，也就是说一刀切在j的后面比切在i的后面要好

```cpp
int minCut(string s) {
    int n = s.size();
    vector<vector<bool> > isPalin(n, vector<bool>(n, false));
    vector<int> min(n+1, -1); //min cut from end

    for (int i = 0; i < n; ++i) isPalin[i][i] = true;

    for (int i = n-1; i >= 0; --i) {
        min[i] = min[i+1] + 1;
        for (int j = i+1; j < n; ++j) {
            if (s[i] == s[j]) {
                if (j == i+1 || isPalin[i+1][j-1]) {
                    isPalin[i][j] = true;
                    if (j == n-1) min[i] = 0;
                    else if (min[i] > min[j+1] + 1) min[i] = min[j+1] + 1;
                }
            }
        }
    }

    return min[0];
}
```

### 133. Clone Graph

Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

Definition for undirected graph:

```text
struct UndirectedGraphNode {
    int label;
    vector<UndirectedGraphNode *> neighbors;
    UndirectedGraphNode(int x) : label(x) {};
 };
```

Solution: DFS递归建造图，并用一个hashmap存储已经构造过的节点，一定要背

```cpp
UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node) {
    if (!node) return nullptr;
    unordered_map<int, UndirectedGraphNode*> node_dict;
    return DFS(node, node_dict);
}
UndirectedGraphNode* DFS (UndirectedGraphNode *cur_node, unordered_map<int, UndirectedGraphNode*>& node_dict) {
    if (node_dict.find(cur_node->label) != node_dict.end())
        return node_dict[cur_node->label];

    UndirectedGraphNode* new_node = new UndirectedGraphNode(cur_node->label);
    node_dict.insert({new_node->label, new_node});
    for (auto neighbor: cur_node->neighbors) {
        new_node->neighbors.push_back(DFS(neighbor, node_dict));
    }
    return new_node;
}
```

### 134. Gas Station

There are N gas stations along a circular route, where the amount of gas at station i is gas\[i\]. You have a car with an unlimited gas tank and it costs cost\[i\] of gas to travel from station i to its next station \(i+1\). You begin the journey with an empty tank at one of the gas stations. Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. Note: If there exists a solution, it is guaranteed to be unique; Both input arrays are non-empty and have the same length; Each element in the input arrays is a non-negative integer.

Example:

```text
Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

Output: 3
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
```

Solution: 贪心法，如果总的offset \(gas - cost\) 是正的，那么一定有解，且解在offset最小的下一位取得；这是因为题目规定了解唯一，那么这个位置一定是解。

```cpp
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int sum = 0, start_idx = 0, min_sum = INT_MAX;
    for (int i = 0; i < gas.size(); ++i) {
        sum += gas[i] - cost[i];
        if (sum < min_sum) {
            min_sum = sum;
            start_idx = (i+1) % gas.size();
        }
    }
    return sum < 0? -1: start_idx;
}
```

### 135. Candy

There are N children standing in a line. Each child is assigned a rating value. You are giving candies to these children subjected to the following requirements: \(1\) Each child must have at least one candy \(2\) Children with a higher rating get more candies than their neighbors. What is the minimum candies you must give?

Example:

```text
Input: [1,0,2]
Output: 5 ([2, 1, 2])
```

Solution: 从左到右从右到左各遍历一遍即可

```cpp
int candy(vector<int>& ratings) {
    int size = ratings.size();
    if (size < 2) return size;
    vector<int> num(size, 1);
    for (int i = 1; i < size; ++i)
        if (ratings[i] > ratings[i-1]) num[i] = num[i-1] + 1;
    for (int i = size - 1; i > 0; --i)
        if (ratings[i] < ratings[i-1]) num[i-1] = max(num[i-1], num[i] + 1);
    return accumulate(num.begin(), num.end(), 0);
}
```

### 136. Single Number

Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Example:

```text
Input: [2,2,1]
Output: 1
```

Solution: XOR bit manipulation

```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (auto i: nums) result ^= i;
    return result;
}
```

### 137. Single Number II

Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

Example:

```text
Input: [2,2,3,2]
Output: 3
```

Solution: bit manipulations。对于N=K的通解可以参考[这个网页](https://leetcode.com/problems/single-number-ii/discuss/43296/An-General-Way-to-Handle-All-this-sort-of-questions.)

```cpp
int singleNumber(vector<int> &nums) {
    int a = 0, b = 0;
    for (auto i : nums) {
        b = (b ^ i) & ~a;
        a = (a ^ i) & ~b;
    }
    return b;
}
```

### 138. Copy List with Random Pointer

A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null. Return a deep copy of the list.

Definition for singly-linked list with a random pointer:

```text
struct RandomListNode {
    int label;
    RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};
```

Solution: 类似Q133，不过既然是链表就不用DFS了，直接遍历到底就可以，利用hashmap辅助储存，一定要背

```cpp
RandomListNode *copyRandomList(RandomListNode *head) {
    unordered_map<RandomListNode*, RandomListNode*>m;
    auto p = head;
    while (p) {
        m[p] = new RandomListNode(p->label);
        p = p->next;
    }
    p = head;
    while (p) {
        m[p]->next = m[p->next];
        m[p]->random = m[p->random];
        p = p->next;
    }
    return m[head];
}
```

### 139. Word Break

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. The same word in the dictionary may be reused multiple times in the segmentation. You may assume the dictionary does not contain duplicate words.

Example:

```text
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
```

Solution: 递归判断容易超时或者溢出，正确方法是用dp，可以遍历字符串\(外loop从左往右，内loop从右往左，hashset加速\)，也可以遍历word dict。

```cpp
// 遍历字符串，稍微慢一些
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
    vector<bool> dp(s.length()+1, false);
    dp[0] = true;
    for (int i = 1; i <= s.length(); ++i) {
        for (int j = i-1; j >= 0; --j) {
            if (dp[j] && wordSet.find(s.substr(j,i-j)) != wordSet.end()) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[s.length()];
}

// 遍历word dict，稍微快一些
bool wordBreak(string s, vector<string>& wordDict) {
    int strlen = s.length();
    vector<bool> node(strlen, false);
    for (int i = 0; i < strlen; ++i) {
        if (!i || node[i-1]) {
            for (string word: wordDict) {
                int wordlen = word.length();
                if (i + wordlen > strlen) continue;
                if (s.substr(i, wordlen) == word) node[i+wordlen-1] = true;
                if (node[strlen-1]) return true;
            }
        }
    }
    return false;
}
```

### 140. Word Break II

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences. The same word in the dictionary may be reused multiple times in the segmentation. You may assume the dictionary does not contain duplicate words.

Example:

```text
Input:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
Output:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
```

Solution: dfs或者递归+hashmap+cache辅助储存

```cpp
vector<string> wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end()); // for easy lookup
    unordered_map<size_t, vector<string>> cache; // memoization cache
    cache[s.length()] = {""}; // base case
    return wordBreak(s, dict, 0UL, cache);
}

vector<string> wordBreak(string& s, unordered_set<string>& dict, size_t index, unordered_map<size_t, vector<string>>& cache) {        
    vector<string> out;
    for (size_t i = index; i < s.length(); ++i) {
        string sub = s.substr(index, i - index + 1);
        if (dict.find(sub) != dict.end()) {
            bool found = (cache.find(i+1) != cache.end());
            vector<string>& result = cache[i+1];
            if (!found) result = wordBreak(s, dict, i+1, cache);
            for (string& subsub : result) {
                out.emplace_back(sub + (subsub.length() ? (string(" ") + subsub) : ""));
            }
        }
    }
    return out; 
}
```

### 141. Linked List Cycle

Given a linked list, determine if it has a cycle in it.

Solution: 快慢指针判圈法

```cpp
bool hasCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    do {
        if (!fast || !fast->next) return false;
        fast = fast->next->next;
        slow = slow->next;
    } while (fast != slow);
    return true;
}
```

### 142. Linked List Cycle II

Given a linked list, return the node where the cycle begins. If there is no cycle, return null

Solution: 快慢指针判圈法

```cpp
​```cpp
ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    do {
        if (!fast || !fast->next) return NULL;
        fast = fast->next->next;
        slow = slow->next;
    } while (fast != slow);
    fast = head;
    while (fast != slow){
        slow = slow->next;
        fast = fast->next;
    }
    return fast;
}
```

### 143. Reorder List

Given a singly linked list L: L0-&gt;L1-&gt;...-&gt;Ln-1-&gt;Ln, reorder it to: L0-&gt;Ln-&gt;L1-&gt;Ln-1-&gt;L2-&gt;Ln-2-&gt;...

Example:

```text
Input: 1->2->3->4->5
Output: 1->5->2->4->3
```

Solution: 分为两步 \(1\) 先找到中点，然后完全反转中点之后的链表，如12345 =&gt; 123 & 54 \(2\) 穿插合并两条分链表

```cpp
void reorderList(ListNode* head) {
    if (!head || !head->next) return;

    ListNode* mid = head, *last = head;
    while (last->next && last->next->next) {
        mid = mid->next;
        last = last->next->next;
    }

    mid->next = reverseList(mid->next, NULL);

    while (head->next && head->next->next) {
        ListNode *temp = head->next;
        head->next = mid->next;
        mid->next = mid->next->next;
        head->next->next = temp;
        head = temp;
    }
}

ListNode* reverseList(ListNode* head, ListNode* newHead) {
    if (!head) return newHead;
    ListNode* next = head->next;
    head->next = newHead;
    return reverseList(next, head);
}
```

### 144. Binary Tree Preorder Traversal

Given a binary tree, return the preorder traversal of its nodes' values.

Example:

```text
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
```

Example: 递归，或stack储存左右树。一定要背和理解

```cpp
// recursion
vector<int> preorderTraversal(TreeNode* root) {
    if (!root) return vector<int>();
    vector<int> left = preorderTraversal(root->left);
    vector<int> right = preorderTraversal(root->right);
    left.insert(left.begin(), root->val);
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

// stack
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> ret;
    if (!root) return ret;

    stack<TreeNode*> st;
    st.push(root);

    while (!st.empty()) {
        TreeNode* tp = st.top();
        st.pop();
        ret.push_back(tp->val);
        if (tp->right) st.push(tp->right);
        if (tp->left) st.push(tp->left);
    }

    return ret;        
}
```

### 145. Binary Tree Postorder Traversal

Given a binary tree, return the postorder traversal of its nodes' values.

Example:

```text
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [3,2,1]
```

Example: 递归；或stack储存左树，每次先打印左枝，再找右枝的左树，且需要一个flag来防止多次遍历右枝。一定要背和理解

```cpp
// recursion
vector<int> postorderTraversal(TreeNode* root) {
    if (!root) return vector<int>();
    vector<int> left = postorderTraversal(root->left);
    vector<int> right = postorderTraversal(root->right);
    left.insert(left.end(), right.begin(), right.end());
    left.push_back(root->val);
    return left;
}

// stack
vector<int> postorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode *> myStack;

    TreeNode *current = root, *lastVisited = NULL;
    while (current|| !myStack.empty()) {
        while (current) {
            myStack.push(current);
            current = current->left;
        }
        current = myStack.top(); 
        if (!current->right || current->right == lastVisited) {
            myStack.pop();
            result.push_back(current->val);
            lastVisited = current;
            current = NULL;
        } else {
            current = current->right;
        }
    }
    return result;
}
```

### 146. LRU Cache

Design and implement a data structure for Least Recently Used \(LRU\) cache. It should support the following operations: get and put. \(1\) get\(key\) - Get the value \(will always be positive\) of the key if the key exists in the cache, otherwise return -1. \(2\) put\(key, value\) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Example:

```text
LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

Solution: hashmap&gt;::iterator&gt; + list&gt;。存iterator的原因是方便调用list的splice函数来直接更新cash hit的pair。一定要背

```cpp
class LRUCache{  
public:  
    LRUCache(int capacity):size(capacity) {}  

    int get(int key) {  
        auto it = hash.find(key);  
        if (it == hash.end()) return -1;  
        cache.splice(cache.begin(), cache, it->second);  
        return it->second->second;  
    }  

    void put(int key, int value) {  
        auto it = hash.find(key);  
        if (it != hash.end()) {  
            it->second->second = value;  
            return cache.splice(cache.begin(), cache, it->second);  
        }  
        cache.insert(cache.begin(), make_pair(key, value));  
        hash[key] = cache.begin();  
        if (cache.size() > size) {  
            hash.erase(cache.back().first);  
            cache.pop_back();  
        }
    }
private:  
    unordered_map<int, list<pair<int, int>>::iterator> hash;  
    list<pair<int, int>> cache;  
    int size;  
};
```

### 147. Insertion Sort List

Sort a linked list using insertion sort.

Solution: 其实录入vector再处理最快，不过还是可以用linkedlist正常做

```cpp
ListNode* insertionSortList(ListNode* head) {
    if (!head|| !head->next) return head;
    ListNode* dummyhead = new ListNode(0);
    dummyhead->next = head;

    ListNode* cur = head;
    ListNode* pre = NULL;

    while (cur) {
        if (cur->next && cur->next->val < cur->val) {
            pre = dummyhead;
            ListNode* t = dummyhead->next;
            while (t && t->val < cur->next->val) {
                pre = t;
                t = t->next;
            }
            pre->next = cur->next;
            cur->next = cur->next->next;
            pre->next->next = t;

        } else cur = cur->next;
    }
    return dummyhead->next;
}
```

### 148. Sort List

Sort a linked list in O\(nlogn\) time using constant space complexity.

Solution: mergesort或quicksort，一定要背

```cpp
// mergesort
ListNode* sortList(ListNode* head) {
    if (!head|| !head->next) return head;
    ListNode *slow = head;
    ListNode *fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    fast = slow->next;
    slow->next = NULL;
    return merge(sortList(head),sortList(fast));
}
ListNode* merge(ListNode *l1, ListNode *l2) {
    ListNode *dummy = new ListNode(INT_MIN);
    ListNode *node = dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            node->next = l1;
            l1 = l1->next;
        } else {
            node->next = l2;
            l2 = l2->next;
        }
        node = node->next;
    }
    node->next = l1? l1: l2; // clever!
    return dummy->next;
}

// quicksort
ListNode* quickSortList(ListNode* head, ListNode *end) {
    ListNode *cur = NULL, *next = NULL, *flag = NULL, *pre = NULL;
    ListNode *temp = NULL, *temp1 = NULL;

    if (head && head != end) {
        cur = head->next;
        flag = head;
        pre = head;
    } else {
        return head;
    }

    while (cur && cur != end) {
        next = cur->next;
        if (cur->val < flag->val) {
            cur->next = head;
            pre->next = next;
            head = cur;
            if (!temp) temp = cur;
        } else if (cur->val == flag->val) {
            pre->next = next;
            if (temp) {
                cur->next = temp->next;
                temp->next = cur;
            } else {
                cur->next = head;
                temp = cur;
                head = cur;
            }
        } else {
            pre = cur;
        }
        cur = next;
    }

    next = flag->next;
    flag->next = NULL;
    if ((temp && temp->next == flag) || !temp)
        head = quickSortList(head, flag);
    else 
        head = quickSortList(head, temp->next);
    next = quickSortList(next, end);
    flag->next = next;

    return head;
}
ListNode* sortList(ListNode* head) {
    return quickSortList(head, NULL);

```

## 149. Max Points on a Line

Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.

Example:

```text
Input: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
^
|
|  o
|     o        o
|        o
|  o        o
+------------------->
0  1  2  3  4  5  6
Output: 4
```

Solution: O\(N^2\)遍历，对每个基准点，记录和它重复的点、相同横线的点、以及每个斜率上的点（用hashmap&lt;斜率，点个数&gt;记录）。一定要背

```cpp
int maxPoints(vector<Point>& points) {
    unordered_map<double, int> slope;
    int max_p = 0, same_p = 1, same_y = 1;
    for (int i = 0; i < points.size(); ++i) {
        same_p = 1, same_y = 1;
        for (int j = i + 1; j < points.size(); ++j) {
            if (points[i].y == points[j].y) {
                ++same_y;
                if (points[i].x == points[j].x) ++same_p;
            } else {
                ++slope[double(points[i].x - points[j].x) / double(points[i].y - points[j].y)];
            }
        }
        max_p = max(max_p, same_y);
        for (auto item : slope) max_p = max(max_p, item.second + same_p);
        slope.clear();
    }
    return max_p;
}
```

