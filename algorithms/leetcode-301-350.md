# LeetCode 301 - 350

### 303. Range Sum Query - Immutable

Given an integer array nums, find the sum of the elements between indices i and j (i <= j), inclusive. There are many calls to sumRegion function.

Example:

```
Given nums = [-2, 0, 3, -5, 2, -1]
sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
```

Solution: 提前做好一个数组a, a[i]表示从头到i位的和，这样sumRange(j, k)就等于a[k] - a[j]。注意在C++可以用partial_sum函数求和

```c++
class NumArray {
public:
    NumArray(vector<int> nums): psum(nums.size() + 1, 0) {
        partial_sum(nums.begin(), nums.end(), psum.begin() + 1);
    }


    int sumRange(int i, int j) {
        return psum[j+1] - psum[i];
    }
private:
    vector<int> psum;    
};
```

### 304. Range Sum Query 2D - Immutable

‌Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2). There are many calls to sumRegion function.

Example:

```
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]
sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12
```

Solution: 类似Q303，但是对于矩形，查询(row1, col1)到(row2, col2)的面积的时候可以用sums\[row2+1\][col2+1] - sums\[row2+1][col1] - sums\[row1][col2+1] + sums\[row1][col1]。这里有个trick，在第一行和第一列始终为0的情况下，对sums数组的重构可以直接resize而不用新建，因为只有第一行和第一列会影响矩阵的求和更新

```c++
class NumMatrix {
private:
    int row, col;
    vector<vector<int>> sums;
public:
    NumMatrix(vector<vector<int>> matrix) {
        row = matrix.size();
        col = row > 0? matrix[0].size(): 0;
        sums.resize(row + 1, vector<int>(col + 1, 0));
        for (int i = 1; i <= row; ++i)
            for (int j = 1; j <= col; ++j)
                sums[i][j] = matrix[i-1][j-1] + sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1];
    }


    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2+1][col2+1] - sums[row2+1][col1] - sums[row1][col2+1] + sums[row1][col1];
    }
};
```

### 306. Additive Number

Additive number is a string whose digits can form additive sequence. A valid additive sequence should contain at least three numbers. Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two. Given a string containing only digits '0'-'9', write a function to determine if it's an additive number. Note: Numbers in the additive sequence cannot have leading zeros, so sequence [1, 2, 03] or [1, 02, 3] is invalid. Please handle overflow for very large input integers.

Example:

```
Input: "199100199"
Output: true (1 + 99 = 100, 99 + 100 = 199)‌
```

Solution: 递归，非常精妙

```c++
bool isAdditiveNumber(string num) {
    int n = num.size();
    for (int i = 1; i <= n / 2; ++i)
        for (int j = 1; j <= (n - i) / 2; ++j)
            if (isAdditiveNumber(num.substr(0, i), num.substr(i, j), num.substr(i + j))) return true;
    return false;
}


bool isAdditiveNumber(string s1, string s2, string res) {
    if (s1.size() > 1 && s1[0] == '0' || s2.size() > 1 && s2[0] == '0') return false;
    string sum = add(s1, s2);
    if (res == sum) return true;
    if (res.size() < sum.size() || res.substr(0, sum.size()) != sum) return false;
    else return isAdditiveNumber(s2, sum, res.substr(sum.size()));
}


string add(string s1, string s2) {
    int n1 = s1.size(), n2 = s2.size(), i = n1 - 1, j = n2 - 1, carry = 0;
    string res;
    while (i >= 0 || j >= 0) {
        int sum = carry + (i >= 0 ? s1[i--] - '0' : 0) + (j >= 0 ? s2[j--] - '0' : 0);
        res.insert(res.begin(), sum % 10 + '0');
        carry = sum / 10;
    }
    if (carry) res.insert(res.begin(), '1');
    return res;
}
```

### 307. Range Sum Query - Mutable

Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive. The update(i, val) function modifies nums by updating the element at index i to val.

Example:

```
Given nums = [1, 3, 5]
sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
```

Solution: 线段树，O(logn)的更新和查询速度，一定要背

```c++
struct SegmentTreeNode {
    int start, end, sum;
    SegmentTreeNode* left;
    SegmentTreeNode* right;
    SegmentTreeNode(int a, int b): start(a), end(b), sum(0), left(NULL), right(NULL) {}
};


class NumArray {
    SegmentTreeNode* root;
public:
    NumArray(vector<int> nums) {
        root = buildTree(nums, 0, nums.size() - 1);
    }


    void update(int i, int val) {
        modifyTree(i, val, root);
    }


    int sumRange(int i, int j) {
        return queryTree(i, j, root);
    }


    SegmentTreeNode* buildTree(vector<int> &nums, int start, int end) {
        if (start > end) return NULL;
        SegmentTreeNode* root = new SegmentTreeNode(start, end);
        if (start == end) {
            root->sum = nums[start];
            return root;
        }
        int mid = start + (end - start) / 2;
        root->left = buildTree(nums, start, mid);
        root->right = buildTree(nums, mid+1, end);
        root->sum = root->left->sum + root->right->sum;
        return root;
    }


    int modifyTree(int i, int val, SegmentTreeNode* root) {
        if (!root) return 0;
        int diff;
        if (root->start == i && root->end == i) {
            diff = val - root->sum;
            root->sum = val;
            return diff;
        }
        int mid = (root->start + root->end) / 2;
        if (i > mid) diff = modifyTree(i, val, root->right);
        else diff = modifyTree(i, val, root->left);
        root->sum = root->sum + diff;
        return diff;
    }


    int queryTree(int i, int j, SegmentTreeNode* root) {
        if (!root) return 0;
        if (root->start == i && root->end == j) return root->sum;
        int mid = (root->start + root->end) / 2;
        if (i > mid) return queryTree(i, j, root->right);
        if (j <= mid) return queryTree(i, j, root->left);
        return queryTree(i, mid, root->left) + queryTree(mid + 1, j, root->right);
    }
};
```

### 309. Best Time to Buy and Sell Stock with Cooldown

Say you have an array for which the ith element is the price of a given stock on day i. Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions: (1) You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again). (2) After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

Example:

```
Input: [1,2,3,0,2]
Output: 3 (buy, sell, cooldown, buy, sell)
```

Solution: 基于两个表达式来做, buy[i] = max(buy[i-1], sell[i-2] - cur_price), sell[i] = max(sell[i-1], buy[i-1] + cur_price)，可以优化成1D，一定要背

```c++
int maxProfit(vector<int>& prices) {
    if (prices.empty()) return 0;
    int pre_sell = 0, cur_sell = 0, pre_buy =0;
    int cur_buy = std::numeric_limits<int>::min();
    for (auto cur_price: prices) {
        pre_buy = cur_buy;
        cur_buy = std::max(pre_buy, pre_sell - cur_price);
        pre_sell = cur_sell;
        cur_sell = std::max(pre_sell, pre_buy + cur_price);
    }
    return cur_sell;
}‌
```

### 310. Minimum Height Trees

For a undirected graph with tree characteristics (without simple cycles), we can choose any node as the root. The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Example:

```
Input: n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]


     0  1  2
      \ | /
        3
        |
        4
        |
        5 


Output: [3, 4]
```

Solution: BFS遍历+记录叶子结点，采用了交替替换叶子结点集合的方法，一定要背

```c++
struct TreeNode{  
    set<int> list;  // 使用set结构方便删除邻居  
    TreeNode() {};  
    bool isLeaf() {return list.size() == 1;};  // 如果是叶子节点，那么邻居大小是1  
};


vector<int> findMinHeightTrees(int n, vector<pair<int, int>> &edges) {  
    if (n == 1) return {0};
    vector<TreeNode> tree(n);  


    // 使用节点来存储这棵树，耗费的空间为O(n+2e)  
    for (auto e: edges) {
        tree[e.first].list.insert(e.second);  
        tree[e.second].list.insert(e.first);  
    }


    vector<int> node1(0);  // 记录当前的叶子节点  
    vector<int> node2(0);  // 记录删除node1叶子节点后，成为新的叶子节点的节点  


    // 记录叶子节点
    for (int i = 0; i < tree.size(); ++i) if (tree[i].isLeaf()) node1.push_back(i);


    // BFS删除叶子节点
    while (true) {  
        for (auto leaf: node1) {   
            for (auto ite = tree[leaf].list.begin(); ite != tree[leaf].list.end(); ++ite) {  
                int neighbor = *ite;  
                tree[neighbor].list.erase(leaf);  // 删除叶子节点  
                if (tree[neighbor].isLeaf()) node2.push_back(neighbor);  // 删除后，如果是叶子节点，则放到node2中  
            }  
        }  
        // 遍历完后，如果node2为空，即node1中的节点不是叶子节点，要么是剩下一个节点，要么剩下两个相互连接的节点  
        if (node2.empty()) return node1;
        node1.clear();  
        swap(node1, node2);  
    }
}
```

### 313. Super Ugly Number

Write a program to find the nth super ugly number. Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k. Note: (1) 1 is a super ugly number for any given primes; (2) The given numbers in primes are in ascending order (3) 0 < k <= 100, 0 < n <= 10^6, 0 < primes[i] < 1000; (4) The nth super ugly number is guaranteed to fit in a 32-bit signed integer.

Example:

```
Input: n = 12, primes = [2,7,13,19]
Output: 32 ([1,2,4,7,8,13,14,16,19,26,28,32])
```

Solution: priority queue，设计很巧妙

```c++
int nthSuperUglyNumber(int n, vector<int>& primes) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> min_heap;
    vector<int> uglies(n);
    uglies[0] = 1;
    vector<int> last_factor(n);
    int k = primes.size();
    vector<int> idx(k);
    for (int i = 0; i < k; ++i) min_heap.emplace(primes[i], i);
    for (int i = 1; i < n; ++i) {
        tie(uglies[i], last_factor[i]) = min_heap.top();
        min_heap.pop();
        int j = last_factor[i];
        do ++idx[j]; while (last_factor[idx[j]] > j);
        min_heap.emplace(primes[j] * uglies[idx[j]], j);
    }
    return uglies.back();
}
```

### 315. Count of Smaller Numbers After Self

You are given an integer array *nums* and you have to return a new *counts* array. The *counts* array has the property where `counts[i]` is the number of smaller elements to the right of `nums[i]`.

Example:

```
Input: [5,2,6,1]
Output: [2,1,1,0] 
```

Solution: 从后往前建立二叉树（最好是AVL），Node记录当前出现次数以及比其小的Node的总出现次数，一定要背

```cpp
struct Node {
    int less_count, self_count, val;
    Node *left, *right;
    Node (int val) : less_count(0), self_count(1), val(val), left(nullptr), right(nullptr) {}
};
class Solution {
public:
    pair<Node*, int> insertVal(Node *node, int val) {
        if (!node) return make_pair(new Node(val), 0);
        if (node->val == val) {
            ++(node->self_count);
            return make_pair(node, node->less_count);
        }
        if (node->val > val) {
            ++(node->less_count);
            auto ret = insertVal(node->left, val);
            node->left = ret.first;
            return make_pair(node, ret.second);
        }
        auto ret = insertVal(node->right, val);
        node->right = ret.first;
        return make_pair(node, node->self_count + node->less_count + ret.second);
    }
    
    vector<int> countSmaller(vector<int>& nums) {
        int n = nums.size();
        if (!n) return {};
        vector<int> ret(n, 0);
        Node *root = new Node(nums.back());
        for (int i = nums.size() - 2; i >= 0; --i) ret[i] = insertVal(root, nums[i]).second;
        return ret;
    }
};
```

### 318. Maximum Product of Word Lengths

Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. You may assume that each word will contain only lower case letters. If no such two words exist, return 0.

Example:

```
Input: ["abcw","baz","foo","bar","xtfn","abcdef"]
Output: 16 ("abcw", "xtfn")
```

Solution: bit manipulation，可以用hashing或者vector记录字典，一定要背

```c++
int maxProduct(vector<string>& words) {
    unordered_map<int, int> maxlen;
    int result = 0;
    for (string word : words) {
        int mask = 0, size = word.size();
        for (char c : word) mask |= 1 << (c - 'a');
        maxlen[mask] = max(maxlen[mask], size);
        for (auto mask_len : maxlen)
            if (!(mask & mask_len.first))
                result = max(result, size * mask_len.second);
    }
    return result;
}
```

### 319. Bulb Switcher

There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the i-th round, you toggle every i bulb. For the n-th round, you only toggle the last bulb. Find how many bulbs are on after n rounds.

Example:

```
Input: 3
Output: 1 ([off, off, off] -> [on, on, on] -> [on, off, on] -> [on, off, off])
```

Solution: 只有平方数是亮着的

```c++
int bulbSwitch(int n) {
    return sqrt(n);
}
```

### 322. Coin Change

You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

Example:

```
Input: coins = [1, 2, 5], amount = 11
Output: 3 (11 = 5 + 5 + 1
```

Solution: dp或者dfs，dfs快一点，dp更好写，但是原理相同。一定要背

```c++
// method 1: dp
int coinChange(vector<int>& coins, int amount) {
    if (coins.empty()) return -1;
    vector<int> dp(amount+1, amount+2);
    dp[0] = 0;
    // the order of for loop can be interchanged
    for (int coin : coins) {
        for (int i = coin; i<= amount; i++) {
            dp[i] = min(dp[i], dp[i-coin] + 1);
        }   
    }
    return dp[amount] == amount + 2? -1: dp[amount];
}


// method 2: dfs
int coinChange(vector<int>& coins, int amount) {
    int res = INT_MAX;
    sort(coins.begin(), coins.end());
    dfs(res, coins, amount, coins.size() - 1, 0);
    return res == INT_MAX? -1: res;
}
void dfs(int& res, vector<int>& coins, int target, int idx, int count) {
    if (idx < 0) return;
    if (target % coins[idx] == 0) res = min(res, count + target / coins[idx]);
    else for (int i = target / coins[idx]; i >= 0; --i) {
        if (count + i >= res - 1) break; // pruing
        dfs(res, coins, target - i * coins[idx], idx - 1, count + i);
    }
}
```

### 324. Wiggle Sort I‌

Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3].... You may assume all input has valid answer. Can you do it in O(n) time and/or in-place with O(1) extra space?

Example:

```
Input: nums = [1, 5, 1, 1, 6, 4]
Output: One possible answer is [1, 4, 1, 5, 1, 6].
```

Solution: find median + index rewriting，一定要背

```c++
void wiggleSort(vector<int>& nums) {
    int n = nums.size();


    // find a median, O(n) in avg
    auto midptr = nums.begin() + n / 2;
    nth_element(nums.begin(), midptr, nums.end());
    int mid = *midptr;


    // index-rewiring: note that (n|1) = n if n is odd else n + 1
    // therefore we have 1, 3, 5, ..., (n|1) - 2, 0, 2, 4, ..., (n|1) - 1
    auto idx = [=](int i) {return (1+2*i) % (n|1);};


    // 3-way-partition-to-wiggle in O(n) time with O(1) space: odd pos > mid, even pos < mid
    int i = 0, j = 0, k = n - 1;
    while (j <= k) {
        if (nums[idx(j)] > mid) swap(nums[idx(i++)], nums[idx(j++)]);
        else if (nums[idx(j)] < mid) swap(nums[idx(j)], nums[idx(k--)]);
        else j++;
    }
}
```

### 326. Power of Three

Given an integer, write a function to determine if it is a power of three. Could you do it without using any loop or recursion?

Example:

```
Input: 45
Output: false
```

Solution: 一种思路是log，一种是找最大的3的倍数的int看能不能除尽（因为3是质数，所以能除尽的话，因数只能是3）

```c++
// method 1: log
bool isPowerOfThree(int n) {
    return fmod(log10(n)/log10(3), 1) == 0;
}


// method 2: max int
bool isPowerOfThree(int n) {
    return n > 0 && 1162261467 % n == 0;
}
```

### 328. Odd Even Linked List

Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes. You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example:

```
Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL
```

Solution: 设置两个head，遍历一遍即可

```c++
ListNode* oddEvenList(ListNode* head) {
    if (!head) return head;
    ListNode *odd=head, *evenhead=head->next, *even = evenhead;
    while (even && even->next) {
        odd->next = odd->next->next;
        even->next = even->next->next;
        odd = odd->next;
        even = even->next;
    }
    odd->next = evenhead;
    return head;
}
```

### 329. Longest Increasing Path in a Matrix

Given an integer matrix, find the length of the longest increasing path. From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

Example:

```
Input: nums = [
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 ([1, 2, 6, 9])
```

Solution: dp的话要遍历两次递增递减，最快的方法是dfs+memoization，对每个点的四向做dfs遍历递增或递减，如果算过就直接调取值，一定要背

```c++
int rows;
int cols;
vector<vector<int>> f;


int DFS(vector<vector<int>>& M, int r, int c) {
    if (f[r][c] > 0) return f[r][c]; // retrieve
    int res = 1;
    if (r+1 < rows && M[r+1][c] > M[r][c]) res = max(res, 1 + DFS(M, r+1, c));
    if (r-1 >= 0   && M[r-1][c] > M[r][c]) res = max(res, 1 + DFS(M, r-1, c));
    if (c+1 < cols && M[r][c+1] > M[r][c]) res = max(res, 1 + DFS(M, r, c+1));
    if (c-1 >= 0   && M[r][c-1] > M[r][c]) res = max(res, 1 + DFS(M, r, c-1));
    f[r][c] = res; // save
    return f[r][c];
}


int longestIncreasingPath(vector<vector<int>>& M) {
    if (M.empty()) return 0;
    rows = M.size(), cols = M[0].size();
    f = vector<vector<int>>(rows, vector<int>(cols, 0));
    int res = 0;
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) res = max(res, DFS(M, r, c));
    return res;
}
```

### 331. Verify Preorder Serialization of a Binary Tree

One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as #. Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.

Example:

```
Input: "9,3,4,#,#,1,#,#,2,#,6,#,#"
     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
Output: true
```

Solution: 对于每个子树，#的个数比node的个数要大1。因为遍历过程中会经过每个子树，所以只要保证在过程中#的个数-node的个数始终大于1即可，一定要背

```c++
bool isValidSerialization(string preorder) {
    int diff = -1;  // diff = (num of # - num of node), diff should be 1 for all subtrees
    vector<string> v = split(preorder, ',');
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] == "#") ++diff;
        else --diff;
        if (diff >= 0 && i != v.size() - 1) return false;
    }
    return !diff;
}


vector<string> split(string str, char delimiter) {
    vector<string> r;
    while (!str.empty()) {
        int ind = str.find_first_of(delimiter);
        if (ind == -1) {
            r.push_back(str);
            str.clear();
        } else {
            r.push_back(str.substr(0, ind));
            str = str.substr(ind + 1, str.size() - ind - 1);
        }
    }
    return r;
}
```

### 332. Reconstruct Itinerary

Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"]. All airports are represented by three capital letters (IATA code). You may assume all tickets form at least one valid itinerary.

Example:

```
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
```

Solution: hashmap+set，用stack遍历，最后反转，一定要背

```c++
vector<string> findItinerary(vector<pair<string, string>> tickets) {
    unordered_map<string, multiset<string>> m;
    vector<string> res;
    if (tickets.empty()) return res;
    for (pair<string, string> p: tickets) m[p.first].insert(p.second);
    stack<string> s;
    s.push("JFK");
    while (!s.empty()) {
        string next = s.top();
        if (m[next].empty()) {
            res.push_back(next);
            s.pop();
        } else {
            s.push(*m[next].begin());
            m[next].erase(m[next].begin());
        }
    }
    reverse(res.begin(), res.end());
    return res;
}
```

### 334. Increasing Triplet Subsequence

Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array. Your algorithm should run in O(n) time complexity and O(1) space complexity.

Example:

```
Input: [1, 5, 2, 4, 3]
Output: true
```

Solution: 只用两个数字就可以当buffer，比较tricky，一定要背

```c++
bool increasingTriplet(vector<int>& nums) {
    int c1 = INT_MAX, c2 = INT_MAX;
    for (int x : nums) {
        if (x <= c1) c1 = x;
        else if (x <= c2) c2 = x;
        else return true;
    }
    return false;
}
```

### 337. House Robber III

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night. Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example:

```
Input:
     3
    / \
   4   5
  / \   \ 
 1   3   1
Output: 9 (4 + 5)
```

‌Solution: DFS，返回值是一个pair<不抢root，抢或不抢root>的当前最大值，一定要背

```c++
int rob(TreeNode* root) {
    return robDFS(root).second;
}
pair<int, int> robDFS(TreeNode* node) {
    if (!node) return make_pair(0, 0);
    auto l = robDFS(node->left), r = robDFS(node->right);
    int f1 = l.second + r.second, f2 = max(f1, l.first + r.first + node->val);
    return make_pair(f1, f2);
}
```

### 338. Counting Bits

Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array. It is very easy to come up with a solution with run time O(n x sizeof(integer)). But can you do it in linear time O(n) possibly in a single pass? Space complexity should be O(n).

Example:

```
Input: num = 5
Output: [0,1,1,2,1,2]
```

Solution: bit manipulation，一定要背

```c++
vector<int> countBits(int num) {
    vector<int> ret(num+1, 0);
    for (int i = 1; i <= num; ++i)
        ret[i] = i & 1? ret[i-1] + 1: ret[i>>1];
        // or equally: ret[i] = ret[i&(i-1)] + 1;
    return ret;
}
```

### 340. Longest Substring with At Most K Distinct Characters

Given a string, find the length of the longest substring T that contains at most *k* distinct characters.

Example:

```
Input: s = "eceba", k = 2
Output: 3 (T is "ece" which its length is 3)
```

Solution: 滑动窗口，与其用vector储存频率+一个int来统计目前不同的字符数，不如直接用hashmap，根据size判断是否超过k，一定要背

```cpp
int lengthOfLongestSubstringKDistinct(string s, int k) {
    unordered_map<char, int> counts;
    int res = 0;
    for (int i = 0, j = 0; j < s.size(); ++j) {
        counts[s[j]]++;
        while (counts.size() > k) if (--counts[s[i++]] == 0) counts.erase(s[i - 1]);
        res = max(res, j - i + 1);
    }
    return res;
}
```

### 341. Flatten Nested List Iterator

Given a nested list of integers, implement an iterator to flatten it.

Example:

```
Input: [[1,1],2,[1,[1]]]
List Iterator treats it as: [1,1,2,1,1]
```

Solution: stack，把flatten的操作放在hasNext是最合适的，注意stack应该反着push，一定要背

```c++
stack<NestedInteger> nodes;


NestedIterator(vector<NestedInteger> &nestedList) {
    for (int i = nestedList.size() - 1; i >= 0; --i) nodes.push(nestedList[i]);
}


int next() {
    int result = nodes.top().getInteger();
    nodes.pop();
    return result;
}


bool hasNext() {
    while (!nodes.empty()) {
        NestedInteger curr = nodes.top();
        if (curr.isInteger()) return true;
        nodes.pop();
        vector<NestedInteger> & adjs = curr.getList();
        for (int i = adjs.size() - 1; i >= 0; --i) nodes.push(adjs[i]);
    }
    return false;
}
```

### 342. Power of Four

Given an integer (signed 32 bits), write a function to check whether it is a power of 4. Could you solve it without loops/recursion?

Example:

```
Input: 16
Output: true
```

Solution: log或bit manipulation

```c++
// method 1: log
bool isPowerOfFour(int num) {
    return fmod(log10(num)/log10(4), 1) == 0;
}


// method 2: bit manipulation
bool isPowerOfFour(int num) {
    return num > 0 && (num & (num - 1)) == 0 && (num - 1) % 3 == 0;
}
```

### 343. Integer Break‌

Given a positive integer n, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get. You may assume that n is not less than 2 and not larger than 58.

Example:

```
Input: n = 10
Output: 36 (3 * 3 * 4 = 36)
```

Solution: 除了几个特殊情况外，分3出来乘积最大

```c++
int integerBreak(int n) {
    if (n == 2) return 1;
    if (n == 3) return 2;
    if (n == 4) return 4;
    if (n == 5) return 6;
    if (n == 6) return 9;
    return 3 * integerBreak(n - 3);
}
```

### 344. Reverse String

Write a function that takes a string as input and returns the string reversed.

Example:

```
Input: "hello"
Output: "olleh"
```

Solution: 简单交换即可

```c++
string reverseString(string s) {
    for (int i = 0, j = s.length() -1; i < j; ++i, --j) swap(s[i], s[j]);
    return s;
}
```

### 345. Reverse Vowels of a String

Write a function that takes a string as input and reverse only the vowels of a string.

Example:

```
Input: "hello"
Output: "holle"
```

Solution: 过滤交换即可，可以拿来练手，一定要背

```c++
bool isVowel(char c) {
    if ('A' <= c && c <= 'Z') c = c - 'A' + 'a';
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}


string reverseVowels(string s) {
    int i = 0, j = s.length() - 1;
    while (i < j) {
        while (i < j && !isVowel(s[i])) ++i;
        while (i < j && !isVowel(s[j])) --j;
        swap(s[i++], s[j--]);
    }
    return s;
}
```

### 347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements. You may assume k is always valid.

Example:

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

Solution: hashmap + bucket sort，一定要背

```c++
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> m;
    for (int num : nums) ++m[num];


    vector<vector<int>> buckets(nums.size() + 1); 
    for (auto p : m) buckets[p.second].push_back(p.first);


    vector<int> ans;
    for (int i = buckets.size() - 1; i >= 0 && ans.size() < k; --i) {
        for (int num : buckets[i]) {
            ans.push_back(num);
            if (ans.size() == k) break;
        }
    }
    return ans;
}
```

### 348. Design Tic-Tac-Toe

Design a Tic-tac-toe game that is played between two players on a *n* x *n* grid.

You may assume the following rules:

1. A move is guaranteed to be valid and is placed on an empty block.
2. Once a winning condition is reached, no more moves is allowed.
3. A player who succeeds in placing *n* of their marks in a horizontal, vertical, or diagonal row wins the game.

Example:

```
Given n = 3, assume that player 1 is "X" and player 2 is "O" in the board.

TicTacToe toe = new TicTacToe(3);

toe.move(0, 0, 1); -> Returns 0 (no one wins)
|X| | |
| | | |    // Player 1 makes a move at (0, 0).
| | | |

toe.move(0, 2, 2); -> Returns 0 (no one wins)
|X| |O|
| | | |    // Player 2 makes a move at (0, 2).
| | | |

toe.move(2, 2, 1); -> Returns 0 (no one wins)
|X| |O|
| | | |    // Player 1 makes a move at (2, 2).
| | |X|

toe.move(1, 1, 2); -> Returns 0 (no one wins)
|X| |O|
| |O| |    // Player 2 makes a move at (1, 1).
| | |X|

toe.move(2, 0, 1); -> Returns 0 (no one wins)
|X| |O|
| |O| |    // Player 1 makes a move at (2, 0).
|X| |X|

toe.move(1, 0, 2); -> Returns 0 (no one wins)
|X| |O|
|O|O| |    // Player 2 makes a move at (1, 0).
|X| |X|

toe.move(2, 1, 1); -> Returns 1 (player 1 wins)
|X| |O|
|O|O| |    // Player 1 makes a move at (2, 1).
|X|X|X|
```

Solution: 类似于九皇后，可以通过开辅助数组统计每个玩家的频次，避免开到`O(n^2)`的空间，一定要背

```cpp
class TicTacToe {
    int sz;
    vector<vector<int>> hor;
    vector<vector<int>> ver;
    vector<int> diag;
    vector<int> antidiag;
public:
    /** Initialize your data structure here. */
    TicTacToe(int n) : sz(n) {
        hor   = vector<vector<int>> (2, vector<int>(n, 0));
        ver   = vector<vector<int>> (2, vector<int>(n, 0));
        diag  = vector<int> (2, 0);
        antidiag = vector<int> (2, 0);
    }
    
    /** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
    int move(int row, int col, int player) {
        if (++hor[player - 1][row] == sz) return player;
        if (++ver[player - 1][col] == sz) return player;
        if (row == col && ++diag[player - 1] == sz) return player;
        if (row + col == sz - 1 && ++antidiag[player - 1] == sz) return player;
        return 0;
    }
};
```

### 349. Intersection of Two Arrays

Given two arrays, write a function to compute their intersection without duplicates.‌

Example:

```
Input: nums1 = [1, 2, 2, 1], nums2 = [2, 2]
Output: [2]
```

Solution: hashset

```c++
vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
    unordered_set<int> hash(nums1.begin(), nums1.end());
    vector<int> res;
    for (auto & n: nums2)
        if (hash.find(n) != hash.end()) {
            res.push_back(n);
            hash.erase(n);
        }
    return res;
}
```

### 350. Intersection of Two Arrays II

Given two arrays, write a function to compute their intersection with duplicates.

Example:

```
Input: nums1 = [1, 2, 2, 1], nums2 = [2, 2]
Output: [2, 2]
```

Solution: hashmap

```c++
vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
    unordered_map<int, int> hash;
    for (auto &n :nums1) ++hash[n];
    vector<int> res;
    for (auto & n: nums2) 
        if (hash[n]) {
            res.push_back(n);
            --hash[n];
        }
    return res;
}
```