LeetCode 851-900

### 857. Minimum Cost to Hire K Workers

There are `N` workers.  The `i`-th worker has a `quality[i]` and a minimum wage expectation `wage[i]`. Now we want to hire exactly `K` workers to form a *paid group*.  When hiring a group of K workers, we must pay them according to the following rules:

1. Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
2. Every worker in the paid group must be paid at least their minimum wage expectation.

Return the least amount of money needed to form a paid group satisfying the above conditions.

Example:

```
Input: quality = [3,1,10,10,1], wage = [4,8,2,2,7], K = 3
Output: 30.66667 (pay 4 to 0-th worker, 13.33333 to 2-th and 3-th workers seperately) 
```

Solution: 先建立一个pair<ratio=wage/quality, quality>的数组，按照ratio排序；然后从前往后做K大小的滑动窗口，并把quality存进priority queue。计算当前滑动窗口最小值时，ratio取窗口右侧（已经排好序所以最大），quality取当前priority queue里所有值的和，且每次新进来右侧的quality后pop掉priority queue里最大的quality。十分巧妙，一定要理解

```cpp
double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int K) {
    int N = quality.size();
    // ratio, quality
    vector<pair<double, int>> vec;
    for (int i = 0; i < N; ++i) vec.push_back(make_pair(wage[i] * 1.0 / quality[i], quality[i])); 
    // sort in ascending order
    sort(vec.begin(), vec.end(), [](auto& a, auto& b){return a.first < b.first;});
    int quality_cnt = 0;
    // max-min heap
    priority_queue<int> q;
    for(int i = 0; i < K; ++i) {
        quality_cnt += vec[i].second;
        q.push(vec[i].second);
    }
    double ans = quality_cnt * vec[K - 1].first;
    for (int i = K; i < N; ++i) {
        q.push(vec[i].second);
        quality_cnt += vec[i].second;
        quality_cnt -= q.top();
        q.pop();
        ans = min(ans, quality_cnt * vec[i].first);
    }
    return ans;
}
```

### 865. Smallest Subtree with all the Deepest Nodes

Given a binary tree rooted at `root`, the *depth* of each node is the shortest distance to the root. A node is *deepest* if it has the largest depth possible among any node in the entire tree. The subtree of a node is that node, plus the set of all descendants of that node. Return the node with the largest depth such that it contains all the deepest nodes in its subtree.

```
Input: [3,5,1,6,2,0,8,null,null,7,4]
     3
   /   \
  5     1
 / \   / \
6   2 0   8
   / \
  7   4
Output: [2,7,4] (We return the node with value 2)
```

Solution: 递归，每次返回一个<node, depth>的pair，一定要背

```cpp
pair<TreeNode*, int> subtreeWithAllDeepestUtil(TreeNode* root){
    if (!root) return make_pair(root, 0);
    auto L = subtreeWithAllDeepestUtil(root->left), R = subtreeWithAllDeepestUtil(root->right);
    if (L.second > R.second) return make_pair(L.first, L.second + 1);
    if (L.second < R.second) return make_pair(R.first, R.second + 1);
    return make_pair(root, L.second + 1);
}

TreeNode* subtreeWithAllDeepest(TreeNode* root) {
    return root? subtreeWithAllDeepestUtil(root).first: NULL;
}
```

### 876. Middle of the Linked List

Given a non-empty, singly linked list with head node `head`, return a middle node of linked list. If there are two middle nodes, return the second middle node.

Solution: 快慢指针判圈法，注意如果只需要是或否，写成do-while的形式会比较方便；本题这种需要指针状态的，最好是把判断先写在while里面，一定要背

```cpp
ListNode* middleNode(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast->next && fast->next->next) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return fast->next? slow->next: slow;
}
```

### 886. Possible Bipartition

Given a set of `N` people (numbered `1, 2, ..., N`), we would like to split everyone into two groups of **any** size. Each person may dislike some other people, and they should not go into the same group. 

Formally, if `dislikes[i] = [a, b]`, it means it is not allowed to put the people numbered `a` and `b` into the same group. Return `true` if and only if it is possible to split everyone into two groups in this way.

Example:

```
Input: N = 4, dislikes = [[1,2],[1,3],[2,4]]
Output: true (group1 [1,4], group2 [2,3])
```

Solution: 染色法，注意要在for loop里面放queue的while loop，一定要背

```cpp
bool possibleBipartition(int N, vector<vector<int>>& dislikes) {
    vector<vector<int>> graph(N, vector<int>());
    vector<int> color(N, -1);
    for (const auto & dislike: dislikes) {
        graph[dislike[0]-1].push_back(dislike[1]-1);
        graph[dislike[1]-1].push_back(dislike[0]-1);
    }
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        if (color[i] == -1) {
            color[i] = 0;
            q.push(i);
        }
        while (!q.empty()) {
            int pos = q.front();
            q.pop();
            for (const auto & neighbor: graph[pos]) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[pos];
                    q.push(neighbor);
                } else if (color[neighbor] == color[pos]) {
                    return false;
                }
            }
        }
    }        
    return true;
}
```

### 888. Fair Candy Swap

Alice and Bob have candy bars of different sizes: `A[i]` is the size of the `i`-th bar of candy that Alice has, and `B[j]` is the size of the `j`-th bar of candy that Bob has. They would like to exchange one candy bar each so that after the exchange, they both have the same total amount of candy. 

Return an integer array `ans` where `ans[0]` is the size of the candy bar that Alice must exchange, and `ans[1]` is the size of the candy bar that Bob must exchange. If there are multiple answers, you may return any one of them.  It is guaranteed an answer exists.

Example:

```
Input: A = [1,2,5], B = [2,4]
Output: [5,4]
```

Solution: hashset，注意要预先算一个diff of sum

```cpp
vector<int> fairCandySwap(vector<int>& A, vector<int>& B) {
    int diff = (accumulate(A.begin(), A.end(), 0) - accumulate(B.begin(), B.end(), 0)) / 2;
    unordered_set<int> hash;
    for (auto a: A) hash.insert(a - diff);
    for (auto b: B) if (hash.find(b) != hash.end()) return vector<int>{b + diff, b};
    return vector<int>();
}
```

### 890. Find and Replace Pattern

You have a list of `words` and a `pattern`, and you want to know which words in `words` matches the pattern.

A word matches the pattern if there exists a permutation of letters `p` so that after replacing every letter `x` in the pattern with `p(x)`, we get the desired word. (*Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and no two letters map to the same letter.*)

Return a list of the words in `words` that match the given pattern. You may return the answer in any order.

Example:

```
Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
Output: ["mee","aqq"]
```

Solution: 遍历一遍即可

```cpp
vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
    vector<string> ret;
    vector<int> wmatch, pmatch;
    int n = pattern.length();
    bool match;
    for (auto word: words) {
        wmatch = vector<int>(128, 0), pmatch = vector<int>(128, 0);
        match = true;
        for (int i = 0; i < n; ++i) {
            if (!wmatch[word[i]] && !pmatch[pattern[i]])
                wmatch[word[i]] = pmatch[pattern[i]] = i + 1;
            else if (wmatch[word[i]] != pmatch[pattern[i]]) {
                match = false;
                break;
            }
        }
        if (match) ret.push_back(word);
    }
    return ret;
}
```

### 892. Surface Area of 3D Shapes

On a `N * N` grid, we place some `1 * 1 * 1 `cubes. Each value `v = grid[i][j]` represents a tower of `v` cubes placed on top of grid cell `(i, j)`. Return the total surface area of the resulting shapes.

Example:

```
Input: [[2,2,2],[2,1,2],[2,2,2]]
Output: 46
```

Solution: 分六面看即可

```cpp
int surfaceArea(vector<vector<int>>& grid) {
    int sum = 0;
    // up and under
    for (auto i: grid) for (auto j: i) if (j) ++sum;
    sum *= 2;
    
    // four directions
    int localsum = 0, prev;
    for (int i = 0; i < grid.size(); ++i) {
        prev = 0;
        for (int j = 0; j < grid.size(); ++j) {
            if (grid[i][j] >= prev) localsum += grid[i][j] - prev;
            prev = grid[i][j];
        }
    }
    sum += localsum;
    
    localsum = 0;
    for (int i = 0; i < grid.size(); ++i) {
        prev = 0;
        for (int j = grid.size() - 1; j >= 0; --j) {
            if (grid[i][j] >= prev) localsum += grid[i][j] - prev;
            prev = grid[i][j];
        }
    }
    sum += localsum;
    
    localsum = 0;
    for (int j = 0; j < grid.size(); ++j) {
        prev = 0;
        for (int i = 0; i < grid.size(); ++i) {
            if (grid[i][j] >= prev) localsum += grid[i][j] - prev;
            prev = grid[i][j];
        }
    }
    sum += localsum;
    
    localsum = 0;
    for (int j = 0; j < grid.size(); ++j) {
        prev = 0;
        for (int i = grid.size() - 1; i >= 0; --i) {
            if (grid[i][j] >= prev) localsum += grid[i][j] - prev;
            prev = grid[i][j];
        }
    }
    sum += localsum;
    
    return sum;
}
```

### 893. Groups of Special-Equivalent Strings

You are given an array `A` of strings. Two strings `S` and `T` are *special-equivalent* if after any number of *moves*, S == T. A *move* consists of choosing two indices `i` and `j` with `i % 2 == j % 2`, and swapping `S[i]` with `S[j]`.

Now, a *group of special-equivalent strings from A* is a non-empty subset S of `A` such that any string not in S is not special-equivalent with any string in S.

Return the number of groups of special-equivalent strings from `A`.

Example:

```
Input: ["abcd","cdab","cbad","xyzz","zzxy","zzyx"]
Output: 3 (["abcd","cdab","cbad"], ["xyzz","zzxy"], ["zzyx"])
```

Solution: 十分巧妙的办法，把奇数字符和偶数字符sort一下之后，中间加个间隔符（比如#）concat到一起，然后放到hashset里，一定要背

```cpp
int numSpecialEquivGroups(vector<string>& A) {
    unordered_set<string> hash;
    for (auto &s: A) {
        string l, r;
        for (int i = 0; i < s.length(); ++i) {
            l += s[i];
            if (i < s.length()) r += s[++i];
        }
        sort(l.begin(), l.end());
        sort(r.begin(), r.end());
        l += '#' + r;
        hash.insert(l);
    }
    return hash.size();
}
```

### 894. All Possible Full Binary Trees

A *full binary tree* is a binary tree where each node has exactly 0 or 2 children. Return a list of all possible full binary trees with `N` nodes.  Each element of the answer is the root node of one possible tree. Each `node` of each tree in the answer **must** have `node.val = 0`. You may return the final list of trees in any order.

Example:

```
Input: 7
Output: [[0,0,0,null,null,0,0,null,null,0,0],[0,0,0,null,null,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,null,null,null,null,0,0],[0,0,0,0,0,null,null,0,0]]
```

Solution: divide and conquer，注意一个细节是需要对root硬拷贝，一定要背

```cpp
vector<TreeNode*> allPossibleFBT(int N) {
    if (N % 2 == 0) return vector<TreeNode*>();
    TreeNode* root = new TreeNode(0);
    if (N == 1) return vector<TreeNode*>{root};
    vector<TreeNode*> ret;
    --N;
    for (int i = 1; i <= N; i += 2) {
        vector<TreeNode*> left = allPossibleFBT(i);
        vector<TreeNode*> right = allPossibleFBT(N-i);
        for (auto l: left) for (auto r: right) {
            root->left = l;
            root->right = r;
            ret.push_back(copyroot(root));
        }
    }
    return ret;
}

TreeNode* copyroot(TreeNode* root) {
    if (!root) return NULL;
    TreeNode* ret = new TreeNode(0);
    ret->left = copyroot(root->left);
    ret->right = copyroot(root->right);
    return ret;
}
```

### 895. Maximum Frequency Stack

Implement `FreqStack`, a class which simulates the operation of a stack-like data structure.

`FreqStack` has two functions:

- `push(int x)`, which pushes an integer `x` onto the stack.
- ``pop()``, which removes and returns the most frequent element in the stack.
  - If there is a tie for most frequent element, the element closest to the top of the stack is removed and returned.

Solution: 用hashmap<key, freq>记录frequency，用hashmap<freq, stack\<key\>>记录按stack排序的frequency group，另外需要一个变量储存max frequency

```cpp
class FreqStack {
public:
    unordered_map<int, int> freq;
    unordered_map<int, stack<int>> group;
    int maxfreq;

    FreqStack() {
        freq.clear();
        group.clear();
        maxfreq = 0;
    }

    void push(int x) {
        int f = freq.find(x) == freq.end()? 1: freq[x] + 1;
        freq[x] = f;
        maxfreq = max(maxfreq, f);
        if (group.find(f) == group.end()) group[f] = stack<int>();
        group[f].push(x);
    }

    int pop() {
        int x = group[maxfreq].top();
        group[maxfreq].pop();
        --freq[x];
        if (group[maxfreq].empty()) --maxfreq;
        return x;
    }
};
```

### 896. Monotonic Array

An array is *monotonic* if it is either monotone increasing or monotone decreasing. An array `A` is monotone increasing if for all `i <= j`, `A[i] <= A[j]`.  An array `A` is monotone decreasing if for all `i <= j`, `A[i] >= A[j]`. Return `true` if and only if the given array `A` is monotonic.

Solution: 遍历一遍即可

```cpp
bool isMonotonic(vector<int>& A) {
    bool isIncreasing = true, isDecreasing = true;
    int prev = A[0], cur, diff;
    for (int i = 1; i < A.size(); ++i) {
        cur = A[i], diff = cur - prev;
        if (diff > 0) isDecreasing = false;
        if (diff < 0) isIncreasing = false; 
        if (!isDecreasing && !isIncreasing) break;
        prev = cur;
    }
    return isIncreasing || isDecreasing;
}
```

### 897. Increasing Order Search Tree

Given a tree, rearrange the tree in **in-order** so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only 1 right child.

Solution:

```
Input: [5,3,6,2,4,null,8,1,null,null,null,7,9]

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

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
            \
             7
              \
               8
                \
                 9  
```

Solution: 设置一个prev和head，一定要注意inorder里面操作的顺序，必须改右->前进->改左，即要在下一次前进前改左，否则在inorder的helper(root->left)时会出现环路，一定要背

```cpp
TreeNode *prev, *head;

TreeNode* increasingBST(TreeNode* root) {
    head = new TreeNode(-1), prev = head;
    helper(root);
    return head->right;
}

void helper(TreeNode* root) {
    if (!root) return;
    helper(root->left);
    prev->right = root;
    prev = prev->right;
    prev->left = NULL;
    helper(root->right);
}
```

### 898. Bitwise ORs of Subarrays

We have an array `A` of non-negative integers. For every (contiguous) subarray `B = [A[i], A[i+1], ..., A[j]]` (with `i <= j`), we take the bitwise OR of all the elements in `B`, obtaining a result `A[i] | A[i+1] | ... | A[j]`. Return the number of possible results.  (Results that occur more than once are only counted once in the final answer.)

Example:

```
Input: [1,1,2]
Output: 3 (The possible subarrays are [1], [1], [2], [1, 1], [1, 2], [1, 1, 2].
These yield the results 1, 1, 2, 1, 3, 3.
There are 3 unique values, so the answer is 3.)
```

Solution: 三个hashmap处理，本题也可换成求和求积等形式，十分巧妙，一定要背

```cpp
int subarrayBitwiseORs(vector<int>& A) {
    unordered_set<int> ret, prev;
    for (int i : A) {
        unordered_set<int> cur({i});
        for (int j : prev) cur.insert(j | i);
        ret.insert(cur.begin(), cur.end());
        prev = cur;
    }
   return ret.size();
}
```

### 899. Orderly Queue

A string `S` of lowercase letters is given.  Then, we may make any number of *moves*. In each move, we choose one of the first `K` letters (starting from the left), remove it, and place it at the end of the string. Return the lexicographically smallest string we could have after any number of moves.

Example:

```
Input: S = "baaca", K = 3
Output: "aaabc" (move "b" to end, and move the 3rd character ("c") to end)
```

Solution: 如果K大于1，直接sort即可；如果K等于1，则找一下从哪里截断后前后交换最小，一定要理解

```cpp
string orderlyQueue(string S, int K) {
    string ret = S;
    if (K == 1) {
        for(int i = 0; i < S.length(); i++) {
            ret = min(ret, S.substr(i + 1) + S.substr(0, i + 1));
        }
    } else if (K > 1) {
        sort(ret.begin(), ret.end());
    }
    return ret;
}
```

### 900. RLE Iterator

Write an iterator that iterates through a run-length encoded sequence.

The iterator is initialized by `RLEIterator(int[] A)`, where `A` is a run-length encoding of some sequence.  More specifically, for all even `i`, `A[i]` tells us the number of times that the non-negative integer value `A[i+1]` is repeated in the sequence.

The iterator supports one function: `next(int n)`, which exhausts the next `n` elements (`n >= 1`) and returns the last element exhausted in this way.  If there is no element left to exhaust, `next` returns `-1` instead.

For example, we start with `A = [3,8,0,9,2,5]`, which is a run-length encoding of the sequence `[8,8,8,5,5]`.  This is because the sequence can be read as "three eights, zero nines, two fives".

Example:

```
Input: ["RLEIterator","next","next","next","next"], [[[3,8,0,9,2,5]],[2],[1],[1],[2]]
Output: [null,8,8,5,-1]
```

Solution: 正常实现即可

```cpp
class RLEIterator {
public:
    vector<int> nums;
    int pos;
    bool isValid;
    RLEIterator(vector<int> A) {
        pos = 0;
        bool isValid = true;
        nums = A;
        if (nums.size() == 0 || nums.size() % 2) isValid = false;
    }
    
    int next(int n) {
        if (!isValid) return -1;
        while (nums[pos] < n) {
            n -= nums[pos];
            pos += 2;
            if (pos == nums.size()) {
                isValid = false;
                return -1;
            }
        }
        nums[pos] -= n;
        return n == 0? nums[pos-1]: nums[pos+1];
    }
};
```