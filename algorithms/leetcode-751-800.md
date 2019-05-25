# LeetCode 751 - 800

### 753. Cracking the Safe

There is a box protected by a password. The password is `n` digits, where each letter can be one of the first `k` digits `0, 1, ..., k-1`. You can keep inputting the password, the password will automatically be matched against the last `n` digits entered.

For example, assuming the password is `"345"`, I can open it when I type `"012345"`, but I enter a total of 6 digits. Please return any string of minimum length that is guaranteed to open the box after the entire string is inputted.

Example:

```
Input: n = 2, k = 2
Output: "00110" ("01100", "10011", "11001" will be accepted too)
```

Solution: hashmap记录何时翻转，十分巧妙

```cpp
string crackSafe(int n, int k) {
    int i, total = pow(k, n);
    unordered_map<string, int> record;
    string ans;
    ans.assign(n, '0' + k - 1);
    for (i = 1; i < total; ++i)
        ans += '0' + record[ans.substr(ans.length() - n + 1, n - 1)]++;
    return ans;
}
```

### 767. Reorganize String

Given a string `S`, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same. If possible, output any possible result.  If not possible, return the empty string.

Example:

```
Input: "aab"
Output: "aba"
```

Solution: 先用桶排序存频率，然后分配

```cpp
string reorganizeString(string S) {
    if (!S.length()) return S;
  
    char maxChar = S[0];
    vector<int> count(26, 0);
    for (auto c : S) {
        count[c-'a']++;
        maxChar = (count[maxChar-'a'] < count[c-'a'] ? c : maxChar);
    }

    if (count[maxChar - 'a'] > (S.length() + 1) / 2) return "";

    int next = 0;
    while (count[maxChar - 'a']-- > 0) {
        S[next] = maxChar;
        next += 2;
    }

    for (int i = 0; i < count.size(); i++) {
        while (count[i]-- > 0) {
            next = (next >= S.length() ? 1 : next);
            S[next] = 'a' + i;
            next += 2;
        }
    }
    return S;
}
```

### 768. Max Chunks To Make Sorted II

Given an array `arr` of integers (**not necessarily distinct**), we split the array into some number of "chunks" (partitions), and individually sort each chunk.  After concatenating them, the result equals the sorted array. What is the most number of chunks we could have made?

Example:

```
Input: [2,1,3,4,4]
Output: 4 ([2, 1], [3], [4], [4])
```

Solution: stack，很难想到，一定要背

```cpp
int maxChunksToSorted(vector<int>& arr) {
    stack<int> st;
    for(int i = 0; i < arr.size(); ++i) {
        int curmax = st.empty()? arr[i]: max(st.top(), arr[i]);
        while (!st.empty() && st.top() > arr[i]) st.pop();
        st.push(curmax);
    }
    return st.size();
}
```

### 769. Max Chunks To Make Sorted

Given an array `arr` that is a permutation of `[0, 1, ..., arr.length - 1]`, we split the array into some number of "chunks" (partitions), and individually sort each chunk.  After concatenating them, the result equals the sorted array. What is the most number of chunks we could have made?

Example:

```
Input: arr = [1,0,2,3,4]
Output: 4 ([1, 0], [2], [3], [4])
```

Solution: 遍历一遍，根据index找limit，一定要背

```cpp
int maxChunksToSorted(vector<int>& arr) {
    int ret = 0, curmax = 0;
    for (int i = 0; i < arr.size(); ++i) {
        curmax = max(curmax, arr[i]);
        if (curmax == i) ++ret;
    }
    return ret;
}
```

### 779. K-th Symbol in Grammar

On the first row, we write a `0`. Now in every subsequent row, we look at the previous row and replace each occurrence of `0` with `01`, and each occurrence of `1` with `10`. Given row `N` and index `K`, return the `K`-th indexed symbol in row `N`. (The values of `K` are 1-indexed)

Examples:

```
Input: N = 1, K = 1
Output: 0 ('0')

Input: N = 2, K = 1
Output: 0 ('01')

Input: N = 2, K = 2
Output: 1 ('0110')

Input: N = 4, K = 5
Output: 1 ('01101001')
```

Solution: recursion

```cpp
int kthGrammar(int N, int K) {
    if (K > pow(2, N-2)) {
        if (K - pow(2, N-2) > pow(2, N-3)) return kthGrammar(N, K - pow(2, N-2) - pow(2, N-3));
        else return kthGrammar(N, K - pow(2, N-3));
    }
    if (N == 1) return 0;
    if (N == 2) return K == 1? 0: 1;
    return kthGrammar(N - 1, K);
}
```

### 783. Minimum Distance Between BST Nodes

Given a Binary Search Tree (BST) with the root node `root`, return the minimum difference between the values of any two different nodes in the tree.

Example:

```
Input: root = [4,2,6,1,3,null,null]
          4
        /   \
      2      6
     / \    
    1   3  
Output: 1
```

Solution: in order traversal

```cpp
int min_diff, prev;
int minDiffInBST(TreeNode* root) {
    if (!root) return 0;
    min_diff = INT_MAX, prev = -1;
    helper(root);
    return min_diff;
}
void helper(TreeNode* root) {
    if (!root) return;
    helper(root->left);
    if (prev != -1) min_diff = min(min_diff, root->val - prev);
    prev = root->val;
    helper(root->right);
}
```

### 785. Is Graph Bipartite?

Given an undirected graph, return true if and only if it is bipartite. A graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B. The graph is given in the following form: graph\[i\] is a list of indexes j for which the edge between nodes i and j exists. Each node is an integer between 0 and graph.length - 1. There are no self edges or parallel edges: graph\[i\] does not contain i, and it doesn't contain any element twice.

Example:

```text
Input: [[1,3], [0,2], [1,3], [0,2]]
0----1
|    |
|    |
3----2
Output: true ({0, 2}, {1, 3})
```

Solution: 染色法，一定要背

```cpp
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    if (!n) return true;
    vector<int> color(n, 0);
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (!color[i]) {
            q.push(i);
            color[i] = 1;
        }
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            for (auto j: graph[node]) {
                if (color[j] == 0) {
                    q.push(j);
                    color[j] = color[node] == 2 ? 1 : 2;
                }
                else if (color[node] == color[j]) return false;
            }
        }
    }
    return true; 
}
```

拓展: 寻找二分图的最大匹配 - [匈牙利算法](https://www.renfei.org/blog/bipartite-matching.html)

```cpp
struct Edge {
    int from;
    int to;
    int weight;
    Edge(int f, int t, int w):from(f), to(t), weight(w) {}
};

vector<vector<int>> G;  // G[i] 存储顶点 i 出发的边的编号
vector<Edge> edges;
int num_nodes, num_left, num_right, num_edges;
vector<int> matching(__maxNodes, 0);  // 存储求解结果
vector<int> check(__maxNodes, 0);  // 交替路

// method 1: dfs
bool dfs(int u) {
    for (auto i: G[u]) { // 对 u 的每个邻接点
        int v = edges[i].to;
        if (!check[v]) {     // 要求不在交替路中
            check[v] = true; // 放入交替路
            if (matching[v] == -1 || dfs(matching[v])) {
                // 如果改点是未匹配点或是增广路起点，说明交替路为增广路，则储存匹配，并返回成功
                matching[v] = u;
                matching[u] = v;
                return true;
            }
        }
    }
    return false; // 不存在增广路，返回失败
}

int hungarian_dfs() {
    int ans = 0;
    fill(matching.begin(), matching.end(), -1);
    for (int u = 0; u < num_left; ++u) {
        if (matching[u] == -1) {
            fill(check.begin(), check.end(), 0);  // 在每一步中清空
            if (dfs(u)) ++ans;
        }
    }
    return ans;
}

// method2: bfs
queue<int> Q;
vector<int> prev(__maxNodes, 0);

int hungarian_bfs() {
    int ans = 0;
    fill(matching.begin(), matching.end(), -1);
    fill(check.begin(), check.end(), -1);
    for (int i = 0; i < num_left; ++i) {
        if (matching[i] == -1) {
            while (!Q.empty()) Q.pop();
            Q.push(i);
            prev[i] = -1; // 设 i 为路径起点
            bool flag = false;  // 尚未找到增广路
            while (!Q.empty() && !flag) {
                int u = Q.front();
                for (auto ix: G[u]) {
                    int v = edges[ix].to;
                    if (check[v] != i) {
                        check[v] = i;
                        Q.push(matching[v]);
                        if (matching[v] >= 0) { // 此点为匹配点
                            prev[matching[v]] = u;
                        } else { // 找到未匹配点，交替路变为增广路
                            flag = true;
                            int d=u, e=v;
                            while (d != -1) {
                                int t = matching[d];
                                matching[d] = e;
                                matching[e] = d;
                                d = prev[d];
                                e = t;
                            }
                            break;
                        }
                    }
                }
                Q.pop();
            }
            if (matching[i] != -1) ++ans;
        }
    }
    return ans;
}
```

### 789. Escape The Ghosts

You are playing a simplified Pacman game. You start at the point `(0, 0)`, and your destination is` (target[0], target[1])`. There are several ghosts on the map, the i-th ghost starts at` (ghosts[i][0], ghosts[i][1])`.

Each turn, you and all ghosts simultaneously *may* move in one of 4 cardinal directions: north, east, west, or south, going from the previous point to a new point 1 unit of distance away. You escape if and only if you can reach the target before any ghost reaches you (for any given moves the ghosts may take.)  If you reach any square (including the target) at the same time as a ghost, it doesn't count as an escape.

Example:

```
Input: ghosts = [[1, 0]], target = [2, 0]
Output: false (You need to reach the destination (2, 0), but the ghost at (1, 0) lies between you and the destination)
```

Solution: 如果人比鬼先能到终点， 那么人一定能赢

```cpp
int distance(vector<int>& a, vector<int>& b) {
    return abs(a[0] - b[0]) + abs(a[1] - b[1]);
}
bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
    int t = abs(target[0]) + abs(target[1]);
    for (auto& v : ghosts) if (distance(v, target) <= t) return false;
    return true;
}
```

### 792. Number of Matching Subsequences

Given string `S` and a dictionary of words `words`, find the number of `words[i]` that is a subsequence of `S`.

Example:

```
Input: S = "abcde", words = ["a", "bb", "acd", "ace"]
Output: 3 ("a", "acd", "ace")
```

Solution: 正常遍历即可，一个小trick是可以加上查重统计

```cpp
int numMatchingSubseq(string S, vector<string>& words) {
    int ret = 0, n = S.length();
    unordered_map<string, int> wordcounts;
    for (auto w: words) {
        if (wordcounts.find(w) != wordcounts.end()) ++wordcounts[w];
        else wordcounts[w] = 1;
    }
    for (auto wc: wordcounts) {
        string w = wc.first;
        int j = 0, m = w.length();
        for (int i = 0; i < n; ++i) {
            if (j == m) break;
            if (S[i] == w[j]) ++j;
        }
        if (j == m) ret += wc.second;
    }
    return ret;
}
```

### 795. Number of Subarrays with Bounded Maximum

We are given an array `A` of positive integers, and two positive integers `L`and `R` (`L <= R`). Return the number of (contiguous, non-empty) subarrays such that the value of the maximum array element in that subarray is at least `L` and at most `R`.

Example:

```
Input: A = [2, 1, 4, 3], L = 2, R = 3
Output: 3 ([2], [2, 1], [3])
```

Solution: dp，先做一个lookup table记录每个位置及左边都小于L的个数，然后用dp[i]表示以i为结尾的能满足要求的最长连续串的长度，最后需要累计，一定要背

```cpp
int numSubarrayBoundedMax(vector<int>& A, int L, int R) {
    if (A.empty()) return 0;
    int n = A.size();
    vector<int> dp(n, 0), lower_than_L(n, 0);

    // build lower than L lookup table
    if (A[0] < L) lower_than_L[0] = 1;
    for (int i = 1; i < n; ++i) if (A[i] < L) lower_than_L[i] = lower_than_L[i-1] + 1;

    // build dp: dp[i] means required subarray that ends at i
    if (A[0] >= L && A[0] <= R) dp[0] = 1;
    for (int i = 1; i < n; ++i) {
        if (A[i] > R) continue;
        if (A[i] >= L) dp[i] = lower_than_L[i-1] + 1;
        dp[i] += dp[i-1];
    }

    return accumulate(dp.begin(), dp.end(), 0);
}
```