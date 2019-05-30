# LeetCode 851-900

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

TODO: 893 - 900