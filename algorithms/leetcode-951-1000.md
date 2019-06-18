# LeetCode 951 - 1000

### 951. Flip Equivalent Binary Trees

For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees. A binary tree X is *flip equivalent* to a binary tree Y if and only if we can make X equal to Y after some number of flip operations. Write a function that determines whether two binary trees are *flip equivalent*.  The trees are given by root nodes `root1` and `root2`.

Example:

```
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
       1               1
    /     \         /     \
   2       3       3       2
  / \     /         \     / \
 4   5   6           6   4   5
    / \                     / \
   7   8                   8   7
Output: true
```

Solution: 递归

```cpp
bool flipEquiv(TreeNode* root1, TreeNode* root2) {
    if (!root1 && !root2) return true;
    if (!root1 || !root2) return false;
    if (root1->val != root2->val) return false;
    return (flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right)) || (flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left));
}
```

### 967. Numbers With Same Consecutive Differences

Return all **non-negative** integers of length `N` such that the absolute difference between every two consecutive digits is `K`. Note that **every** number in the answer **must not** have leading zeros **except** for the number `0` itself. You may return the answer in any order.

Example:

```
Input: N = 3, K = 7
Output: [181,292,707,818,929]
```

Solution: 递归或者hashing判重

```cpp
// method 1: pure recursion
vector<int> numsSameConsecDiff(int N, int K) {
    vector<int> res;
    for (int i = 1; i <= 9; ++i) add(res, K, i, N - 1);
    if (N == 1) res.push_back(0);
    return res;
}

void add(vector<int>& res, int diff, int curr, int n_rem) {
    if (!n_rem) {
        res.push_back(curr);
        return;
    }
    int last = curr % 10;
    curr *= 10;
    if (last + diff <= 9)
        add(res, diff, curr + last + diff, n_rem - 1);
    if (diff && last >= diff && (last != 0 || last - diff != 0))
        add(res, diff, curr + last - diff, n_rem - 1);
}

// method 2: hashing
vector<int> numsSameConsecDiff(int N, int K) {
    vector<vector<vector<string>>> wordmap(N, vector<vector<string>>(10, vector<string>()));
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < 10; ++j) {
            if (j - K >= 0) {
                if (wordmap[i-1][j-K].empty()) {
                    if (i == 1) wordmap[i][j].push_back(to_string(j-K));
                } else {
                    for (const auto & s: wordmap[i-1][j-K]) {
                        wordmap[i][j].push_back(to_string(j-K) + s);
                    }
                }
            }
            if (j + K < 10 && K != 0) {
                if (wordmap[i-1][j+K].empty()) {
                    if (i == 1) wordmap[i][j].push_back(to_string(j+K));
                } else {
                    for (const auto & s: wordmap[i-1][j+K]) {
                        wordmap[i][j].push_back(to_string(j+K) + s);
                    }
                }
            }
        }
    }
    vector<int> ret;
    if (N == 1) ret = vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    else for (int j = 1; j < 10; ++j) {
        for (const auto & s: wordmap[N-1][j]) {
            ret.push_back(stoi(to_string(j) + s));
        }
    }
    return ret;
}
```

### 969. Pancake Sorting

Given an array `A`, we can perform a *pancake flip*: We choose some positive integer `k <= A.length`, then reverse the order of the first **k** elements of `A`.  We want to perform zero or more pancake flips (doing them one after another in succession) to sort the array `A`.

Return the k-values corresponding to a sequence of pancake flips that sort `A`.  Any valid answer that sorts the array within `10 * A.length` flips will be judged as correct.

Example:

```
Input: [3,2,4,1]
Output: [4,2,4,3] (We perform 4 pancake flips, with k values 4, 2, 4, and 3.
Starting state: A = [3, 2, 4, 1]
After 1st flip (k=4): A = [1, 4, 2, 3]
After 2nd flip (k=2): A = [4, 1, 2, 3]
After 3rd flip (k=4): A = [3, 2, 1, 4]
After 4th flip (k=3): A = [1, 2, 3, 4], which is sorted.)
```

Solution: 每次把当前数字串的最大值放到最后：找当前最大值，然后翻转到第一位再反转到当前最后一位

```cpp
vector<int> pancakeSort(vector<int>& A) {
    vector<int> ret;
    for (int k = A.size(); k > 1; --k) {
        int maxindex = findmax(A, k);
        ret.push_back(maxindex+1);
        reverse(A.begin(), A.begin()+maxindex+1);
        ret.push_back(k);
        reverse(A.begin(), A.begin()+k);
    }
    return ret;
}

int findmax(const vector<int>& A, int k) {
    int maxindex = 0, maxn = 0;
    for (int i = 0; i < k; ++i) {
        if (A[i] > maxn) {
            maxn = A[i];
            maxindex = i;
        }
    }
    return maxindex;
}
```

### 970. Powerful Integers

Given two positive integers `x` and `y`, an integer is *powerful* if it is equal to `x^i + y^j` for some integers `i >= 0` and `j >= 0`. Return a list of all *powerful* integers that have value less than or equal to `bound`. You may return the answer in any order without duplicates.

Example:

```
Input: x = 2, y = 3, bound = 10
Output: [2,3,4,5,7,9,10] (
    2 = 2^0 + 3^0
    3 = 2^1 + 3^0
    4 = 2^0 + 3^1
    5 = 2^1 + 3^1
    7 = 2^2 + 3^1
    9 = 2^3 + 3^0
    10 = 2^0 + 3^2
)
```

Solution: 两个while loop即可，注意1的N次方还是1

```cpp
vector<int> powerfulIntegers(int x, int y, int bound) {
    unordered_set<int> res_set;
    int i = 0, j = 0, sum = pow(x, i) + pow(y, j);
    while (sum <= bound) {
        if (x == 1 && i == 1) break;
        while (sum <= bound) {
            if (y == 1 && j == 1) break;
            res_set.insert(sum);
            sum = pow(x, i) + pow(y, ++j);
        }
        j = 0, sum = pow(x, ++i) + pow(y, j);
    }
    return vector<int>(res_set.begin(), res_set.end());
}
```

### 971. Flip Binary Tree To Match Preorder Traversal

Given a binary tree with `N` nodes, each node has a different value from `{1, ..., N}`. A node in this binary tree can be *flipped* by swapping the left child and the right child of that node.

Consider the sequence of `N` values reported by a preorder traversal starting from the root.  Call such a sequence of `N` values the *voyage* of the tree. Our goal is to flip the **least number** of nodes in the tree so that the voyage of the tree matches the `voyage` we are given.

If we can do so, then return a list of the values of all nodes flipped.  You may return the answer in any order. If we cannot do so, then return the list `[-1]`.

Example:

```
Input: root = [1,2,3], voyage = [1,3,2]
  1
 / \
2   3
Output: [1]
```

Solution: dfs，利用一个swap可以巧妙地只对root而不是voyage做操作，一定要背

```cpp
vector<int> flipMatchVoyage(TreeNode* root, vector<int>& voyage) {
    if (!root) return vector<int>{-1};
    vector<int> res;
    int idx = 0;
    traversal(root, voyage, res, idx);
    return res;
}
void traversal(TreeNode* root, vector<int>& voyage, vector<int>& res, int& idx) {
    if (root->val != voyage[idx]) {
        res = {-1};
        return;
    }
    if (root->left) {
        if (voyage[idx+1] != root->left->val) {
            res.push_back(root->val);
            swap(root->left, root->right);
        }
        if (root->left) traversal(root->left, voyage, res, ++idx);
    }
    if (root->right) traversal(root->right, voyage, res, ++idx);
}
```

### 973. K Closest Points to Origin

We have a list of `points` on the plane.  Find the `K` closest points to the origin `(0, 0)`. You may return in any order.

Example:

```
Input: points = [[3,3],[5,-1],[-2,4]], K = 2
Output: [[3,3],[-2,4]]
```

Solution: quick selection

```cpp
// method 1: with algorithm.h
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
    nth_element(points.begin(), points.begin() + K - 1, points.end(),
                [](vector<int>& p, vector<int>& q) {
        return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
    });
    return vector<vector<int>>(points.begin(), points.begin() + K);
}
// method 2: without algorithm.h
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
    int l = 0, r = points.size() - 1;
    while (l < r) {
        int mid = quick_selection(points, l, r);
        if (mid == K) break;
        if (mid < K) l = mid + 1;
        else r = mid - 1;
    }
    return vector<vector<int>>(points.begin(), points.begin()+K);
}

int quick_selection(vector<vector<int>>& points, int l, int r) {
    int i = l + 1, j = r;
    while (true) {
        while (i < r && dist(points[i]) <= dist(points[l])) ++i;
        while (l < j && dist(points[j]) >= dist(points[l])) --j;
        if (i >= j) break;
        swap(points[i], points[j]);
    }
    swap(points[l], points[j]);
    return j;
}

double dist(vector<int> point) {
    return point[0]*point[0] + point[1]*point[1];
}
```

### 974. Subarray Sums Divisible by K

Given an array `A` of integers, return the number of (contiguous, non-empty) subarrays that have a sum divisible by `K`.

Example:

```
Input: A = [4,5,0,-2,-3,1], K = 5
Output: 7 ([4,5,0,-2,-3,1], [5], [5,0], [5,0,-2,-3], [0], [0,-2,-3], [-2,-3])
```

Solution: 前缀和，一个小技巧是可以用另外一个vector统计每个余数出现的频率f，这样每个余数的组合方式就是C(f, 2)，一定要背

```cpp
int subarraysDivByK(vector<int>& A, int K) {
    int n = A.size(), res = 0;
    vector<int> prefixSum(n + 1);
    for (int i = 0; i < n; ++i) prefixSum[i+1] = prefixSum[i] + A[i];
    
    vector<int> freq(K);
    for (int x: prefixSum) ++freq[(x % K + K ) % K];

    int ans = 0;
    for (int f: freq) ans += f * (f - 1) / 2;

    return ans;
}
```

### 976. Largest Perimeter Triangle

Given an array `A` of positive lengths, return the largest perimeter of a triangle with **non-zero area**, formed from 3 of these lengths. If it is impossible to form any triangle of non-zero area, return `0`.

Example:

```
Input: [3,6,2,3]
Output: 8 (2, 3, 3)
```

Solution: 先排序，然后从后往前看连续的三个，只要满足三角准则即可，否则各前推一位

```cpp
int largestPerimeter(std::vector<int> &A) {
    ssort(A.begin(), A.end());
    for (int i = A.size() - 3; i >= 0; --i)
        if (A[i] + A[i+1] > A[i+2])
            return A[i] + A[i+1] + A[i+2];
    return 0;
}
```

### 977. Squares of a Sorted Array

Given an array of integers `A` sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.

Example:

```
Input: [-7,-3,2,3,11]
Output: [4,9,9,49,121]
```

Solution: 先二分法找0，然后双指针向左向右推进

```cpp
vector<int> sortedSquares(vector<int>& A) {
    int n = A.size();
    if (n == 1) return vector<int>{A.back()*A.back()};
    vector<int> ret;
    int pos = lower_bound(A.begin(), A.end(), 0) - A.begin();
    int i = pos, j = pos - 1;
    while (j != -1 && i != n) {
        if (abs(A[i]) <= abs(A[j])) {
            ret.push_back(A[i]*A[i]);
            ++i;
        } else {
            ret.push_back(A[j]*A[j]);
            --j;
        }
    }
    while (j != -1) {
        ret.push_back(A[j]*A[j]);
        --j;
    }
    while (i != n) {
        ret.push_back(A[i]*A[i]);
        ++i;
    }
    return ret;
}
```

### 978. Longest Turbulent Subarray

A subarray `A[i], A[i+1], ..., A[j]` of `A` is said to be *turbulent* if and only if:

- For `i <= k < j`, `A[k] > A[k+1]` when `k` is odd, and `A[k] < A[k+1]` when `k` is even;
- **OR**, for `i <= k < j`, `A[k] > A[k+1]` when `k` is even, and `A[k] < A[k+1]` when `k` is odd.

That is, the subarray is turbulent if the comparison sign flips between each adjacent pair of elements in the subarray. Return the **length** of a maximum size turbulent subarray of A.

Example:

```
Input: [9,4,2,10,7,8,8,1,9]
Output: 5 (A[1] > A[2] < A[3] > A[4] < A[5])
```

Solutiion: 正常遍历即可

```cpp
int maxTurbulenceSize(vector<int>& A) {
    int n = A.size(), localmax = 1, globalmax = 1;
    if (n <= 2) return n;
    bool larger = A[0] > A[1];
    for (int i = 0; i < n - 1; ++i) {
        if (A[i] == A[i+1]) {
            while (A[i] == A[i+1] && i < n-1) ++i;
            if (i == n-1) break;
            larger = A[i] > A[i+1];
            globalmax = max(globalmax, localmax);
            localmax = 2;
        } else {
            if (i == 0) {
                ++localmax;
            } else if (larger) {
                if (A[i] < A[i+1]) {
                    ++localmax;
                    larger = !larger;
                } else {
                    globalmax = max(globalmax, localmax);
                    localmax = 2;
                }
            } else {
                if (A[i] > A[i+1]) {
                    ++localmax;
                    larger = !larger;
                } else {
                    globalmax = max(globalmax, localmax);
                    localmax = 2;
                }
            }
            if (i == n-2) globalmax = max(globalmax, localmax);
        }
    }
    return globalmax;
}
```

### 979. Distribute Coins in Binary Tree

Given the `root` of a binary tree with `N` nodes, each `node` in the tree has `node.val` coins, and there are `N` coins total. In one move, we may choose two adjacent nodes and move one coin from one node to another.  (The move may be from parent to child, or from child to parent.) Return the number of moves required to make every node have exactly one coin.

Example:

```
Input: [1,0,2]
    1
   / \
  0   2
Output: 2
```

Solution: dfs，注意技巧，一定要背

```cpp
int dfs (TreeNode* root, int & ret) {
    int sum = root? root->val + dfs(root->left, ret) + dfs(root->right, ret) - 1: 0;
    ret += abs(sum);
    return sum;
}
int distributeCoins(TreeNode* root) {
    int ret = 0;
    dfs(root, ret);
    return ret;
}
```

### 980. Unique Paths III

On a 2-dimensional `grid`, there are 4 types of squares:

- `1` represents the starting square.  There is exactly one starting square.
- `2` represents the ending square.  There is exactly one ending square.
- `0` represents empty squares we can walk over.
- `-1` represents obstacles that we cannot walk over.

Return the number of 4-directional walks from the starting square to the ending square, that **walk over every non-obstacle square exactly once**.

Example:

```
Input: [
		[1,0,0,0],
		[0,0,0,0],
		[0,0,2,-1]
]
Output: 2 (We have the following two paths: 
1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)
2. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2))
```

Solution: backtracking

```cpp
int dfs(vector<vector<int>>& grid, int x, int y, int n) {
    if (x < 0 || x >= grid[0].size() || y < 0 || y >= grid.size() || grid[y][x] == -1) return 0;
    if (grid[y][x] == 2) return !n? 1: 0;
    grid[y][x] = -1;
    int sum = dfs(grid, x + 1, y, n - 1) + dfs(grid, x, y + 1, n - 1) +
        dfs(grid, x - 1, y, n - 1) + dfs(grid, x, y - 1, n - 1);
    grid[y][x] = 0;
    return sum;
}
int uniquePathsIII(vector<vector<int>>& grid) {
    int sx = -1, sy = -1;
    int n = 1;
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[0].size(); ++j) {
            if (grid[i][j] == 0) ++n;
            else if (grid[i][j] == 1) {
                sy = i;
                sx = j;
            }
        }
    }
    return dfs(grid, sx, sy, n);
}
```

### 981. Time Based Key-Value Store

Create a timebased key-value store class `TimeMap`, that supports two operations.

`set(string key, string value, int timestamp)`

- Stores the `key` and `value`, along with the given `timestamp`.

`get(string key, int timestamp)`

- Returns a value such that `set(key, value, timestamp_prev)` was called previously, with `timestamp_prev <= timestamp`.

- If there are multiple such values, it returns the one with the largest `timestamp_prev`.

- If there are no values, it returns the empty string (`""`).

Example:

```
Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
Output: [null,null,"bar","bar",null,"bar2","bar2"] (The process is
TimeMap kv;   
kv.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1   
kv.get("foo", 1);  // output "bar"   
kv.get("foo", 3);  // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
kv.set("foo", "bar2", 4);   
kv.get("foo", 4);  // output "bar2"   
kv.get("foo", 5);  // output "bar2")
```

Solution: hashmap + 二分

```cpp
class TimeMap {
public:
    unordered_map<string, map<int, string>> hashmap;
    TimeMap() {}
    
    void set(string key, string value, int timestamp) {
        if (hashmap.count(key) == 0) {
            hashmap[key] = map<int, string>{make_pair(timestamp, value)};
        } else {
            hashmap[key][timestamp] = value;
        }
    }
    
    string get(string key, int timestamp) {
        if (hashmap.count(key) == 0) return "";
        auto pair = hashmap[key].lower_bound(timestamp);
        if (pair == hashmap[key].begin() && pair->first > timestamp) return "";
        if (pair->first > timestamp) --pair;       
        if (pair == hashmap[key].end()) return (hashmap[key].rbegin())->second;
        return pair->second;
    }
};
```

### 983. Minimum Cost For Tickets

In a country popular for train travel, you have planned some train travelling one year in advance.  The days of the year that you will travel is given as an array `days`.  Each day is an integer from `1`to `365`.

Train tickets are sold in 3 different ways:

- a 1-day pass is sold for `costs[0]` dollars;
- a 7-day pass is sold for `costs[1]` dollars;
- a 30-day pass is sold for `costs[2]` dollars.

The passes allow that many days of consecutive travel.  For example, if we get a 7-day pass on day 2, then we can travel for 7 days: day 2, 3, 4, 5, 6, 7, and 8.

Return the minimum number of dollars you need to travel every day in the given list of `days`.

Example:

```
Input: days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
Output: 17 (For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 30-day pass for costs[2] = $15 which covered days 1, 2, ..., 30.
On day 31, you bought a 1-day pass for costs[0] = $2 which covered day 31.
In total you spent $17 and covered all the days of your travel.)
```

Solution: dp

```cpp
int mincostTickets(vector<int>& days, vector<int>& costs) {
    int n = days.size(), maxday = days.back();
    vector<int> mincurrent (maxday+1, 0);
    vector<bool> visited(maxday+1, false);
    for (int i = 0; i<n; ++i) visited[days[i]] = true;
    for (int i = 1; i <= maxday; ++i)
        mincurrent[i] = !visited[i]? mincurrent[i-1]: min(mincurrent[i-1]+costs[0], min(mincurrent[max(0, i-7)]+costs[1], mincurrent[max(0, i-30)]+costs[2]));
    return mincurrent.back();
}
```

### 984. String Without AAA or BBB

Given two integers `A` and `B`, return **any** string `S` such that:

- `S` has length `A + B` and contains exactly `A` `'a'` letters, and exactly `B` `'b'` letters;
- The substring `'aaa'` does not occur in `S`;
- The substring `'bbb'` does not occur in `S`

Example:

```
Input: A = 4, B = 1
Output: "aabaa"
```

Solution: 长度有差的部分用"aab"或"bba"，补齐之后直接"ab"

```cpp
string strWithout3a3b(int A, int B) {
    int major = A < B? B: A;
    int minor = A < B? A: B;
    string majorStr = A < B? "b": "a";
    string minorStr = A < B? "a": "b";
    string res = "";
    while (major >= 2 && minor >= 1) {
        if (major - minor > 2) {
            res += majorStr + majorStr + minorStr;
            major = major - 2;
        } else {
            res += majorStr + minorStr;
            --major;
        }
        --minor;
    }
    while (major) {
        res += majorStr;
        --major;
    }
    while (minor) {
        res += minorStr;
        --minor;
    }
    return res;
}
```

### 985. Sum of Even Numbers After Queries

We have an array `A` of integers, and an array `queries` of queries. For the `i`-th query `val = queries[i][0], index = queries[i][1]`, we add val to `A[index]`.  Then, the answer to the `i`-th query is the sum of the even values of `A`. Return the answer to all queries.  Your `answer` array should have `answer[i]` as the answer to the `i`-th query.

Example:

```
Input: A = [1,2,3,4], queries = [[1,0],[-3,1],[-4,0],[2,3]]
Output: [8,6,2,4] (At the beginning, the array is [1,2,3,4].
After adding 1 to A[0], the array is [2,2,3,4], and the sum of even values is 2 + 2 + 4 = 8.
After adding -3 to A[1], the array is [2,-1,3,4], and the sum of even values is 2 + 4 = 6.
After adding -4 to A[0], the array is [-2,-1,3,4], and the sum of even values is -2 + 4 = 2.
After adding 2 to A[3], the array is [-2,-1,3,6], and the sum of even values is -2 + 6 = 4.)
```

Solution: brute force

```cpp
vector<int> sumEvenAfterQueries(vector<int>& A, vector<vector<int>>& queries) {
    vector<int> res;
    int n = queries.size();
    int evensum = 0;
    for (auto i: A) if (!(i % 2)) evensum += i;
    for (auto p: queries) {
        int val = p[0], pos = p[1];
        if (A[pos] % 2) {
            if (val % 2) evensum += A[pos] + val;
        } else {
            if (val % 2) evensum -= A[pos];
            else evensum += val;
        }
        A[pos] += val;
        res.push_back(evensum);
    }
    return res;
}
```

### 986. Interval List Intersections

Given two lists of **closed** intervals, each list of intervals is pairwise disjoint and in sorted order. Return the intersection of these two interval lists.

Example:

```
Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

Solution: 双指针

```cpp
vector<Interval> intervalIntersection(vector<Interval>& A, vector<Interval>& B) {
    vector<Interval> result;
    int i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        int lo = max(A[i].start, B[j].start);
        int hi = min(A[i].end, B[j].end);
        if (lo <= hi) result.emplace_back(lo, hi);
        if (A[i].end > B[j].end) ++j;
        else ++i;
    }
    return result;
}
```

### 987. Vertical Order Traversal of a Binary Tree

Given a binary tree, return the *vertical order* traversal of its nodes values.

Example:

```
Input: [3,9,20,null,null,15,7]
      3
     / \
    9   20
       /  \
      15   7
Output: [[9],[3,15],[20],[7]]
```

Solution: bfs

```cpp
vector<vector<int>> verticalTraversal(TreeNode* root) {
    vector<vector<int>> answer;
    if (!root) return answer;
    map<int, vector<pair<int, int>>> myMap;

    int h_dist = 0, v_dist = 0;

    queue<pair<TreeNode*, pair<int,int>>> myQueue;
    myQueue.push(make_pair(root, make_pair(h_dist, v_dist)));

    while (!myQueue.empty()) {
        auto front = myQueue.front();
        myQueue.pop();
        auto currentNode = front.first;
        auto currentDistance = front.second;
        h_dist = (currentDistance).first, v_dist = (currentDistance).second;
        myMap[h_dist].push_back( make_pair(v_dist, currentNode->val) );
        auto leftChild = currentNode->left, rightChild = currentNode->right;
        if (leftChild) myQueue.push(make_pair(leftChild, make_pair(h_dist-1, v_dist+1)));
        if (rightChild) myQueue.push(make_pair(rightChild, make_pair(h_dist+1, v_dist+1)));
    }

    answer.resize(myMap.size());
    int index = 0;

    for (auto ele: myMap) {
        auto row_vec = ele.second;
        sort(row_vec.begin(), row_vec.end());
        for(auto val: row_vec) answer[index].push_back(val.second);
        ++index;
    }

    return answer;
}
```

### 988. Smallest String Starting From Leaf

Given the `root` of a binary tree, each node has a value from `0` to `25` representing the letters `'a'` to `'z'`: a value of `0` represents `'a'`, a value of `1` represents `'b'`, and so on. Find the lexicographically smallest string that starts at a leaf of this tree and ends at the root.

Example:

```
Input: [0,1,2,3,4,3,4] (
        a
      /   \
     b     c
    / \   / \
   d   e d   e
)
Output: "dba"
```

Solution: dfs

```cpp
string smallestFromLeaf(TreeNode* r, string s = "") {
    if (!r) return "|";
    s = char('a' + r->val) + s;
    return r->left == r->right? s: min(smallestFromLeaf(r->left, s), smallestFromLeaf(r->right, s));
}
```

### 989. Add to Array-Form of Integer

For a non-negative integer `X`, the *array-form of X* is an array of its digits in left to right order.  For example, if `X = 1231`, then the array form is `[1,2,3,1]`. Given the array-form `A` of a non-negative integer `X`, return the array-form of the integer `X+K`.

Example:

```
Input: A = [1,2,0,0], K = 34
Output: [1,2,3,4]
Explanation: 1200 + 34 = 1234
```

Solution: 正常从前往后进位

```cpp
vector<int> addToArrayForm(vector<int>& A, int K) {
    int carry = 0;
    for (int i = A.size() - 1; i>=0; --i){
        carry = A[i] + K % 10 + carry;
        A[i] = carry % 10;
        carry /= 10;
        K /= 10;
    }
    while (K) {
        carry = K % 10 + carry;
        A.insert(A.begin(), carry % 10);
        carry /= 10;
        K /= 10;
    }
    if (carry) A.insert(A.begin(), 1);
    return A;
}
```

### 990. Satisfiability of Equality Equations

Given an array equations of strings that represent relationships between variables, each string `equations[i]` has length `4` and takes one of two different forms: `"a==b"` or `"a!=b"`.  Here, `a`and `b` are lowercase letters (not necessarily different) that represent one-letter variable names. Return `true` if and only if it is possible to assign integers to variable names so as to satisfy all the given equations.

Example:

```
Input: ["a==b","b!=a"]
Output: false
```

Solution: union and find

```cpp
class UF {
public:
    vector<int> id;
    UF(int N) {
        id = vector<int>(N+1);
        for (int i = 0; i < id.size(); ++i) id[i] = i;
    }
    void connect(int u, int v) {
        int uID = find(u);
        int vID = find(v);
        if (uID == vID) return;
        for (int i = 0; i < id.size(); ++i) if (id[i] == uID) id[i] = vID;
    }
    int find(int p) {
        return id[p];
    }
    bool is_connected(int u, int v) {
        return find(u) == find(v);
    }
};

class Solution {
public:
    bool equationsPossible(vector<string>& equations) {
        UF *uf = new UF(26);
        for (string & s: equations) {
            int a = s[0] - 'a', b = s[3] - 'a';
            if (s[1] == '=') uf->connect(a, b);
        }
        for (string & s: equations) {
            int a = s[0] - 'a', b = s[3] - 'a';
            if (s[1] == '!') if (uf->is_connected(a, b)) return false;
        }
        return true;
    }
};
```

### 991. Broken Calculator

On a broken calculator that has a number showing on its display, we can perform two operations:

- **Double**: Multiply the number on the display by 2, or;
- **Decrement**: Subtract 1 from the number on the display.

Initially, the calculator is displaying the number `X`. Return the minimum number of operations needed to display the number `Y`.

Example:

```
Input: X = 5, Y = 8
Output: 2 (5 -> 4 -> 8)
```

Solution: 数学题

```cpp
int brokenCalc(int X, int Y) {
    return X >= Y? X - Y: Y % 2? 1 + brokenCalc(X, Y + 1): 1 + brokenCalc(X, Y / 2);
}
```

### 993. Cousins in Binary Tree

In a binary tree, the root node is at depth `0`, and children of each depth `k` node are at depth `k+1`. Two nodes of a binary tree are *cousins* if they have the same depth, but have **different parents**. Given two values `x` and `y`, return if the nodes corresponding to the values `x` and `y` are cousins. No nodes share the same value.

Example:

```
Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
  1
 / \
2   3
 \   \
  4   5
Output: true
```

Solution: dfs，可以用pair<depth, parent>来记录深度和parent

```cpp
void getInfo(TreeNode * root, vector<pair<int, TreeNode*>> & map, const int x, const int y, TreeNode * parent, int depth) {
    if (!root) return;
    if (root->val == x) map[0] = make_pair(depth, parent);
    if (root->val == y) map[1] = make_pair(depth, parent);
    getInfo(root->left, map, x, y, root, depth + 1);
    getInfo(root->right, map, x, y, root, depth + 1);
}
bool isCousins(TreeNode* root, int x, int y) {   
    vector<pair<int, TreeNode*>> map(2);
    getInfo(root, map, x, y, NULL, 0);
    return map[0].first == map[1].first && map[0].second != map[1].second;
}
```

### 994. Rotting Oranges

In a given grid, each cell can have one of three values:

- the value `0` representing an empty cell;
- the value `1` representing a fresh orange;
- the value `2` representing a rotten orange.

Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no cell has a fresh orange.  If this is impossible, return `-1` instead.

Example:

```
Input: [
    [2,1,1],
    [1,1,0],
    [0,1,1]
]
Output: 4
```

Solution: bfs

```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int steps = 0;
    int N = grid.size(), M = grid[0].size();
    queue<pair<int, int>> rotten;
    set<pair<int, int>> fresh;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (grid[i][j] == 2) rotten.push({i, j});
            else if (grid[i][j] == 1) fresh.insert({i, j});
        }
    }
    int count = 0;
    while (rotten.size()) {
        count = rotten.size();
        while (count) {
            int i = rotten.front().first, j = rotten.front().second;
            rotten.pop();
            if (i > 0 && fresh.find({i-1, j}) != fresh.end()) {
                    fresh.erase({i-1, j});
                    rotten.push({i-1, j});
            }
            if (j > 0 && fresh.find({i, j-1}) != fresh.end()) {
                    fresh.erase({i, j-1});
                    rotten.push({i, j-1});
            }
            if (i < N && fresh.find({i+1, j}) != fresh.end()) {
                    fresh.erase({i+1, j});
                    rotten.push({i+1, j});
            }
            if (j < M && fresh.find({i, j+1}) != fresh.end()) {
                    fresh.erase({i, j+1});
                    rotten.push({i, j+1});
            }
            --count;
        }
        ++steps;
    }
    if (!fresh.empty()) return -1;
    if (steps == 0) return 0;
    return steps-1;
}
```

### 997. Find the Town Judge

In a town, there are `N` people labelled from `1` to `N`.  There is a rumor that one of these people is secretly the town judge. If the town judge exists, then:

1. The town judge trusts nobody.
2. Everybody (except for the town judge) trusts the town judge.
3. There is exactly one person that satisfies properties 1 and 2.

You are given `trust`, an array of pairs `trust[i] = [a, b]`representing that the person labelled `a` trusts the person labelled `b`. If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return `-1`.

Example:

```
Input: N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
Output: 3
```

Solution: 遍历一遍即可

```cpp
int findJudge(int N, vector<vector<int>>& trust) {
    vector<int> trusted_cnt(N+1, 0);
    vector<bool> trust_someone(N+1, false);
    for(vector<int>& p: trust){
         trusted_cnt[p[1]]++;
         trust_someone[p[0]] = true;
    }
    for (int i = 1; i <= N; ++i){
        if (!trust_someone[i] && trusted_cnt[i] == N-1) return i;
    }
    return -1;
}
```

### 998. Maximum Binary Tree II

We are given the `root` node of a *maximum tree:* a tree where every node has a value greater than any other value in its subtree. Just as in the [654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/), the given tree was constructed from an list `A` (`root = Construct(A)`) recursively with the following `Construct(A)` routine:

- If `A` is empty, return `null`.
- Otherwise, let `A[i]` be the largest element of `A`.  Create a `root` node with value `A[i]`.
- The left child of `root` will be `Construct([A[0], A[1], ..., A[i-1]])`
- The right child of `root` will be `Construct([A[i+1], A[i+2], ..., A[A.length - 1]])`
- Return `root`.

Note that we were not given A directly, only a root node `root = Construct(A)`. Suppose `B` is a copy of `A` with the value `val` appended to it.  It is guaranteed that `B` has unique values. Return `Construct(B)`.

Example:

```
Input: root = [4,1,3,null,null,2], val = 5
  4
 / \
1   3
   /
  2
Output: [5,4,null,1,3,null,null,2]
    5
   /
  4
 / \
1   3
   /
  2
```

Solution: dfs，注意细节

```cpp
TreeNode* insertIntoMaxTree(TreeNode* root, int val) {
    if (!root || val > root->val) {
        TreeNode* n = new TreeNode(val);
        n->left = root;
        return n;
    }
    root->right = insertIntoMaxTree(root->right, val);
    return root;
}
```

### 999. Available Captures for Rook

On an 8 x 8 chessboard, there is one white rook.  There also may be empty squares, white bishops, and black pawns.  These are given as characters 'R', '.', 'B', and 'p' respectively. Uppercase characters represent white pieces, and lowercase characters represent black pieces.

The rook moves as in the rules of Chess: it chooses one of four cardinal directions (north, east, west, and south), then moves in that direction until it chooses to stop, reaches the edge of the board, or captures an opposite colored pawn by moving to the same square it occupies.  Also, rooks cannot move into the same square as other friendly bishops.

In short, 'R' eats 'p' and gets block by 'B'. Return the number of pawns the rook can capture in one move.

Example:

```
Input: [
    [".",".",".",".",".",".",".","."],
    [".",".",".","p",".",".",".","."],
    [".",".",".","p",".",".",".","."],
    ["p","p",".","R",".","p","B","."],
    [".",".",".",".",".",".",".","."],
    [".",".",".","B",".",".",".","."],
    [".",".",".","p",".",".",".","."],
    [".",".",".",".",".",".",".","."]
]
Output: 3 (The rook can capture the pawns at positions b5, d6 and f5)
```

Solution: 先定位车，然后上下左右各遍历一遍即可

```cpp
int cap(vector<vector<char>>& b, int x,int y, int dx, int dy) {
    while (x >= 0 && x < b.size() && y >= 0 && y < b[x].size() && b[x][y] != 'B') {
        if (b[x][y] == 'p') return 1;
        x += dx;
        y += dy;
    }    
    return 0;
}

int numRookCaptures(vector<vector<char>>& board) {
    for (int i = 0; i < board.size(); ++i)
        for (int j = 0; j < board[i].size(); ++j)
            if (board[i][j] == 'R')
                return cap(board, i, j, -1, 0) + cap(board, i, j, 0, -1) + cap(board, i, j, 0, 1) + cap(board, i, j, 1, 0);
    return 0;
}
```