# LeetCode 251 - 300

### 251. Flatten 2D Vector

Design and implement an iterator to flatten a 2d vector. It should support the following operations: `next` and `hasNext`.

Example:

```
Vector2D iterator = new Vector2D([[1,2],[3],[4]]);
iterator.next(); // return 1
iterator.next(); // return 2
iterator.next(); // return 3
iterator.hasNext(); // return true
iterator.hasNext(); // return true
iterator.next(); // return 4
iterator.hasNext(); // return false
```

Solution: 设置指针遍历即可

```cpp
class Vector2D {
public:
    Vector2D(vector<vector<int>>& v) : vec(v), i(-1), j(0) {
        while (++i < vec.size() && vec[i].empty()) continue;
    }
    
    int next() {
        int next_val = vec[i][j];
        if (++j >= vec[i].size()) {
            j = 0;
            while (++i < vec.size() && vec[i].empty()) continue;
        }
        return next_val;
    }
    
    bool hasNext() {
        return i < vec.size();
    }
private:
    int i, j;
    vector<vector<int>> vec;
};
```

### 253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times `[[s1,e1],[s2,e2],...]` (si < ei), find the minimum number of conference rooms required.

Example:

```
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2
```

Solution: 排序+贪心

```cpp
int minMeetingRooms(vector<vector<int>>& intervals) {
    vector<pair<int, int>> pairs;
    for (const auto & v: intervals) {
        pairs.emplace_back(v[0], 1);
        pairs.emplace_back(v[1], -1);
    }
    int cnt = 0, max_cnt = 0;
    sort(pairs.begin(), pairs.end());
    for (const auto & p: pairs) {
        cnt += p.second;
        max_cnt = max(max_cnt, cnt);
    }
    return max_cnt;
}
```

### 257. Binary Tree Paths

Given a binary tree, return all root-to-leaf paths.

Example:

```text
Input:
   1
 /   \
2     3
 \
  5
Output: ["1->2->5", "1->3"]
```

Solution: 递归dfs，容易眼高手低，一定要背

```cpp
// without backtracking
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> res;
    if (!root) return res;
    helper(res, root, "");
    return res;
}

void helper(vector<string> &res, TreeNode* root, string prev) {
    if (!prev.empty()) prev += "->";
    prev += to_string(root->val);
    if (!root->left && !root->right) {res.push_back(prev); return;}
    if (root->left) helper(res, root->left, prev);
    if (root->right) helper(res, root->right, prev);
}

// with backtracking
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> res;
    vector<int> path;
    if (!root) return res;
    backtracking(res, root, path);
    return res;
}

void backtracking(vector<string> &res, TreeNode* root, vector<int> &path) {
    path.push_back(root->val);
    if (!root->left && !root->right) {
        string base;
        for (const int& i: path) {
            if (!base.empty()) base += "->";
            base += to_string(i);
        }
        res.push_back(base);
    } else {
        if (root->left) backtracking(res, root->left, path);
        if (root->right) backtracking(res, root->right, path);
    }
    path.pop_back();
}
```

### 258. Add Digits

Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

Example:

```text
Input: 38
Output: 2 (3 + 8 = 11, 1 + 1 = 2)
```

Solution: 根据[Congruence formula](https://en.wikipedia.org/wiki/Digital_root#Congruence_formula)，结果是1 + \(num - 1\) % 9

```cpp
int addDigits(int num) {
    return 1 + (num - 1) % 9;
}
```

### 260. Single Number III

Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.

Example:

```text
Input:  [1,2,1,3,2,5]
Output: [3,5]
```

Solution: 先XOR找出那两个数的diff，运用位运算diff &= -diff 得到出 diff 最右侧不为 0 的位，也就是不存在重复的两个元素在位级表示上最右侧不同的那一位，利用这一位就可以将两个元素区分开来

```cpp
vector<int> singleNumber(vector<int>& nums) {
    int diff = accumulate(nums.begin(), nums.end(), 0, bit_xor<int>());
    diff &= -diff;  // 得到最右一位
    int intA = 0, intB = 0;
    for (auto item : nums) {
        if (item & diff) intA = intA ^ item;
        else intB = intB ^ item;
    }
    return vector<int>{intA, intB};   
}
```

### 263. Ugly Number

Write a program to check whether a given number is an ugly number. Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 1 is typically treated as an ugly number.

Example:

```text
Input: 6
Output: true (6 = 2 × 3)
```

Solution: 注意int取值范围

```cpp
bool isUgly(int num) {
    if (num <= 0) return false;
    while (!(num % 5)) num /= 5;
    while (!(num % 3)) num /= 3;
    while (!(num % 2)) num >>= 1;
    return num == 1;
}
```

### 264. Ugly Number II

Write a program to find the n-th ugly number. Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 1 is typically treated as an ugly number.

Example:

```text
Input: n = 10
Output: 12 (1, 2, 3, 4, 5, 6, 8, 9, 10, 12)
```

Solution: dp with indexing，一定要背

```cpp
int nthUglyNumber(int n) {
    vector<int> dp(n, 1);
    int i2 = 0, i3 = 0, i5 = 0, next2 = 2, next3 = 3, next5 = 5;
    for (int i = 1; i < n; ++i) {
        dp[i] = min(min(next2, next3), next5);
        if (dp[i] == next2) next2 = dp[++i2] * 2;
        if (dp[i] == next3) next3 = dp[++i3] * 3;
        if (dp[i] == next5) next5 = dp[++i5] * 5;
    }
    return dp.back();
}
```

### 268. Missing Number

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array. Implement it using only constant extra space.

Example:

```text
Input: [9,6,4,2,3,5,7,0,1]
Output: 8
```

Solution: bit manipulation或求和公式法，一定要背

```cpp
// method 1: bit manipulation
int missingNumber(vector<int>& nums) {
    int missing = nums.size();
    for (int i = 0; i < nums.size(); ++i) missing ^= i ^ nums[i];
    return missing;
}

// method 2: Gauss' Formula
int missingNumber(vector<int>& nums) {
    return nums.size()*(nums.size() + 1)/2 - accumulate(nums.begin(), nums.end(), 0);
}
```

### 269. Alien Dictionary

There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of **non-empty** words from the dictionary, where **words are sorted lexicographically by the rules of this new language**. Derive the order of letters in this language.

Example:

```
Input: [
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]

Output: "wertf"
```

Solution: 先记录先后关系，然后拓扑排序

```cpp
string alienOrder(vector<string>& strings) {
    string order;
    vector<int> toVisit;
    vector<int> inCount(26, 0);
    unordered_map<int, unordered_set<int>> graph;
    int n = strings.size();
    for (auto str: strings) {
        for (auto c: str){
            int cur = c - 'a';
            if (graph.find(cur) == graph.end())
                graph[cur] = unordered_set<int>();
        }
    }
    string prevstr = strings[0], curstr;
    for (int i = 1; i < n; ++i) {
        curstr = strings[i];
        for (int j = 0; j < min(curstr.length(), prevstr.length()); ++j) {
            int left = prevstr[j] - 'a', right = curstr[j] - 'a';
            if (left == right) continue;
            if (graph[left].find(right) == graph[left].end()){
                graph[left].insert(right);
                ++inCount[right];
            }
            break;
        }
        prevstr = curstr;
    }
    for (auto i: graph) {
        if (!inCount[i.first]) toVisit.push_back(i.first);
    }
    while (!toVisit.empty()){
        int cur = toVisit.back();
        toVisit.pop_back();
        order.push_back(cur + 'a');
        for (auto i: graph[cur]) {
             if (--inCount[i] == 0) {
                toVisit.push_back(i);
            }
        }
    }
    for (auto i: graph) {
        if (inCount[i.first]) return "";
    }
    return order;
}
```

### 274. H-Index

Given an array of citations \(each citation is a non-negative integer\) of a researcher, write a function to compute the researcher's h-index. According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

Example:

```text
Input: citations = [3,0,6,1,5]
Output: 3
```

Solution: 先遍历一遍进行桶排序（大于数组大小n的算n），然后从后往前遍历，当累计个数 &gt; 数组index的时候的时候即为h

```cpp
int hIndex(vector<int>& citations) {
    if (citations.empty()) return 0;
    int n = citations.size();
    vector<int> buckets (n+1,0);
    for (int c: citations) {
        if (c >= n) ++buckets[n];
        else ++buckets[c];
    }
    int h_index = 0;
    for (int i = n; i >= 0; --i) {
        h_index += buckets[i];
        if (h_index >= i) return i;
    }
    return h_index;
}
```

### 275. H-Index II

Given an array of citations sorted in ascending order \(each citation is a non-negative integer\) of a researcher, write a function to compute the researcher's h-index. According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

Example:

```text
Input: citations = [0,1,3,5,6]
Output: 3
```

Solution: 二分法，一定要背

```cpp
// 不推荐左闭右闭，但这题这么写好写一点
int hIndex(vector<int>& citations) {
    int n = citations.size();
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (citations[m] == n - m) return n - m;
        if (citations[m] >= n - m) r = m - 1;
        else l = m + 1;
    }
    return n - l;
}
```

### 277. Find the Celebrity

Suppose you are at a party with `n` people (labeled from `0` to `n - 1`) and among them, there may exist one celebrity. The definition of a celebrity is that all the other `n - 1` people know him/her but he/she does not know any of them.

Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" to get information of whether A knows B. You need to find out the celebrity (or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

You are given a helper function `bool knows(a, b)` which tells you whether A knows B. Implement a function `int findCelebrity(n)`. There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return `-1`.

Example:

```
Input: graph = [
  [1,1,0],
  [0,1,0],
  [1,1,1]
]
Output: 1 (There are three persons labeled with 0, 1 and 2. graph[i][j] = 1 means person i knows person j, otherwise graph[i][j] = 0 means person i does not know person j. The celebrity is the person labeled as 1 because both 0 and 2 know him but 1 does not know anybody.)
```

Solution: 可以用visited + single pass

```cpp
// Forward declaration of the knows API.
bool knows(int a, int b);

class Solution {
public:
    int findCelebrity(int n) {
        vector<bool> visited(n, false);
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            for (int j = 0; j <= n; ++j) {
                if (i == j) continue;
                if (j == n) return i;
                if (!knows(j, i)) {
                    visited[i] = true;
                    break;
                }
                visited[j] = true;
                if (knows(i, j)) {
                    visited[i] = true;
                    break;
                }
            }
        }
        return -1;
    }
};
```

### 278. First Bad Version

Suppose you have n versions \[1, 2, ..., n\] and you want to find out the first bad one, which causes all the following ones to be bad. You are given an API bool isBadVersion\(version\) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example:

```text
Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version.
```

Solution: 二分法，想好了再写，一定要背

```cpp
int firstBadVersion(int n) {
    long l = 0, r = n, mid;
    while (l < r) {
        mid = (l + r) / 2;
        if (isBadVersion(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```

### 279. Perfect Squares

Given a positive integer n, find the least number of perfect square numbers \(for example, 1, 4, 9, 16, ...\) which sum to n.

Example:

```text
Input: n = 13
Output: 2 (13 = 4 + 9)
```

Solution: static dp或者Lagrange's Four Square theorem，用static dp是为了防止多次调用重复打表

```cpp
// method 1: static dp
// dp[i] = the least number of perfect square numbers which sum to i. Since it is a static vector, if 
// its size > n, we have already calculated the result during previous function calls.
int numSquares(int n)  {
    if (n <= 0) return 0;
    static vector<int> dp({0});
    while (dp.size() <= n) {
        int m = dp.size(), cntSqr = INT_MAX;
        for (int i = 1; i * i <= m; ++i) cntSqr = min(cntSqr, dp[m - i*i] + 1);
        dp.push_back(cntSqr);
    }
    return dp[n];
}

// method 2: Lagrange's Four Square theorem, there are only 4 possible results: 1, 2, 3, 4
int is_square(int n) {  
    int sqrt_n = (int)sqrt(n);  
    return sqrt_n*sqrt_n == n;  
}

int numSquares(int n) {
    // if n is a perfect square, return 1.
    if (is_square(n))  return 1;  

    // result is 4 if and only if n can be written in the form of 4^k*(8*m + 7)
    while (!(n & 3)) n >>= 2;  // or equally !(n % 4)
    if ((n & 7) == 7) return 4; // or equally n % 8 == 7

    // check whether 2 is the result.
    int sqrt_n = (int)sqrt(n); 
    for (int i = 1; i <= sqrt_n; ++i) if (is_square(n - i * i)) return 2;

    return 3;  
}
```

### 280. Wiggle Sort

Given an unsorted array `nums`, reorder it **in-place** such that `nums[0] <= nums[1] >= nums[2] <= nums[3]...`.

Example:

```
Input: nums = [3,5,2,1,6,4]
Output: One possible answer is [3,5,1,6,2,4]
```

Solution: 直接遍历即可，容易想复杂，一定要背

```cpp
void wiggleSort(vector<int>& nums) {
    if (nums.empty()) return;
    for (int i = 0; i < nums.size() - 1; ++i) 
        if ((i % 2) == (nums[i] < nums[i+1])) swap(nums[i], nums[i+1]);
}
```

### 283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements. You must do this in-place without making a copy of the array. Minimize the total number of operations.

Example:

```text
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

Solution: 记录几个0，遍历先把非0的移动，然后剩余的改成0即可

```cpp
void moveZeroes(vector<int>& nums) {
    int pos = 0;
    for (int i = 0; i < nums.size(); ++i) {
        if (nums[i]) {
            if (i > pos) nums[pos] = nums[i];
            ++pos;
        }
    }
    for (int i = pos; i < nums.size(); ++i) nums[i] = 0;
}
```

### 284. Peeking Iterator

Given an Iterator class interface with methods: next\(\) and hasNext\(\), design and implement a PeekingIterator that support the peek\(\) operation -- it essentially peek\(\) at the element that will be returned by the next call to next\(\).

Example:

```text
class Iterator {
    struct Data;
    Data* data;
public:
    Iterator(const vector<int>& nums);
    Iterator(const Iterator& iter);
    virtual ~Iterator();
    int next();
    bool hasNext() const;
};

Assume that the iterator is initialized to the beginning of the list: [1,2,3].

Call next() gets you 1, the first element in the list.
Now you call peek() and it returns 2, the next element. Calling next() after that still return 2. 
You call next() the final time and it returns 3, the last element. 
Calling hasNext() after that should return false.
```

Solution: 两种方法，一种是维护一个cache变量，一种是给Iterator输入this，一定要背

```cpp
// method 1: cache
class PeekingIterator : public Iterator {
public:
    PeekingIterator(const vector<int>& nums) : Iterator(nums) {
        _flag = false;
    }

    int peek() {
        if (!_flag) {
            _value = Iterator::next();
            _flag = true;
        }
        return _value;
    }

    int next() {
        if (!_flag) return Iterator::next();
        _flag = false;
        return _value;
    }

    bool hasNext() const {
        if (_flag) return true;
        if (Iterator::hasNext()) return true;
        return false;
    }
private:
    int _value;
    bool _flag;
};

// method 2: *this
class PeekingIterator : public Iterator {
public:
    PeekingIterator(const vector<int>& nums) : Iterator(nums) {}

    int peek() {
        return Iterator(*this).next();
    }

    int next() {
        return Iterator::next();
    }

    bool hasNext() const {
        return Iterator::hasNext();
    }
};
```

### 285. Inorder Successor in BST

Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

The successor of a node `p` is the node with the smallest key greater than `p.val`.

Example:

```
Input: root = [2,1,3], p = 1
  2
 / \
1   3
Output: 2
```

Solution: 中序遍历，注意return空指针的细节

```cpp
TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
    if (!root) return root;
    TreeNode* left = inorderSuccessor(root->left, p);
    if (left) return left;
    if (p->val < root->val) return root;
    TreeNode* right = inorderSuccessor(root->right, p);
    return right;
}
```

### 287. Find the Duplicate Number

Given an array nums containing n + 1 integers where each integer is between 1 and n \(inclusive\), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one. You must not modify the array.

Example:

```text
Input: [1,3,4,2,2]
Output: 2
```

Solution: 因为如果有相同数字，表示多个映射都指向它，因此可以用快慢指针，一定要背

```cpp
int findDuplicate(vector<int>& nums) {
    int slow = 0, fast = 0;
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);
    int find = 0;
    while (find != slow) {
        slow = nums[slow];
        find = nums[find];
    }
    return find;
}
```

### 289. Game of Life

Given a board with m by n cells, each cell has an initial state live \(1\) or dead \(0\). Each cell interacts with its eight neighbors \(horizontal, vertical, diagonal\) using the following four rules : \(1\) Any live cell with fewer than two live neighbors dies, as if caused by under-population; \(2\) Any live cell with two or three live neighbors lives on to the next generation; \(3\) Any live cell with more than three live neighbors dies, as if by over-population; \(4\) Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction. Write a function to compute the next state \(after one update\) of the board given its current state. The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Could you solve it in-place?

Example:

```text
Input: 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
Output: 
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]
```

Solution: 不in-place怎么做都可以, in-place的话可以先改二进制的第二位，比较的时候只与1做&操作; 等都运算完了再&gt;&gt;=。一定要背

```cpp
void gameOfLife(vector<vector<int>>& board) {
    int d[][2] = {{1,-1},{1,0},{1,1},{0,-1},{0,1},{-1,-1},{-1,0},{-1,1}};
    for (int i = 0; i < board.size(); ++i ) {
        for (int j = 0; j < board[0].size(); ++j) {
            int live = 0;
            for (int k = 0; k < 8; k++) {
                int x = d[k][0] + i, y = d[k][1] + j;
                if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size()) continue;
                if (board[x][y] & 1) ++live;
            }
            if (board[i][j] == 0) {
                if (live == 3) board[i][j] = 2;
            } else {
                if (live < 2 || live > 3) board[i][j] = 1;
                else board[i][j] = 3;
            }
        }
    }
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            board[i][j] >>= 1;
        }
    }
}
```

### 290. Word Pattern

Given a pattern and a string str, find if str follows the same pattern.

Example:

```text
Input: pattern = "abba", str = "dog cat cat dog"
Output: true
Input: pattern = "abba", str = "dog dog dog dog"
Output: false
```

Solution: 因为是双射，所以要做双向map。注意map/hashmap在做\[\]取值的时候，如果没有对应键，会自动插入一个默认的空值；如果这样不影响结果的话，那么比写find要好写一点

```cpp
bool wordPattern(string pattern, string str) {
    istringstream ss(str);
    string s;
    vector<string> v;
    while (ss >> s) v.push_back(s);
    if (v.size() != pattern.size()) return false;

    map<string, char> s2c;
    map<char, string> c2s;
    for (int i = 0; i < v.size(); ++i) {
        if (s2c[v[i]] == 0 && c2s[pattern[i]] == "") { 
            s2c[v[i]] = pattern[i];
            c2s[pattern[i]] = v[i];
            continue;
        }
        if (s2c[v[i]] != pattern[i]) return false;
        // trick here, safe to omit checking (c2s[pattern[i]] != match)
    }
    return true;
}
```

### 292. Nim Game

You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones. Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.

Example:

```text
Input: 4
Output: false
```

Solution: 不是4的倍数都可以。方法是玩家拿走1到3个，使剩余成石头个数为4的倍数，这样无论对方拿走几个，玩家下一回合还可以使其变成4的倍数，最终对方就会遇到4颗石头的情况，无论拿几个都一定会输

```cpp
bool canWinNim(int n) {
    return n % 4;
}
```

### 295. Find Median from Data Stream

Design a data structure that supports the following two operations: void addNum\(int num\) and double findMedian\(\).

Solution: 维护两个pq: 一个最大堆，一个最小堆，最大堆最大的比最小堆最小的小，注意移动的方法和目标，一定要背

```cpp
class MedianFinder {
    struct compare {bool operator()(int a, int b) const {return a > b;}};
    priority_queue<int, vector<int>, compare> big;
    priority_queue<int, vector<int>> small;
public:
    MedianFinder() {};
    void addNum(int num) {
        if (big.empty() || num > big.top()) { 
            big.emplace(num); 
            if (big.size() > small.size() + 1) {
                small.emplace(big.top());
                big.pop();
            }
        }
        else {
            small.emplace(num);
            if (small.size() > big.size() + 1) {
                big.emplace(small.top());
                small.pop();
            }
        }
    };
    double findMedian() const {
        auto bigSize = big.size();
        auto smallSize = small.size();
        auto allSize = smallSize + bigSize;
        if (allSize & 1) return bigSize > smallSize ? big.top() : small.top();
        return (big.top() + small.top()) / 2.0;
    };
};
```

### 300. Longest Increasing Subsequence

Given an unsorted array of integers, find the length of longest increasing subsequence. Could you improve it to O\(n log n\) time complexity?

Example:

```text
Input: [10,9,2,5,3,7,101,18]
Output: 4 ([2,3,7,101])
```

Solution: dp只能O\(n^2\)，想要O\(nlogn\)可以遍历+lowerbound，每次用lowerbound修改已知序列，一定要背

```cpp
// method 1: lowerbound
int lengthOfLIS(vector<int>& nums) {
    int len = nums.size();
    if (len <= 1) return len;
    vector<int> before;
    before.push_back(nums[0]);
    for (int i = 1; i < len; ++i) {
        if (before.back() < nums[i]) before.push_back(nums[i]);
        else before[lower_bound(before.begin(), before.end(), nums[i]) - before.begin()] = nums[i];
    }
    return before.size();
}

// method 2: dp
int lengthOfLIS(vector<int>& nums) {
    int curmax = 0, n = nums.size();
    if (!n) return 0;
    if (n == 1) return 1;
    vector<int> dp(n, 1);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < i; ++j){
            if (nums[i] > nums[j])
                dp[i] = max(dp[i], dp[j]+1);
        }
        curmax = max(curmax, dp[i]);
    }
    return curmax;
}
```

