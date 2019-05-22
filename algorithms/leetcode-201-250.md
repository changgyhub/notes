# LeetCode 201 - 250

### 201. Bitwise AND of Numbers Range

Given a range \[m, n\] where 0 &lt;= m &lt;= n &lt;= 2147483647, return the bitwise AND of all numbers in this range, inclusive.

Example:

```text
Input: [5,7]
Output: 4
```

Solution: 等同于找common binary prefix，一定要背和理解

```cpp
int rangeBitwiseAnd(int m, int n) {
    int offset = 0;
    while (m != n) {
        m >>= 1;
        n >>= 1;
        ++offset;
    }
    return m << offset;
}
```

### 202. Happy Number

Write an algorithm to determine if a number is "happy": A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 \(where it will stay\), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

Example:

```text
Input: 19
Output: true (1^2 + 9^2 = 82, 8^2 + 2^2 = 68, 6^2 + 8^2 = 100, 1^2 + 0^2 + 0^2 = 1)
```

Solution: 快慢指针判圈法，一定要背

```cpp
bool isHappy(int n) {

    auto digitsum = [](int a)->int{
        int sum = 0;
        while (a) {
            sum += (a % 10) * (a % 10);
            a /= 10;
        }
        return sum;
    };

    int slow = n, fast = n;
    do {
        slow = digitsum(slow);
        fast = digitsum(digitsum(fast));
    } while (slow != fast);

    return slow == 1;
}
```

### 203. Remove Linked List Elements

Remove all elements from a linked list of integers that have value val.

Example:

```text
Input:  1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5
```

Solution: 可以加也可以不加dummy，遍历一遍即可，一定要背

```cpp
// with dummy
ListNode* removeElements(ListNode* head, int val) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head; head = dummy;
    while (head->next) {
        if (head->next->val == val) head->next = head->next->next;
        else head = head->next;
    } 
    return dummy->next;
}

// without dummy
ListNode* removeElements(ListNode* head, int val) {
    while (head) {
        if (head->val == val) head = head->next;
        else break;
    }
    if (!head) return NULL;
    ListNode* cur = head;
    while (cur->next) {
        if (cur->next->val == val) cur->next = cur->next->next;
        else cur = cur->next;
    } 
    return head;
}
```

### 204. Count Primes

Count the number of prime numbers less than a non-negative number, n.

Example:

```text
Input: 10
Output: 4 (2,3,5,7)
```

Solution: 埃式筛法，但是有很多trick可以提速，一定要背和理解

```cpp
// optimized
int countPrimes(int n) {
    if (n <= 2) return 0;
    vector<bool> visited(n, false);
    int i = 3, sqrtn = sqrt(n), count = n / 2;         // avoid even numbers
    while (i <= sqrtn) {                               // divisors must less than sqrtn
        for (int j = i * i; j < n; j += (i << 1)) {    // to avoid even numbers and counted numbers
            if (!visited[j]) --count;                  // avoid repeated visit
            visited[j] = true;
        }
        do i += 2; while (i <= sqrtn && visited[i]);   // avoid repeated visit and even number
    }
    return count;
}

// unoptimized
int countPrimes(int n) {
    if (n <= 2) return 0;
    int count = n - 2;  // 1 is not prime
    vector<bool> prime(n, true);
    for (int i = 2; i <= sqrt(n); ++i) {
        if (prime[i]) {
            for (int j = 2*i; j < n; j += i) {
                if (prime[j]) {
                    prime[j] = false;
                    --count;
                }
            }
        }
    }
    return count;
}
```

### 205. Isomorphic Strings

Given two strings s and t, determine if they are isomorphic. Two strings are isomorphic if the characters in s can be replaced to get t. All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself. You may assume both s and t have the same length.

Example:

```text
Input: s = "paper", t = "title"
Output: true
```

Solution: 打两个表即可，用两个是因为必须满足双摄，可以一个表记录映射，一个表记录是否重复映射

```cpp
bool isIsomorphic(string s, string t) {
    vector<char> dict(127, 0);
    vector<bool> used(127, false);
    int spos = 0, tpos = 0;
    for (int i = 0; i < s.length(); ++i) {
        spos = s[i], tpos = t[i];
        if (dict[spos] == 0) {
            if (used[tpos]) return false;
            else {
                dict[spos] = tpos;
                used[tpos] = true;
            }
        }
        else if (dict[spos] != tpos) return false;
    }
    return true;
}
```

### 206. Reverse Linked List

Reverse a singly linked list.

Solution: 正常操作，一定要背

```cpp
ListNode* reverseList(ListNode* head) {
    if (!head) return NULL;

    ListNode *left = NULL, *right = NULL;

    while (head) {
        right = head->next;
        head->next = left;
        left = head;
        head = right;
    }

    return left;
}
```

### 207. Course Schedule

There are a total of n courses you have to take, labeled from 0 to n-1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: \[0,1\]. Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example:

```text
Input: 2, [[1,0],[0,1]]
Output: false
```

Solution: dfs找环路，或拓扑排序找环路，一定要背

```cpp
// dfs
bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    if (!prerequisites.size()) return true;
    vector<vector<int>> map(numCourses, vector<int>());
    vector<bool> onStack(numCourses, false);
    vector<bool> isVisited(numCourses, false);
    for (int i = 0; i < prerequisites.size(); ++i)
        map[prerequisites[i].first].push_back(prerequisites[i].second);
    for (int i = 0; i < numCourses; ++i)
        if (!isVisited[i] && hasCycle(map, i, isVisited, onStack)) return false;
    return true;
}

bool hasCycle(vector<vector<int>> &map, int i, vector<bool> &isVisited, vector<bool> &onStack) {
    isVisited[i] = true;
    onStack[i] = true;
    for (int k : map[i]) {
        if (onStack[k]) return true;
        else if (hasCycle(map, k, isVisited, onStack)) return true;
    }
    onStack[i] = false;
    return false;
}

// topological sort
bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> graph(numCourses,vector<int>(0));
    vector<int> indegree(numCourses, 0);
    for (auto u: prerequisites) {
        graph[u.second].push_back(u.first);
        ++indegree[u.first];
    }
    queue<int> q;
    for (int i = 0; i < indegree.size(); ++i) {
        if (!indegree[i]) q.push(i);
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto v: graph[u]) {
            --indegree[v];
            if (!indegree[v]) q.push(v);
        }

    }
    for (int i = 0; i < indegree.size(); ++i) {
        if (indegree[i]) return false;
    }
    return true;
}
```

### 208. Implement Trie \(Prefix Tree\)

Implement a trie with insert, search, and startsWith methods.

Example:

```text
Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true
```

Solution: 设置一个TrieNode树，每个TrieNode含一个flag和一个TrieNode\* childNode\[26\]。一定要背

```cpp
class TrieNode {
public:
    TrieNode* childNode[26];
    bool isVal;
    TrieNode() {
        isVal = false;
        for (int i = 0; i < 26; ++i) childNode[i] = NULL;
    }
};

class Trie {
public:
    Trie() {root = new TrieNode();}

    // Inserts a word into the trie
    void insert(string word) {
        TrieNode* temp = root;
        for (int i = 0; i < word.size(); ++i) {
            if (!temp->childNode[word[i]-'a']) temp->childNode[word[i]-'a'] = new TrieNode();
            temp = temp->childNode[word[i]-'a'];
        }
        temp->isVal = true;
    }

    // Returns if the word is in the trie
    bool search(string word) {
        TrieNode* temp = root;
        for (int i = 0; i < word.size(); ++i) {
            if (!temp) break;
            temp = temp->childNode[word[i]-'a'];
        }
        return temp? temp->isVal: false;
    }

    // Returns if there is any word in the trie that starts with the given prefix
    bool startsWith(string prefix) {
        TrieNode* temp = root;
        for (int i = 0; i < prefix.size(); ++i) {
            if (!temp) break;
            temp = temp->childNode[prefix[i]-'a'];
        }
        return temp;
    }

private:
    TrieNode* root;
};
```

### 209. Minimum Size Subarray Sum

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example:

```text
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2 ([4,3])
```

Solution: 从左到右遍历一遍即可，更新left、right和最小差

```cpp
int minSubArrayLen(int s, vector<int>& nums) {
    int minLen = INT_MAX, left = 0, right = 0;
    for (; right < nums.size(); ++right) {
        s -= nums[right];
        if (s <= 0) {
            while (s + nums[left] <= 0) s += nums[left++];
            minLen = min(minLen, right - left + 1);
        }
    }
    return s > 0 ? 0: minLen;
}
```

### 210. Course Schedule II

There are a total of n courses you have to take, labeled from 0 to n-1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: \[0,1\] Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses. There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example:

```text
Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
```

Solution: 拓扑排序，同Q207

```cpp
vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> graph(numCourses,vector<int>(0));
    vector<int> indegree(numCourses, 0), res;
    for (auto u: prerequisites) {
        graph[u.second].push_back(u.first);
        ++indegree[u.first];
    }
    queue<int> q;
    for (int i = 0; i < indegree.size(); ++i) {
        if (!indegree[i]) q.push(i);
    }
    while (!q.empty()) {
        int u = q.front();
        res.push_back(u);
        q.pop();
        for (auto v: graph[u]) {
            --indegree[v];
            if (!indegree[v]) q.push(v);
        }

    }
    for (int i = 0; i < indegree.size(); ++i) {
        if (indegree[i]) return vector<int>();
    }
    return res;
}
```

### 211. Add and Search Word - Data structure design

Design a data structure that supports add and search a word. search\(word\) can search a literal word or a regular expression string containing only letters a-z or '.', where '.' means it can represent any one letter. You may assume that all words are consist of lowercase letters a-z.

Example:

```text
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
```

Solution: Trie或者hashmap，优劣势都不同

```cpp
// hashmap, key is word len
class WordDictionary {
public:
    WordDictionary() {}

    void addWord(string word) {
        int n = word.size();
        if (map.find(n) == map.end()) {
            vector<string> cur;
            cur.push_back(word);
            map[n] = cur;
        } else {
            map[n].push_back(word);
        }
    }

    bool search(string word) {
        int n = word.size();
        if (map.find(n) == map.end()) return false;
        vector<string> words = map[n];
        for (auto s: words) {
            int i = 0;
            for (; i < n; ++i) {
                if (word[i] != '.' && word[i] != s[i]) break;
            }
            if (i == n) return true;
        }
        return false;
    }

private:
    unordered_map<int, vector<string>> map;
};

// Trie
class TrieNode{
    TrieNode* children[26];
    bool ending = false;
public:
    TrieNode() {
        for (int i = 0; i < 26; ++i) children[i] = NULL;
        ending = false;
    }

    bool find(string word, int index) {
        if (index == word.size()) return ending;
        if (word[index] == '.') {
            for (int i = 0; i < 26; ++i) {
                if (children[i] && children[i]->find(word, index+1)) {
                    return true;
                }
            }
            return false;
        }
        int c = word[index] - 'a';
        return children[c] ? children[c]->find(word, index+1): false;
    }

    void insert(string word, int index) {
        if (word.size() == index) {
            ending = true;
        } else {
            int c = word[index] - 'a';
            if (!children[c]) children[c] = new TrieNode();
            children[c]->insert(word, index+1);
        }
    }

};

class WordDictionary {
    TrieNode* root;
public:
    WordDictionary() {
        root = new TrieNode();
    }

    void addWord(string word) {
        root->insert(word, 0);
    }

    bool search(string word) {
        return root->find(word, 0);
    }
};
```

### 212. Word Search II

Given a 2D board and a list of words from the dictionary, find all words in the board. Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word. You may assume that all inputs are consist of lowercase letters a-z.

Example:

```text
Input: 
words = ["oath","pea","eat","rain"] and board =
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

Output: ["eat","oath"]
```

Solution: TrieNode + 四向搜索，注意搜索的时候需要设立一个used table，防止搜过去又回来

```cpp
struct TrieNode {
    TrieNode* children[26];
    int wordId = -1;
};

class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        // build Trie
        TrieNode* root = new TrieNode();
        for (int i = 0; i < words.size(); ++i) {
            TrieNode* cur = root;
            for (char c: words[i]) {
                if (!cur->children[c-'a']) cur->children[c-'a'] = new TrieNode();
                cur = cur->children[c-'a'];
            }
            cur->wordId = i;
        }

        unordered_set<int> found;
        int h = board.size();
        int w = board[0].size();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                GetWordsAt(board, root, i, j, h, w, found);
            }
        }

        vector<string> ans;
        for (auto id: found) ans.push_back(words[id]); 
        return ans;
    }

    void GetWordsAt(vector<vector<char>>& board, TrieNode* root, int i, int j, int h, int w, unordered_set<int>& found) {
        // if the a neighbor char can be found in the Trie and the neighbor hasn't been used, proceed.
        TrieNode* next = root->children[board[i][j]-'a'];
        if (next) {
            // if current char is the end of a word, add current word to found
            if (next->wordId >= 0) found.insert(next->wordId);
            char temp = board[i][j];
            board[i][j] = 0;  // important: mark as used!
            if (i > 0 && board[i-1][j]) GetWordsAt(board, next, i - 1, j, h, w, found);
            if (i < h - 1 && board[i+1][j]) GetWordsAt(board, next, i + 1, j, h, w, found);
            if (j > 0 && board[i][j-1]) GetWordsAt(board, next, i, j - 1, h, w, found);
            if (j < w - 1 && board[i][j+1]) GetWordsAt(board, next, i, j + 1, h, w, found);
            board[i][j] = temp;
        }
    }
};
```

### 213. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night. Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example:

```text
Input: [1,2,3,1]
Output: 4 (Rob house 1 (money = 1) and then rob house 3 (money = 3). Total amount you can rob = 1 + 3 = 4.)
```

Solution: 同Q198，在循环的情况下，只需要分成两个dp就可，分别为是否rob第0个房子，也可以把dp转化成一个prev一个cur

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (!n) return 0;
    if (n == 1) return nums[0];
    vector<int> dp1(n+1, 0), dp2(n+1, 0);
    for (int i = 0; i < n - 1; ++i) {
        dp1[i+1] = max(dp1[i], nums[i] + dp1[i-1]);
        dp2[i+2] = max(dp2[i+1], nums[i+1] + dp2[i]);
    }
    return max(dp1[n-1], dp2[n]);
}
```

### 215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element. You may assume k is always valid.

Example:

```text
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

Solution: 很多解法，如median of medians \(最大O\(n\)复杂度，但是空间要求大，数据小的时候不一定好\), c++自带的nth\_element，quicksort更新pivot当二分用，priority queue/multiset自动排序等等

```cpp
// qsort
bool greater(int inc, int a, int b) {
    return inc > 0? a > b: a < b;
}
int quicksort(vector<int> &nums, int k, int l, int r) {
    int pivot = l + rand() % (r - l), end = r - 1;
    swap(nums[pivot], nums[l]);
    pivot = l; 

    int inc = 1;
    while (pivot != end) {
        if (greater(inc, nums[pivot], nums[end])) {
            swap(nums[pivot], nums[end]);
            swap(pivot, end);
            inc = inc > 0 ? -1 : 1;
        }
        pivot += inc;
    }

    if (nums.size() - k == pivot) return nums[pivot];
    else if (nums.size() - k < pivot) return quicksort(nums, k, l, pivot);
    return quicksort(nums, k, pivot + 1, r);
}

int findKthLargest(vector<int> &nums, int k) {
    return quicksort(nums, k, 0, nums.size());
}

// multiset
int findKthLargest(vector<int>& nums, int k) {
    multiset<int> mset;
    int n = nums.size();
    for (int i = 0; i < n; i++) { 
        mset.insert(nums[i]);
        if (mset.size() > k) mset.erase(mset.begin());
    }
    return *mset.begin();
}

// priority queue
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int> pq(nums.begin(), nums.end());
    for (int i = 0; i < k - 1; ++i) pq.pop(); 
    return pq.top();
}
```

### 216. Combination Sum III

Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Example:

```text
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
```

Solution: backtrack

```cpp
void dfs(int t, int start, vector<vector<int>>& res, vector<int>& tmp, int k) {
    if (tmp.size() == k) {
        if (!t) res.push_back(tmp);
        return;
    } else if (tmp.size() > k || t < 0) {
        return;
    }
    for (int i = start; i < 10; ++i) {
        tmp.push_back(i);
        dfs(t-i, i+1, res, tmp, k);
        tmp.pop_back();
    }
}

vector<vector<int>> combinationSum3(int k, int n) {
    vector<vector<int>> res;
    vector<int> tmp;
    dfs(n, 1, res, tmp, k);
    return res;
}
```

### 217. Contains Duplicate

Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

Example:

```text
Input: [1,2,3,1]
Output: true
```

Solution: sort或者hashset

```cpp
bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> hash;
    for (int i: nums) {
        if (hash.find(i) != hash.end()) return true;
        hash.insert(i);
    }
    return false;
}
```

### 219. Contains Duplicate II

Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums\[i\] = nums\[j\] and the absolute difference between i and j is at most k.

Example:

```text
Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

Solution: hashmap

```cpp
bool containsNearbyDuplicate(vector<int>& nums, int k) {
    unordered_map<int, int> hash;
    for (int i = 0; i < nums.size(); ++i) {
        auto find = hash.find(nums[i]);
        if (find != hash.end() && i - find->second <= k) return true;
        hash[nums[i]] = i;
    }
    return false;
}
```

### 220. Contains Duplicate III

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums\[i\] and nums\[j\] is at most t and the absolute difference between i and j is at most k.

Example:

```text
Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
```

Solution: 移动窗口或者set+lowerbound

```cpp
// 移动窗口，t=0可以变成hashmap
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    if (!t) {
        unordered_set<int> hash;
        for (int i: nums) {
            if (hash.find(i) != hash.end()) return true;
            hash.insert(i);
        }
        return false;
    }
    if (nums.size() < 2) return false;
    for (int i = 0; i < nums.size(); ++i)
        for (int j = i + 1; j < min(i + k + 1, (int)nums.size()); ++j)
            if (i != j)
                if (abs((long)nums[i] - nums[j]) <= t) return true;
    return false;
}

// set + lowerbound
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    set<long long> window;
    for (int i = 0; i < nums.size(); ++i) {
        if (i > k && i-k-1 < nums.size()) window.erase(nums[i-k-1]);
        auto it = window.lower_bound((long long)nums[i] - t);
        if (it != window.cend() && *it - nums[i] <= t) return true;
        window.insert(nums[i]);
    }
    return false;
}
```

### 221. Maximal Square

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example:

```text
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
```

Solution: dp, 以当前节点'1'为右下角的的边长为: dp\[i\]\[j\] = min\(dp\[i-1\]\[j\], dp\[i\]\[j-1\], dp\[i-1\]\[j-1\]\) + 1，可以优化为1D

```cpp
int maximalSquare(vector<vector<char>>& matrix) {
    if (matrix.empty()) return 0;
    int m = matrix.size(), n = matrix[0].size();
    vector<int> dp(m + 1, 0);
    int maxsize = 0, pre = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 1; i <= m; i++) {
            int temp = dp[i];
            if (matrix[i - 1][j] == '1') {
                dp[i] = min(dp[i], min(dp[i - 1], pre)) + 1;
                maxsize = max(maxsize, dp[i]);
            }
            else dp[i] = 0; 
            pre = temp;
        }
    }
    return maxsize * maxsize;
}
```

### 222. Count Complete Tree Nodes

Given a complete binary tree, count the number of nodes. In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2^h nodes inclusive at the last level h.

Example:

```text
Input: 
    1
   / \
  2   3
 / \  /
4  5 6
Output: 6
```

Solution: 正常高度计算，一定要背

```cpp
// method 1, faster
int count_height(TreeNode* root) {
    return root? 1 + count_height(root->left): 0;
}

int countNodes(TreeNode* root) {
    if (!root) return 0;
    int h = count_height(root), c = 0;
    for (int i = h - 1; i > 0; --i) {
        if (count_height(root->right) == i) {
            c += int(pow(2, i - 1));
            root = root->right;
        }
        else{
            root = root->left;
        }
    }
    if (root) c++;
    return int(pow(2, h - 1)) - 1 + c;
}

// method 2, slower
int maxdepth = 0,  maxnum = 0;
int countNodes(TreeNode* root) {
    if (!root) return 0;
    int l = getLeft(root) + 1;  
    int r = getRight(root) + 1; 
    if (l==r) return (2 << (l-1)) - 1;  
    return countNodes(root->left) + countNodes(root->right) + 1;  
}

int getLeft(TreeNode* root) {
    int count = 0;  
    while (root->left) {  
        root = root->left;  
        ++count;  
    }  
    return count;  
}

int getRight(TreeNode* root) {
    int count = 0;  
    while (root->right) {  
        root = root->right;  
        ++count;  
    }  
    return count;  
}
```

### 223. Rectangle Area

Find the total area covered by two rectilinear rectangles in a 2D plane. Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.

Example:

```text
Input: A = -3, B = 0, C = 3, D = 4, E = 0, F = -1, G = 9, H = 2
Output: 45
```

Solution: 计算是否overlap，背

```cpp
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
    long right = (C < G)? C: G, left = (E > A)? E: A, top = (H < D)? H:D, bottom = (B > F)? B:F;
    long overlap_width = right - left, overlap_height = top - bottom, overlap_area = 0;
    if (overlap_width > 0 && overlap_height > 0) {
        overlap_area = overlap_width * overlap_height;
    }
    return (C - A) * (D - B) + (G - E) * (H - F) - overlap_area;
}
```

### 225. Implement Stack using Queues

Implement the following operations of a stack using queues: push\(x\), pop\(\), top\(\), empty\(\)

Solution: 可以用两个来回倒，也可以只用一个，通过q.push\(q.front\(\)\)来移动

```cpp
class MyStack {
private:
    queue<int>q;
public:
    MyStack() {}

    void push(int x) {
        int a = q.size();
        q.push(x);
        while (a--) {
            q.push(q.front());
            q.pop();
        }
    }

    int pop() {
        int x = q.front();
        q.pop();
        return x;
    }

    int top() {
        return q.front();
    }

    bool empty() {
        return q.empty();
    }
};
```

### 226. Invert Binary Tree

Invert a binary tree.

Example:

```text
Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

Solution: 递归

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (!root) return NULL;
    TreeNode* left = invertTree(root->left), *right = invertTree(root->right);
    root->right = left, root->left = right;
    return root;
}
```

### 228. Summary Ranges

Given a sorted integer array without duplicates, return the summary of its ranges.

Example:

```text
Input:Input:  [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: 2,3,4 form a continuous range; 8,9 form a continuous range.
```

Solution: 遍历一遍即可

```cpp
vector<string> summaryRanges(vector<int>& nums) {
    vector<string> res;
    if (nums.empty()) return res;
    int left = nums[0], right = 0;
    for (int i = 1; i < nums.size(); ++i) {
        if (nums[i] == left + right + 1) {
            ++right;
        } else if (!right) {
            res.push_back(to_string(left));
            left = nums[i];
        } else {
            res.push_back(to_string(left) + "->" + to_string(left + right));
            left = nums[i];
            right = 0;
        }
    }
    if (right) res.push_back(to_string(left) + "->" + to_string(left + right));
    else res.push_back(to_string(left));
    return res;
}
```

### 229. Majority Element II

Given an integer array of size n, find all elements that appear more than ⌊n/3⌋ times. The algorithm should run in linear time and in O\(1\) space.

Example:

```text
Input: [1,1,1,3,3,2,2,2]
Output: [1,2]
```

Solution: Boyer-Moore Majority Vote algorithm \(Q169的general case版本，有多大k就开多大的candidates\) 或者hashmap

```cpp
// Boyer-Moore Majority Vote algorithm
vector<int> majorityElement(vector<int>& nums) {
    vector<int> res;
    int candidate1 = INT_MAX, candidate2 = INT_MAX, count1 = 0, count2 = 0;
    for (int i: nums) {
        if (count1 <= 0 && candidate2 != i) candidate1 = i;
        if (count2 <= 0 && candidate1 != i) candidate2 = i;  
        count1 += (candidate1 == i)? 2: -1;
        count2 += (candidate2 == i)? 2: -1;
    }
    count1 = 0, count2 = 0;
    for (int i : nums) {
        if (candidate1 == i) count1++;
        if (candidate2 == i) count2++;
    }
    if (count1 > nums.size() / 3)  res.push_back(candidate1);
    if (candidate1 != candidate2 && count2 > nums.size() / 3)  res.push_back(candidate2);
    return res;
}

// hashmap
vector<int> majorityElement(vector<int>& nums) const {
    vector<int> result;
    int s = (int) nums.size();
    if (s) {
        int min = s / 3;
        unordered_map<int, int> c;
        for (auto n : nums) if (c[n]++ == min) result.push_back(n);
    }
    return result;
};
```

### 230. Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it. You may assume k is always valid

Example:

```text
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
```

Solution: inorder traversal，一定要背

```cpp
int kthSmallest(TreeNode* root, int k) {
    stack<TreeNode *> st;
    TreeNode *p = root;
    while (p || !st.empty()) {
        while (p) {
            st.push(p);
            p = p->left;
        }
        p = st.top();
        if (!--k) return p->val;
        st.pop();
        p = p->right;
    }
}
```

### 231. Power of Two

Given an integer, write a function to determine if it is a power of two.

Example:

```text
Input: 16
Output: true (2^4 = 16)
```

Solution: 如果是平方数，则n&\(n-1\)为0

```cpp
bool isPowerOfTwo(int n) {
    return n > 0 && !(n & (n-1));
}
```

### 232. Implement Queue using Stacks

Implement the following operations of a queue using stacks: push\(x\), pop\(\), peek\(\), empty\(\)

Solution: 类似Q225，一个stack就可以完成，不过空间需求和两个stack是相同的

```cpp
class MyQueue {
    stack<int> s;
public:
    MyQueue() {}

    void push(int x) {
        if (s.empty()) s.push(x);
        else {
            int data = s.top();
            s.pop();
            push(x);
            s.push(data);
        }
    }

    int pop() {
        int x = s.top();
        s.pop();
        return x;
    }

    int peek() {
        return s.top();
    }

    bool empty() {
        return s.empty();
    }
};
```

### 234. Palindrome Linked List

Given a singly linked list, determine if it is a palindrome. Do it in O\(n\) time and O\(1\) space.

Example:

```text
Input: 1->2->2->1
Output: true
```

Solution: reverse+快慢指针，一定要背

```cpp
bool isPalindrome(ListNode* head) {
    if (!head || !head->next) return true;
    ListNode *slow = head, *fast = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    slow->next = reverseList(slow->next);
    slow = slow->next;
    while (slow) {
        if (head->val != slow->val) return false;
        head = head->next;
        slow = slow->next;
    }
    return true;
}
ListNode* reverseList(ListNode* head) {
    ListNode *pre = NULL, *next = NULL;
    while (head) {
        next = head->next;
        head->next = pre;
        pre = head;
        head = next;
    }
    return pre;
}
```

### 235. Lowest Common Ancestor of a Binary Search Tree

Given a binary search tree \(BST\), find the lowest common ancestor \(LCA\) of two given nodes in the BST. The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants \(where we allow a node to be a descendant of itself\).

Example:

```text
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
        _______6______
       /              \
    ___2__          ___8__
   /      \        /      \
   0      _4       7       9
         /  \
         3   5
Output: 6
```

Solution: 如果pq比root小, 则LCA必定在左子树; 如果pq比root大, 则LCA必定在右子树; 如果一大一小, 则root即为LCA

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root|| !p|| !q) return NULL;
    if (max(p->val, q->val) < root->val) return lowestCommonAncestor(root->left, p, q);  
    if (min(p->val, q->val) > root->val) return lowestCommonAncestor(root->right, p, q);  
    return root;  
}
```

### 236. Lowest Common Ancestor of a Binary Tree

Given a binary tree, find the lowest common ancestor \(LCA\) of two given nodes in the tree. The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants \(where we allow a node to be a descendant of itself\).

Example:

```text
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
Output: 3
```

Solution: 与Q235不同，这里需要return p/q是否等于root，当两者return第一次递归回遇到的时候，那个点就是LCA

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root) return NULL;  
    if (root == p || root == q) return root;  
    TreeNode* l1 = lowestCommonAncestor(root->left, p, q);  
    TreeNode* l2 = lowestCommonAncestor(root->right, p, q);  
    if (l1 && l2) return root;  
    return !l1 ? l2 : l1;  
}
```

### 237. Delete Node in a Linked List

Write a function to delete a node \(except the tail\) in a singly linked list, given only access to that node.

Example:

```text
Input: head = [4,5,1,9], node = 1
Output: [4,5,9]
```

Solution: 因为不能删除本身的node，要么复制下个node的值倒这个node上，要么修改地址

```cpp
// method 1
void deleteNode(ListNode* node) {
    node->val = node->next->val;
    node->next = node->next->next;
}

// method 2
void deleteNode(ListNode *node) {
    *node = *node->next;
}
```

### 238. Product of Array Except Self

Given an array nums of n integers where n &gt; 1, return an array output such that output\[i\] is equal to the product of all the elements of nums except nums\[i\]. Please solve it without division, in O\(n\) time, and in O\(1\) space \(the output array does not count as extra space for the purpose of space complexity analysis\). Example:

```text
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

Solution: 定义一个fromBegin和fromLast，对于每个i，乘上之前和之后的即可，一定要背

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size(), fromBegin = 1, fromLast = 1;
    vector<int> res(n,1);
    for (int i = 0; i < n; ++i) {
        res[i] *= fromBegin;
        fromBegin *= nums[i];
        res[n-1-i] *= fromLast;
        fromLast *= nums[n-1-i];
    }
    return res;
}
```

### 240. Search a 2D Matrix II

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties: \(1\) Integers in each row are sorted in ascending from left to right. \(2\) Integers in each column are sorted in ascending from top to bottom.

Example:

```text
Input: target = 5, matrix =
[[1,   4,  7, 11, 15],
 [2,   5,  8, 12, 19],
 [3,   6,  9, 16, 22],
 [10, 13, 14, 17, 24],
 [18, 21, 23, 26, 30]]
Output: true
```

Solution: 从右上角开始，一定要背

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    if (!m) return false;
    int n = matrix[0].size();
    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
        if (matrix[i][j] == target) return true;
        else if (matrix[i][j] > target) --j;
        else ++i;
    }
    return false;
}
```

### 241. Different Ways to Add Parentheses

Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and x.

Example:

```text
Input: "2*3-4*5"
Output: [-34, -14, -10, -10, 10], for
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```

Solution: dp，dp\[i\]\[j\]表示从第i到第j的算子的结合结果集合。对每个算子k，可以结合dp\[i\]\[k\]和dp\[k+1\]\[j\]; dp\[i\]\[i\]为第i个数字。也可以用divide and conqure。一定要背

```cpp
vector<int> diffWaysToCompute(string input) {
    vector<int> data;
    vector<char> ops;
    int num = 0;
    char op = ' ';
    istringstream ss(input + "+");
    while (ss >> num && ss >> op) {
        data.push_back(num);
        ops.push_back(op);
    }
    const int size_i = data.size();
    vector<vector<vector<int>>> dp(size_i, vector<vector<int>>(size_i, vector<int>()));
    for (int i = 0; i < size_i; ++i) {
        for (int j = i; j >= 0; --j) {
            if (i == j) dp[j][i].push_back(data[i]);
            else for (int k = j; k < i; k += 1) {
                for (auto left : dp[j][k]) {
                    for (auto right : dp[k+1][i]) {
                        int val = 0;
                        switch (ops[k]) {
                           case '+': val = left + right; break;
                           case '-': val = left - right; break;
                           case '*': val = left * right; break;
                       }
                       dp[j][i].push_back(val);
                    }
                }
            }
        }
     }
   return dp[0][size_i-1];
}
```

### 242. Valid Anagram

Given two strings s and t , write a function to determine if t is an anagram of s.

Example:

```text
Input: s = "anagram", t = "nagaram"
Output: true
```

Solution: 开一个26大小的char数组即可，对s的字符++，对t的字符--，最后判断是否数组全为零

```cpp
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;
    int counts[26] = {0};
    for (int i = 0; i < s.length(); ++i) { 
        ++counts[s[i]-'a'];
        --counts[t[i]-'a'];
    }
    for (int i = 0; i < 26; ++i) if (counts[i]) return false;
    return true;
}
```

