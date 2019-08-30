# LeetCode 401 - 450

### 402. Remove K Digits

Given a non-negative integer *num* represented as a string, remove *k* digits from the number so that the new number is the smallest possible.

**Note:**

- The length of *num* is less than 10002 and will be ≥ *k*.
- The given *num* does not contain any leading zero.

Example:

```
Input: num = "1432219", k = 3
Output: "1219" (Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest)

Input: num = "10200", k = 1
Output: "200" (Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes)
```

Solution: stack，特别考虑0，一定要背（很难想到是stack，原理是应该优先移除前面的大数）

```cpp
string removeKdigits(string num, int k) {
       string ans = "";
       for (const char & c : num) {
           while (!ans.empty() && ans.back() > c && k) {
               ans.pop_back();
               --k;
           }
           if (!ans.empty() || c != '0') ans.push_back(c);
       }
       while (!ans.empty() && k--) ans.pop_back();
       return ans.empty() ?"0" :ans;
}
```

### 403. Frog Jump

A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.

If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

Example:

```
[0,1,3,5,6,8,12,17]


There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.


Return true. The frog can jump to the last stone by jumping 
1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.
```

Solution: dp，拿位置做dp，拿步长做遍历，巧妙运用hashing

```c++
bool canCross(vector<int>& stones) {
    unordered_map<int, unordered_set<int>> dp;
    dp[0].insert(0);
    for (int pos : stones) {
        for (int step : dp[pos]) {
            if (step-1 > 0) dp[pos + step-1].insert(step-1);
            dp[pos + step].insert(step);
            dp[pos + step+1].insert(step+1);
        }
    }
    return !dp[stones.back()].empty();
}
```

### 404. Sum of Left Leaves

Find the sum of all left leaves in a given binary tree.

Example:

```
Input:
    3
   / \
  9  20
    /  \
   15   7
Output: 24 (9+15)
```

Solution: 递归，需要传一个flag

```c++
int sumOfLeftLeaves(TreeNode* root) {
    return helper(root, false);
}
int helper(TreeNode* root, bool isleft) {
    if (!root) return 0;
    if (!root->left && !root->right && isleft) return root->val;
    return helper(root->left, true) + helper(root->right, false);
}
```

### 406. Queue Reconstruction by Height

Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue. The number of people is less than 1,100.

Example:

```
Input: [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
Output: [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

Solution: 首先按照高度由高到低、相同高度下次序小优先进行排序，然后依次插入次序位，一定要背和理解

```c++
vector<pair<int, int>> reconstructQueue(vector<pair<int, int>>& people) {
    sort(people.begin(), people.end(), [](const pair<int, int>& p1, const pair<int, int>& p2) {return p1.first > p2.first || (p1.first == p2.first && p1.second < p2.second);});
    vector<pair<int, int>> res;
    for (auto& p: people) res.insert(res.begin() + p.second, p);
    return res;
}
```

### 409. Longest Palindrome

Given a string which consists of lowercase or uppercase letters, find the length of the longest (case sensitive) palindromes that can be built with those letters.

Example:

```
Input: "abccccdd"
Output: 7 ("dccaccd")
```

Solution: 统计，偶数次*2 + 一次可以单独在中间

```c++
int longestPalindrome(string s) {
    vector<bool> used(52, false);
    int len = 0, pos;
    for (auto c: s) {
        if (c - 'A' >= 0 && c - 'Z' <= 0) pos = c - 'A' + 26;
        else pos = c - 'a';
        if (used[pos]) len += 2;
        used[pos] = !used[pos];
    }
    for (auto u: used) if (u) return len + 1;
    return len;
}
```

### 412. Fizz Buzz

Write a program that outputs the string representation of numbers from 1 to n. But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”. For numbers which are multiples of both three and five output “FizzBuzz”.

Example:

```
Input: n = 15,
Output: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
```

Solution: 正常遍历

```c++
vector<string> fizzBuzz(int n) {
    vector<string> res;
    for (int i = 1; i <= n; ++i) {
        if (!(i % 3)) {
            if (!(i % 5)) res.push_back("FizzBuzz");
            else res.push_back("Fizz");
        } else if (!(i % 5)) res.push_back("Buzz");
        else res.push_back(to_string(i));
    }
    return res;
}
```

### 413. Arithmetic Slices

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same. A zero-indexed array A consisting of N numbers is given. A slice of that array is any pair of integers (P, Q) such that 0 <= P < Q < N. A slice (P, Q) of array A is called arithmetic if the sequence: A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular, this means that P + 1 < Q. The function should return the number of arithmetic slices in the array A.

Example:

```
Input: A = [1, 2, 3, 4]
Output: 3 ([1, 2, 3], [2, 3, 4], [1, 2, 3, 4])
```

Solution: dp + 累计计算，dp[i]表示以i位截止的等差数列数量，累计即可得出最终排列种类，一定要背

```c++
int numberOfArithmeticSlices(vector<int>& nums) {
    int n = nums.size();
    if (n < 3) return 0;
    vector<int> dp(n, 0);
    for (int i = 2; i < n; ++i) if (nums[i] - nums[i-1] == nums[i-1] - nums[i-2]) dp[i] = dp[i-1] + 1;
    return accumulate(dp.begin(), dp.end(), 0);
}
```

### 415. Add Strings

Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2. The length of both num1 and num2 is < 5100.

Solution: 正常操作，不过注意细节处理和进位

```c++
string addStrings(string num1, string num2) {
    string output("");
    reverse(num1.begin(), num1.end());
    reverse(num2.begin(), num2.end());
    int onelen = num1.length();
    int twolen = num2.length();
    if (onelen <= twolen){
        swap(num1, num2);
        swap(onelen, twolen);
    }
    int addbit = 0;
    for (int i = 0; i < twolen; ++i){
        int cur_sum = (num1[i]-'0') + (num2[i]-'0') + addbit;
        output += to_string((cur_sum)%10);
        if (cur_sum >= 10) addbit = 1;
        else addbit = 0;
    }
    for (int i = twolen; i < onelen; ++i){
        int cur_sum = (num1[i]-'0') + addbit;
        output += to_string((cur_sum)%10);
        if (cur_sum >= 10) addbit = 1;
        else addbit = 0;
    }
    if (addbit) output += "1";
    reverse(output.begin(), output.end());
    return output;
}
```

### 416. Partition Equal Subset Sum

Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal. Each of the array element will not exceed 100. The array size will not exceed 200

Example:

```
Input: [1, 5, 11, 5]
Output: true ([1, 5, 5], [11])
```

Solution: 两种方法: (1) dp knapsack, dp[i][j](https:i可以被空间压缩)表示前i个数字的和是否能达到j; (2) DFS + memoization，写法和combination sum差不多，只是不需要backtrack。一定要背

```c++
// method 1: dp knapsack
bool canPartition(vector<int> &nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2) return false;
    int target = sum / 2;
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int i = 0; i < nums.size(); ++i) {
        // 注意j要反向，因为原本是dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
        for (int j = target; j >= nums[i]; --j) {
            dp[j] = dp[j] || dp[j-nums[i]];
        }
    }
    return dp.back();
}


// method 2: DFS + memoization
bool canPartition(vector<int> &nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2) return false;
    int target = sum / 2;
    sort(nums.begin(), nums.end());
    vector<int> dp(target + 1, -1);
    return dfs(nums, target, 0, dp);
}


bool dfs(vector<int> &nums, int target, int start, vector<int> &dp){
    if (!target) return true;
    if (dp[target] != -1) return dp[target];
    for (int i = start; i < nums.size() && nums[i] <= target; ++i){
        if (dfs(nums, target - nums[i], i + 1, dp)) return dp[target] = true;
    }
    return dp[target] = false;
}
```

### 418. Sentence Screen Fitting

Given a `rows x cols` screen and a sentence represented by a list of **non-empty** words, find **how many times** the given sentence can be fitted on the screen.

Note:

1. A word cannot be split into two lines.
2. The order of words in the sentence must remain unchanged.
3. Two consecutive words **in a line** must be separated by a single space.
4. Total words in the sentence won't exceed 100.
5. Length of each word is greater than 0 and won't exceed 10.
6. 1 ≤ rows, cols ≤ 20,000.

Example:

```
Input: rows = 3, cols = 6, sentence = ["a", "bcd", "e"]
Output:  2 (
    a-bcd- 
    e-a---
    bcd-e-
    The character '-' signifies an empty space on the screen
)
```

Solution: 正常处理

```c++
int wordsTyping(vector<string>& sentence, int rows, int cols) {
    int n = sentence.size();
    vector<int> counts(n, 0);
    
    for (int i = 0; i < n; ++i) {
        int length = 0, words = 0, index = i;
        while (length + sentence[index % n].length() <= cols) {
            length += sentence[index % n].length();
            ++length;
            ++index;
            ++words;
        }
        counts[i] = words;
    }
    
    int words = 0;
    for (int i = 0, index = 0; i < rows; ++i) {
        words += counts[index];
        index = (counts[index] + index) % n;
    }
    
    return words / n;
}
```

### 426. Convert Binary Search Tree to Sorted Doubly Linked List

Convert a BST to a sorted circular doubly-linked list in-place. Think of the left and right pointers as synonymous to the previous and next pointers in a doubly-linked list.

Solution: 用prev记录左边节点，注意这里的定义的左右是链表的左右。一个关键点是，没有左子树的第一个inorder节点应为新链表的最左端

```c++
void inorder(Node* n, Node*& prev, Node*& head) {
    // First node without left children is head
    if (!head && !n->left) head = prev = n; 
    Node *left = n->left, *right = n->right;
    // left
    if (left) inorder(left, prev, head);
    // process
    prev->right = n;
    n->left = prev;
    prev = n;
    // right
    if (right) inorder(right, prev, head);
}
Node* treeToDoublyList(Node* root) {
    if (!root) return NULL;
    Node *prev = NULL, *head = NULL;
    inorder(root, prev, head);
    prev->right = head;
    head->left = prev;
    return head;
}
```

### 432. All O(1) Data Structure

Implement a data structure supporting the following operations:

1. Inc(Key) - Inserts a new key with value 1. Or increments an existing key by 1. Key is guaranteed to be a **non-empty** string.
2. Dec(Key) - If Key's value is 1, remove it from the data structure. Otherwise decrements an existing key by 1. If the key does not exist, this function does nothing. Key is guaranteed to be a **non-empty** string.
3. GetMaxKey() - Returns one of the keys with maximal value. If no element exists, return an empty string `""`.
4. GetMinKey() - Returns one of the keys with minimal value. If no element exists, return an empty string `""`.

Challenge: Perform all these in O(1) time complexity.

Solution: LRU + bucket，一定要背

```cpp
class AllOne {
public:
    void inc(string key) {
        // If the key doesn't exist, insert it with value 0.
        if (!bucketOfKey.count(key)) bucketOfKey[key] = buckets.insert(buckets.begin(), {0, {key}});
        // Insert the key in next bucket and update the lookup.
        auto next = bucketOfKey[key], bucket = next++;
        if (next == buckets.end() || next->value > bucket->value + 1) next = buckets.insert(next, {bucket->value + 1, {}});
        next->keys.insert(key);
        bucketOfKey[key] = next;
        // Remove the key from its old bucket.
        bucket->keys.erase(key);
        if (bucket->keys.empty()) buckets.erase(bucket);
    }

    void dec(string key) {
        // If the key doesn't exist, just leave.
        if (!bucketOfKey.count(key)) return;
        // Maybe insert the key in previous bucket and update the lookup.
        auto prev = bucketOfKey[key], bucket = prev--;
        bucketOfKey.erase(key);
        if (bucket->value > 1) {
            if (bucket == buckets.begin() || prev->value < bucket->value - 1) prev = buckets.insert(bucket, {bucket->value - 1, {}});
            prev->keys.insert(key);
            bucketOfKey[key] = prev;
        }
        // Remove the key from its old bucket.
        bucket->keys.erase(key);
        if (bucket->keys.empty()) buckets.erase(bucket);
    }

    string getMaxKey() {
        return buckets.empty() ? "" : *(buckets.rbegin()->keys.begin());
    }
    
    string getMinKey() {
        return buckets.empty() ? "" : *(buckets.begin()->keys.begin());
    }

private:
    struct Bucket { int value; unordered_set<string> keys; };
    list<Bucket> buckets;
    unordered_map<string, list<Bucket>::iterator> bucketOfKey;
};
```

### 433. Minimum Genetic Mutation

A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T". Suppose we need to investigate about a mutation (mutation from "start" to "end"), where ONE mutation is defined as ONE single character changed in the gene string. For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation. Also, there is a given gene "bank", which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string. Now, given 3 things - start, end, bank, your task is to determine what is the minimum number of mutations needed to mutate from "start" to "end". If there is no such a mutation, return -1. You may assume start and end string is not the same.

Example:

```
Input: start = "AACCGGTT", end = "AAACGGTA", bank = ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
Output: 2
```

Solution: BFS, 可以用trick设立两个set，每次只扩展小的那个，一定要背

```c++
int minMutation(string start, string end, vector<string>& bank) {
    unordered_set<string> dict(bank.begin(), bank.end());
    if (!dict.count(end)) return -1;
    unordered_set<string> bset, eset, *set1, *set2;
    bset.insert(start), eset.insert(end);
    int step = 0, n = start.size();
    while (!bset.empty() and !eset.empty()) {
        if (bset.size() <= eset.size()) set1 = &bset, set2 = &eset;
        else set2 = &bset, set1 = &eset;
        unordered_set<string> tmp;
        ++step;
        for (auto itr = set1->begin(); itr != set1->end(); ++itr) {
            for (int i = 0; i < n; ++i) {
                string dna = *itr;
                for (auto g : string("ATGC")) {
                    dna[i] = g;
                    if (set2->count(dna)) return step;
                    if (dict.count(dna)) {
                        tmp.insert(dna);
                        dict.erase(dna);
                    }
                }
            }
        }
        *set1 = tmp;
    }
    return -1;
}
```

### 435. Non-overlapping Intervals

Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

‌Example:

```
Input: [[1,2], [2,3], [3,4], [1,3]]
Output: 1 (remove [1,3])
```

Solution: 遍历，注意要按照end排序来获得最多的interval

```c++
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    int n = intervals.size();
    sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b) {
        return a[1] < b[1];
    });
    int total = 0, prev = intervals[0][1];
    for (int i = 1; i < n; ++i) {
        if (intervals[i][0] < prev) ++total;
        else prev = intervals[i][1];
    }
    return total;
}
```

### 436. Find Right Interval‌

Given a set of intervals, for each of the interval i, check if there exists an interval j whose start point is bigger than or equal to the end point of the interval i, which can be called that j is on the "right" of i.

‌For any interval i, you need to store the minimum interval j's index, which means that the interval j has the minimum start point to build the "right" relationship for interval i. If the interval j doesn't exist, store -1 for the interval i. Finally, you need output the stored value of each interval as an array.

Example:

```
Input: [[3,4], [2,3], [1,2]]
Output: [-1, 0, 1] 
```

Solution: 建一个map<start, index>即可，通过lower_bound找位置，十分巧妙

```c++
vector<int> findRightInterval(vector<Interval>& intervals) {
    map<int, int> m; //<start, idx>
    for (int i = 0; i < intervals.size(); i++) {
        m[intervals[i].start] = i;
    }
    vector<int> res(intervals.size(), -1);
    for (int i = 0; i < intervals.size(); i++) {
        auto it = m.lower_bound(intervals[i].end);
        if (it != m.end()) res[i] = (it->second);
    }
    return res;
}
```

### 437. Path Sum III

You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value. The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes). The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

```
Imput: root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
Output: 3 (5 -> 3, 5 -> 2 -> 1, -3 -> 11)
```

Solution: DFS，需要分情况是否开始

```c++
int pathSum(TreeNode* root, int sum) {
    return root? pathSumStartWithRoot(root, sum) + pathSum(root->left, sum) + pathSum(root->right, sum): 0;
}

int pathSumStartWithRoot(TreeNode* root, int sum) {
    if (!root) return 0;
    int ret = root->val == sum? 1: 0;
    ret += pathSumStartWithRoot(root->left, sum - root->val);
    ret += pathSumStartWithRoot(root->right, sum - root->val);
    return ret;
}
```

### 438. Find All Anagrams in a String

Given a string s and a non-empty string p, find all the start indices of p's anagrams in s. Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100. The order of output does not matter.

Example:

```
Input: s: "cbaebabacd" p: "abc"
Output: [0, 6] ("cba", "bac")
```

Solution: 滑动窗口，一定要背

```c++
vector<int> findAnagrams(string s, string p) {
    vector<int> pv(26,0), sv(26,0), res;
    if (s.size() < p.size()) return res;
    for (int i = 0; i < p.size(); ++i) {
        ++pv[p[i]-'a'];
        ++sv[s[i]-'a'];
    }
    if (pv == sv) res.push_back(0);
    for (int i = p.size(); i < s.size(); ++i) {
        ++sv[s[i]-'a'];
        --sv[s[i-p.size()]-'a']; 
        if (pv == sv) res.push_back(i-p.size()+1);
    }
    return res;
}
```

### 441. Arranging Coins

You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins. Given n, find the total number of **full** staircase rows that can be formed (n is a non-negative integer and fits within the range of a 32-bit signed integer)‌

Example:

```
Imput: n = 5
Output: 2 (
    ¤
    ¤ ¤
    ¤ ¤
)
```

Solution: 二分法，类似找平方根

```c++
int arrangeCoins(int n) {
    if (!n) return 0;
    int l = 0, r = n, mid;
    while (l < r) {
        mid = l + (r - l) / 2;
        if (countTotal(mid) > n) r = mid;
        else l = mid + 1;
    }
    return max(1, l - 1);
}


long countTotal(long n) {
    return n * (n + 1) / 2;
}
```

### 442. Find All Duplicates in an Array

Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once. Find all the elements that appear twice in this array. Could you do it without extra space and in O(n) runtime?

Example:

```
Input: [4,3,2,7,8,2,3,1]
Output: [2,3]
```

Solution: bucket sort变形，不需要移动位置，只需要记录对应数字位是否被遍历过，遍历过则设为负数标记即可，一定要背

```c++
vector<int> findDuplicates(vector<int>& nums) {
    vector<int> res;
    for(int i = 0; i < nums.size(); ++i){
        if (nums[abs(nums[i])-1] > 0) nums[abs(nums[i])-1] = -nums[abs(nums[i])-1];
        else res.push_back(abs(nums[i]));
    }
    return res;
}
```

### 447. Number of Boomerangs

Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points `(i, j, k)` such that the distance between `i` and `j`equals the distance between `i` and `k` (**the order of the tuple matters**). Find the number of boomerangs.

Example:

```
Input: [[0,0],[1,0],[2,0]]
Output: 2 ([[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]])
```

Solution: 按距离做hashmap，然后计算排列可能，一定要好好理解

```cpp
int numberOfBoomerangs(vector<pair<int, int>>& points) {
    int res = 0;
    for (int i = 0; i < points.size(); ++i) {
        unordered_map<long, int> group(points.size());
        for (int j = 0; j < points.size(); ++j) {
            if (j == i) continue;
            int dy = points[i].second - points[j].second, dx = points[i].first - points[j].first;
            ++group[dy*dy+dx*dx];
        }
        // for all the groups of points, number of ways to select 2 from n is nP2
        for (auto& p : group) if (p.second > 1) res += p.second * (p.second - 1);
    }
    return res;
}
```

### 448. Find All Numbers Disappeared in an Array‌

Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once. Find all the elements of [1, n] inclusive that do not appear in this array. Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

Example:

```
Input: [4,3,2,7,8,2,3,1]
Output: [5,6]
```

Solution: 类似Q442，负数bucket sort插入

```c++
vector<int> findDisappearedNumbers(vector<int>& nums) {
    vector<int> res;
    for (auto i: nums) if (nums[abs(i)-1] > 0) nums[abs(i)-1] = -nums[abs(i)-1];
    for (int i = 0; i < nums.size(); ++i) if (nums[i] > 0) res.push_back(i + 1);
    return res;
}
```

### 450. Delete Node in a BST‌

Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST. Time complexity should be O(height of tree).

Solution: 递归+分情况讨论

```c++
TreeNode* deleteNode(TreeNode* root, int val) {
    if (!root) return root;
    if (val < root->val) root->left = deleteNode(root->left, val);
    else if (val > root->val) root->right = deleteNode(root->right, val);
    else {
        if (!root->left && !root->right) {
            delete(root);
            return NULL;
        }
        /* 1 child case */
        if (!root->left || !root->right) {
            TreeNode *ret = root->left ? root->left : root->right;
            delete(root);
            return ret;
        }
        /* 2 child case */
        if (root->left && root->right) {
            TreeNode *tmp = root->right;
            while (tmp->left) tmp = tmp->left;
            root->val = tmp->val;
            root->right = deleteNode(root->right, root->val);
        }
    }
    return root;
}
```