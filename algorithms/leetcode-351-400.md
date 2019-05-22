# LeetCode 351 - 400

### 355. Design Twitter

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods: \(1\) postTweet\(userId, tweetId\): Compose a new tweet. \(2\) getNewsFeed\(userId\): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. \(3\) follow\(followerId, followeeId\): Follower follows a followee. \(4\) unfollow\(followerId, followeeId\): Follower unfollows a followee.

Example:

```text
Twitter twitter = new Twitter();

// User 1 posts a new tweet (id = 5).
twitter.postTweet(1, 5);

// User 1's news feed should return a list with 1 tweet id -> [5].
twitter.getNewsFeed(1);

// User 1 follows user 2.
twitter.follow(1, 2);

// User 2 posts a new tweet (id = 6).
twitter.postTweet(2, 6);

// User 1's news feed should return a list with 2 tweet ids -> [6, 5].
// Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.getNewsFeed(1);

// User 1 unfollows user 2.
twitter.unfollow(1, 2);

// User 1's news feed should return a list with 1 tweet id -> [5],
// since user 1 is no longer following user 2.
twitter.getNewsFeed(1);
```

Solution: 对每个user做一个set存followees、做一个vector存tweets，在getNewsFeed时用一个priority queue/multiset存自己followees的每个人的最新的tweet，pop queue的时候再插入对应用户的次最新tweet，以节约时间和内存开销，一定要背

```cpp
class Twitter {
public:
    /** Initialize your data structure here. */
    Twitter() {}

    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        if (users.find(userId) == users.end()) users[userId].id = userId;
        users[userId].ts.push_back(Tweet(tweetId, timestamp++));
    }

    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the use followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        multiset<TweetwithID> pq;
        User& u = users[userId];
        // only insert the most recent tweet of a user
        if (!u.ts.empty()) pq.insert(TweetwithID(u.ts.back(), userId, u.ts.size()-1));
        for (int f : u.fs) {
            User& uu = users[f];
            if (!uu.ts.empty()) pq.insert(TweetwithID(uu.ts.back(), f, uu.ts.size()-1));
        }
        vector<int> res;
        for (int i = 0; i < 10 && !pq.empty(); ++i) {
            auto it = pq.rbegin();
            int pos = it->pos, userId = it->userId;
            res.push_back(it->tweet.id);
            pq.erase(--pq.end());
            if (pos >= 1) pq.insert(TweetwithID(users[userId].ts[pos-1], userId, pos-1));
        }
        return res;
    }

    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        if (followerId != followeeId) users[followerId].fs.insert(followeeId);
    }

    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        users[followerId].fs.erase(followeeId);
    }

private:
    struct Tweet {
        Tweet(int i=-1, int t=-1) : id(i), time(t) {}
        int id;
        int time;
    };

    struct User {
        User(int i=-1) : id(i) {}
        int id;
        unordered_set<int> fs;
        vector<Tweet> ts;
    };

    struct TweetwithID {
        TweetwithID(Tweet t, int u=-1, int p=-1): tweet(t), userId(u), pos(p) {}
        Tweet tweet;
        int userId;
        int pos;
        bool operator < (const TweetwithID& t) const { return tweet.time < t.tweet.time; }
    };

    unordered_map<int, User> users;
    int timestamp = 0;
};
```

### 357. Count Numbers with Unique Digits

Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x &lt; 10^n.

Example:

```text
Input: n = 2
Output: 91 (100 - [11,22,33,44,55,66,77,88,99])
```

Solution: 数学，n位数的unique digits个数为，\(n-1\)位数的unique digits个数，即开始数字为0的情况，加上9x9x8x7...的情况，即开始数字为1-9。也可以用dp加速，不过影响不大

```cpp
// method 1: recurssion
int fac_from_9(int n) {
    int res = 9, base = 9;
    while(--n) res *= base--;
    return res;
}
int countNumbersWithUniqueDigits(int n) {
    return n ? fac_from_9(n) + countNumbersWithUniqueDigits(n-1): 1;
}
```

### 365. Water and Jug Problem

You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need to determine whether it is possible to measure exactly z litres using these two jugs. If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end. Operations allowed are: \(1\) Fill any of the jugs completely with water; \(2\) Empty any of the jugs; \(3\) Pour water from one jug into another till the other jug is completely full or the first jug itself is empty.

Example:

```text
Input: x = 3, y = 5, z = 4
Output: True
```

Solution: gcd

```cpp
bool canMeasureWater(int x, int y, int z) {
    return x + y >= z && (x == z || y == z || x + y == z || z % gcd(x, y) == 0);
}
int gcd(int a, int b) {
    int r;
    while (b) {
        r = b;
        b = a % b;
        a = r;
    }
    return a;
}
```

扩展: gcd和lcm算法，一定要背

```cpp
int gcd(int x, int y) {
    int r;
    while (b) {
        r = b;
        b = a % b;
        a = r;
    }
    return a;
}

int lcm(int x, int y) {
    int gcd = gcd(a, b);
    return gcd? (a * b / gcd) : 0;
}
```

### 367. Valid Perfect Square

Given a positive integer num, write a function which returns True if num is a perfect square else False. Do not use any built-in library function such as sqrt.

Example:

```text
Input: 16
Output: True
```

Solution: 牛顿法

```cpp
bool isPerfectSquare(int x) {
    long r = x;
    while (r * r > x) r = (r + x / r) / 2;
    return r * r == x;
}
```

### 368. Largest Divisible Subset

Given a set of distinct positive integers, find the largest subset such that every pair \(Si, Sj\) of elements in this subset satisfies: Si % Sj = 0 or Sj % Si = 0. If there are multiple solutions, return any subset is fine.

Example:

```text
Input: [1,2,3,4,8]
Output: [1,2,4,8]
```

Solution: dp，dp\[i\]表示以i为结尾的最大subset大小，dp\[i\] = max{1 + dp\[j\] if a\[i\] % a\[j\] == 0 else 1}；同时用一个parent来跟踪，注意对j的loop要从大到小，否则parent会设置成parent\[i\] = \[i\]，导致跟踪失败。一定要背

```cpp
vector<int> largestDivisibleSubset(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<int> dp(nums.size(), 0), parent(nums.size(), 0);
    int len = 0, pos = 0;

    for (int i = 0; i < nums.size(); ++i) {
        for (int j = i; j >= 0; --j) {
            if (nums[i] % nums[j] == 0 && dp[i] < 1 + dp[j]) {
                dp[i] = 1 + dp[j];
                parent[i] = j;
                if (dp[i] > len) {
                    len = dp[i];
                    pos = i;
                }
            }
        }
    }

    vector<int> ret;
    for(int i = 0; i < len; ++i) {
        ret.push_back(nums[pos]);
        pos = parent[pos];
    }
    reverse(ret.begin(), ret.end());
    return ret;
}
```

### 371. Sum of Two Integers

Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

Solution: bit manipulation

```cpp
int getSum(int a, int b) {
    return b? getSum(a ^ b, (a & b) << 1): a;
}
```

### 372. Super Pow

Your task is to calculate ab mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.

Example:

```text
a = 2
b = [1,0]

Result: 1024
```

Solution: ab % k = \(a%k\)\(b%k\) % k，一定要背

```cpp
int superPow(int a, int k)  {  
    if (!k) return 1;  
    int ans = a;  
    for (int i = 1; i < k; ++i) ans = (ans*a) % 1337;  
    return ans;  
}

int superPow(int a, vector<int>& b) {  
    if (b.empty()) return 1;  
    a = a % 1337;  
    int lastBit = b.back();  
    b.pop_back();  
    return (superPow(superPow(a, b), 10) * superPow(a, lastBit)) % 1337;  
}
```

### 373. Find K Pairs with Smallest Sums

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k. Define a pair \(u,v\) which consists of one element from the first array and one element from the second array. Find the k pairs \(u1,v1\),\(u2,v2\) ...\(uk,vk\) with the smallest sums.

Example:

```text
Input: nums1 = [1,7,11], nums2 = [2,4,6],  k = 3
Output: [1,2],[1,4],[1,6] (first 3 of [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6])
```

Solution: priority queue会导致多余的插入，正确方法应该是对于nums1的每一个数，记录对于nums2的使用位数，这样复杂度是O\(km\)，一定要背

```cpp
vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
    vector<pair<int, int>> res;
    int m = nums1.size(), n = nums2.size(), index;
    long local_sum;
    k = min(k, m * n);
    vector<int> indice(m, 0);
    while(k--){
        index = 0;
        local_sum = LONG_MAX;
        for (int i = 0; i < m; i++) {
            if (indice[i] < n && local_sum >= nums1[i] + nums2[indice[i]]){
                index = i;
                local_sum = nums1[i] + nums2[indice[i]];
            }
        }
        res.push_back(make_pair(nums1[index], nums2[indice[index]]));
        indice[index]++;
    }
    return res;
}
```

### 374. Guess Number Higher or Lower

I pick a number from 1 to n. You have to guess which number I picked. You call a pre-defined API guess\(int num\) which returns 3 possible results: -1, my num is lower; 1, my num is higher; or 0, you got the answer.

Solution: 二分，注意如果用左闭右开的话，需要换成long，否则INT\_MAX超界

```cpp
int guessNumber(int n) {
    long left = 1, right = long(n) + 1, mid, res;
    while (left < right) {
        mid = (left + right) / 2;
        res = guess(mid);
        if (!res) return mid;
        if (res == 1) left = mid + 1;
        else right = mid;
    }
    return left;
}
```

### 375. Guess Number Higher or Lower II

I pick a number from 1 to n. You have to guess which number I picked. Each time you guess a particular number x and wrong, you pay $x. Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.

Example:

```text
Input: n = 10.
Output: 21 ($5 + $7 + $9)
```

Solution: dp，dp\[j\]\[i\] = min{k + max\(dp\[j\]\[k-1\], dp\[k+1\]\[i\]\)}，一定要背和注意loop的顺序

```cpp
int getMoneyAmount(int n) {
    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
    for (int i = 1; i <= n; ++i){
        for (int j = i - 1; j >= 1; --j){
            int min_value = INT_MAX;
            for (int k = j; k <= i; ++k){
                int tmp = k + max(dp[j][k - 1], dp[k + 1][i]);
                min_value = min(min_value, tmp);
            }
            dp[j][i] = min_value;
        }
    }
    return dp[1][n];        
}
```

### 376. Wiggle Subsequence

A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference \(if one exists\) may be either positive or negative. A sequence with fewer than two elements is trivially a wiggle sequence. Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence. A subsequence is obtained by deleting some number of elements \(eventually, also zero\) from the original sequence, leaving the remaining elements in their original order.

Example:

```text
Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7 (One is [1,17,10,13,10,16,8], which wiggle sequence is [16,-7,3,-3,6,-8])
```

Solution: 记录两个值p和q，分别表示递增和递减位的交替最大次数（而非递增和递减单独的个数），然后最后输出二者的max即可，一定要背

```cpp
int wiggleMaxLength(vector<int>& nums) {
    int p = 1, q = 1, n = nums.size();
    for (int i = 1; i < n; ++i) {
        if (nums[i] > nums[i-1]) p = q + 1;
        else if (nums[i] < nums[i-1]) q = p + 1;
    }
    return min(n, max(p, q));  // add min here to consider nums.empty() 
}
```

### 377. Combination Sum IV

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target. Note that different sequences are counted as different combinations.

Example:

```text
Input: nums = [1, 2, 3], target = 4
Output: 7 (The possible combination ways are: (1, 1, 1, 1) (1, 1, 2) (1, 2, 1) (1, 3) (2, 1, 1) (2, 2) (3, 1))
```

Solution: dp，dp\[i\]表示target=i时的permutation个数 \(虽然题目叫combination，实际上是permutation\), dp\[i\] = sum\(dp\[i-n\] for n in nums\)。这是因为假设多一位新的只放在排列最前面，那么对每个因子做dp然后求和即可，一定要背

```cpp
int combinationSum4(vector<int>& nums, int target) {  
    if (!nums.size()) return 0;  
    vector<int> dp(target + 1, 0);  
    dp[0] = 1;  
    for (int i = 1; i <= target; ++i) for (auto val: nums) if(val <= i) dp[i] += dp[i-val];
    return dp[target];  
}
```

### 378. Kth Smallest Element in a Sorted Matrix

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Example:

```text
Input: k = 8, matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
]
Output: 13
```

Solution: 二分，对mid关于每一行做upper\_bound然后求和，就可得到准确位置来做二分处理，一定要背

```cpp
int kthSmallest(vector<vector<int>>& matrix, int k) {
    int n = matrix.size(), l = matrix[0][0], r = matrix[n-1][n-1];
    int mid, num;
    while (l < r) {
        mid = l + (r - l) / 2, num = 0;
        for (int i = 0; i < n; ++i) {
            int pos = upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
            num += pos;
        }
        if (num < k) l = mid + 1;
        else r  = mid;
    }
    return l;
}
```

### 380. Insert Delete GetRandom O\(1\)

Design a data structure that supports all following operations in average O\(1\) time: \(1\) insert\(val\): Inserts an item val to the set if not already present; \(2\) remove\(val\): Removes an item val from the set if present \(3\) getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.

Example:

```cpp
// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();
// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);
// Returns false as 2 does not exist in the set.
randomSet.remove(2);
// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);
// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();
// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);
// 2 was already in the set, so return false.
randomSet.insert(2);
// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();
```

Solution: hashmap + vector; map的key是值，val是在vector的位置; vector push back存值。注意remove的时候需要交换hashmap里key和back的pos，然后在vector里交换它们再pop back。一定要背

```cpp
class RandomizedSet {
public:
    RandomizedSet() {}

    bool insert(int val) {
        if (hash.count(val)) return false;
        hash[val] = vec.size();
        vec.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!hash.count(val)) return false;
        int pos = hash[val];
        hash[vec.back()] = pos;
        hash.erase(val);
        swap(vec[pos], vec[vec.size()-1]);
        vec.pop_back();
        return true;
    }

    int getRandom() {
        return vec[rand()%vec.size()];
    }

private:
    unordered_map<int, int> hash;
    vector<int> vec;
};
```

### 381. Insert Delete GetRandom O\(1\) - Duplicates allowed

Design a data structure that supports all following operations in average O\(1\) time. Note: Duplicate elements are allowed.

Example:

```cpp
// Init an empty collection.
RandomizedCollection collection = new RandomizedCollection();
// Inserts 1 to the collection. Returns true as the collection did not contain 1.
collection.insert(1);
// Inserts another 1 to the collection. Returns false as the collection contained 1. Collection now contains [1,1].
collection.insert(1);
// Inserts 2 to the collection, returns true. Collection now contains [1,1,2].
collection.insert(2);
// getRandom should return 1 with the probability 2/3, and returns 2 with the probability 1/3.
collection.getRandom();
// Removes 1 from the collection, returns true. Collection now contains [1,2].
collection.remove(1);
// getRandom should return 1 and 2 both equally likely.
collection.getRandom();
```

Solution: 类似Q380，只是hashmap的val变成了一个hashset

```cpp
class RandomizedCollection {
public:
    RandomizedCollection() {}

    bool insert(int val) {
        hash[val].insert(vec.size());
        vec.push_back(val);
        return hash[val].size() == 1;
    }

    bool remove(int val) {
        if (hash[val].empty() || !hash.count(val)) return false;
        if (vec.back() == val) {
            hash[val].erase(vec.size()-1);
            vec.pop_back();
        } else {
            int index = *hash[val].begin();
            hash[val].erase(index);
            swap(vec[index],vec[vec.size()-1]);
            vec.pop_back();
            hash[vec[index]].erase(vec.size());
            hash[vec[index]].insert(index);

        }
        return true;
    }

    int getRandom() {
        return vec[rand()%vec.size()];
    }
private:
    unordered_map<int, unordered_set<unsigned long> > hash;
    vector<int> vec;
};
```

### 382. Linked List Random Node

Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.

Example:

```cpp
// Init a singly linked list [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);
// getRandom() should return either 1, 2, or 3 randomly. Each element should have equal probability of returning.
solution.getRandom();
```

Solution: [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)，可以在不count长度的情况下使用，一定要背

```cpp
class Solution {
    ListNode* head;
public:
    Solution(ListNode* n): head(n) {}

    int getRandom() {
        int res = head->val;
        ListNode* node = head->next;
        int i = 2;
        while (node) {
            int j = rand()%i;
            if (!j) res = node->val;
            ++i;
            node = node->next;
        }
        return res;
    }
};
```

### 383. Ransom Note

Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines; otherwise, it will return false. Each letter in the magazine string can only be used once in your ransom note. You may assume that both strings contain only lowercase letters.

Example:

```text
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
```

Solution: 做个26长度的array, ++--计算即可

```cpp
bool canConstruct(string ransomNote, string magazine) {
    int dict[26] = {};
    for (auto c: ransomNote) ++dict[c-'a'];
    for (auto c: magazine) if (dict[c-'a']) --dict[c-'a'];
    for (auto i: dict) if (i) return false;
    return true;
}
```

### 384. Shuffle an Array

Shuffle a set of numbers without duplicates.

Example:

```cpp
// Init an array with set 1, 2, and 3.
int[] nums = {1,2,3};
Solution solution = new Solution(nums);
// Shuffle the array [1,2,3] and return its result. Any permutation of [1,2,3] must equally likely to be returned.
solution.shuffle();
// Resets the array back to its original configuration [1,2,3].
solution.reset();
// Returns the random shuffling of array [1,2,3].
solution.shuffle();
```

Solution: 从前往后swap一遍即可

```cpp
class Solution {
    vector<int> origin;
public:
    Solution(vector<int> nums) {
        origin = std::move(nums);
    }

    vector<int> reset() {
        return origin;
    }

    vector<int> shuffle() {
        if (origin.empty()) return {};  
        vector<int> shuffled(origin);
        int len = origin.size();
        for (int i = 0; i < len; ++i) {  
            int pos = rand() % (len - i);  
            swap(shuffled[i], shuffled[i+pos]);  
        }  
        return shuffled;  
    }
};
```

### 386. Lexicographical Numbers

Given an integer n, return 1 - n in lexicographical order. Please optimize your algorithm to use less time and space. The input size may be as large as 5,000,000.

Example:

```text
Input: 13
Output: [1,10,11,12,13,2,3,4,5,6,7,8,9]
```

Solution: 正常遍历，注意规则

```cpp
vector<int> lexicalOrder(int n) {
    vector<int> res(n);
    int cur = 1;
    for (int i = 0; i < n; ++i) {
        res[i] = cur;
        if (cur * 10 <= n) cur *= 10;
        else {
            if (cur >= n) cur /= 10;
            ++cur;
            while (!(cur % 10)) cur /= 10;
        }
    }
    return res;
}
```

### 387. First Unique Character in a String

Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1. You may assume the string contain only lowercase letters.

Example:

```text
Input: "leetcode"
Output: 0
```

Solution: 拿一个vector做hash即可

```cpp
int firstUniqChar(string s) {
    vector<int> pos(26, -2);
    for (int i = 0; i < s.length(); ++i) {
        int c = s[i]-'a';
        if (pos[c] == -2) pos[c] = i;
        else pos[c] = -1;
    }
    int res = s.length();
    for (auto i: pos) if (i >= 0 && i < res) res = i;
    return res == s.length()? -1: res;
}
```

### 388. Longest Absolute File Path

Given a string representing the file system in the above format, return the length of the longest absolute path to file in the abstracted file system. If there is no file in the system, return 0. Notice that a/aa/aaa/file1.txt is not the longest file path, if there is another path aaaaaaaaaaaaaaaaaaaaa/sth.png.

Example:

```text
Input: "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext", which means
Output: 32 ("dir/subdir2/subsubdir2/file2.ext"), the tree is
dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
```

Solution: 正常遍历，拿一个stack&gt;记录即可，一定要背

```cpp
int lengthLongestPath(string input) {
    input += '\n';
    int len = input.size(), ans = 0, level_cnt = 0, cur_path_len = 0, cur_len = 0, is_file = 0;  
    stack<pair<int, int> > st;  // (len，level)
    for (int i = 0; i < len; ++i) {  
        if (input[i] != '\n') {  
            if (input[i] != '\t') ++cur_len;
            if (input[i] == '\t') ++level_cnt;  
            if (input[i] == '.') is_file = 1;  
            continue;  
        }    
        while (!st.empty() && level_cnt <= st.top().second) {  
            cur_path_len -= st.top().first;  
            st.pop();  
        }  
        if (is_file) ans = max(cur_path_len + cur_len + 1, ans);  
        else {  
            st.push(make_pair(cur_len + 1, level_cnt));  
            cur_path_len += (cur_len + 1);  
        }  
        is_file = level_cnt = cur_len = 0;  
    }  
    return ans? ans - 1: 0;  
}
```

### 389. Find the Difference

Given two strings s and t which consist of only lowercase letters. String t is generated by random shuffling string s and then add one more letter at a random position. Find the letter that was added in t.

Example:

```text
Input: s = "abcd", t = "abcde"
Output: 'e'
```

Solution: 拿一个vector做hash, ++--即可

```cpp
char findTheDifference(string s, string t) {
    vector<int> hash(26, 0);
    for (auto c: t) ++hash[c-'a'];
    for (auto c: s) --hash[c-'a'];
    for (int i = 0; i < 26; ++i) if (hash[i]) return i+'a';
}
```

### 390. Elimination Game

There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list. Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers. We keep repeating the steps again, alternating left to right and right to left, until a single number remains. Find the last number that remains starting with a list of length n.

Example:

```text
Input: n = 9
Output: 6 (1 2 3 4 5 6 7 8 9 -> 2 4 6 8 -> 2 6 -> 6)
```

Solution: 递归，每次除以2然后镜像查找，一定要理解

```cpp
int lastRemaining(int n) {
    return n == 1 ? 1 : 2 * (1 + n / 2 - lastRemaining(n / 2));
}
```

### 392. Is Subsequence

Given a string s and a string t, check if s is subsequence of t. You may assume that there is only lower case English letters in both s and t. t is potentially a very long \(length around 500,000\) string, and s is a short string \(length around 100\).

Example:

```text
Input: s = "abc", t = "ahbgdc"
Output: Ture
```

Output: 从左到右遍历一遍即可，这道题很容易想复杂

```cpp
bool isSubsequence(string s, string t) {
    int j = 0, n = s.length(), m = t.length();
    for (auto c: t) if (j < n && s[j] == c) j++;            
    return j == n;
}
```

### 394. Decode String

Given an encoded string, return it's decoded string. The encoding rule is: k\[encoded\_string\], where the encoded\_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer. You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Example

```text
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
```

Solution: stack，分类讨论下一个字符是\[、\]、数字、还是字母，一定要背

```cpp
string decodeString(string s) {
    if (!s.length()) return s;
    int times = 0;
    vector<pair<int, string>> stack(1, {1, ""});
    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == ']') {
            string back = repeat(stack.back().first, stack.back().second);
            stack.pop_back();
            stack.back().second += back;
        }
        else if (!isDigit(s[i])) stack.back().second += s[i];
        else {
            do times = times * 10 + (s[i] - '0'); while (s[++i] != '[');
            stack.push_back({times, ""});
            times = 0;
        }
    }
    return stack.back().second;
}

bool isDigit (char c) {
    int pos = c - '0';
    return pos >= 0 && pos < 10;
}

string repeat(int times, const string& input) {
    ostringstream os;
    fill_n(ostream_iterator<string>(os), times, input);
    return os.str();
}
```

### 395. Longest Substring with At Least K Repeating Characters

Find the length of the longest substring T of a given string \(consists of lowercase letters only\) such that every character in T appears no less than k times.

Example:

```text
Input: s = "ababbc", k = 2
Output: 5 ("ababb")
```

Solution: 递归，一定要好好理解

```cpp
int longestSubstring(string s, int k) {
    int count[26] = {};
    for (auto c: s) ++count[c-'a'];
    return longestSubstring_recur(s, k, count, 0, s.size());
}

int longestSubstring_recur(const string& s, int k, int *count, int first, int last) {
    int max_len = 0;
    for (int i = first; i < last;) {
        while (i < last && count[s[i]-'a'] < k) ++i;
        if (i == last) break;
        int j = i, tmp[26] = {};
        while (j < last && count[s[j]-'a'] >= k) ++tmp[s[j++]-'a'];
        if (i == first && j == last) return last - first; 
        max_len = max(max_len, longestSubstring_recur(s, k, tmp, i, j));
        i = j;
    }
    return max_len;
}
```

### 396. Rotate Function

Given an array of integers A and let n to be its length. Assume Bk to be an array obtained by rotating the array A k positions clock-wise, we define a "rotation function" F on A as follow: F\(k\) = 0  _Bk\[0\] + 1_  Bk\[1\] + ... + \(n-1\) \* Bk\[n-1\]. Calculate the maximum value of F\(0\), F\(1\), ..., F\(n-1\).

Example:

```text
Input: A = [4, 3, 2, 6]
Output: 26, since
F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25,
F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16,
F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23,
F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26.
```

Solution: 遍历一遍求和以及加权和，再遍历一遍做数值移动就可以，这种题大概率都是O\(n\)的

```cpp
int maxRotateFunction(vector<int>& nums) {
    int n = nums.size();
    if (!n) return 0;
    long long wsum = 0, sum = 0;
    for (int i = 0; i < n; ++i) {
        wsum += nums[i] * i;
        sum += nums[i];
    }
    long long result = wsum;
    for (int i = 0; i < n; ++i) {
        wsum += nums[i] * n - sum;
        result = max(wsum, result);
    }
    return result;
}
```

### 397. Integer Replacement

Given a positive integer n and you can do operations as follow: \(1\) If n is even, replace n with n/2; \(2\) If n is odd, you can replace n with either n + 1 or n - 1. What is the minimum number of replacements needed for n to become 1?

Example:

```text
Input: 8
Output: 3 (8 -> 4 -> 2 -> 1)
```

Solution: 简单递归即可，可以用memoization，不要想复杂

```cpp
unordered_map<int, int> visited;
int integerReplacement(int n) {        
    if (n == 1) return 0;
    if (!visited.count(n)) {
        if (n & 1 == 1) {
            visited[n] = 2 + min(integerReplacement(n / 2), integerReplacement(n / 2 + 1));
        } else {
            visited[n] = 1 + integerReplacement(n / 2);
        }
    }
    return visited[n];
}
```

### 398. Random Pick Index

Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Example:

```cpp
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);
// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);
// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
```

Solution: reservoir sampling \(快\) 或hashmap \(函数调用次数少的时候慢\), 一定要背

```cpp
// method 1: reservoir sampling
vector<int> data;
Solution(vector<int> nums): data(nums) {}

int pick(int target) {
    int cnt = 0, ans = -1;
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] == target && !(rand() % ++cnt)) ans = i;
    }
    return ans;
}

// method 2: hashmap
unordered_map<int, vector<int>> hash;

Solution(vector<int> nums) {
    for (int i = 0; i < nums.size(); ++i) {
        if (hash.count(nums[i])) hash[nums[i]].push_back(i);
        else hash[nums[i]] = vector<int>(1, i);
    }
}

int pick(int target) {
    vector<int> vec = hash[target];
    return vec[rand()%vec.size()];
}
```

### 400. Nth Digit

Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...

Example:

```text
Input: 11
Output: 0 (the second digit of 10)
```

Solution: 先找位数，再找数，最后确定位置

```cpp
int findNthDigit(int n) {
    // step 1. calculate how many digits the number has.
    long base = 9, digits = 1;
    while (n - base * digits > 0) {
        n -= base * digits;
        base *= 10;
        ++digits;
    }

    // step 2. calculate what the number is.
    int index = n % digits;
    if (!index) index = digits;
    long num = 1;
    for (int i = 1; i < digits; ++i) num *= 10;
    num += (index == digits) ? n / digits - 1 : n / digits;

    // step 3. find out which digit in the number is we wanted.
    for (int i = index; i < digits; ++i) num /= 10;
    return num % 10;
}
```

