# LeetCode 601 - 650

### 617. Merge Two Binary Trees

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example:

```text
Input:
    Tree 1           Tree 2
          1            2
         / \          / \
        3   2        1   3
       /              \   \
      5                4   7
Output:
         3
        / \
       4   5
      / \   \
     5   4   7
```

Solution: 正常递归即可，可以当练手

```cpp
TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
    if (!t1 && !t2) return t1;
    if (!t1) return t2;
    if (!t2) return t1;
    t1->val += t2->val;
    t1->left = mergeTrees(t1->left, t2->left);
    t1->right = mergeTrees(t1->right, t2->right);
    return t1;
}
```

### 621. Task Scheduler

Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle. However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle. You need to return the least number of intervals the CPU will take to finish all the given tasks.

Example:

```text
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8 (A -> B -> idle -> A -> B -> idle -> A -> B)
```

Solution: 先遍历一遍存入hashmap并记录最高频字符，结果为max\(数据集大小，\(最高频字符的出现次数-1\) \* \(间隔长度+1\) + \(与最高频字符出现次数相同的字符数\)\)。原理：先把这个最高频字符放到开头，每个间隔单元填充其它字符，然后最后一个间隔单元填充与最高频字符出现次数相同的字符。注意需要max一下数据集大小，因为有的时候分不完数字。一定要背和理解

```cpp
int leastInterval(vector<char>& tasks, int n) {
    unordered_map<char, int> hash;
    int count = 0;
    for (auto t: tasks) {
        ++hash[t];
        count = max(count, hash[t]);
    }
    int ans = (count - 1) * (n + 1);
    for (auto kv: hash) if (kv.second == count) ++ans;
    return max(ans, (int)tasks.size());
}
```

### 638. Shopping Offers

In LeetCode Store, there are some kinds of items to sell. Each item has a price. However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item. The job is to output the lowest price you have to pay for **exactly** certain items as given, where you could make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to pay for this special offer, other numbers represents how many specific items you could get if you buy this offer. You could use any of special offers as many times as you want.

Example:

```
Input: [2,3,4], [[1,1,0,4],[2,2,1,9]], [1,2,1]
Output: 11
(The price of A is $2, and $3 for B, $4 for C. 
You may pay $4 for 1A and 1B, and $9 for 2A ,2B and 1C. 
You need to buy 1A ,2B and 1C, so you may pay $4 for 1A and 1B (special offer #1), and $3 for 1B, $4 for 1C. )
```

Solution: 递归+用hashmap做memoization（也可dp）

```cpp
unordered_map<string, int> map;

int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs) {
    string key;
    for (int i = 0; i < needs.size(); ++i) {
        key += to_string(needs[i]) + "#";
    }

    auto it = map.find(key);
    if (it != map.end()) return it->second;

    int m = 0;
    for (int i = 0; i < needs.size(); ++i) {
        m += price[i]*needs[i];
    }

    for (int i = 0; i < special.size(); ++i) {
        int s = special[i][needs.size()];
        if (m > s) {
            bool found = true;
            for (int j = 0; j < needs.size(); ++j) {
                needs[j] = needs[j] - special[i][j];
                if (needs[j] < 0) found = false;
            }

            if (found) {
                int n = shoppingOffers(price, special, needs) + s;
                m = min(m, n);
            }

            for (int j = 0; j < needs.size(); ++j) {
                needs[j] = needs[j] + special[i][j];
            }
        }
    }

    map[key] = m;
    return m;
}
```



### 647. Palindromic Substrings

Given a string, your task is to count how many palindromic substrings in this string. The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example:

```text
Input: "aaa"
Output: 6 ("a", "a", "a", "aa", "aa", "aaa")
```

Solution: 两种方法: \(1\) [manacher algorithm](https://leetcode.com/problems/palindromic-substrings/solution/), O\(n\) \(2\) 两个loop\(dp，半径扩展等方法\)。一定要背

```cpp
// method 1: manacher algorithm
string preprocess(string  s) {
    if (s.empty()) return "^$";
    int n = s.length();
    string str = "^";
    for (int i = 0; i < n; ++i) str += "#" + s.substr(i, 1);
    str += "#$";
    return str;
}

vector<int> manacher(const string& s) {
    string str = preprocess(s);
    int n = str.length(), c = 0, r = 0;
    vector<int> p(n);
    for(int i = 1; i < n - 1; ++i) {
        int i_mirror = 2 * c - i;
        p[i] = (r > i) ? min(r - i, p[i_mirror]) : 0;
        while (str[i+1+p[i]] == str[i-1- p[i]]) p[i]++;
        if (i + p[i] > r) {
            r = i + p[i];
            c = i;
        }
    }
    return p;
}

int countSubstrings(string s) {
    auto result = 0;
    for (const auto & max_len : manacher(s)) result += (max_len + 1) / 2;
    return result;
}

// method 2: expand from center
int countSubstrings(string s) {
    int num = s.size();
    for (float center = 0.5; center < s.size(); center += 0.5) {
        int left = int(center - 0.5), right = int(center + 1);
        while (left >= 0 && right < s.size() && s[left--] == s[right++])  ++num;
    }
    return num;
}

// method 3: expand from pos
int countSubstrings(string s) {
    int res = 0, n = s.length();
    for (int i = 0; i < n; i++) {
        for (int j = 0;   i-j >= 0 && i+j < n &&   s[i-j] == s[i+j]; j++) ++res; //substring s[i-j, ..., i+j]
        for (int j = 0; i-1-j >= 0 && i+j < n && s[i-1-j] == s[i+j]; j++) ++res; //substring s[i-1-j, ..., i+j]
    }
    return res;
}
```

