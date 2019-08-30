# LeetCode 1 - 50

### 1. Two Sum

Given an array of integers, return indices of the two numbers such that they add up to a specific target. You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

```text
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

Solution: \(hash\)map存每个值，然后map.find\(target-map\[i\]\)找是否不等于end

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> um;
    for (int i = 0; i < nums.size(); ++i) {
        int num = nums[i];
        auto pos = um.find(target - num);
        if (pos != um.end()) {
            return vector<int>{pos->second, i};
        } else {
            um.insert({num, i});
        }
    }
}
```

### 2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

```text
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

Solution: 用carry记录进位，需要处理最后的carry

```cpp
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* head = new ListNode(0);
    ListNode* cur = head;
    int carry = 0;
    while (l1 || l2) {
        if (l1) {
            carry += l1->val;
            l1 = l1 -> next;
        }
        if (l2) {
            carry += l2->val;
            l2 = l2 -> next;
        }
        cur -> next = new ListNode(carry%10);
        carry /= 10;
        cur = cur -> next;
    }
    // do not forget the remaining carry
    if (carry > 0) {
        cur->next = new ListNode(carry);
    }
    return head->next;
}
```

### 3. Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating characters.

Example:

```text
Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3.
```

Solution: 用一个额外数组\(对应每个char\)，记录该char最后一次出现的位置；遍历字符串，每次query对应char的位置，若重复则更新位置；用一个额外的int记录上一次不出现重复的位置，返回最大的历史差

```cpp
int lengthOfLongestSubstring(string s) {
    int n = s.length(), ans = 0;
    // current index of character
    int index[128] = {0};
    // try to extend the range [i, j]
    for (int j = 0, i = 0; j < n; ++j) {
        i = max(index[s[j]], i);
        ans = max(ans, j - i + 1);
        index[s[j]] = j + 1;
    }
    return ans;
}
```

### 4. Median of Two Sorted Arrays

There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O\(log \(m+n\)\).

Example:

```text
nums1 = [1, 3]
nums2 = [2]
The median is 2.0

nums1 = [1, 2]
nums2 = [3, 4]
The median is (2 + 3)/2 = 2.5
```

Solution:

O\(m+n\): merge sort简单变形

```cpp
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int median = 0, median_prev = 0;
    int newsize = nums1.size() + nums2.size(), p = 0, q = 0;
    for (int i = 0; i <= newsize/2; ++i) {
        median_prev = median;
        if (q >= nums2.size() ||(p < nums1.size() && nums1[p] <= nums2[q]))
            median = nums1[p++];
        else median = nums2[q++];
    }
    if (newsize%2) return median;
    else return (median + median_prev)/2.0;
}
```

O\(log\(m+n\)\): 二分。设总长度为m+n。则array1的i位数字，和array2的j = \(m+n\)/2 - i位数字，必须满足比当前array次大的那一个数字更接近对方。若满足，则这两位就是中位候选。因为array已经排好序，所以可以用二分确定i的位置。

```cpp
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size();
    // to ensure m <= n
    if (m > n) {
        nums1.swap(nums2);
        swap(m, n);
    }
    int iMin = 0, iMax = m, halfLen = (m + n + 1) / 2;
    while (iMin <= iMax) {
        int i = (iMin + iMax) / 2, j = halfLen - i;
        if (i < iMax && nums2[j-1] > nums1[i]) ++iMin;  // i is too small
        else if (i > iMin && nums1[i-1] > nums2[j]) --iMax;  // i is too big
        else {
            // i is perfect
            int maxLeft = 0;
            if (i == 0) maxLeft = nums2[j-1];
            else if (j == 0) maxLeft = nums1[i-1];
            else maxLeft = max(nums1[i-1], nums2[j-1]);
            if ((m + n) % 2 == 1) return maxLeft;

            int minRight = 0;
            if (i == m) minRight = nums2[j];
            else if (j == n) minRight = nums1[i];
            else minRight = min(nums2[j], nums1[i]);

            return (maxLeft + minRight) / 2.0;
        }
    }
    return 0.0;
}
```

### 5. Longest Palindromic Substring

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example:

```text
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Input: "cbbd"
Output: "bb"
```

Solution: O\(n^2\): 从后往前dp，核心为dp\[i\]\[j\] = dp\[i+1\]\[j-1\] && s\[i\] == s\[j\]，可反向压缩成1维dp\[j\] = dp\[j-1\] && s\[i\] == s\[j\]

```cpp
string longestPalindrome(string s) {
    int startpos = 0, maxlen = 0;
    const int slen = s.length();
    bool dp[slen];
    for (int i = slen - 1; i >= 0; --i) {  // 因为需要i+1轮的信息，所以反着走
        dp[i] = true;
        for (int j = slen - 1; j > i; --j) {  // 因为需要i+1轮的j-1个信息，所以反着走防止覆盖
            if (j == i + 1) dp[j] = s[i] == s[j];
            else dp[j] = dp[j-1] && s[i] == s[j];
            if (dp[j] && j - i > maxlen) {
                maxlen = j - i;
                startpos = i;
            }
        }
    }
    return s.substr(startpos, maxlen + 1);
}
```

O\(n\): Manacher's Algorithm

```cpp
string longestPalindrome(string s) {
    int n = s.size(), len = 0, start = 0;
    for (int i = 0; i < n; ++i) {
        int left = i, right = i;
        while (right < n && s[right+1] == s[right]) right++;
        i = right;
        while (left > 0 && right < n-1 && s[left-1] == s[right+1]) {
            --left;
            ++right;
        }
        if (len < right-left+1) {
            len = right - left + 1;
            start = left;
        }
    }
    return s.substr(start, len);
}
```

### 6. ZigZag Conversion

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:

```text
P   A   H   N
A P L S I I G
Y   I   R
```

Write the code that will take a string and make this conversion given a number of rows: `convert("PAYPALISHIRING", 3)` should return "PAHNAPLSIIGYIR".

Solution: 分第一行，中间行，最后行，跳跃输出

```cpp
string convert(string s, int numRows) {
    if (numRows == 1) return s;
    string res;

    // print first row
    int interval = 2 * (numRows - 1);
    for (int i = 0; i < s.length(); i += interval)
        res += s[i];

    // print middle rows (conditionally)
    for (int row = 1; row < numRows - 1; ++row)
        for (int i = row, j = 0; i < s.length(); j += interval, i = j - i)
            res += s[i];

    // print last row (conditionally)
    if (numRows > 1)
        for (int i = numRows - 1; i < s.length(); i += interval)
        res += s[i];

    return res;
}
```

### 7. Reverse Integer

Given a 32-bit signed integer, reverse digits of an integer.

Example:

```text
Input: 123
Output: 321

Input: -123
Output: -321

Input: 120
Output: 21
```

Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: \[−2^31, 2^31 − 1\]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.

Solution: 用预运算判断是否overflow

```cpp
int reverse(int x) {
    int result = 0;
    while (x) {
        preproc = result * 10 + x % 10;
        if (result != preproc / 10) return 0;
        result = preproc;
        x /= 10;
    }
    return result;
}
```

### 8. String to Integer

Implement atoi to convert a string to an integer.

Solution: 需要注意 \(1\) discards all leading whitespaces \(2\) sign of the number \(3\) overflow \(4\) invalid input

```cpp
int myAtoi(string str) {
    int sign = 1, base = 0, i = 0;
    while (str[i] == ' ') i++;
    if (str[i] == '-' || str[i] == '+') {
        sign = 1 - 2 * (str[i++] == '-');
    }
    while (str[i] >= '0' && str[i] <= '9') {
        if (base >  INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0' > 7)) {
            if (sign == 1) return INT_MAX;
            else return INT_MIN;
        }
        base  = 10 * base + (str[i++] - '0');
    }
    return base * sign;
}
```

### 9. Palindrome Number

Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example:

```text
Input: 121
Output: true

Input: -121
Output: false

Input: 10
Output: false
```

Solution: 分成一半来比

```cpp
bool isPalindrome(int x) {
    if (x == 0) return true;
    if (x < 0 || x % 10 == 0) return false;
    int right = 0;
    while (x > right) {
        right = right * 10 + x % 10;
        x /= 10;
    }
    return x == right || x == right / 10;
}
```

### 10. Regular Expression Matching

Given an input string \(of lowercase letters a-z\) and a pattern, implement regular expression matching with support for '.' \(matches any single character\) and '\*' \(matches zero or more of the preceding element\).

Example:

```text
Input:
s = "aa"
p = "a"
Output: false

Input:
s = "aa"
p = "a*"
Output: true

Input:
s = "ab"
p = ".*"
Output: true

Input:
s = "aab"
p = "c*a*b"
Output: true

Input:
s = "mississippi"
p = "mis*is*p*."
Output: false
```

Solution: 用DP，dp\[i\]\[j\] 表示 s 和 p 是否 match。当 p\[j\] != '\*'时，b\[i + 1\]\[j + 1\] == b\[i\]\[j\] 且 s\[i\] == p\[j\]；当p\[j\] == '\*'时，b\[i\]\[j + 2\] = b\[i\]\[j\]或考虑下一位情况。

```cpp
vector<vector<int> > vec;
bool isMatch(string s, string p) {
    vec = vector<vector<int> >(s.length() + 1, vector<int>(p.length() + 1, -1));
    return dp(0, 0, s, p);
}
bool dp(int i, int j, string s, string p) {
    if (vec[i][j] != -1) return vec[i][j] == 1;
    bool ans;
    if (j == p.length()) {
        ans = i == s.length();
    } else {
        bool first_match = i < s.length() && (p[j] == s[i] || p[j] == '.');
        if (j + 1 < p.length() && p[j + 1] == '*')
            ans = dp(i, j + 2, s, p) || (first_match && dp(i + 1, j, s, p));
        else
            ans = first_match && dp(i + 1, j + 1, s, p);
    }
    vec[i][j] = ans ? 1 : 0;
    return ans;
}
```

### 11. Container With Most Water

Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate \(i, ai\). n vertical lines are drawn such that the two endpoints of line i is at \(i, ai\) and \(i, 0\). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

Solution: 双指针，从两端开始向中间靠拢，如果左端线段短于右端，那么左端右移，反之右端左移；直到左右两端移到中间重合，记录这个过程中每一次组成木桶的容积，返回其中最大的。

```cpp
int maxArea(vector<int>& height) {
    int left = 0, right = height.size() - 1, water = 0;
    while (left < right) {
        water = max(water, (right - left) * min(height[left], height[right]));
        if (height[left] > height[right]) right--;
        else left++;
    }
    return water;
}
```

### 13. Roman to Integer

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.

Example:

```
Input: "MCMXCIV"
Output: 1994 (M = 1000, CM = 900, XC = 90 and IV = 4)
```

Solution: 从后往前遍历

```cpp
int romanToInt(string s) {
    unordered_map<char, int> hash {{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}};
    if (s.empty()) return 0;
    int res = hash[s.back()], post = hash[s.back()], cur;
    for (int i = s.size() - 2; i >= 0; --i) {
        cur = hash[s[i]];
        if (post > cur) res -= cur;
        else res += cur;
        post = cur;
    }
    return res;
}
```

### 14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string "". All given inputs are in lowercase letters a-z.

Example:

```text
Input: ["flower","flow","flight"]
Output: "fl"

Input: ["dog","racecar","car"]
Output: ""
```

Solution: 逐位比较。注意如果对string的access index等于length，则返回'\0' \(a default value of charT in std::basic\_string instantiation, for char it is \0\)

```cpp
string longestCommonPrefix(vector<string>& strs) {
    if (strs.empty()) return "";
    for (int i = 0; i < strs[0].length(); ++i)
        for (int j = 1; j < strs.size(); ++j)
            if (strs[0][i] != strs[j][i])
                return strs[0].substr(0, i);
    return strs[0];
}
```

### 15. 3Sum

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. Note that the solution set must not contain duplicate triplets.

Example:

```text
Input = [-1, 0, 1, 2, -1, -4],
Output =[[-1, 0, 1],[-1, -1, 2]]
```

Solution: for loop固定一个，然后双指针扫右面剩余段，比值的负数小就++left，反之--right

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] > 0) break;
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int target = -1 * nums[i];
        int left = i + 1, right = nums.size() - 1;
        while (left < right) {
            if (nums[left] + nums[right] == target) {
                result.push_back({nums[i], nums[left++], nums[right--]});
                while (left < right && nums[left] == nums[left-1]) ++left;
                while (left < right && nums[right] == nums[right]+1) --right;
            } else if (nums[left] + nums[right] < target) {
                ++left;
            } else --right;
        }
    }
    return result;
}
```

### 16. 3Sum Closest

Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

Example:

```text
Given array nums = [-1, 2, 1, -4], and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
```

Solution: 同3Sum，改为记录最小abs差

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    int ans, left, right, total;
    sort(nums.begin(), nums.end());
    ans = nums[nums.size()-1] + nums[nums.size()-2] + nums[nums.size()-3];
    for (int i = 0; i < nums.size()-2; i++) {
        if (i != 0 && nums[i] == nums[i-1]) continue;
        left = i+1;
        right = nums.size() - 1;
        while (left < right) {
            total = nums[i] + nums[left] + nums[right];
            if (total > target) right--;
            else if (total < target) left++;
            else return target;
            if (abs(ans - target) > abs(total - target)) ans = total;
        }
    }
    return ans;
}
```

### 17. Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. A mapping of digit to letters \(just like on the telephone buttons\) is given below: 2-abc, 3-def, 4-ghi, 5-jkl, 6-mno, 7-pqrs, 8-tuv, 9-wxyz. Note that 1 does not map to any letters.

Example:

```text
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

Solution: 正常添加，也可backtrack不过可能栈溢出

```cpp
vector<string> letterCombinations(string digits) {
    vector<string> result;
    if (digits.empty()) return vector<string>();
    static const vector<string> v = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    result.push_back("");
    for (int i = 0 ; i < digits.size(); ++i) {
        int num = digits[i]-'0';
        if (num < 0 || num > 9) break;
        const string& candidate = v[num];
        if (candidate.empty()) continue;
        vector<string> tmp;
        for (int j = 0 ; j < candidate.size() ; ++j) {
            for (int k = 0 ; k < result.size() ; ++k) {
                tmp.push_back(result[k] + candidate[j]);
            }
        }
        result.swap(tmp);
    }
    return result;
}
```

### 19. Remove Nth Node From End of List

Given a linked list, remove the n-th node from the end of list and return its head. Given n will always be valid. Can you do it in one pass?

Example:

```text
Given linked list: 1->2->3->4->5, and n = 2.
Output: 1->2->3->5.
```

Solution: 快慢指针，始终相差n，最后移除慢指针的nextnext。注意需要在head前面多加一个dummy node，防止溢出

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* start = new ListNode(0); // avoid overflow when list = [1] and n = 1
    start->next = head;
    ListNode* fast = start;
    ListNode* slow = start;
    while (n-- >= 0) fast = fast->next;
    while (fast) {
        slow = slow->next;
        fast = fast->next;
    }
    slow->next = slow->next->next;
    return start->next;
}
```

### 20. Valid Parentheses

Given a string containing just the characters '\(', '\)', '{', '}', '\[' and '\]', determine if the input string is valid. An input string is valid if: \(1\) Open brackets must be closed by the same type of brackets. \(2\) Open brackets must be closed in the correct order. Note that an empty string is also considered valid.

Example:

```text
Input: "()"
Output: true

Input: "()[]{}"
Output: true

Input: "(]"
Output: false

Input: "([)]"
Output: false

Input: "{[]}"
Output: true
```

Solution: stack

```cpp
bool isValid(string s) {
    vector<char> v;
    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == '{' || s[i] == '[' || s[i] == '(') v.push_back(s[i]);
        else {
            if (v.empty()) return false;
            char c = v.back();
            if ((s[i] == '}' && c == '{') || (s[i] == ']' && c == '[') || (s[i] == ')' && c == '(')) v.pop_back();
            else return false;
        }
    }
    return v.empty();
}
```

### 21. Merge Two Sorted Lists

Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:

```text
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```

Solution: 可以recursive，也可以直接merge，直接merge时先做一个dummy node，然后往后加l1或者l2

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if (!l2) return l1;
    if (!l1) return l2;
    if (l1->val > l2->val) {
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    } else {
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }
}
```

or

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if (!l1) return l2;
    if (!l2) return l1;
    ListNode* head = new ListNode(-1);
    ListNode* tmp = head;

    while (l1 && l2) {
        if (l1->val > l2->val) {
            tmp->next = l2;
            l2 = l2->next;
        } else {
            tmp->next = l1;
            l1 = l1->next;
        } tmp = tmp->next;
    }
    tmp->next = l2? l2: l1;
    return head->next;
}
```

### 22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example:

```text
Input: n = 3
Output: [
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

Solution: 正常添加，也可backtrack不过可能栈溢出

```cpp
vector<string> generateParenthesis(int n) {
    vector<string> ret;
    add(ret, "", n, n);
    return ret;
}
void add(vector<string> & ret, string s, int left, int right) {
    if (!left && !right) {
        ret.push_back(s);
        return;
    }
    if (left > 0) add(ret, s + "(", left-1, right);
    if (right > left) add(ret, s + ")", left, right-1);
}
```

### 23. Merge k Sorted Lists

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

```text
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```

Solution: 用std::make\_heap或者priority\_queue会比较慢；最快的方法是divide and conquer，不断merge相邻两个list

```cpp
ListNode* mergeKLists(vector<ListNode*> &lists) {
    int size = lists.size();
    if (!size) return NULL;
    if (size == 1) return lists[0];

    int i = 2, j;
    while (i / 2 < size) {
        for (j = 0; j < size; j += i) {
            ListNode* p = lists[j];
            if (j + i / 2 < size) {
                p = mergeTwoLists(p, lists[j + i / 2]);
                lists[j] = p;
                // optional: lists[j + i / 2] = NULL;
            }
        }
        i *= 2;
    }
    return lists[0];
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if (!l2) return l1;
    if (!l1) return l2;
    if (l1->val > l2->val) {
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    } else {
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }
}
```

### 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head. Your algorithm should use only constant extra space. You may not modify the values in the list's nodes, only nodes itself may be changed.

Example:

```text
Input: 1->2->3->4
Output: 2->1->4->3
```

Solution: 直接交换，但是一定要想好！此题考验细致耐心

```cpp
ListNode* swapPairs(ListNode* head) {
    ListNode *p=head, *s;  
    if (p && p->next) {  
        s = p->next;  
        p->next = s->next;  
        s->next = p;  
        head = s;  
        while (p->next && p->next->next) {  
            s = p->next->next;  
            p->next->next = s->next;  
            s->next = p->next;  
            p->next = s;  
            p = s->next;  
        }  
    }  
    return head;
}
```

### 26. Remove Duplicates from Sorted Array

Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O\(1\) extra memory.

Example:

```text
Input: [0,0,1,1,1,2,2,3,3,4]
Output: 5
Note: nums will become [0,1,2,3,4] for the first 5 positions
```

Solution: 双指针更新修改

```cpp
int removeDuplicates(vector<int>& nums) {
    int size = nums.size();
    if (!size) return 0;
    int r = 0, l = 0;
    while (++r < size) {
        if (nums[r] > nums[l]) {
            nums[++l] = nums[r];
        }
    }
    return l+1;
}
```

### 27. Remove Element

Given an array nums and a value val, remove all instances of that value in-place and return the new length. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O\(1\) extra memory. The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example:

```text
Input: nums = [0,1,2,2,3,0,4,2], val = 2,
Output: 5
Note: nums will become [0,1,3,0,4] for the first 5 positions
```

Solution: 双指针更新修改

```cpp
int removeElement(vector<int>& nums, int val) {
    int i = 0;
    for (int j = 0; j < nums.size(); ++j) {
        if (nums[j] != val) {
            if (i != j) nums[i++] = nums[j];
            else ++i;
        }
    }
    return i;
}
```

### 28. Implement strStr

Implement strStr\(\). Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example:

```text
Input: haystack = "hello", needle = "ll"
Output: 2

Input: haystack = "aaaaa", needle = "bba"
Output: -1
```

Clarification: What should we return when needle is an empty string? This is a great question to ask during an interview. For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr\(\) and Java's indexOf\(\).

Solution: 暴力或[KMP](https://blog.csdn.net/starstar1992/article/details/54913261)，有两种写法，分别在算next table时用的是前一位和当前位，本质上没差别但算前一位的next table要好写一点\(即version 1\)，一定要背

```cpp
// KMP version 1
int find_next(vector<int> &next, const string &needle, const int k) {
    // memoization
    if (k < next.size()) return next[k];

    // find palindrome and build next table for position j-1
    // e.g. next table of ababaca is [-1,0,0,1,2,3,0]
    for (int j = next.size(), p = next[j-1]; j <= k; ++j) {           
        while (p >= 0 && needle[p] != needle[j-1]) p = next[p];
        next.push_back(p+1);
        p = next[j];
    }
    return next[k];
}

int strStr(string haystack, string needle) {
    int n = haystack.size(), p = needle.size();
    if (p > n) return -1;

    // build next table for needle
    vector<int> next {-1, 0};

    // begin matching
    int x = 0;
    while (x < n - p + 1) {
        int k = 0;
        while (x < n - p + 1 && k < p && haystack[x+k] == needle[k]) ++k;
        if (k == p) return x;
        x += k - find_next(next, needle, k);
    }
    return -1;        
}

// KMP version 2
void cal_next(const string &needle, vector<int> &next) {
    // e.g. ababaca 的 next 数组是 [-1,-1,0,1,2,-1,0]
    for (int j = 1, p = -1; j < needle.length(); ++j) {
        while (p > -1 && needle[p+1] != needle[j]) p = next[p];  // 如果下一位不同，往前回溯
        if (needle[p+1] == needle[j]) ++p;  // 如果下一位相同
        next[j] = p;  // 把算的k的值（就是相同的最大前缀和最大后缀长）赋给next[q]
    }
}

int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;
    vector<int> next(needle.length(), -1); // next[0]初始化为-1，-1表示不存在相同的最大前缀和最大后缀
    cal_next(needle, next);  //计算next数组

    int k = -1, n = haystack.length(), p = needle.length();
    for (int i = 0; i < n; ++i) {
        while (k > -1 && needle[k+1] != haystack[i]) k = next[k]; // 有部分匹配，往前回溯
        if (needle[k+1] == haystack[i]) ++k;
        if (k == p-1) return i-p+1;  // 说明k移动到needle的最末端，返回相应的位置
    }
    return -1;         
}
```

### 29. Divide Two Integers

Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator. Return the quotient after dividing dividend by divisor. Note: Both dividend and divisor will be 32-bit signed integers; The divisor will never be 0; Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: \[−2^31, 2^31 − 1\]. For the purpose of this problem, assume that your function returns 2^31 − 1 when the division result overflows.

Example:

```text
Input: dividend = 10, divisor = 3
Output: 3

Input: dividend = 7, divisor = -3
Output: -2
```

Solution: 要求不能乘除，进行加减就行了，但是一个问题是加减有可能速度太慢，因此需要转换，由于任何一个数都能表示成二进制，所以有dividend = divisor\*\(a\*2^1 + b\*2^2 + ...... + m\*2^k\)。所以只要计算出所有divisor\*2^k，然后减去即可

```cpp
int divide(int dividend, int divisor) {
    if (!divisor) return INT_MAX;
    long long res = 0;
    long long dd = llabs(dividend);
    long long ds = llabs(divisor);

    long long origin_ds = ds;
    int sign = ((dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0))? 1 : -1;
    int offset = -1;
    while (offset) {
        offset = 0;
        ds = origin_ds;
        while (dd >= (ds << 1)) {
            ds <<= 1;  // 左移一位 相当于乘以二
            offset++;
        }
        if (dd < ds) break;
        dd -= ds;
        res += (1ll << offset);
    }
    res = res * sign;
    return (res > INT_MAX || res < INT_MIN) ? INT_MAX : (int)res;
}
```

### 30. Substring with Concatenation of All Words

You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring\(s\) in s that is a concatenation of each word in words exactly once and without any intervening characters.

Example:

```text
Input:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
Output: [0,9]

Input:
  s = "wordgoodstudentgoodword",
  words = ["word","stud"]
Output: []
```

Solution: hashmap，但trick和操作比较繁复

```cpp
vector<int> findSubstring(string s, vector<string>& words) {
    unordered_map<string, int> map;
    int i, v_size = words.size(),
        w_len = words[0].length(),
        start, cur,
        s_len = s.length(), w_cnt;
    vector<int> res;
    if (s_len < w_len * v_size) return res;

    for (i = 0; i < v_size; ++i)
        map[words[i]] = map.count(words[i]) > 0? ++map[words[i]]: 1;  // trick here

    for (i = 0; i < w_len; ++i) {
        start = cur = i;
        w_cnt = v_size;
        while (start <= s_len - w_len * v_size) {
            if (map.count(s.substr(cur, w_len)) == 0) {
                w_cnt = v_size;
                for (; start != cur; start += w_len)
                    ++map[s.substr(start, w_len)];
                start += w_len;
            } else if (map[s.substr(cur, w_len)] == 0) {
                for (; s.substr(cur, w_len) != s.substr(start, w_len); start += w_len) {
                    ++map[s.substr(start, w_len)];
                    ++w_cnt;
                }
                start += w_len;
            } else {
                --map[s.substr(cur, w_len)];
                if (--w_cnt == 0) {
                    res.push_back(start);
                    ++map[s.substr(start, w_len)];
                    start += w_len;
                    ++w_cnt;
                }
            }
            cur += w_len;
        }
        for (; start<cur; start+= w_len)  ++map[s.substr(start, w_len)];
    }
    return res;
}
```

### 31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers. If such arrangement is not possible, it must rearrange it as the lowest possible order \(ie, sorted in ascending order\). The replacement must be in-place and use only constant extra memory.

Example:

```text
1,2,3 -> 1,3,2
3,2,1 -> 1,2,3
1,1,5 -> 1,5,1
```

Solution: 比较复杂，涉及到比较和对称交换: \(1\)从数列的右边向左寻找连续递增序列，中止位置记为i; \(2\)从上述序列中找一个比a\[i-1\]大的最小数，并交换这两个数; \(3\)将该连续递增序列反序

```cpp
void nextPermutation(vector<int>& nums) {
    int nsize = nums.size() - 1;
    int i = nsize;
    for (; i > 0; i--) {
        if (nums[i-1] < nums[i]) {
            for (int j = nsize; j >= i; j--) {
                if (nums[j] > nums[i-1]) {
                    swap(nums[i-1], nums[j]);
                    break;
                }
            }
            int left = i, right = nsize;
            while (left < right) swap(nums[left++], nums[right--]);
            break;
        }
    }
    if (!i) {
        int left = 0, right = nsize;
        while (left < right) swap(nums[left++], nums[right--]);
    }
}
```

### 32. Longest Valid Parentheses

Given a string containing just the characters '\(' and '\)', find the length of the longest valid \(well-formed\) parentheses substring.

Example:

```text
Input: "(()"
Output: 2

Input: ")()())"
Output: 4
```

Solution: dp or stack

```cpp
// dp
int longestValidParentheses(string s) {
    int n = s.length();
    if ( n < 2) return 0;
    int cur_max = 0;
    vector<int> dp(n, 0);
    for (int i = 1; i < n; i++) {
        if (s[i] == ')' && i-dp[i-1]-1 >= 0 && s[i-dp[i-1]-1] == '(') {
                dp[i] = dp[i-1] + 2 + ((i-dp[i-1]-2 >= 0)? dp[i-dp[i-1]-2]: 0);
                cur_max = max(dp[i], cur_max);
        }
    }
    return cur_max;
}

// stack
int longestValidParentheses(string s) {
    int res = 0, l = 0;
    stack<int> sk;
    for (int i = 0; i < s.size(); ++ i) {
        if (s[i] == '(') sk.push(i);
        else {
            if (sk.empty()) l = i + 1;
            else {
                sk.pop();
                if (sk.empty()) res = max(res, i - l + 1);
                else res = max(res, i - sk.top());
            }

        }

    }
    return res;
}
```

### 33. Search in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. \(i.e., \[0,1,2,4,5,6,7\] might become \[4,5,6,7,0,1,2\]\). You are given a target value to search. If found in the array return its index, otherwise return -1. You may assume no duplicate exists in the array. Your algorithm's runtime complexity must be in the order of O\(log n\).

Example:

```text
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

Solution: 二分，非常精妙，一定要背

```cpp
int search(vector<int>& nums, int target) {
    int start = 0, end = nums.size() - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < nums[end]) {
            // right half sorted
            if (target > nums[mid] && target <= nums[end]) start = mid+1;
            else end = mid-1;
        } else {
            // left half sorted
            if (target >= nums[start] && target < nums[mid]) end = mid-1;
            else start = mid+1;
        }
    }
    return -1;
}
```

### 34. Search for a Range

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value. Your algorithm's runtime complexity must be in the order of O\(log n\). If the target is not found in the array, return \[-1, -1\].

Example:

```text
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

Solution: 二分法，熟能生巧，注意二分保持左闭右开的习惯，一定要背

```cpp
// direct
vector<int> searchRange(vector<int> &nums, int target) {
    vector<int> res;
    int l = 0, r = nums.size() - 1;
    if (nums.empty()) {
        res.push_back(-1);
        res.push_back(-1);
        return res;     
    } else if (nums.size() == 1) {
        if (nums[0] == target) {
            res.push_back(0);
            res.push_back(0);
            return res;
        } else {
            res.push_back(-1);
            res.push_back(-1);
            return res;
        }
    }
    while (l <= r) {
        if (nums[l] < target) ++l;
        if (nums[r] > target) --r;
        if (nums[l] == target && nums[r] == target) {
            res.push_back(l);
            res.push_back(r);
            return res;
        }
    }
    res.push_back(-1);
    res.push_back(-1);
    return res;
}

// indirect
int lower_bound(vector<int> &nums, int target) {
    int mid;
    int left = 0, right = nums.size();
    while (left < right) {
        mid = (left + right) / 2;
        if (nums[mid] >= target) right = mid;
        else left = mid + 1;
    }
    return left;

}
int upper_bound(vector<int> &nums, int target) {
    int mid;
    int left = 0, right = nums.size();
    while (left < right) {
        mid = (left + right) / 2;
        if (nums[mid] > target) right = mid;
        else left = mid + 1;
    }
    return left - 1; // important!!!

}
vector<int> searchRange(vector<int>& nums, int target) {
    vector<int> ans;
    if (nums.empty()) {
        ans.push_back(-1);
        ans.push_back(-1);
        return ans;     
    }
    int lower = lower_bound(nums, target);
    int upper = upper_bound(nums, target);
    if (lower == nums.size() || nums[lower] != target) { // or lower == nums.size() || lower > upper
        ans.push_back(-1);
        ans.push_back(-1);
    } else {
        ans.push_back(lower);
        ans.push_back(upper);
    }
    return ans;
}
```

### 35. Search Insert Position

Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order. You may assume no duplicates in the array.

Example:

```text
Input: [1,3,5,6], 5
Output: 2

Input: [1,3,5,6], 2
Output: 1

Input: [1,3,5,6], 7
Output: 4

Input: [1,3,5,6], 0
Output: 0
```

Solution: 二分，可以用来练手

```cpp
int searchInsert(vector<int>& nums, int target) {
    if (nums.empty()) return 0;
    int l = 0, r = nums.size();
    while (l < r) {
        int mid = (l + r) / 2;
        if (nums[mid] >= target) r = mid;
        else l = mid + 1;
    }
    return l;
}
```

### 36. Valid Sudoku

Determine if a 9x9 Sudoku board is valid \(each row, column, and sub-boxe contain 1-9 without repetition\). The Sudoku board could be partially filled, where empty cells are filled with the character '.'.

Example:

```text
Input:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: true
```

Solution: 做三个row, column和sub-box的2D数组，然后判断对应digit是否重复

```cpp
bool isValidSudoku(vector<vector<char>>& board) {
    vector<vector<bool>> a(9, vector<bool> (9,false));
    vector<vector<bool>> b(9, vector<bool> (9,false));
    vector<vector<bool>> c(9, vector<bool> (9,false));
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == '.') continue;
            int k = board[i][j] - '1';

            int p = (i/3) * 3 + j/3;
            if (a[i][k] || b[j][k] || c[p][k]) return false;
            else {
                a[i][k] = true;
                b[j][k] = true;
                c[p][k] = true;
            }
        }
    }
    return true;
}
```

### 37. Sudoku Solver

Write a program to solve a Sudoku puzzle by filling the empty cells \(each row, column, and sub-boxe contain 1-9 without repetition\). Empty cells are indicated by the character '.'.

Solution: backtrack，一定要背

```cpp
bool isValid(vector<vector<char> > &board, int x, int y) {  
        int i, j;  
        for (i = 0; i < 9; i++)  
            if (i != x && board[i][y] == board[x][y])  
                return false;  
        for (j = 0; j < 9; j++)  
            if (j != y && board[x][j] == board[x][y])  
                return false;  
        for (i = 3 * (x / 3); i < 3 * (x / 3 + 1); i++)  
            for (j = 3 * (y / 3); j < 3 * (y / 3 + 1); j++)  
                if (i != x && j != y && board[i][j] == board[x][y])  
                    return false;  
        return true;  
}
bool solveSudoku(vector<vector<char> > &board) {  
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {  
            if ('.' == board[i][j]) {  
                for (int k = 1; k <= 9; ++k) {  
                    board[i][j] = '0' + k;  
                    if (isValid(board, i, j) && solveSudoku(board))  
                        return true;  
                    board[i][j] = '.';  
                }  
                return false;  
            }  
        }  
    }
    return true;  
}
```

### 38. Count and Say

The count-and-say sequence is the sequence of integers with the first five terms as following:

```text
1.     1
2.     11
3.     21
4.     1211
5.     111221
```

1 is read off as "one 1" or 11. 11 is read off as "two 1s" or 21. 21 is read off as "one 2, then one 1" or 1211. Given an integer n, generate the nth term of the count-and-say sequence.

Solution: 暴力处理

```cpp
string countAndSay(int n) {
    if (n < 1) return "";
    string ret = "1";
    for (int i = 2; i <= n; i++) {
        string temp = "";
        int count = 1;
        char prev = ret[0];
        for (int j = 1; j < ret.size(); j++) {
            if (ret[j] == prev) count++;
            else {
                temp += to_string(count);
                temp.push_back(prev);
                count = 1;
                prev = ret[j];
            }
        }
        temp += to_string(count);
        temp.push_back(prev);
        ret = temp;
    }
    return ret;
}
```

### 39. Combination Sum

Given a set of candidate numbers \(candidates\) \(without duplicates\) and a target number \(target\), find all unique combinations in candidates where the candidate numbers sums to target. The same repeated number may be chosen from candidates unlimited number of times. Note: All numbers \(including target\) will be positive integers; The solution set must not contain duplicate combinations.

Example:

```text
Input: candidates = [2,3,6,7], target = 7
Output: [
  [7],
  [2,2,3]
]
```

Solution: backtrack，每次递归要传当前值，目标值（或者合并为差值），当前迭代位置

```cpp
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> result;
    vector<int> path;
    backtrack(candidates, 0, 0, target, path, result);
    return result;
}

void backtrack(vector<int> &nums, int pos, int base, int target, vector<int>& path, vector<vector<int>> & result) {
    if (base == target) {
        result.push_back(path);
        return;
    }
    if (base > target) return;
    for (int i = pos; i < nums.size(); ++i) {
        path.push_back(nums[i]);
        backtrack(nums, i, base + nums[i], target, path, result);
        path.pop_back();
    }
}
```

### 40. Combination Sum II

Given a collection of candidate numbers \(candidates\) and a target number \(target\), find all unique combinations in candidates where the candidate numbers sums to target. Each number in candidates may only be used once in the combination. Note: All numbers \(including target\) will be positive integers; The solution set must not contain duplicate combinations.

Example:

```text
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: [
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

Solution: backtrack，每次递归要传当前值，目标值（或者合并为差值），当前迭代位置；注意要判断是否重复使用。一定要背

```cpp
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>> result;
    vector<int> path;
    sort(candidates.begin(), candidates.end());
    backtrack(candidates, 0, 0, target, path, result);
    return result;
}

void backtrack(vector<int> &nums, int pos, int base, int target, vector<int>& path, vector<vector<int>> & result) {
    if (base == target) {
        result.push_back(path);
        return;
    }
    if (base > target) return;
    for (int i = pos; i < nums.size(); ++i) {
        if (i != pos && nums[i] == nums[i-1]) continue;  // new in Q40 from Q39
        path.push_back(nums[i]);
        backtrack(nums, i + 1, base + nums[i], target, path, result);  // Q39: i; Q40: i + 1
        path.pop_back();
    }
}
```

### 41. First Missing Positive

Given an unsorted integer array, find the smallest missing positive integer.

Example:

```text
Input: [1,2,0]
Output: 3

Input: [3,4,-1,1]
Output: 2

Input: [7,8,9,11,12]
Output: 1
```

Solution: 桶排序（鉴于数字大小小于总长度）

```cpp
int firstMissingPositive(vector<int>& nums) {
    int i = 0, n = nums.size();
    while (i < n) {  
        if (nums[i] != i+1 && nums[i] >= 1 && nums[i] <= n && nums[nums[i]-1] != nums[i]) swap(nums[i], nums[nums[i]-1]);  
        else i++;  
    }  
    for (int i = 0; i < n; ++i)
        if (nums[i] != i + 1) return i + 1;  
    return n + 1;  
}
```

### 42. Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

Example:

```text
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

Solution: 水填满后的形状是先升后降的塔形，因此，先遍历一遍找到塔顶，然后分别从两边开始，往塔顶所在位置遍历；水位只会增高不会减小，且一直和最近遇到的最大高度持平，这样知道了实时水位，就可以边遍历边计算面积。或者，也可以不用找顶峰，直接从两边双指针向内遍历，每次移动较小的那个

```cpp
int trap(vector<int>& height) {
    int n = height.size();
    if (n <= 2) return 0;
    int max = -1, maxInd = 0;
    for (int i = 0; i < n; ++i) {
        if (height[i] > max) {
            max = height[i];
            maxInd = i;
        }
    }
    int area = 0, root = height[0];
    for (int i = 0; i < maxInd; ++i) {
        if (root < height[i]) root = height[i];
        else area += (root - height[i]);
    }
    for (int i = n-1, root = height[n-1]; i > maxInd; --i) {
        if (root < height[i]) root = height[i];
        else area += (root - height[i]);
    }
    return area;
}
```

### 45. Jump Game II

Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Your goal is to reach the last index in the minimum number of jumps. You can assume that you can always reach the last inde.

Example:

```text
Input: [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
    Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

Solution: 从左到右记录两个值ab，a是上一次移动到当前位置可以移动的最长距离\(这里有个小技巧，与其记录差值，不如记录a\[i\]+i，即绝对跳跃长度，这样更好比较也不用开另外值记录开始位置\)，b是当前能移动到的最长距离。每当b==i的时候，跳跃一次，b更新为a

```cpp
int jump(vector<int>& nums) {
    int ret = 0, cur_max = 0, cur_rch = 0;
    for (int i = 0; i < nums.size(); ++i) {
        if (cur_rch < i) {
            ++ret;
            cur_rch = cur_max;
        }
        cur_max = max(cur_max, nums[i] + i);
    }
    return ret;
}
```

### 46. Permutations

Given a collection of distinct integers, return all possible permutations.

Solution: backtrack, 一定要背

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> res;
    backtrack(nums, 0, res);
    return res;
}

void backtrack(vector<int> &nums, int level, vector<vector<int>> &res) {
    if (level == nums.size() - 1) {
        res.push_back(nums);
        return;
    }
    for (int i = level; i < nums.size(); i++) {
        swap(nums[i], nums[level]);
        backtrack(nums, level+1, res);
        swap(nums[i], nums[level]);
    }
}
```

### 47. Permutations II

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Solution: backtrack, 一定要背

```cpp
vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> res;
    backtrack(nums, 0, res);
    return res;
}

// nums here pass by copy, since later sorted
void backtrack(vector<int> nums, int level, vector<vector<int>> &res) {
    if (level == nums.size() - 1) {
        res.push_back(nums);
        return;
    }
    sort(nums.begin() + level, nums.end()); // new in Q47 from Q46
    for (int i = level; i < nums.size(); i++) {
        if (i != level && nums[i] == nums[i-1]) continue; // new in Q47 from Q46
        swap(nums[i], nums[level]);
        backtrack(nums, level+1, res);
        swap(nums[i], nums[level]);
    }
}
```

### 48. Rotate Image

You are given an n x n 2D matrix representing an image. Rotate the image by 90 degrees \(clockwise\). Note: You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example:

```text
Input: [
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 
Output: [
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

Solution: 从外层到内层，顺时针同时转四个位置

```cpp
void rotate(vector<vector<int>>& matrix) {
    int temp = 0, n = matrix.size()-1;
    for (int i = 0; i <= n/2; i++) {
        for (int j = i; j < n-i; j++) {
            temp = matrix[j][n-i];
            matrix[j][n-i] = matrix[i][j];
            matrix[i][j] = matrix[n-j][i];
            matrix[n-j][i] = matrix[n-i][n-j];
            matrix[n-i][n-j] = temp;
        }
    }
}
```

### 49. Group Anagrams

Given an array of strings, group anagrams together. Note: All inputs will be in lowercase; The order of your output does not matter.

Example:

```text
Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```

Solution: 用一个26个0组成的string来表示计数，存到map&gt;里面；或者sort字符串，然后存在unordered\_map&gt;里面

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> group;
    for (const auto &s: strs) {
        string key = s;
        sort(key.begin(), key.end());
        group[key].push_back(s);
    }

    vector<vector<string>> anagrams;
    for (const auto m: group) { 
        vector<string> anagram(m.second.begin(), m.second.end());
        anagrams.push_back(anagram);
    }
    return anagrams;
}
```

### 50. Power Function

Implement pow\(x, n\). Note: -100.0 &lt; x &lt; 100.0; n is a 32-bit signed integer, within the range \[−2^31, 2^31 − 1\]

Example:

```text
Input: 2.10000, 3
Output: 9.26100

Input: 2.00000, -2
Output: 0.25000
```

Solution: 递归，一定要背

```cpp
double myPow(double x, int n) {
    if (!n) return 1;
    if (!x) return 0;
    if (n == INT_MIN) return 1/(myPow(x, INT_MAX)*x);
    if (n < 0) return 1/myPow(x, -n);
    if (n % 2) return x*myPow(x, n-1);
    double temp = myPow(x, n >> 1);
    return temp*temp;
}
```

