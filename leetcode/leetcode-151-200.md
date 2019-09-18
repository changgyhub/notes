# LeetCode 151 - 200

### 151. Reverse Words in a String

Given an input string, reverse the string word by word.

Example:

```text
Input: "the sky is blue",
Output: "blue is sky the".
```

Solution: 需要考虑前后的空格，图方便可以开一个新字符串，如果想inplace可以换做char\* swap操作

```cpp
void reverseWords(string &s) {
    int size = s.size();
    if (size <= 0) return;
    int index = size - 1;
    string str;
    while (index >= 0) {
        while (index >= 0 && s[index] == ' ') --index;
        if (index < 0) break;
        if (str.size()) str.push_back(' ');
        string tmp;
        while (index >= 0 && s[index] != ' ') {
            tmp.push_back(s[index]);
            --index;
        }
        reverse(tmp.begin(),tmp.end());
        str.append(tmp);
    }
    s = str;
}
```

### 152. Maximum Product Subarray

Given an integer array nums, find the contiguous subarray within an array \(containing at least one number\) which has the largest product.

Example:

```text
Input: [2,3,-2,4]
Output: 6
```

Solution: 维护local\_min, local\_max, global\_max，做贪心

```cpp
int maxProduct(vector<int>& nums) {
    if (nums.empty()) return 0;
    int max_local = nums[0], min_local = nums[0], global = nums[0];  
    for (int i = 1; i < nums.size(); ++i) {  
        int max_copy = max_local;  
        max_local = max(max(nums[i]*max_copy, nums[i]), nums[i]*min_local);  
        min_local = min(min(nums[i]*max_copy, nums[i]), nums[i]*min_local);  
        global = max(global, max_local);  
    }  
    return global; 
}
```

### 153. Find Minimum in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

Example:

```text
Input: [4,5,6,7,0,1,2]
Output: 0
```

Solution: 二分查找，一定要背

```cpp
int findMin(vector<int>& nums) {
    if (nums.empty()) return 0;
    int n = nums.size(), low = 0, high = n - 1;
    while (low < high && nums[low] >= nums[high]) {
        int mid = low + (high - low) / 2;
        if (nums[mid] < nums[low]) high = mid;
        // else if (nums[mid] == nums[low]) ++low;  // with duplicates
        else low = mid + 1;
    }
    return nums[low];
}
```

### 154. Find Minimum in Rotated Sorted Array II

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. The array may contain duplicates.

Example:

```text
Input: [2,2,2,0,1]
Output: 0
```

Solution: 二分查找，一定要背

```cpp
int findMin(vector<int>& nums) {
    if (nums.empty()) return 0;
    int n = nums.size(), low = 0, high = n - 1;
    while (low < high && nums[low] >= nums[high]) {
        int mid = low + (high - low) / 2;
        if (nums[mid] < nums[low]) high = mid;
        else if (nums[mid] == nums[low]) ++low;  // with duplicates
        else low = mid + 1;
    }
    return nums[low];
}
```

### 155. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Example:

```text
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
```

Solution: 维护两个stack，一定要背

```cpp
class MinStack {
public:
    MinStack() {}

    void push(int x) {
        sta.push(x);
        if (min.empty() || min.top() >= x) min.push(x);
    }

    void pop() {
        if (!min.empty() && min.top() == sta.top()) min.pop();
        sta.pop();
    }

    int top() {
        return sta.top();
    }

    int getMin() {
        return min.top();
    }
private:
    stack<int> sta, min;
};
```

### 160. Intersection of Two Linked Lists

Write a program to find the node at which the intersection of two singly linked lists begins. You may assume there are no cycles anywhere in the entire linked structure. Return null if no intersection.

Example:

```text
Input:
A:          a1 -> a2
                   |
                   V
                     c1 -> c2 -> c3
                   A
                   |            
B:     b1 -> b2 -> b3
Output: c1
```

Solution: 由于一旦连一起，后面一定一样长，所以先测定长度，去掉前面多余的（对应例子中的b1），等长度相同后，两头一齐前进即可

```cpp
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *l1 = headA, *l2 = headB;
    while (l1 != l2) {
        l1 = l1? l1->next: headB;
        l2 = l2? l2->next: headA;
    }
    return l1;
}
```

### 162. Find Peak Element

A peak element is an element that is greater than its neighbors. Given an input array nums, where neighbors are unequal, find a peak element and return its index. You may imagine that nums\[-1\] = nums\[n\] = -\inf.

Example:

```text
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5
```

Solution: 二分或者暴力

```cpp
// brute force
int findPeakElement(vector<int>& nums) {
    int i;
    for (i = 0; i < nums.size() - 1; ++i) {
        if (nums[i] > nums[i+1]) break;
    }
    return i;
}

// bisection
int findPeakElement(vector<int> &nums) {
    int n = nums.size(), l = 0, r = n - 1;
    while (r - l > 1) {
        int mid = l + (r - l) / 2;
        if (nums[mid] < nums[mid + 1]) l = mid;
        else r = mid;
    }
    return nums[r] > nums[l] ? r : l;
}
```

### 163. Missing Ranges

Given a sorted integer array **nums**, where the range of elements are in the **inclusive range** **[lower, upper]**, return its missing ranges.

Example:

```
Input: nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99,
Output: ["2", "4->49", "51->74", "76->99"]
```

Solution: 正常遍历处理，注意细节

```cpp
string print(long long x, long long y){
    return x == y? to_string(x): to_string(x) + "->" + to_string(y);
}
vector<string> findMissingRanges(vector<int>& nums, int lower, int upper) {
    if (nums.empty()) return {print(lower, upper)};
    vector<string> res;
    if (nums.front() != lower) res.push_back(print(lower, nums.front() - 1));        
    for (int i = 0; i < int(nums.size())-1; ++i){
        long long l = static_cast<long long>(nums[i]) + 1;
        long long r = static_cast<long long>(nums[i+1]) - 1;
        if (l <= r) res.push_back(print(l, r));
    }
    if (nums.back() != upper) res.push_back(print(nums.back() + 1, upper));
    return res;
}
```

### 164. Maximum Gap

Given an unsorted array, find the maximum difference between the successive elements in its sorted form.

Example:

```text
Input: [3,6,9,1]
Output: 3 ([1,3,6,9] => 6 - 3 = 3 or 9 - 6 = 3)
```

Solution: O\(nlgn\)可以先排序再过一遍。O\(n\)可以用Radix Sort（基数排序），或者Buckets & The Pigeonhole Principle（按鸽子笼原理，gap至少为n/m，则可以开这么多个buckets，比较前一个bucket的最小值和后一个bucket的最大值）

```cpp
// sort and compare
int maximumGap(vector<int>& nums) {
    sort(begin(nums),end(nums));
    int maxGap = 0;
    for (int i = 0; i < nums.size() - 1; i++)
        maxGap = max(nums[i + 1] - nums[i], maxGap);
    return maxGap;
}

// radix sort: time O(d(n+k)) space O(n+k)
int maximumGap(vector<int>& nums) {
    if (nums.empty() || nums.size() < 2) return 0;
    int maxVal = *max_element(nums.begin(), nums.end());
    int exp = 1;                                 // 1, 10, 100, 1000 ...
    int radix = 10;                              // base 10 system

    vector<int> aux(nums.size());

    /* LSD Radix Sort */
    while (maxVal / exp > 0) {                   // Go through all digits from LSD to MSD
        vector<int> count(radix, 0);

        for (int i = 0; i < nums.size(); i++)    // Counting sort
            count[(nums[i] / exp) % 10]++;

        for (int i = 1; i < count.size(); i++)   // you could also use partial_sum()
            count[i] += count[i - 1];

        for (int i = nums.size() - 1; i >= 0; i--)
            aux[--count[(nums[i] / exp) % 10]] = nums[i];

        for (int i = 0; i < nums.size(); i++)
            nums[i] = aux[i];

        exp *= 10;
    }

    int maxGap = 0;
    for (int i = 0; i < nums.size() - 1; i++)
        maxGap = max(nums[i + 1] - nums[i], maxGap);
    return maxGap;
}

// buckets & the Pigeonhole Principle: time O(n+b) space O(2b)
class Bucket {
public:
    bool used = false;
    int minval = numeric_limits<int>::max();        // same as INT_MAX
    int maxval = numeric_limits<int>::min();        // same as INT_MIN
};
int maximumGap(vector<int>& nums) {
    if (nums.empty() || nums.size() < 2) return 0;

    int mini = *min_element(nums.begin(), nums.end()),
        maxi = *max_element(nums.begin(), nums.end());

    int bucketSize = max(1, (maxi - mini) / ((int)nums.size() - 1));        // bucket size or capacity
    int bucketNum = (maxi - mini) / bucketSize + 1;                         // number of buckets
    vector<Bucket> buckets(bucketNum);

    for (auto && num : nums) {
        int bucketIdx = (num - mini) / bucketSize;                          // locating correct bucket
        buckets[bucketIdx].used = true;
        buckets[bucketIdx].minval = min(num, buckets[bucketIdx].minval);
        buckets[bucketIdx].maxval = max(num, buckets[bucketIdx].maxval);
    }

    int prevBucketMax = mini, maxGap = 0;
    for (auto && bucket : buckets) {
        if (!bucket.used) continue;

        maxGap = max(maxGap, bucket.minval - prevBucketMax);
        prevBucketMax = bucket.maxval;
    }

    return maxGap;
}
```

### 166. Fraction to Recurring Decimal

Given two integers representing the numerator and denominator of a fraction, return the fraction in string format. If the fractional part is repeating, enclose the repeating part in parentheses.

Example:

```text
Input: numerator = 2, denominator = 3
Output: "0.(6)"
```

Solution: 用一个hashmap储存每次remainder乘10除以分母的余数，直到这个余数重复出现，则为一个循环

```cpp
string fractionToDecimal(int numerator, int denominator) {
    if (!numerator) return "0";
    long long n = abs((long long)numerator), d = abs((long long)denominator);
    long long q = n / d, r = n % d; 

    string s1, s2;
    s1 = (numerator >= 0) ^ (denominator >= 0)? "-": "";
    s1 += to_string(q);
    if (!r) return s1;

    unordered_map<int, int> map;
    while (r && !map.count(r)) {          
        map[r] = s2.size();
        r *= 10;
        s2 += to_string(r / d);
        r %= d;
    }
    if (r) {            
         s2.insert(map[r], "(");
         s2 += ")";
    }
    return s1 + "." + s2; 
}
```

### 167. Two Sum II - Input array is sorted

Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number. The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Assume one unqie solution exists.

Example:

```text
Input: numbers = [2,7,11,15], target = 9
Output: [1,2] (2 + 7 = 9)
```

Solution: 双指针

```cpp
vector<int> twoSum(vector<int>& numbers, int target) {
    int l = 0, r = numbers.size() - 1, sum;
    while (l < r) {
        sum = numbers[l] + numbers[r];
        if (sum == target) break;
        if (sum < target) ++l;
        else --r;
    }
    return vector<int>{l + 1, r + 1};
}
```

### 168. Excel Sheet Column Title

Given a positive integer, return its corresponding column title as appear in an Excel sheet.

Example:

```text
1 -> A
2 -> B
3 -> C
...
26 -> Z
27 -> AA
28 -> AB 
...
```

Solution: 正常mod+div操作，注意这和进制类题不太一样，没有0，需要减去。一定要背和体会

```cpp
string convertToTitle (int n) {
    if (n == 0) return "";
    --n;
    return convertToTitle(n / 26) + (char) (n % 26 + 'A');
}
```

### 169. Majority Element

Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times. You may assume that the array is non-empty and the majority element always exist in the array.

Example:

```text
Input: [2,2,1,1,1,2,2]
Output: 2
```

Solution: 设一个count和一个elem，遍历数组，如果和elem相同++count，否则--count，count为0时更新elem

```cpp
int majorityElement(vector<int> &num) {
    int elem = 0, count = 0;
    for (int i = 0; i < num.size(); ++i)  {
        if (!count) {
            elem = num[i];
            count = 1;
        } else {
            if (elem == num[i]) count++;
            else count--;
        }
    }
    return elem;
}
```

### 171. Excel Sheet Column Number

Given a column title as appear in an Excel sheet, return its corresponding column number.

Example:

```text
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...
```

Solution: 同Q168

```cpp
// recursive
int titleToNumber(string s) {
    return s.empty()? 0: titleToNumber(s.substr(0, s.length()-1))*26 + (s[s.length()-1] + 1 - 'A');
}

// iterative
int titleToNumber(string s) {
    int num = 0;
    for (char c : s) {
        num += num * 25 + (c - 'A' + 1); // or num = num * 26 + (c - 'A' + 1);
    }
    return num;
}
```

### 172. Factorial Trailing Zeroes

Given an integer n, return the number of trailing zeroes in n!.

Example:

```text
Input: 5
Output: 1 (5! = 120, one trailing zero)
```

Solution: 迭代除5，算有多少个（因为出现2的次数一定比5多，所以尾数为0的个数就是所有数字能被多少个5除尽）

```cpp
int trailingZeroes(int n) {
    int res = 0;
    while (n) {
        n /= 5;
        res += n;
    }
    return res;
}
```

### 173. Binary Search Tree Iterator

mplement an iterator over a binary search tree \(BST\). Your iterator will be initialized with the root node of a BST. Calling next\(\) will return the next smallest number in the BST. Note: next\(\) and hasNext\(\) should run in average O\(1\) time and uses O\(h\) memory, where h is the height of the tree.

Example:

```text
Your BSTIterator will be called like this:
BSTIterator i = BSTIterator(root);
while (i.hasNext()) cout << i.next();
```

Solution: 用stack存右树，一定要背

```cpp
class BSTIterator {
    stack<TreeNode*> s;
public:
    BSTIterator(TreeNode *root) {            
        while (root) {
            s.push(root);
            root = root->left;
        }
    }

    bool hasNext() {
        return !s.empty();
    }

    int next() {
        TreeNode *smallest = s.top();
        s.pop(); 
        if (smallest->right) {
            TreeNode* tmp = smallest->right;
            while (tmp) {
                s.push(tmp);
                tmp = tmp->left;
            }
        }
        return smallest->val;
    }
};
```

### 179. Largest Number

Given a list of non negative integers, arrange them such that they form the largest number. Note: The result may be very large, so you need to return a string instead of an integer.

Example:

```text
Input: [3,30,34,5,9]
Output: "9534330"
```

Solution: 自定义compare函数, a比b大的条件是a.concat\(b\) &gt; b.concat\(a\)

```cpp
string largestNumber(vector<int>& nums) {
    vector<string> snums;
    for (auto i: nums) snums.push_back(to_string(i));
    sort(snums.begin(), snums.end(), [](const string &a, const string &b) {return a + b > b + a;});
    if (snums[0] == "0") return "0"; // avoid cases of 000000...
    string res = "";
    for (auto s: snums) res += s;
    return res;
}
```

### 187. Repeated DNA Sequences

All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA. Write a function to find all the 10-letter-long sequences \(substrings\) that occur more than once in a DNA molecule.

Example:

```text
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC", "CCCCCAAAAA"]
```

Solution: hashset即可，也可以用bitset等优化

```cpp
vector<string> findRepeatedDnaSequences(string s) {
    unordered_set<string> seen, repeated;
    for (int i = 0; i + 9 < s.length(); ++i) {
        string ten = s.substr(i, 10);
        if (seen.find(ten) == seen.end()) {
            seen.insert(ten);
        } else {
            repeated.insert(ten);
        }
    }
    vector<string> ret;
    for (string s: repeated) ret.push_back(s);
    return ret;
}
```

### 188. Best Time to Buy and Sell Stock IV

Say you have an array for which the ith element is the price of a given stock on day i. Design an algorithm to find the maximum profit. You may complete at most k transactions.

Example:

```text
Input: [3,2,6,5,0,3], k = 2
Output: 7 (Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.)
```

Solution: 维护local和global：local\[i\]\[j\]为在到达第i天时最多可进行j次交易并且最后一次交易在最后一天卖出的最大利润，此为局部最优；global\[i\]\[j\]为在到达第i天时最多可进行j次交易的最大利润，此为全局最优。递推关系为\(1\) local\[i\]\[j\] = max\(global\[i-1\]\[j-1\] + max\(diff, 0\), local\[i-1\]\[j\] + diff\) \(2\) global\[i\]\[j\] = max\(local\[i\]\[j\], global\[i - 1\]\[j\]\)。local\[i\]\[j\]和global\[i\]\[j\]的区别是：local\[i\]\[j\]意味着在第i天一定有交易（卖出）发生，当第i天的价格高于第i-1天（即diff &gt; 0）时，那么可以把这次交易（第i-1天买入第i天卖出）跟第i-1天的交易（卖出）合并为一次交易，即local\[i\]\[j\]=local\[i-1\]\[j\]+diff；当第i天的价格不高于第i-1天（即diff&lt;=0）时，那么local\[i\]\[j\]=global\[i-1\]\[j-1\]+diff，而由于diff&lt;=0，所以可写成local\[i\]\[j\]=global\[i-1\]\[j-1\]。global\[i\]\[j\]就是我们所求的前i天最多进行k次交易的最大收益，可分为两种情况：如果第i天没有交易（卖出），那么global\[i\]\[j\]=global\[i-1\]\[j\]；如果第i天有交易（卖出），那么global\[i\]\[j\]=local\[i\]\[j\]。注意如果k&gt;=n，则相当于可以交易无数次

```cpp
int maxProfit(int k, vector<int>& prices) {
    int days = prices.size();
    if (days < 2) return 0;
    if (k >= days) return maxProfitUnlimited(prices);
    vector<int> local(k+1, 0), global(k+1, 0);
    for (int i = 1; i < days ; ++i) {
        int diff = prices[i] - prices[i-1];
        for (int j = k; j > 0; --j) {
            local[j] = max(global[j-1], local[j] + diff);
            global[j] = max(global[j], local[j]);
        }
    }

    return global[k];
}

int maxProfitUnlimited(vector<int> prices) {
    int maxProfit = 0;
    for (int i = 1; i < prices.size(); ++i) {
        if (prices[i] > prices[i-1]) {
            maxProfit += prices[i] - prices[i-1];
        }
    }
    return maxProfit;
}
```

### 189. Rotate Array

Given an array, rotate the array to the right by k steps, where k is non-negative.

Example:

```text
Input: [-1,-100,3,99] and k = 2
Output: [3,99,-1,-100]
```

Solution:

```cpp
void rotate(vector<int>& nums, int k) {
    if (nums.empty()) return;
    k %= nums.size();
    nums.insert(nums.begin(), nums.end()-k, nums.end());
    nums.erase(nums.end()-k, nums.end());
}
```

### 190. Reverse Bits

Reverse bits of a given 32 bits unsigned integer. Follow up: If this function is called many times, how would you optimize it?

Example:

```text
Input: 43261596   (00000010100101000001111010011100)
Output: 964176192 (00111001011110000010100101000000)
```

Solution: 背，学会运用&lt;&lt;=和&gt;&gt;=

```cpp
uint32_t reverseBits(uint32_t n) {
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
        res <<= 1;
        res += n & 1;
        n >>= 1;
    }
    return res;
}
```

### 191. Number of 1 Bits

Write a function that takes an unsigned integer and returns the number of '1' bits it has \(also known as the Hamming weight\).

Example:

```text
Input: 128
Output: 1 (00000000000000000000000010000000)
```

Solution: 按位&和&gt;&gt;

```cpp
int hammingWeight(uint32_t n) {
    int res = 0;
    while (n) {
        res += n & 1;
        n >>= 1;
    }
    return res;
    // or equally in one line
    // return n? (n & 1) + hammingWeight(n >> 1): 0;
}
```

### 198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night. Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example:

```text
Input: [2,7,9,3,1]
Output: 12 (Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1). Total amount you can rob = 2 + 9 + 1 = 12.)
```

Solution: dp\[i+1\] = max\(dp\[i\], nums\[i\] + dp\[i-1\]\)，也可以转化成一个prev一个cur。很容易想复杂，一定要背

```cpp
int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    int n = nums.size();
    vector<int> dp(n+1, 0);
    dp[1] = nums[0];
    for (int i = 1; i < n; ++i) {
        dp[i+1] = max(dp[i], nums[i] + dp[i-1]);
    }
    return dp[n];
}
```

### 199. Binary Tree Right Side View

Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example:

```text
Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```

Solution: preorder traversal，每层更新最后遇到的节点，一定要背

```cpp
void rightSideView(TreeNode *root, int level, vector<int> &view) {
    if (!root) return;
    if (view.size() < level + 1) view.push_back(root->val);
    else view[level] = root->val;
    rightSideView(root->left, level + 1, view);
    rightSideView(root->right, level + 1, view);
}
vector<int> rightSideView(TreeNode* root) {
    vector<int> view;
    rightSideView(root, 0, view);
    return view;
}
```

### 200. Number of Islands

Given a 2d grid map of '1's \(land\) and '0's \(water\), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example:

```text
Input:
11000
11000
00100
00011
Output: 3
```

Solution: dfs/bfs常规题目，一定要背

```cpp
int numIslands(vector<vector<char>>& grid) {
    int count = 0;  
    for (int i = 0; i < grid.size(); ++i) {  
        for (int j = 0; j < grid[0].size(); ++j) {  
            if (grid[i][j] =='1') {  
                search(grid, i, j);  
                ++count;  
            }  
        }  
    }  
    return count;  
}
void search(vector<vector<char>> &grid, int x, int y) {  
    if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] != '1') return;  
    grid[x][y] = '0';  
    search(grid, x - 1, y);  
    search(grid, x + 1, y);  
    search(grid, x, y - 1);  
    search(grid, x, y + 1);  
}
```

