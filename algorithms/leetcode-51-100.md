# LeetCode 51 - 100

### 51. N-Queens

Given an integer n, return all distinct solutions to the n-queens puzzle. Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.

Example:

```text
Input: 4
Output: [
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
```

Solution: backtrack, 一定要背

```cpp
void backtrack(vector<vector<string>> &ret, vector<string> &board, vector<bool> &column, vector<bool> &ldiag, vector<bool> &rdiag, int row, int n) {  
    if (row == n) {  
        ret.push_back(board);  
        return;  
    }
    for (int i = 0; i < n; i++) {
        if (column[i] || ldiag[n-row+i-1] || rdiag[row+i+1]) continue;
        board[row][i] = 'Q'; column[i] = ldiag[n-row+i-1] = rdiag[row+i+1] = true;
        backtrack(ret, board, column, ldiag, rdiag, row+1, n);
        board[row][i] = '.'; column[i] = ldiag[n-row+i-1] = rdiag[row+i+1] = false;
    }  
}  
vector<vector<string>> solveNQueens(int n) {  
    vector<vector<string>> ret;
    if (n == 0) return ret;  
    vector<string> board(n, string(n, '.'));
    vector<bool> column(n, false), ldiag(2*n-1, false), rdiag(2*n-1, false);
    backtrack(ret, board, column, ldiag, rdiag, 0, n);  
    return ret;  
}
```

### 52. N-Queens II

Given an integer n, return the number of distinct solutions to the n-queens puzzle.

Example:

```text
Input: 4
Output: 2
```

Solution: Q51的输出改成纪录次数即可

```cpp
void backtrack(int &ret, vector<string> &board, vector<bool> &column, vector<bool> &ldiag, vector<bool> &rdiag, int row, int n) {  
    if (row == n) {  
        ret++;
        return;  
    }
    for (int i = 0; i < n; i++) {
        if (column[i] || ldiag[n-row+i-1] || rdiag[row+i+1]) continue;
        board[row][i] = 'Q'; column[i] = ldiag[n-row+i-1] = rdiag[row+i+1] = true;
        backtrack(ret, board, column, ldiag, rdiag, row+1, n);
        board[row][i] = '.'; column[i] = ldiag[n-row+i-1] = rdiag[row+i+1] = false;
    }  
}  
int totalNQueens(int n) {  
    int ret = 0;
    if (n == 0) return ret;  
    vector<string> board(n, string(n, '.'));
    vector<bool> column(n, false), ldiag(2*n-1, false), rdiag(2*n-1, false);
    backtrack(ret, board, column, ldiag, rdiag, 0, n);  
    return ret;  
}
```

### 53. Maximum Subarray

Given an integer array nums, find the contiguous subarray \(containing at least one number\) which has the largest sum and return its sum. If you have figured out the O\(n\) solution, try coding another solution using the divide and conquer approach, which is more subtle.

Example:

```text
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

Solution: \(1\) 变量dp，核心为 sum = max\(nums\[i\], sum + nums\[i\]\); maxSum = max\(maxSum, sum\); 本题有一点要非常注意：这种需要连续片段的题\(见Q152\)，dp时的i或j不应该表示区间范围，而应该是以i或j终止的optimal情况。如本题max的情况里是一定要有nums\[i\]的，即i此处表示以i结尾的最大和片段；否则下一次遍历时会有空隔。\(2\) divide and conquer: T\(n\) = 2T\(n / 2\) + O\(1\), 详见解

```cpp
// dp
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    if (!n) return 0;
    int sum = nums[0];
    int maxSum = sum;
    for (int i = 1; i < n; ++i) {
        sum = max(nums[i], sum + nums[i]);
        maxSum = max(maxSum, sum);
    }
    return maxSum;
}

/*
 * divide and conquer
 * T(n) = 2T(n / 2) + O(1)
 *  mx: largest sum of this subarray
 * lmx: largest sum starting from the left most element
 * rmx: largest sum ending with the right most element
 * sum: the sum of the total subarray
 */
void maxSubArray(vector<int>& nums, int l, int r, int& mx, int& lmx, int& rmx, int& sum) {
    if (l == r) {
        mx = lmx = rmx = sum = nums[l];
    } else {
        int m = (l + r) / 2;
        int mx1, lmx1, rmx1, sum1;
        int mx2, lmx2, rmx2, sum2;
        maxSubArray(nums, l, m, mx1, lmx1, rmx1, sum1);
        maxSubArray(nums, m + 1, r, mx2, lmx2, rmx2, sum2);
        mx = max(max(mx1, mx2), rmx1 + lmx2);
        lmx = max(lmx1, sum1 + lmx2);
        rmx = max(rmx2, sum2 + rmx1);
        sum = sum1 + sum2;
    }
}
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    if (!n) return 0;
    int mx, lmx, rmx, sum;
    maxSubArray(nums, 0, n-1, mx, lmx, rmx, sum);
    return mx;
}
```

### 54. Spiral Matrix

Given a matrix of m x n elements \(m rows, n columns\), return all elements of the matrix in spiral order.

Example:

```text
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```

Solution: 对于每一层（记为层i），以\(i,i\)位置出发，向右到达\(i,n-1-i\)，向下到达\(m-1-i,n-1-i\)，向左到达\(m-1-i,i\)，再向上回到起点。所有层次遍历完成后，即得到所求数组。注意：i=m-1-i 或者 i=n-1-i时，不要来回重复遍历

```cpp
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> ret;
    if (matrix.empty() || matrix[0].empty()) return ret;
    int m = matrix.size();
    int n = matrix[0].size();
    int layer = (min(m,n)+1) / 2;
    for (int i = 0; i < layer; ++i) {
        //row i: top-left --> top-right
        for (int j = i; j < n-i; ++j) ret.push_back(matrix[i][j]);

        //col n-1-i: top-right --> bottom-right
        for (int j = i+1; j < m-i; ++j) ret.push_back(matrix[j][n-1-i]);

        //row m-1-i: bottom-right --> bottom-left
        if (m-1-i > i) for (int j = n-1-i-1; j >= i; --j) ret.push_back(matrix[m-1-i][j]);

        //col i: bottom-left --> top-left
        if (n-1-i > i) for (int j = m-1-i-1; j > i; --j) ret.push_back(matrix[j][i]);
    }
    return ret;
}
```

### 55. Jump Game

Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Determine if you are able to reach the last index.

Example:

```text
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
```

Solution: 从后往前贪心，对于当前位置i，向左找，如果能到达i就跳；判断最终位置是否为0

```cpp
bool canJump(vector<int>& nums) {
    int lastPos = nums.size() - 1;
    for (int i = lastPos; i >= 0; --i) {
        if (i + nums[i] >= lastPos) {
            lastPos = i;
        }
    }
    return !lastPos;
}
```

### 56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example:

```text
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

Solution: 先把区间按开始数字大小排序，然后merge，判断下一个的start是否在上一个的end之前

```cpp
vector<Interval> merge(vector<Interval>& intervals) {
    vector<Interval> ret;
    if (intervals.empty()) return ret;
    sort(intervals.begin(), intervals.end(), [](Interval a, Interval b) {return a.start < b.start;});
    ret.push_back(intervals[0]);
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i].start <= ret.back().end)
            ret.back().end = max(ret.back().end, intervals[i].end);
        else
            ret.push_back(intervals[i]);
    }
    return ret;
}
```

### 57. Insert Interval

Given a set of non-overlapping intervals, insert a new interval into the intervals \(merge if necessary\). You may assume that the intervals were initially sorted according to their start times.

Example:

```text
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
```

Solution: 三种情况：新区间开头比当前区间结尾都大则过，新区间结尾比当前区间开头小则直接插入，若有并集则合并start和end，继续考虑下一个区间

```cpp
vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
    int start = newInterval.start;
    int end = newInterval.end;
    vector<Interval> ans;
    for (int i = 0; i < intervals.size(); ++i) {
        if (intervals[i].end < start) {
            ans.push_back(intervals[i]);
        }
        else if (intervals[i].start > end) {
            ans.push_back(Interval(start, end));
            copy(intervals.begin() + i, intervals.end(), back_inserter(ans));
            return ans;
        }
        else {
            start = min(start, intervals[i].start);
            end = max(end, intervals[i].end);
        }
    }
    ans.push_back(Interval(start, end));
    return ans;
}
```

### 58. Length of Last Word

Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string. If the last word does not exist, return 0. Note: A word is defined as a character sequence consists of non-space characters only.

Example:

```text
Input: "Hello World"
Output: 5
```

Solution: 从后往前遍历，不要忘记考虑字符串最后有多余空格的情况

```text
int lengthOfLastWord(string s) {
    int len = 0;
    int i = s.size() - 1;
    while (i >= 0 && s[i] == ' ') {
        --i;
    }
    while (i >= 0 && s[i] != ' ') {
        ++len;
        --i;
    }
    return len;
}
```

### 59. Spiral Matrix II

Given a positive integer n, generate a square matrix filled with elements from 1 to n^2 in spiral order.

Example:

```text
Input: 3
Output:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
```

Solution: 类似Q54，但是要考虑奇偶n

```cpp
vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0));
    int layer = n/2, cnt = 1;
    for (int t = 0; t < layer; ++t) {
        int walk_i = t, walk_j = t;
        while (walk_j+1 < n-t) res[walk_i][walk_j++] = cnt++;
        while (walk_i+1 < n-t) res[walk_i++][walk_j] = cnt++;
        while (walk_j > t) res[walk_i][walk_j--] = cnt++;
        while (walk_i > t) res[walk_i--][walk_j] = cnt++;
    }
    if (n%2) res[layer][layer] = cnt++;
    return res;
}
```

### 60. Permutation Sequence

Given n and k, return the k-th permutation sequence. Note: Given n will be between 1 and 9 inclusive; Given k will be between 1 and n! inclusive.

Example:

```text
Input: n = 3, k = 3
Output: "213"

Input: n = 4, k = 9
Output: "2314"
```

Solution: 从左往右逐位确定数字：第k个排列的第一个元素在0-n中的位置为\(k-1\)/fac\(n-1\)，[详见](http://www.cnblogs.com/houkai/p/3675270.html)

```cpp
string getPermutation(int n, int k) {
    string vec;
    string answer;
    int fact = 1;
    --k;
    for (int i = 1; i < n + 1; ++i) {
        vec.push_back(i + '0');
        fact *= i;
    }
    answer.resize(n);
    for (int i = 0; i < n; ++i) {
        fact /= (n - i);
        int tmp = k / fact;
        answer[i] = vec[tmp];
        vec.erase(vec.begin() + tmp);
        k -= tmp * fact;
    }
    return answer;
}
```

### 61. Rotate List

Given a linked list, rotate the list to the right by k places, where k is non-negative.

Example:

```text
Input: 1->2->3->4->5->NULL, k = 2
Output: 4->5->1->2->3->NULL

Input: 0->1->2->NULL, k = 4
Output: 2->0->1->NULL
```

Solution: 先遍历一遍算长度，再遍历一遍找节点

```cpp
ListNode *rotateRight(ListNode *head, int k) {  
    if (!head || !k) return head;

    // first loop over
    int length = 1;
    ListNode *node = head;
    while (node->next) {
        ++length;
        node = node->next;  
    }
    node->next = head;

    // second loop over
    int m = length - k % length;
    for (int i = 0; i < m; ++i) node = node->next;
    head = node->next;
    node->next = NULL;

    return head;  
}
```

### 62. Unique Paths

A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. How many possible unique paths are there?

Example:

```text
Input: m = 7, n = 3
Output: 28
```

Solution: 结果是C\(m+n-2, m-1\)。dp计算组合数: \(1\) C\(n, 0\) = C\(n, n\) = 1 for all n &gt; 0; \(2\) C\(n, k\) = C\(n − 1, k − 1\) + C\(n − 1, k\) for all 0 &lt; k &lt; n。可以缩成一维计算

```cpp
int uniquePaths(int m, int n) {
    return comb(m+n-2, m-1);
}
int comb(int n, int k) {
    if (!k) return 1;
    vector<int> result (n+1, 1);
    for (int i=1; i<=n; ++i) for (int j=i-1; j>=1; --j) result[j] += result[j-1];
    return result[k];
}
```

### 63. Unique Paths II

A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. Now consider if some obstacles are added to the grids. How many unique paths would there be?

Example:

```text
Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2
```

Solution: 类似Q62, 在dp上加一层判断是否有障碍，有则到当前格子的sum变成0

```cpp
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    vector<int> result(obstacleGrid.size() + 1, 0);
    int m = obstacleGrid.size() - 1;
    int n = obstacleGrid[0].size() - 1;

    result[m] = 1;

    for (int j = n; j >= 0; --j) {
        for (int i = m; i >= 0; --i) {
            if (obstacleGrid[i][j]) result[i] = 0;
            else result[i] += result[i + 1];
        }
    }

    return result[0];
}
```

### 64. Minimum Path Sum

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path. Note: You can only move either down or right at any point in time.

Example:

```text
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
```

Solution: dp: min\_sum = min\(left\_sum, up\_sum\) + cur

```cpp
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<int> dp(n, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == 0 && i == 0) dp[j] = grid[i][j];
            else if (j == 0) dp[j] = dp[j] + grid[i][j];
            else if (i == 0) dp[j] = dp[j-1] + grid[i][j];
            else dp[j] = std::min(dp[j], dp[j-1]) + grid[i][j];
        }
    }
    return dp[n-1];
}
```

### 66. Plus One

Given a non-empty array of digits representing a non-negative integer, plus one to the integer. The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit. You may assume the integer does not contain any leading zero, except the number 0 itself.

Example:

```text
Input: [1,2,3]
Output: [1,2,4]
```

Solution: 正常操作，注意进位和9999的情况

```cpp
vector<int> plusOne(vector<int>& digits) {
    int n = digits.size() - 1;
    while (n >= 0) {
        if (digits[n]<9) {
            ++digits[n];
            return digits;
        }
        digits[n--] = 0;
    }
    vector<int> ret(digits.size()+1, 0);
    ret[0] = 1;
    return ret;
}
```

### 67. Add Binary

Given two binary strings, return their sum \(also a binary string\). The input strings are both non-empty and contains only characters 1 or 0.

Example:

```text
Input: a = "1010", b = "1011"
Output: "10101"
```

Solution: 正常操作，注意进位

```cpp
string addBinary(string a, string b, int asz, int bsz) {
    int carry = 0;
    for (int i = 0; i < bsz; i++) {
        a[asz - 1 - i] = a[asz - 1 - i] - '0' + b[bsz - 1 - i] - '0' + carry + '0';
        if (a[asz - 1 - i] >= '2') {
            a[asz - 1 - i] -= 2;
            carry = 1;
        } else {
            carry = 0;
        }
    }
    if (carry == 0) return a;
    for (int i = bsz; i < asz; i++) {
        a[asz - 1 - i] = a[asz - 1 - i] - '0' + carry + '0';
        if (a[asz - 1 - i] == '2') {
            a[asz - 1 - i] = '0';
            carry = 1;
        } else {
            return a;
        }
    }
    return carry ? to_string(1) + a : a; 
}
string addBinary (string a, string b) {
    int asz = a.size(), bsz = b.size();
    return asz >= bsz ? addBinary(a, b, asz, bsz): addBinary(b, a, bsz, asz);
}
```

### 68. Text Justification

Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully \(left and right\) justified. You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters. Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right. For the last line of text, it should be left justified and no extra space is inserted between words. Note: A word is defined as a character sequence consisting of non-space characters only; Each word's length is guaranteed to be greater than 0 and not exceed maxWidth; The input array words contains at least one word.

Example:

```text
Input:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do","haha"]
maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do haha             "
]
```

Solution: 正常操作

```cpp
vector<string> fullJustify(vector<string>& words, int maxWidth) {
    vector<string> rs;  
    int L = maxWidth;  

    for (int i = 0; i < words.size();) {    
        int j = i+1;    
        int len = words[i].length();    
        for (; j < words.size() && len+words[j].length()<L; j++)    
            len += 1 + words[j].length();    

        if (j == words.size()) {    
            string s(words[i]);    
            for (i++ ; i < j; i++) s +=" "+words[i];    
            while (s.length() < L) s.push_back(' ');    
            rs.push_back(s);    
            return rs;    
        }    
        if (j-i == 1) {    
            rs.push_back(words[i++]);    
            rs.back().append(L-rs.back().length(), ' ');    
            continue;    
        }    

        int a = (L-len) / (j-i-1) + 1;    
        int b = (L-len) % (j-i-1);    
        string s(words[i]);    
        for (i++; i < j; i++, b--) {    
            s.append(a,' ');    
            if (b>0) s.push_back(' ');    
            s.append(words[i]);    
        }    
        rs.push_back(s);    
    }    
    return rs;
}
```

### 69. Sqrt

Implement int sqrt\(int x\). Compute and return the square root of x, where x is guaranteed to be a non-negative integer. Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example:

```text
Input: 4
Output: 2

Input: 8
Output: 2
```

Solution: 牛顿迭代法x\_n+1 = x\_n - f\(x\_n\)/f'\(x\_n\), 其中f = x^2 - a = 0

```cpp
int mySqrt(int sq) {
    long x = sq, a = sq;
    while (x * x > a) {
        x = (x + a / x) / 2;
    }
    return x;
}
```

### 70. Climbing Stairs

You are climbing a stair case. It takes n steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top? Note: Given n will be a positive integer.

Example:

```text
Input: 3
Output: 3
```

Solution: fibonacci

```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int pre2 = 1, pre1 = 2, cur;
    for (int i = 2; i < n; ++i) {
        cur = pre1 + pre2;
        pre2 = pre1;
        pre1 = cur;
    }
    return pre1;
}
```

### 71. Simplify Path

Given an absolute path for a file \(Unix-style\), simplify it.

Example:

```text
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"

Did you consider the case where path = "/../"?
In this case, you should return "/".

Another corner case is the path might contain multiple slashes '/' together, such as "/home//foo/".
In this case, you should ignore redundant slashes and return "/home/foo".
```

Solution: stack，考虑corner cases

```cpp
string simplifyPath(string path) {
    if (!path.size()|| path[0] != '/') return "";

    stack<string> ans;
    int last = 1, idx = 1;
    do {
        idx = path.find("/", last);
        idx = idx == -1? path.size(): idx;

        if (idx - last == 0) {
            last = idx + 1;
            continue;
        }

        string tmp = path.substr(last, idx - last);
        last = idx + 1;

        if (tmp.size() == 1 && tmp == ".") continue;
        if (tmp.size() == 2 && tmp == "..") {
            if (!ans.empty()) ans.pop();
            continue;
        }
        ans.push(tmp);
    } while (idx != path.size());

    if (ans.empty()) return "/";

    string res = "";
    while (!ans.empty()) {
        res = "/" + ans.top() + res;
        ans.pop();
    }
    return res;
}
```

### 72. Edit Distance

Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2. You have the following 3 operations permitted on a word: Insert a character; Delete a character; Replace a character.

Example:

```text
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
```

Solution: dp\[i\]\[j\]=min\(dp\[i-1\]\[j-1\]+\(S\[i\]==T\[j\]?0,1\),dp\[i-1\]\[j\]+1,dp\[i\]\[j-1\]+1\), 详见代码

```cpp
int minDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> distance(m + 1, vector<int>(n + 1));
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (!i) distance[i][j] = j;
            else if (!j) distance[i][j] = i;
            else distance[i][j] = min(
                distance[i-1][j-1] + ((word1[i-1] == word2[j-1])? 0: 1),
                min(distance[i-1][j] + 1, distance[i][j-1] + 1));
        }        
    }
    return distance[m][n];
}
```

### 73. Set Matrix Zeroes

Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Example:

```text
Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

Solution: 用第一行第一列做Buffer，不过需要从后往前遍历，防止修改过后的行列影响检测

```cpp
void setZeroes(vector<vector<int>>& matrix) {
    int col0 = 1;
    for (int i = 0; i < matrix.size(); ++ i) {
        if (!matrix[i][0]) col0 = 0;
        for (int j = 1; j < matrix[i].size(); ++j)
            if (!matrix[i][j]) matrix[i][0] = matrix[0][j] = 0;
    }
    // i需要从后往前，否则matrix[0][j]的值可能被改变
    for (int i = matrix.size() - 1; i >= 0; --i) {
        for (int j = 1; j < matrix[i].size(); ++ j)
            if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
        if (col0 == 0) matrix[i][0] = 0;
    }
}
```

### 74. Search a 2D Matrix

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties: \(1\) Integers in each row are sorted from left to right. \(2\) The first integer of each row is greater than the last integer of the previous row.

Example:

```text
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
```

Solution: 二分法

```cpp
bool binarySearch(const vector<vector<int>> &matrix, int l, int r, int target, int n) {
    if (l >= r && matrix[l/n][l%n] != target) return false;
    int mid = l + (r-l)/2;
    if (target > matrix[mid/n][mid%n]) return binarySearch(matrix, mid+1, r, target, n);
    if (target < matrix[mid/n][mid%n]) return binarySearch(matrix, l, mid, target, n);
    if (target == matrix[mid/n][mid%n]) return true;
}

bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    if (!m) return false;
    int n = matrix[0].size();
    if (!n) return false;
    int l = 0, r = n*m-1;
    return binarySearch(matrix, l, r, target, n);
}
```

### 75. Sort Colors

Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue. Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Example:

```text
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```

Solution: 维护三个数low=0, mid=0, high=size-1；然后while \(mid &lt; high\)判断a\[mid\]，如果是0，交换a\[mid\]和a\[low\]，然后low和mid各++；如果是1，则mid++；如果是2，则交换a\[mid\]和a\[high\]，然后high--

```cpp
void sortColors(vector<int>& vec) {
    int low = 0, mid = 0, high = vec.size() - 1;
    while (mid <= high) {
        switch(vec[mid]) {
        case 0:
            swap(vec[low], vec[mid]);
            low++; mid++; break;
        case 1:
            mid++; break;
        case 2:
            swap(vec[mid], vec[high]);
            high--; break;  
        }
    }
}
```

### 76. Minimum Window Substring

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O\(n\). Note: If there is no such window in S that covers all characters in T, return the empty string ""; If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

Example:

```text
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
```

Solution: 设置一个char c\[128\]记录字符使用情况。设下标l=0和r=1，把左开右闭\[l, r\)想象成一个窗口。当窗口包含所有T中的字符时，则此时的窗口是一个可行解，l++; 当窗口没有包含所有T中的字符时，则r++

```cpp
string minWindow(string S, string T) {
    int c[128] = {0};
    bool flag[128] = {false};
    for (int i = 0; i < T.size(); ++i) {
        flag[T[i]] = true;
        ++c[T[i]];
    }
    int cnt = 0, l = 0, minl = 0, minsize = S.size() + 1;
    for (int r = 0; r < S.size(); ++ r) {
        if (flag[S[r]]) {
            if (--c[S[r]] >= 0) ++cnt;
            while (cnt == T.size()) {
                if (r - l + 1 < minsize) minl = l, minsize = r - l + 1;
                if (flag[S[l]] && ++c[S[l]] > 0) --cnt;
                ++l;
            }
        }
    }
    if (minsize > S.size()) return "";
    return S.substr(minl, minsize);
}
```

### 77. Combinations

Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

Example:

```text
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

Solution: backtrack

```cpp
void myCombine(vector<vector<int>>& res, vector<int>& cur, int pos, int count, int k, int n) {
    if (!k) {
        res.push_back(cur);
        return;
    }
    for (int i = pos; i <= n; i++) {
        cur[count] = i;
        myCombine(res, cur, i + 1, count + 1, k - 1, n);
    }
}
vector<vector<int>> combine(int n, int k) {
    vector<vector<int>> res;
    vector<int> cur(k, 0);
    myCombine(res, cur, 1, 0, k, n);
    return res;
}
```

### 78. Subsets

Given a set of distinct integers, nums, return all possible subsets \(the power set\). Note: The solution set must not contain duplicate subsets.

Example:

```text
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

Solution: bit manipulation或backtrack，一定要背

```cpp
// bit manipulation
vector<vector<int>> subsets(vector<int>& nums) {
    int cnt = 0, n = nums.size();
    int sub = 1 << n;
    vector<vector<int>> result(sub, vector<int>());
    for (int i = 0; i < n; i ++)
        for (int j = 0; j < sub; j++)
            if (j >> i & 1)
                result[j].push_back(nums[i]);
    return result;
}

// backtrack
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> t;
    helper(nums, res, t, 0);
    return res;
}
void helper(vector<int> & nums, vector<vector<int>>& res, vector<int> & t, int begin) {
    res.push_back(t);
    for (int i = begin; i < nums.size(); i++) {
        t.push_back(nums[i]);
        helper(nums, res, t, i+1);
        t.pop_back();
    }   
}
```

### 79. Word Search

Given a 2D board and a word, find if the word exists in the grid. The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:

```text
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

Solution: dfs，用一个额外2d矩阵表示是否搜过，这个矩阵backtrack更新，一定要背

```cpp
bool exist(vector<vector<char>>& board, string word) {
    if (board.empty()) return false;
    int m = board.size(), n = board[0].size();
    vector<vector<bool>> searched(m, vector<bool>(n, false));
    bool find = false;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            dfs(i, j, board, word, 0, find, searched);
    return find;
}
void dfs(int i, int j, vector<vector<char>>& board, string& word, int pos, bool& find, vector<vector<bool>>& searched) {
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size()) return;
    if (searched[i][j] || find || board[i][j] != word[pos]) return;
    if (pos == word.size()-1) { 
        find = true;
        return;
    }
    searched[i][j] = true;

    dfs(i+1, j, board, word, pos+1, find, searched);
    dfs(i-1, j, board, word, pos+1, find, searched);
    dfs(i, j+1, board, word, pos+1, find, searched);
    dfs(i, j-1, board, word, pos+1, find, searched);
    searched[i][j] = false;
}
```

### 80. Remove Duplicates from Sorted Array II

Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length. Do not allocate extra space for another array, you must do this by modifying the input array in-place with O\(1\) extra memory.

Example:

```text
Input: [1,1,1,2,2,3]
Output: 5, first five elems of the array should be [1,1,2,2,3]
```

Solution: Q26变形，双指针推进，比较r和l-2

```cpp
int removeDuplicates(vector<int>& nums) {
    int i = 0;
    for (int n : nums)
        if (i < 2 || n > nums[i-2])
            nums[i++] = n;
    return i;
}
// or
int removeDuplicates(vector<int>& nums) {
    int n = nums.size();
    if (n <= 2) return n;
    int l = 2, r = 1;
    while (++r < n)
        if (nums[r] > nums[l-2])
            nums[l++] = nums[r];
    return l;
}
```

### 81. Search in Rotated Sorted Array II

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. You are given a target value to search. If found in the array return true, otherwise return false.

Example:

```text
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
```

Solution: Q33变形\(加了重复数字\)，多考虑两个corner case

```cpp
bool search(vector<int>& nums, int target) {
    int start = 0, end = nums.size() - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (nums[mid] == target) return true;
        if (nums[start] == nums[mid]) {
            // new in Q81: cannot tell if start == mid
            start++;
        } else if (nums[mid] <= nums[end]) {
            // right half sorted; new in Q81: less than -> less than or equal
            if (target > nums[mid] && target <= nums[end]) start = mid+1;
            else end = mid-1;
        } else {
            // left half sorted
            if (target >= nums[start] && target < nums[mid]) end = mid-1;
            else start = mid+1;
        }
    }
    return false;
}
```

### 82. Remove Duplicates from Sorted List II

Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

Example:

```text
Input: 1->2->3->3->4->4->5
Output: 1->2->5
```

Solution: 可以双指针推进，但是为了删掉额外内存，可以引入第三个指针

```cpp
ListNode* deleteDuplicates(ListNode* head) {
    if (!head) return head;
    ListNode pre(head->val-1);
    pre.next = head;
    ListNode *l = &pre, *r = l, *rnext;
    while (r) {
        bool isDuplicate = false;
        rnext = r->next;
        while (rnext && r->val == rnext->val) {
            isDuplicate = true;
            r->next = rnext->next;
            delete rnext;
            rnext = r->next;
        }
        if (isDuplicate) {
            l->next = r->next;
            delete r;
            r = l->next;
        } else {
            l = r;
            r = r->next;
        }
    }
    return pre.next;
}
```

### 83. Remove Duplicates from Sorted List

Given a sorted linked list, delete all duplicates such that each element appear only once.

Example:

```text
Input: 1->1->2
Output: 1->2
```

Solution: 可以单指针推进，但是为了删掉额外内存，可以引入第二个指针

```cpp
ListNode* deleteDuplicates(ListNode* head) {
    if (!head) return head;
    ListNode *l = head, *r;
    while (l) {
        r = l->next;
        while (r && l->val == r->val) {
            l->next = r->next;
            delete r;
            r = l->next;
        }
        l = r;
    }
    return head;
}
// or with recursion
ListNode* deleteDuplicates(ListNode* head) {
    if (!head|| !head->next) return head;
    head->next = deleteDuplicates(head->next);
    if (head->val == head->next->val) {
        ListNode* next = head->next;
        delete head;
        head = next;
    }
    return head;
}
```

### 84. Largest Rectangle in Histogram

Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

Example:

```text
Input: [2,1,5,6,2,3]
Output: 10
```

Solution: stack, 1D推进: \(1\) 在height尾部添加一个0，也就是一个高度为0的立柱。作用是在最后也能凑成“波峰图”。\(2\) 定义了一个stack，然后遍历时如果height\[i\] 大于stack.top\(\)，进栈。反之，出栈直到栈顶元素小于height\[i\]。由于出栈的这些元素高度都是递增的，我们可以求出这些立柱中所围成的最大矩形。\(3\) 由于比height\[i\]大的元素都出完了，height\[i\]又比栈顶元素大了，因此再次进栈。如此往复，直到遍历到最后那个高度为0的柱，触发最后的弹出以及最后一次面积的计算，此后stack为空。\(4\) 返回面积最大值。注意栈中存的不是高度，而是height的索引，这样做的好处是不会影响宽度的计算，索引值相减 = 宽度

```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<int> area;
    heights.push_back(0);
    int result = 0;

    for (int i = 0; i < heights.size();) {
        if (area.empty() || heights[i] > heights[area.top()]) {
            area.push(i++);
        } else {
            int tmp = area.top();
            area.pop();
            result = max(result, heights[tmp] * (area.empty() ? i : i - area.top() - 1));
        }
    }
    return result;
}
```

### 85. Maximal Rectangle

Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Example:

```text
Input:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

Solution: 对Q84每行扫一遍; 也可以用dp\(参考最大正方形\), 不过要额外记录边长

```cpp
int maximalRectangle(vector<vector<char>>& matrix) {
    if (!matrix.size()|| !matrix[0].size()) return 0;
    int H = matrix.size(), W = matrix[0].size();
    int height[W+1];
    int i, j, result = 0;
    stack<int> area;
    for (j = 0; j <= W; height[j] = 0, ++j);
    for (i = 0; i < H; ++i) {
        while (!area.empty()) area.pop();
        for (j = 0; j < W; ++j) {
            if (matrix[i][j] == '1') height[j]++;
            else height[j] = 0;
        }
        for (j = 0; j <= W;) {
            if (area.empty() || height[j] > height[area.top()]) {
                area.push(j++);
            } else {
                int tmp = area.top();
                area.pop();
                result = max(result, height[tmp] * (area.empty() ? j : j - area.top() - 1));
            }
        }
    }
    return result;
}
```

### 86. Partition List

Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x. You should preserve the original relative order of the nodes in each of the two partitions.

Example:

```text
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```

Solution: 新建两个指针引导，注意要在最前面加个dummy指针，一定要背

```cpp
ListNode* partition(ListNode* head, int x) {
    ListNode* dummyLeft = new ListNode(-1);
    ListNode* dummyRight = new ListNode(-1);
    ListNode* leftCur = dummyLeft;
    ListNode* rightCur = dummyRight;
    while (head) {
        if (head->val < x) {
            leftCur->next = head;
            leftCur = leftCur->next;
        }
        else{
            rightCur->next=head;
            rightCur=rightCur->next;
        }
        head = head->next;
    }
    leftCur->next = dummyRight->next;
    rightCur->next = nullptr;
    return dummyLeft->next;

}
```

### 87. Scramble String

Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively. Below is one possible representation of s1 = "great":

```text
    great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
```

To scramble the string, we may choose any non-leaf node and swap its two children. For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".

```text
    rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
```

We say that "rgeat" is a scrambled string of "great". Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".

```text
    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
```

We say that "rgtae" is a scrambled string of "great". Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.

Example:

```text
Input: s1 = "great", s2 = "rgeat"
Output: true
```

Solution: 法1：递归：对于每个位置i，判断 if \(\(isScramble\(s1.substr\(0, i\), s2.substr\(0, i\)\) \&& isScramble\(s1.substr\(i\), s2.substr\(i\)\)\)\|\| \(isScramble\(s1.substr\(0, i\), s2.substr\(s1.size\(\)-i\)\) && isScramble\(s1.substr\(i\), s2.substr\(0, s1.size\(\)-i\)\)\)\) return true; 法2:：dp：使用三维数组boolean result\[len\]\[len\]\[len\], 其中第一维为子串的长度，第二维为s1的起始索引，第三维为s2的起始索引。result\[k\]\[i\]\[j\]表示s1\[i...i+k\]是否可以由s2\[j...j+k\]变化得来

```cpp
bool isScramble(string s1, string s2) {
    if (s1 == s2) return true;
    if (s1.size() != s2.size()) return false;
    vector<int> count(26, 0);
    for (int i = 0; i < s1.size(); ++i) {
        ++count[s1[i]-'a'];
        --count[s2[i]-'a'];
    }
    for (int i = 0; i < 26; ++i) if (count[i]) return false;
    for (int i = 1; i < s1.size(); ++i) 
        if ((isScramble(s1.substr(0, i), s2.substr(0, i)) && isScramble(s1.substr(i), s2.substr(i)))|| 
         (isScramble(s1.substr(0, i), s2.substr(s1.size()-i)) && isScramble(s1.substr(i), s2.substr(0, s1.size()-i))))
            return true;
    return false;
}
```

### 88. Merge Sorted Array

Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array. You may assume that nums1 has enough space to hold additional elements from nums2.

Example:

```text
Input: num1 = [1,2,3,0,0,0], num2 = [2,5,6]
Output: [1,2,2,3,5,6]
```

Solution: 从后往前merge，不要忘了还剩余一段的情况，一定要背和练习

```cpp
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int pos = m-- + n-- - 1;
    while (m >= 0 && n >= 0) nums1[pos--] = nums1[m] > nums2[n]? nums1[m--]: nums2[n--];
    while (n >= 0) nums1[pos--] = nums2[n--];
}
```

### 89. Gray Code

The gray code is a binary numeral system where two successive values differ in only one bit. Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0. Note: For a given n, a gray code sequence is not uniquely defined. For example, \[0,2,3,1\] is also a valid gray code sequence for the following example.

Example:

```text
Input: n = 2
Output: [0,1,3,2]
Note: Its gray code sequence is:
00 - 0
01 - 1
11 - 3
10 - 2
```

Solution: 规律是flip第一位，剩余位数是上下对称的

```cpp
vector<int> grayCode(int n) {
    if (!n) return vector<int>{0};
    if (n == 1) return vector<int>{0, 1};
    vector<int> prev = grayCode(n-1);
    int base = 1 << (n-1);
    for (int i = 0; i < base; ++i) prev.push_back(prev[base-i-1] + base);
    return prev;
}
```

### 90. Subsets II

Given a collection of integers that might contain duplicates, nums, return all possible subsets \(the power set\). Note: The solution set must not contain duplicate subsets.

Example:

```text
Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

Solution: 对没重复Q78的做法，先排序，然后加last或者去重\(同样数字左边或右边的组合都舍弃\)做backtract

```cpp
vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> t;
    sort(nums.begin(), nums.end());  // new in Q90
    helper(nums, res, t, 0);
    return res;
}
void helper(vector<int> & nums, vector<vector<int>>& res, vector<int> & t, int begin) {
    res.push_back(t);
    for (int i = begin; i < nums.size(); i++) {
        if (i != begin && nums[i] == nums[i-1]) continue;  // new in Q90
        t.push_back(nums[i]);
        helper(nums, res, t, i+1);
        t.pop_back();
    }   
}
```

### 91. Decode Ways

A message containing letters from A-Z is being encoded to numbers using the following mapping:

```text
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example:

```text
Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).

Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

Solution: fibonacci dp, 但是因为字符串不保证合法\(有0的情况\)和需要考虑31和27这些超范围的组合, 所以要做if-else处理

```cpp
int numDecodings(string s) {
    int n = s.length();
    if (!n) return 0;
    int prev = s[0] - '0';
    if (!prev) return 0;
    if (n == 1) return 1;
    vector<int> dp(n+1, 1);
    for (int i = 2; i <= n; ++i) {
        int cur = s[i-1] - '0';
        if ((prev == 0 || prev > 2) && cur == 0) return 0;
        if ((prev < 2 && prev > 0) || prev == 2 && cur < 7) {
            if (cur) dp[i] = dp[i-2] + dp[i-1];
            else dp[i] = dp[i-2];
        }
        else dp[i] = dp[i-1];
        prev = cur;
    }
    return dp.back();
}
```

### 92. Reverse Linked List II

Reverse a linked list from position m to n. Do it in one-pass. Assume 1 &lt;= m &lt;= n &lt;= len.

Example:

```text
Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
```

Solution: 三指针，很重要一定要背!

```cpp
ListNode* reverseBetween(ListNode* head, int m, int n) {
    ListNode* dummy = new ListNode(-1);
    dummy->next = head;
    ListNode* left = dummy, *right;
    n -= m;
    while (--m) left = left->next;
    head = left->next;
    while (n--) {
        right = head->next->next;       // left -> leftnext -> ... -> head -> headnext -> right
        head->next->next = left->next;  // left -> leftnext -> ... -> head -> headnext -> leftnext right
        left->next = head->next;        // left -> headnext -> leftnext -> ... -> head -> headnext right
        head->next = right;             // left -> headnext -> leftnext -> ... -> head -> right
    }
    return dummy->next;
}
```

### 94. Binary Tree Inorder Traversal

Given a binary tree, return the inorder traversal of its nodes' values.

Example:

```text
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
```

Solution: 分治、stack、或者[Morris Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/solution/)\，一定要背

```cpp
// divide and conquer
vector<int> inorderTraversal(TreeNode* root) {
    if (!root) return vector<int>();
    vector<int> left = inorderTraversal(root->left);
    vector<int> right = inorderTraversal(root->right);
    left.push_back(root->val);
    left.insert(left.end(), right.begin(), right.end());
    return left;
}
// stack
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> ret;
    if (!root) return ret;
    stack<TreeNode*> s;
    TreeNode* cur = root;
    while (cur || !s.empty()) {
        while (cur) {
            s.push(cur);
            cur = cur->left;
        }
        TreeNode* node = s.top(); s.pop();
        ret.push_back(node->val);
        cur = node->right;
    }
    return ret;
}
// Morris Traversal
// Step 1: Initialize current as root
// Step 2: While current is not NULL,
// If current does not have left child
//   a. Add current’s value
//   b. Go to the right child
// Else
//   a. In current's left subtree, make current the right child of the rightmost node
//   b. Go to this left child
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> ret;
    TreeNode *cur = root, *pre;
    while (cur) {
        if (!cur->left) {
            ret.push_back(cur->val);
            cur = cur->right;
        } else {
            pre = cur->left;
            while (pre->right) pre = pre->right;
            pre->right = cur;
            TreeNode* temp = cur;
            cur = cur->left;
            temp->left = NULL;
        }
    }
    return ret;
}
```

### 95. Unique Binary Search Trees II

Given an integer n, generate all structurally unique BST's \(binary search trees\) that store values 1 ... n.

Example:

```text
Input: 3
Output:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
Explanation:
The above output corresponds to the 5 unique BST's shown below:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

Solution: 递归，一定要背

```cpp
vector<TreeNode*> generateTrees(int n) {
    if (!n) return vector<TreeNode*>();
    return genBST(1, n);
}

vector<TreeNode *> genBST(int min, int max) {
    vector<TreeNode*> ret;
    if (min > max) {
        ret.push_back(NULL);  // do not forget!!!
        return ret;
    }
    for (int i = min; i <= max; ++i) {
        vector<TreeNode*> leftBST = genBST(min, i-1);
        vector<TreeNode*> rightBST = genBST(i+1, max);
        // note: faster than using for loop with ijk
        for (const auto &l: leftBST) {
            for (const auto &r: rightBST) {
                TreeNode *root = new TreeNode(i);
                root->left = l;
                root->right = r;
                ret.push_back(root);
            }
        }
    }

    return ret;
}
```

### 96. Unique Binary Search Trees

Given n, how many structurally unique BST's \(binary search trees\) that store values 1 ... n?

Example:

```text
Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

Solution: f\(n\) = f\(0\)f\(n-1\) + f\(1\)f\(n-2\) + ... + f\(n-2\)f\(1\) + f\(n-1\)f\(0\)

```cpp
int numTrees(int n) {
    vector<int> dp(n+1,0);
    dp[0] = 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            dp[i] += dp[j] * dp[i-1-j];
        }
    }
    return dp[n];
}
```

### 97. Interleaving String

Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

Example:

```text
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
```

Solution: dp, 其实不难但要注意写法

```cpp
bool isInterleave(string s1, string s2, string s3) {
    int m = s1.length(), n = s2.length();
    if (m + n != s3.length()) return false;
    vector<vector<bool> > dp(m + 1, vector<bool> (n + 1, true)); 
    for (int i = 1; i <= m; ++i) dp[i][0] = dp[i-1][0] && (s3[i-1] == s1[i-1]); 
    for (int i = 1; i <= n; ++i) dp[0][i] = dp[0][i-1] && (s3[i-1] == s2[i-1]); 
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
            dp[i][j] = (dp[i-1][j] && s3[i+j-1] == s1[i-1]) || (dp[i][j-1] && s3[i+j-1] == s2[j-1]); 
    return dp[m][n];
}
```

### 98. Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree \(BST\). Note that in BST no duplication is allowed.

Example:

```text
Input:
    2
   / \
  1   3
Output: true
```

Solution: 递归带着一个大小区间，每次递归时更新这个区间，一定要背

```cpp
bool isValidBST(TreeNode* root) {
    return validateBST(root, LLONG_MIN, LLONG_MAX);
}
bool validateBST(TreeNode* node, long min, long max) {
    if (!node) return true;
    if (node->val <= min || node->val> = max) return false;
    return validateBST(node->left, min, node->val) && validateBST(node->right, node->val, max);
}
```

### 99. Recover Binary Search Tree

Two elements of a binary search tree \(BST\) are swapped by mistake. Recover the tree without changing its structure.

Example:

```text
Input: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

Output: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```

Solution: 递归中序遍历二叉树，设置一个prev指针，记录当前节点中序遍历时的前节点，如果当前节点大于prev节点的值，说明需要调整次序。有一个技巧是如果遍历整个序列过程中只出现了一次次序错误，说明就是这两个相邻节点需要被交换；如果出现了两次次序错误，那就需要交换这两个节点

```cpp
TreeNode *mistake1, *mistake2, *prev;
void recursive_traversal(TreeNode* root) {  
    if (!root) return;
    if (root->left) recursive_traversal(root->left);
    if (prev && root->val < prev->val) {  
        if (!mistake1) {  
            mistake1 = prev;  
            mistake2 = root;  
        } else {  
            mistake2 = root;  
        }  
    }  
    prev = root; // in-order traversal here!
    if (root->right) recursive_traversal(root->right);
}  

void recoverTree(TreeNode* root) {
    recursive_traversal(root);  
    if (mistake1 && mistake2) {  
        int tmp = mistake1->val;  
        mistake1->val = mistake2->val;  
        mistake2->val = tmp;  
    }  
}
```

### 100. Same Tree

Given two binary trees, write a function to check if they are the same or not.

Example:

```text
Input:     1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

Output: false
```

Solution: 递归

```cpp
bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p || !q) return p == q;
    return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}
```

