# LeetCode 701 - 750

### 714. Best Time to Buy and Sell Stock with Transaction Fee

Your are given an array of integers `prices`, for which the `i`-th element is the price of a given stock on day `i`; and a non-negative integer `fee` representing a transaction fee.

You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction. You may not buy more than 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)

Return the maximum profit you can make.

Example:

```
Input: prices = [1, 3, 2, 8, 4, 9], fee = 2
Output: 8 (The maximum profit can be achieved by:
Buying at prices[0] = 1
Selling at prices[3] = 8
Buying at prices[4] = 4
Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8)
```

Solution: 状态机

![](../.gitbook/assets/image%20%2810%29.png)

```cpp
int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    if (n == 0) return 0;
    vector<int> buy(n), sell(n), s1(n), s2(n);
    s1[0] = buy[0] = -prices[0];
    sell[0] = s2[0] = 0;
    for (int i = 1; i < n; i++) {
        buy[i] = max(sell[i-1], s2[i-1]) - prices[i];
        s1[i] = max(buy[i-1], s1[i-1]);
        sell[i] = max(buy[i-1], s1[i-1]) - fee + prices[i];
        s2[i] = max(s2[i-1], sell[i-1]);
    }
    return max(sell[n-1], s2[n-1]);
}
```

### 716. Max Stack

Design a max stack that supports push, pop, top, peekMax and popMax.

1. push(x) -- Push element x onto stack.
2. pop() -- Remove the element on top of the stack and return it.
3. top() -- Get the element on the top.
4. peekMax() -- Retrieve the maximum element in the stack.
5. popMax() -- Retrieve the maximum element in the stack, and remove it. If you find more than one maximum elements, only remove the top-most one.

Example:

```
MaxStack stack = new MaxStack();
stack.push(5); 
stack.push(1);
stack.push(5);
stack.top(); -> 5
stack.popMax(); -> 5
stack.top(); -> 1
stack.peekMax(); -> 5
stack.pop(); -> 1
stack.top(); -> 5
```

Solution: 类似LRU，不过把hashmap换成treemap，一定要背

```cpp
class MaxStack {
public:
    MaxStack() {
        hash.clear();
        stack.clear();
    }
    
    void push(int x) {
        stack.push_front(x);
        hash[x].push_back(stack.begin());
    }
    
    int pop() {
        int x = stack.front();
        hash[x].pop_back();
        if (hash[x].empty()) hash.erase(x);
        stack.pop_front();
        return x;
    }
    
    int top() {
        return stack.front();
    }
    
    int peekMax() {
        return hash.rbegin()->first;
    }
    
    int popMax() {
        int x = hash.rbegin()->first;
        stack.erase(hash[x].back());
        hash[x].pop_back();
        if (hash[x].empty()) hash.erase(x);
        return x;
    }
private:
    map<int, vector<list<int>::iterator>> hash;
    list<int> stack;
};
```

### 718. Maximum Length of Repeated Subarray

Given two integer arrays `A` and `B`, return the maximum length of an subarray that appears in both arrays.

Example:

```
Input: A: [1,2,3,2,1], B: [3,2,1,4,7]
Output: 3 ([3, 2, 1])
```

Solution: dp，十分经典，一定要背

```cpp
int findLength(vector<int>& A, vector<int>& B) {
    int res = 0, na = A.size(), nb = B.size();
    vector<vector<int>> dp(na, vector<int>(nb, 0));
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < nb; ++j) {
            if (i == 0 || j == 0) dp[i][j] = A[i] == B[j]? 1: 0;
            else dp[i][j] = A[i] == B[j]? dp[i-1][j-1] + 1: 0;
            res = max(res, dp[i][j]);
        }
    }
    return res;
}
```

### 727. Minimum Window Subsequence

Given strings `S` and `T`, find the minimum (contiguous) **substring** `W` of `S`, so that `T` is a **subsequence** of `W`. If there is no such window in `S` that covers all characters in `T`, return the empty string `""`. If there are multiple such minimum-length windows, return the one with the left-most starting index.

Example:

```
Input: S = "abcdebdde", T = "bde"
Output: "bcde" ("bcde" occurs before "bdde")
```

Solution: dp，十分巧妙，一定要背

```cpp
string minWindow(string S, string T) {
    int ns = S.size(), nt = T.size();
    vector<vector<int>> dp(nt, vector<int>(ns, 0));
    if (T[0] == S[0]) dp[0][0] = 1;
    for (int j = 1; j < ns; ++j) {
        if (T[0] != S[j]) dp[0][j] = dp[0][j-1] == 0? 0: dp[0][j-1] + 1;
        else dp[0][j] = 1;
    }
    for (int i = 1; i < nt; ++i) {
        for (int j = i; j < ns; ++j) {
            if (T[i] != S[j]) dp[i][j] = dp[i][j-1] == 0? 0: dp[i][j-1] + 1;
            else dp[i][j] = dp[i-1][j-1] == 0? 0: dp[i-1][j-1] + 1;
        }
    }
    string res = "";
    for (int j = 0; j < ns; ++j)
        if (dp[nt-1][j])
            if (res.empty() || dp[nt-1][j] < res.size())
                res = S.substr(j - dp[nt-1][j] + 1, dp[nt-1][j]);
    return res;
}
```

### 729. My Calendar I

Implement a `MyCalendar` class to store your events. A new event can be added if adding the event will not cause a double booking.

Your class will have the method, `book(int start, int end)`. Formally, this represents a booking on the half open interval `[start, end)`, the range of real numbers `x` such that `start <= x < end`.

A *double booking* happens when two events have some non-empty intersection (ie., there is some time that is common to both events.)

For each call to the method `MyCalendar.book`, return `true` if the event can be added to the calendar successfully without causing a double booking. Otherwise, return `false` and do not add the event to the calendar.

Example:

```
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(15, 25); // returns false
MyCalendar.book(20, 30); // returns true
```

Solution: 用一个map<start, end>即可

```cpp
class MyCalendar {
private:
    map<int, int> schedule;
public:
    MyCalendar() {
        schedule.clear();
    }
    bool book(int start, int end) {
        auto it = schedule.lower_bound(start);
        if (it != schedule.begin()) {
            auto last = next(it, -1);
            if (last->second > start) return false;   
        }
        if (it != schedule.end() && end > it->first) return false;
        schedule[start] = end;
        return true;
    }
};
```

### 733. Flood Fill

An `image` is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535). Given a coordinate `(sr, sc)` representing the starting pixel (row and column) of the flood fill, and a pixel value `newColor`, "flood fill" the image.

To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.

Example:

```
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
```

Solution: dfs

```cpp
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
    if (image[sr][sc] != newColor) helper(image, sr, sc, image[sr][sc], newColor);
    return image;
}

void helper(vector<vector<int>>& image, int i, int j, int oldColor, int newColor) {
    if (i < 0 || j < 0 || i == image.size() || j == image[0].size() || image[i][j] != oldColor) return;
    image[i][j] = newColor;
    helper(image, i - 1, j, oldColor, newColor);
    helper(image, i + 1, j, oldColor, newColor);
    helper(image, i, j - 1, oldColor, newColor);
    helper(image, i, j + 1, oldColor, newColor);
}
```

### 735. Asteroid Collision

We are given an array `asteroids` of integers representing asteroids in a row. For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

Example:

```
Input: asteroids = [5, 10, -5]
Output: [5, 10] (10 and -5 collide resulting in 10, 5 and 10 never collide)
```

Solution: stack，注意细节

```cpp
vector<int> asteroidCollision(vector<int>& a) {
    vector<int> st;
    for (int i = 0; i < a.size(); i++) {
        if (st.empty() || a[i] > 0) {
            st.push_back(a[i]);
            continue;
        }
        while (true) {
            int prev = st.back();
            if (prev < 0) {
                st.push_back(a[i]);
                break;
            }
            if (prev == -a[i]) {
                st.pop_back();
                break;
            }
            if (prev > -a[i]) {
                break;
            }
            st.pop_back();
            if (st.empty()) {
                st.push_back(a[i]);
                break;
            }
        }
    }
    return st;
}
```

### 739. Daily Temperatures

Given a list of daily temperatures `T`, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put `0` instead.

Example:

```cpp
Input: T = [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

Solution: 单调栈

```cpp
vector<int> dailyTemperatures(vector<int>& T) {
    int n = T.size(); 
    vector<int> res(n);
    stack<int> indices;
    for (int i = 0; i < n; ++i) {
        while (!indices.empty()) {
            int pre_index = indices.top();
            if (T[i] <= T[indices.top()]) break;
            indices.pop();
            res[pre_index] = i - pre_index;
        }
        indices.push(i);
    }
    return res;
}
```

### 744. Find Smallest Letter Greater Than Target

Given a list of sorted characters `letters` containing only lowercase letters, and given a target letter `target`, find the smallest element in the list that is larger than the given target.

Letters also wrap around. For example, if the target is `target = 'z'` and `letters = ['a', 'b']`, the answer is `'a'`.

Example:

```
Input: letters = ["c", "f", "j"], target = "k"
Output: "c"
```

Solution: 二分法

```cpp
char nextGreatestLetter(vector<char>& letters, char target) {
    int l = 0, r = letters.size();
    while (l < r) {
        int m = l + (r - l) / 2;
        if (letters[m] < target + 1) l = m + 1;
        else r = m;
    }
    return l < letters.size() ? letters[l] : letters[0];
}
```