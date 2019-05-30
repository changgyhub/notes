# LeetCode 801-850

### 803. Bricks Falling When Hit

We have a grid of 1s and 0s; the 1s in a cell represent bricks.  A brick will not drop if and only if it is directly connected to the top of the grid, or at least one of its (4-way) adjacent bricks will not drop.

We will do some erasures sequentially. Each time we want to do the erasure at the location (i, j), the brick (if it exists) on that location will disappear, and then some other bricks may drop because of that erasure.

Return an array representing the number of bricks that will drop after each erasure in sequence.

Example:

```
Input: grid = [[1,0,0,0],[1,1,0,0]], hits = [[1,1],[1,0]]
Output: [0, 0] (When we erase the brick at (1, 0), the brick at (1, 1) has already disappeared due to the last move. So each erasure will cause no bricks dropping.  Note that the erased brick (1, 0) will not be counted as a dropped brick.)
```

Solution: 先把所有要拿掉的砖块都拿掉，判断最后有哪些砖块在；然后从后往前加砖块。十分巧妙，一定要背

```cpp
vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
    int m = grid.size(), n = grid.front().size(), hs = hits.size();
    vector<int> res (hs, 0), val (hs, 0);
    for (int i = 0; i < hs; ++i) {
        val[i] = grid[hits[i][0]][hits[i][1]];
        grid[hits[i][0]][hits[i][1]] = 0;
    }
    // step 1: counted finally connected bricks
    for (int j = 0; j < n; ++j) {
        if (grid[0][j] == 1) {
            countOne(grid, 0, j, m, n);
        }
    }
    // step 2: add bricks in reversed order
    for (int i = hs - 1; i >= 0; --i) {
        int a = hits[i][0], b = hits[i][1];
        grid[a][b] = val[i];
        if (grid[a][b] == 0) continue;  // skip hitting on 0 cases
        if (isConnected(grid, a, b, m, n)) {
            res[i] = countOne(grid, a, b, m, n) - 1;
        }
    }
    return res;
}

bool isConnected(vector<vector<int>>& grid, int i, int j, int m, int n) {
    // if connected to the root
    if (i == 0) return true;
    static int dir[5] = {1, 0, -1, 0, 1};
    for (int k = 0; k < 4; ++k) {
        int ni = i + dir[k];
        int nj = j + dir[k+1];
        if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
        // if there is any direction eventually connected to the root.
        if (grid[ni][nj] == 3) return true;
    }
    return false;
}

int countOne(vector<vector<int>>& g, int i, int j, int m, int n) {
    if (i < 0 || i >= m || j < 0 || j >= n || g[i][j] != 1) return 0;
    g[i][j] = 3;
    return 1 + countOne(g, i + 1, j, m, n) + countOne(g, i, j + 1, m, n) + countOne(g, i - 1, j, m, n) + countOne(g, i, j - 1, m, n);
}
```

### 818. Race Car

Your car starts at position 0 and speed +1 on an infinite number line.  (Your car can go into negative positions.) Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse). When you get an instruction "A", your car does the following: `position += speed, speed *= 2`. When you get an instruction "R", your car does the following: if your speed is positive then `speed = -1` , otherwise `speed = 1`.  (Your position stays the same.)

Now for some target position, say the **length** of the shortest sequence of instructions to get there.

Example:

```
Input: target = 6
Output: 5 ("AAARA": 0->1->3->7->7->6)
```

Solution: dp + bit manipulation

```cpp
int dp[10001];
int racecar(int t) {
    if (dp[t] > 0) return dp[t];
    int n = floor(log2(t)) + 1, res;
    if (1 << n == t + 1) dp[t] = n;
    else {
        dp[t] = racecar((1 << n) - 1 - t) + n + 1;
        for (int m = 0; m < n - 1; ++m)
            dp[t] = min(dp[t], racecar(t - (1 << (n - 1)) + (1 << m)) + n + m + 1);
    }
    return dp[t];
}
```

### 843. Guess the Word

This problem is an **interactive problem** new to the LeetCode platform.

We are given a word list of unique words, each word is 6 letters long, and one word in this list is chosen as **secret**. You may call `master.guess(word)` to guess a word.  The guessed word should have type `string` and must be from the original list with 6 lowercase letters.

This function returns an `integer` type, representing the number of exact matches (value and position) of your guess to the **secret word**.  Also, if your guess is not in the given wordlist, it will return `-1` instead.

For each test case, you have 10 guesses to guess the word. At the end of any number of calls, if you have made 10 or less calls to `master.guess` and at least one of these guesses was the **secret**, you pass the testcase.

Example:

```
Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"]
Possible Guesses:
master.guess("aaaaaa") returns -1, because "aaaaaa" is not in wordlist.
master.guess("acckzz") returns 6, because "acckzz" is secret and has all 6 matches.
master.guess("ccbazz") returns 3, because "ccbazz" has 3 matches.
master.guess("eiowzz") returns 2, because "eiowzz" has 2 matches.
master.guess("abcczz") returns 4, because "abcczz" has 4 matches.
We made 5 calls to master.guess and one of them was the secret, so we pass the test case.
```

Solution: 每次随机从list里选一个猜，然后根据list里的词和这个词的重字数，是否等于这个词和正确答案的重字数，来做筛除；几次下来一般就可以找到答案了

```cpp
int num_of_matches(string str,string guessWord) {
    int sum = 0;
    for (int i = 0; i < str.length(); ++i) sum += str[i] == guessWord[i];
    return sum;
}
void shrinkList(vector<string>& wordlist, string guessWord, int matches) {
    vector<string> tmp;
    for (auto str: wordlist) if (num_of_matches(str, guessWord) == matches) tmp.push_back(str);
    wordlist = tmp;
}
void findSecretWord(vector<string>& wordlist, Master& master) {
    string guessWord;
    for (int i = 1; i <= 10; ++i) {
        guessWord = wordlist[rand() % wordlist.size()];
        int matches = master.guess(guessWord);
        if (matches == 6) return;
        shrinkList(wordlist, guessWord, matches);
    }
}
```

### 844. Backspace String Compare

Given two strings `S` and `T`, return if they are equal when both are typed into empty text editors. `#` means a backspace character.

Example:

```
Input: S = "a##c", T = "#a#c"
Output: true
```

Solution: stack

```cpp
bool backspaceCompare(string S, string T) {
    stack<int> s, t;
    for (auto c : S) {
        if (c == '#') {
            if (!s.empty()) s.pop();
        } else s.push(c);
    }
    for (auto c : T) {
        if (c == '#') {
            if (!t.empty()) t.pop();
        } else t.push(c);
    }
    return s == t;
}
```

### 846. Hand of Straights

Alice has a `hand` of cards, given as an array of integers. Now she wants to rearrange the cards into groups so that each group is size `W`, and consists of `W` consecutive cards. Return `true` if and only if she can.

Example:

```
Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true ([1,2,3],[2,3,4],[6,7,8])
```

Solution: sort + hashmap，十分经典，一定要背

```cpp
bool isNStraightHand(vector<int>& hand, int W) {
    sort(hand.begin(), hand.end());
    unordered_map<int, int> cards;
    for (int i: hand) ++cards[i];
    for (int i = 0; i < hand.size(); ++i) {
        int count = cards[hand[i]];
        for (int j = hand[i] + 1; j < hand[i] + W; ++j) {
            if (cards[j] < count) return false;
            cards[j] -= count;
        }
        while (i < hand.size() && (hand[i] == hand[i+1] || cards[hand[i+1]] == 0)) ++i;
    }
    return true;
}
```

### 849. Maximize Distance to Closest Person

In a row of `seats`, `1` represents a person sitting in that seat, and `0`represents that the seat is empty. There is at least one empty seat, and at least one person sitting. Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. Return that maximum distance to closest person.

Example:

```
Input: [1,0,0,0,1,0,1]
Output: 2 ([1,0,1,0,1,0,1])
```

Solution: 遍历一遍就好

```cpp
int maxDistToClosest(vector<int> seats) {
    int res = 0, n = seats.size(), i = 0, j = n - 1;
    while (i < n) if (seats[i]) break; else ++i;
    res = max(res, i);
    while (j >= 0) if (seats[j]) break; else --j;
    res = max(res, n - j - 1);
    int prev = i;
    for (int k = i; k <= j; ++k) {
        if (seats[k] == 1) {
            res = max(res, (k - prev) / 2);
            prev = k;
        }
    }
    return res;
}
```