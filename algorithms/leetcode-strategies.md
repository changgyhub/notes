# LeetCode Strategies

面试时，如果想不到好的思路，先考虑给出一个brute force solution（也有很多题其实也只有brute force）

### dp问题

1. knapsack类问题，为了方便迭代，0-1背包对物品的迭代放在最外层，完全背包对物品的迭代放在最里层。0-1背包不可利用当轮信息，1D压缩时从后往前；而完全背包利用当轮信息，1D压缩时从前往后
2. 区间问题\(两种情况，i到j的 vs 以i或j结尾的，有的时候需要加个中间位k
3. 复杂状态转移方程，如局部和全局最优，买卖区分。dp时如果发现i-2需要用，可以把数组变成n+1大小，其他情况以此类推

### 要点

1. 有时直接双指针或者stack也很快：stack可以用vector来写，很多贪心、特殊字符匹配等题目都可以用stack
2. 需要中间比两边大或小的题，可以正反各遍历一次
3. DFS+Memoization基本等价DP，需要个数的时候用DP，需要输出的时候用DFS/backtracking
4. Backtracking修改一般有两种情况，一种是修改最后一位输出，比如排列组合；一种是修改visited，比如矩阵里搜字符串
5. 记录重复，可以换用set储存再转成要求的形式，可以加flag，也可以开一个bool数组，实在不行就排序删重。
6. 水漫dfs时注意防止重复遍历，要么隐藏式加判断条件，要么开一个visited
7. set的实现是红黑树，可以用来做快速insert、delete
8. 重复数字类
   1. find duplicate in \[1, n\]  \(e.g. \[1,3,4,2,2\]\)：快慢指针
   2. find all duplicates in \[1, n\] ：遍历一遍，翻转标记
      * nums\[abs\(nums\[i\]\)-1\] = -nums\[abs\(nums\[i\]\)-1\]）
   3.  \[1, n\]一个数被替换为另一个数，找出重复的数和丢失的数 \(e.g. \[1,2,2,4\]\)：持续交换交换第 i 位和 nums\[i\] - 1 位置上的元素，然后遍历一边
      * 类似题目：寻找所有丢失的元素、寻找所有重复的元素
9. subarray/sliding window有的题可以直接遍历，用hashmap/hashset记录当前和，这样\[i, j\]的信息可以用hash\[j\]-hash\[i\]得到，每个subarray都遍历过
10. 正方形定义是对角线垂直、平分、相等
11. leetcode hack: `static int x=[]() {std::ios::sync_with_stdio(false);cin.tie(NULL);return 0;}();`

### bit manipulation

见这个[链接](https://leetcode.com/problems/sum-of-two-integers/discuss/84278/A-summary:-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently)

### 二分法

1. 可以左闭右开也可以左闭右闭，但是注意左闭右闭时要用right = mid和left = mid + 1，而不是right = mid - 1和left = mid，因为算mid整除2的时候会向下取整。最后可以用-1来修正最终取整方式。
3. 注意分清要lower bound还是upper bound，分别如下

```cpp
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
```

### 树操作

1. 树需要return结果时，大概有三种做法：\(1\) 对当前node，递归左右，再对递归得到的左右进行操作 \(2\) dfs，信息当参数传递，到leaf时操作积累的信息 \(3\) 将信息放在全局，通过traversal或dfs直接更新

BST删除节点

```cpp
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

Given a binary tree, return all root-to-leaf paths.

```cpp
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
```

寻找节点

```cpp
TreeNode* find_val(TreeNode* root, int key) {
    if (!root || root->key == key) return root;
    TreeNode* ret = find_val(root->left, key);
    if (!ret) ret = find_val(root->right, key);
    return ret;
}
```

树按in order压成一条右枝（注意：`prev->left = NULL`这一句一定要放在最后，即前进一步之后立即更新，否则会出现环路）

```cpp
TreeNode *prev, *head;
    
TreeNode* increasingBST(TreeNode* root) {
    head = new TreeNode(-1), prev = head;
    helper(root);
    return head->right;
}

void helper(TreeNode* root) {
    if (!root) return;
    helper(root->left);
    prev->right = root;
    prev = prev->right;
    prev->left = NULL;
    helper(root->right);
}
```

### 数学

1. extended gcd

```cpp
int xGCD(int a, int b, int &x, int &y) {
    if (!b) {
       x = 1, y = 0;
       return a;
    }
    int x1, y1, gcd = xGCD(b, a % b, x1, y1);
    x = y1, y = x1 - (a / b) * y1;
    return gcd;
}
```

2. Minimum Moves to Equal Array Elements

```cpp
int minMoves2(vector<int>& nums) {
    int n = nums.size();
    if (n < 2) return 0;
    if (n == 2) return abs(nums[0] - nums[1]); 
    int ret = 0, median = find_median(nums);
    for (auto i: nums) ret += abs(i - median);
    return ret;
}

int find_median(vector<int>& nums) {
    int l = 0, r = nums.size() - 1, target = (nums.size() - 1)/2;
    while (l < r) {
        int mid = quick_partition(nums, l, r);
        if (mid == target) return nums[mid];
        if (mid < target) l = mid + 1;
        else r = mid - 1;
    }
    return nums[l];
}

int quick_partition(vector<int>& nums, int l, int r) {
    int i = l + 1, j = r;
    while (true) {
        while (i < r && nums[i] <= nums[l]) ++i;
        while (l < j && nums[j] >= nums[l]) --j;
        if (i >= j) break;
        swap(nums[i], nums[j]);
    }
    swap(nums[l], nums[j]);
    return j;
}
```

### 链表

1. 翻转

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode *prev = NULL, *next;
    while (head) {
        next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}

// call node = reverseList(mode, NULL);
ListNode* reverseList(ListNode* head, ListNode* prev) {
    if (!head) return prev;
    ListNode* next = head->next;
    head->next = prev;
    return reverseList(next, head);
}
```

2. 排序

```cpp
ListNode* sortList(ListNode* head) {
    if (!head|| !head->next) return head;
    ListNode *slow = head, *fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    fast = slow->next;
    slow->next = NULL;
    return merge(sortList(head),sortList(fast));
}
ListNode* merge(ListNode *l1, ListNode *l2) {
    ListNode *dummy = new ListNode(INT_MIN);
    ListNode *node = dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            node->next = l1;
            l1 = l1->next;
        } else {
            node->next = l2;
            l2 = l2->next;
        }
        node = node->next;
    }
    node->next = l1? l1: l2; // clever!
    return dummy->next;
}
```

### 拓展

1. 有1000瓶老鼠药和10只老鼠，其中有一瓶老鼠药是有毒的，老鼠喝了有毒的老鼠药2天就会挂掉，如何在两天之内找到哪瓶老鼠药是有毒的？老鼠列成一排，每只老鼠看成一个bit位，1-1000给对应bit位是1的老鼠喝，最后挂掉的老鼠是药瓶编号的bit位1\(其它是0\)，该瓶老鼠药是有毒的



