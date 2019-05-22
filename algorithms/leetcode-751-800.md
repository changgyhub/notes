# LeetCode 751 - 800

### 785. Is Graph Bipartite?

Given an undirected graph, return true if and only if it is bipartite. A graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B. The graph is given in the following form: graph\[i\] is a list of indexes j for which the edge between nodes i and j exists. Each node is an integer between 0 and graph.length - 1. There are no self edges or parallel edges: graph\[i\] does not contain i, and it doesn't contain any element twice.

Example:

```text
Input: [[1,3], [0,2], [1,3], [0,2]]
0----1
|    |
|    |
3----2
Output: true ({0, 2}, {1, 3})
```

Solution: 染色法，一定要背

```cpp
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    if (!n) return true;
    vector<int> color(n, 0);
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (!color[i]) {
            q.push(i);
            color[i] = 1;
        }
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            for (auto j: graph[node]) {
                if (color[j] == 0) {
                    q.push(j);
                    color[j] = color[node] == 2 ? 1 : 2;
                }
                else if (color[node] == color[j]) return false;
            }
        }
    }
    return true; 
}
```

拓展: 寻找二分图的最大匹配 - [匈牙利算法](https://www.renfei.org/blog/bipartite-matching.html)

```cpp
struct Edge {
    int from;
    int to;
    int weight;
    Edge(int f, int t, int w):from(f), to(t), weight(w) {}
};

vector<vector<int>> G;  // G[i] 存储顶点 i 出发的边的编号
vector<Edge> edges;
int num_nodes, num_left, num_right, num_edges;
vector<int> matching(__maxNodes, 0);  // 存储求解结果
vector<int> check(__maxNodes, 0);  // 交替路

// method 1: dfs
bool dfs(int u) {
    for (auto i: G[u]) { // 对 u 的每个邻接点
        int v = edges[i].to;
        if (!check[v]) {     // 要求不在交替路中
            check[v] = true; // 放入交替路
            if (matching[v] == -1 || dfs(matching[v])) {
                // 如果改点是未匹配点或是增广路起点，说明交替路为增广路，则储存匹配，并返回成功
                matching[v] = u;
                matching[u] = v;
                return true;
            }
        }
    }
    return false; // 不存在增广路，返回失败
}

int hungarian_dfs() {
    int ans = 0;
    fill(matching.begin(), matching.end(), -1);
    for (int u = 0; u < num_left; ++u) {
        if (matching[u] == -1) {
            fill(check.begin(), check.end(), 0);  // 在每一步中清空
            if (dfs(u)) ++ans;
        }
    }
    return ans;
}

// method2: bfs
queue<int> Q;
vector<int> prev(__maxNodes, 0);

int hungarian_bfs() {
    int ans = 0;
    fill(matching.begin(), matching.end(), -1);
    fill(check.begin(), check.end(), -1);
    for (int i = 0; i < num_left; ++i) {
        if (matching[i] == -1) {
            while (!Q.empty()) Q.pop();
            Q.push(i);
            prev[i] = -1; // 设 i 为路径起点
            bool flag = false;  // 尚未找到增广路
            while (!Q.empty() && !flag) {
                int u = Q.front();
                for (auto ix: G[u]) {
                    int v = edges[ix].to;
                    if (check[v] != i) {
                        check[v] = i;
                        Q.push(matching[v]);
                        if (matching[v] >= 0) { // 此点为匹配点
                            prev[matching[v]] = u;
                        } else { // 找到未匹配点，交替路变为增广路
                            flag = true;
                            int d=u, e=v;
                            while (d != -1) {
                                int t = matching[d];
                                matching[d] = e;
                                matching[e] = d;
                                d = prev[d];
                                e = t;
                            }
                            break;
                        }
                    }
                }
                Q.pop();
            }
            if (matching[i] != -1) ++ans;
        }
    }
    return ans;
}
```

