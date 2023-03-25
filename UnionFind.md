## LeetCode Problem - UnionFind

[toc]

### [128.Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

Java

```java
class Solution {
    HashMap<Integer,Integer> parents;
    HashMap<Integer,Integer> counts;
    public int longestConsecutive(int[] nums) {
        parents = new HashMap<>();
        counts = new HashMap<>();
        for (int num : nums) counts.put(num, 1);
        for (int num : nums) union(num, num+1);
        int res = 0;
        for (int num : counts.values()) res = Math.max(res, num);
        return res;
    }

    public int find(int x){
        while (x != parents.getOrDefault(x,x))
            x = parents.get(x);
        return x;
    }

    public void union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        int x_size = counts.getOrDefault(x_parent, 0);
        int y_size = counts.getOrDefault(y_parent, 0);
        if (x_parent != y_parent){
            if (x_size < y_size){
                parents.put(x_parent,y_parent);
                counts.put(y_parent,counts.get(y_parent) + x_size);
            }else{
                parents.put(y_parent,x_parent);
                counts.put(x_parent,counts.get(x_parent) + y_size);
            }
        }
    }
}
```

python

``` python
class Solution:
    def __init__(self):
        self.parents = {}
        self.count = {}

    def longestConsecutive(self, nums: List[int]) -> int:
        for num in nums:
            self.count[num] = 1
        for num in nums:
            self.union(num, num+1)
        if nums:
            return max(self.count.values())
        else: 
            return 0
    
    def find(self, x: int) -> int:
        while(x != self.parents.get(x,x)):
            x = self.parents[x]
        return x

    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.count.get(a_parent, 0), self.count.get(b_parent, 0)
        
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.count[b_parent] += a_size
            else:
                self.parents[b_parent] = a_parent
                self.count[a_parent] += b_size
```



### [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/description/)

Java

``` java
class Solution {
    HashMap<Integer,Integer> parents;
    public int[] findRedundantConnection(int[][] edges) {
        parents = new HashMap<>();
        for (int[] edge : edges) {
            if (find(edge[0]) == find(edge[1]))
                return edge;
            union(edge[0], edge[1]);
        }
        return edges[edges.length-1];
    }

    public int find(int x){
        if (x != parents.getOrDefault(x,x))
            parents.put(x, find(parents.get(x)));
        return parents.getOrDefault(x,x);
    }

    public void union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        if (x_parent != y_parent){
            parents.put(x_parent,y_parent);
        }
    }
}
```

python

``` python
class Solution:
    def __init__(self):
        self.parents = {}
    
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        for edge in edges:
            start, end = edge[0], edge[1]
            if self.find(start)==self.find(end):
                return edge
            self.union(start, end)
        return edge

    def find(self, x: int):
        if x != self.parents.get(x, x):
            self.parents[x] = self.find(self.parents.get(x))
        return self.parents.get(x, x)

    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        if a_parent != b_parent:
            self.parents[a_parent] = b_parent
```





### [685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/description/)

```java
class Solution {
    HashMap<Integer,Integer> parents;

    public int[] findRedundantDirectedConnection(int[][] edges) {
        parents = new HashMap<>();
        int[] parent = new int[edges.length+1];
        int edgeRemoved  = -1;
        int edgeMakesCycle = -1;
        for (int i=0; i<edges.length; i++){
            if (parent[edges[i][1]] != 0){
                edgeRemoved = i;
                break;
            }else
                parent[edges[i][1]] = edges[i][0];
        }
        for (int i=0; i<edges.length; i++){
            if (i == edgeRemoved) continue;
            if (! union(edges[i][0],edges[i][1])){
                edgeMakesCycle = i;
                break;
            }
        }
        if (edgeRemoved == -1)
            return edges[edgeMakesCycle];
        if (edgeMakesCycle != -1){
            int v = edges[edgeRemoved][1];
            int u = parent[v];
            return new int[]{u,v};
        }
        return edges[edgeRemoved];
    }

    public int find(int x){
        while (x != parents.getOrDefault(x, x))
            x = parents.get(x);
        return x;
    }

    public boolean union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        if (x_parent == y_parent)
            return false;
        parents.put(y_parent, x_parent);
        return true;
    }
}
```

python

```python
class Solution:
    def __init__(self):
        self.parents = {}
        self.edgeRemoved = -1
        self.edgeMakesCycle = -1

    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        p = [0 for i in range(len(edges)+1)]
        for i,edge in enumerate(edges):
            parent, child = edge[0], edge[1]
            if p[child] != 0:
                self.edgeRemoved = i
                break
            else:
                p[child] = parent
        for i,edge in enumerate(edges):
            if i == self.edgeRemoved:
                continue
            parent, child = edge[0], edge[1]
            if not self.union(parent, child):
                self.edgeMakesCycle = i
                break
        if (self.edgeRemoved == -1):
            return edges[self.edgeMakesCycle]
        if (self.edgeMakesCycle != -1):
            v = edges[self.edgeRemoved][1]
            u = p[v]
            return [u,v]
        return edges[self.edgeRemoved]
    
    def find(self, a):
        while(a != self.parents.get(a,a)):
            a = self.parents[a]
        return a
    
    def union(self, a, b):
        a_parent, b_parent = self.find(a), self.find(b)
        if a_parent == b_parent:
            return False
        self.parents[b_parent] = a_parent
        return True
```



### [1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/)

[![img](https://darktiantian.github.io/LeetCode%E7%AE%97%E6%B3%95%E9%A2%98%E6%95%B4%E7%90%86%EF%BC%88%E5%B9%B6%E6%9F%A5%E9%9B%86%E7%AF%87%EF%BC%89UnionFind/1319q.png)](https://darktiantian.github.io/LeetCode算法题整理（并查集篇）UnionFind/1319q.png)

```
Input: n = 4, connections = [[0,1],[0,2],[1,2]]
Output: 1
Explanation: Remove cable between computer 1 and 2 and place between computers 1 and 3.
```

```java
class Solution {
    HashMap<Integer, Integer> parents;
    HashMap<Integer, Integer> count;
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n-1) return -1;
        parents = new HashMap<>();
        count = new HashMap<>();
        for (int i=0; i<n; i++){
            parents.put(i,i);
            count.put(i,1);
        }
        for (int[] connection : connections){
            union(connection[0], connection[1]);
        }
        int res = 0;
        for (int i=0; i<n; i++){
            if(find(i)==i) res++;
        }
        return res-1;
    }

    public int find(int x){
        while (x != parents.get(x))
            x = parents.get(x);
        return x;
    }

    public void union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        int x_size = count.get(x_parent), y_size = count.get(y_parent);
        if (x_parent != y_parent){
            if (x_size < y_size){
                parents.put(x_parent, y_parent);
                count.put(y_parent, x_size + y_size);
            }else{
                parents.put(y_parent, x_parent);
                count.put(x_parent, x_size + y_size);
            }
        }
    }
}
```

python

``` python
class Solution:
    def __init__(self):
        self.parents = []
        self.count = []

    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n-1:
            return -1  
        self.parents = [i for i in range(n)]
        self.count = [1 for _ in range(n)]
        for connection in connections:
            a, b = connection[0], connection[1]
            self.union(a, b)
        return len({self.find(i) for i in range(n)}) - 1

    def find(self, node : int) -> int:
        while(node != self.parents[node]):
            node = self.parents[node]
        return node

    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.count[a_parent], self.count[b_parent]
        
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.count[b_parent] += a_size
            else:
                self.parents[b_parent] = a_parent
                self.count[a_parent] += b_size
```



### [1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/)

找到最小生成树的关建边和伪关建边。关建边指去掉这条边以后，路径和会增加。伪关建边，因为生成树图不唯一，指有的生成树图中有这条边，有的没有。还有一种是冗余边，指的是必须要去掉的边。分别返回关建边和伪关建边的索引。

[![img](https://darktiantian.github.io/LeetCode%E7%AE%97%E6%B3%95%E9%A2%98%E6%95%B4%E7%90%86%EF%BC%88%E5%B9%B6%E6%9F%A5%E9%9B%86%E7%AF%87%EF%BC%89UnionFind/1489q.png)](https://darktiantian.github.io/LeetCode算法题整理（并查集篇）UnionFind/1489q.png)

```
Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
Output: [[0,1],[2,3,4,5]]
Explanation: The figure above describes the graph.
The following figure shows all the possible MSTs:
```

Union-Find & Kruskal’s 算法。按照Kruskal算法，我们先将所有的边按权重排列。先将所有点边传入得到一个最小生成树的路径和S。再将每条边去掉剩下的边求路径和S1，如果大于S，则说明这是一条关建边；否则它可能是一条伪关建边或者冗余边。然后我们将这条边加到图中，生成一个路径和S2。如果S2和S1相等，则说明是伪关建边。

```python
class Solution:
    def __init__(self):
        self.parents = {}
        self.counts = {}
    
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        edges = [(u, v, w, i) for i, (u, v, w) in enumerate(edges)]
        edges = sorted(edges, key=lambda x: x[2])
        result, min_weight = [[], []], self.kruskal(edges, n)
        for i, (nodea,nodeb,weight, index) in enumerate(edges):
            if self.kruskal(edges, n, i) > min_weight:
                result[0].append(index)
            elif self.kruskal(edges, n, None, (nodea,nodeb)) == min_weight:
                result[1].append(index)
        return result

    def kruskal(self, edges, n, edge_not_use=None, edge_must_use=None):
        totalweight, c = 0, 1
        self.parents = {}
        if edge_must_use:
            self.union(edge_must_use[0],edge_must_use[1])
        for i, (nodea,nodeb,weight, _) in enumerate(edges):
            if i == edge_not_use:
                continue
            if self.find(nodea) != self.find(nodeb) or (nodea,nodeb) == edge_must_use:
                self.union(nodea,nodeb)
                totalweight += weight
                c += 1
            if c == n: 
                break

        return totalweight if c == n else float('inf')

    def find(self, x: int):
        while(x != self.parents.get(x,x)):
            x = self.parents[x]
        return x
    
    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.counts.get(a_parent, 1), self.counts.get(b_parent, 1)
        
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.counts[b_parent] = a_size + b_size
            else:
                self.parents[b_parent] = a_parent
                self.counts[a_parent] = a_size + b_size
```



### [1579. Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)

找到能使Alice，Bob任意节点都能遍历完整图的最多删除的边的条数。

[![img](https://darktiantian.github.io/LeetCode%E7%AE%97%E6%B3%95%E9%A2%98%E6%95%B4%E7%90%86%EF%BC%88%E5%B9%B6%E6%9F%A5%E9%9B%86%E7%AF%87%EF%BC%89UnionFind/1579q.png)](https://darktiantian.github.io/LeetCode算法题整理（并查集篇）UnionFind/1579q.png)

```
Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
Output: 2
Explanation: If we remove the 2 edges [1,1,2] and [1,1,3]. The graph will still be fully traversable by Alice and Bob. Removing any additional edge will not make it so. So the maximum number of edges we can remove is 2.
```

方法一：贪心+并查集。优先保留类型3的边。注意类型3的边也可能增加结果。

```
class UF:
    def __init__(self, n):
        self.uf = list(range(n+1))
        self.n = n

    def union(self, x, y):
            self.uf[self.find(x)] = self.find(y)

    def find(self, x):
        if self.uf[x] != x:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]

    def all_connected(self):
        return len({self.find(i) for i in range(1, self.n+1)}) == 1

class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        self.res = 0
        edges_3 = ([t, u, v] for t, u, v in edges if t==3)
        edges_2 = ([t, u, v] for t, u, v in edges if t==2)
        edges_1 = ([t, u, v] for t, u, v in edges if t==1)

        uf = UF(n)
        def connect(uf, edges):
            for _, u, v in edges:
                if uf.find(u) == uf.find(v):
                    self.res += 1
                else:
                    uf.union(u, v)
            return uf, uf.all_connected()

        cp_uf, _ = connect(uf, edges_3)
        cp_uf = copy.deepcopy(uf)
        _, connected = connect(uf, edges_2)
        if not connected: return -1
        _, connected = connect(cp_uf, edges_1)
        if not connected: return -1
        return self.res
```





### [1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/)

给你一个 n 个点组成的无向图边集 edgeList ，其中 edgeList[i] = [ui, vi, disi] 表示点 ui 和点 vi 之间有一条长度为 disi 的边。请注意，两个点之间可能有 超过一条边 。给你一个查询数组queries ，其中 queries[j] = [pj, qj, limitj] ，你的任务是对于每个查询 queries[j] ，判断是否存在从 pj 到 qj 的路径，且这条路径上的每一条边都 严格小于 limitj 。请你返回一个 布尔数组 answer ，其中 answer.length == queries.length ，当 queries[j] 的查询结果为 true 时， answer 第 j 个值为 true ，否则为 false 。

```
输入：n = 3, edgeList = [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], queries = [[0,1,2],[0,2,5]]
输出：[false,true]
解释：上图为给定的输入数据。注意到 0 和 1 之间有两条重边，分别为 2 和 16 。
对于第一个查询，0 和 1 之间没有小于 2 的边，所以我们返回 false 。
对于第二个查询，有一条路径（0 -> 1 -> 2）两条边都小于 5 ，所以这个查询我们返回 true 。
```

方法：离线算法+并查集。比赛中没时间做，此题要用并查集。

> 在线算法，可以用来处理数据流。算法不需要一次性地把所有的 query 都收集到再处理。大家也可以想象成：把这个算法直接部署到线上，尽管在线上可能又产生了很多新的 query，也不影响，算法照常运行。
>
> 离线算法则不同。离线算法需要把所有的信息都收集到，才能运行。处理当前 query 的计算过程，可能需要使用之后 query 的信息。
>
> 以排序算法为例，插入排序算法是一种在线算法。因为可以把插入排序算法的待排序数组看做是一个数据流。插入排序算法顺次把每一个数据插入到当前排好序数组部分的正确位置。在排序过程中，即使后面源源不断来新的数据也不怕，整个算法照常进行。
>
> 选择排序算法则是一种离线算法。因为选择排序算法一上来要找到整个数组中最小的元素；然后找第二小元素；以此类推。这就要求不能再有新的数据了。因为刚找到最小元素，再来的新数据中有更小的元素，之前的计算就不正确了。

按照*queries*的*limit*排序，这样我们只需要查看当前的边是否联通来知道是否这条路径上的每条边都小于*limit*。

```
def distanceLimitedPathsExist(self, n: int, edgeList: List[List[int]], queries: List[List[int]]) -> List[bool]:
    edgeList.sort(key=itemgetter(2))

    def union(x, y):
        uf[find(x)] = find(y)

    def find(x):
        uf.setdefault(x, x)
        if x != uf[x]:
            uf[x] = find(uf[x])
        return uf[x]

    uf = {}
    Q = len(queries)
    E = len(edgeList)

    ans = [None] * Q
    j = 0

    for l, i, a, b in sorted(((l, i, a, b) for i, (a, b, l) in enumerate(queries))):
        while j < E and edgeList[j][2] < l:
            x, y, _ = edgeList[j]
            union(x, y)
            j += 1

        ans[i] = find(a)==find(b)

    return ans
```



### [2316. Count Unreachable Pairs of Nodes in an Undirected Graph](https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/description/)

Java

``` java
class Solution {
    int[] parents;
    int[] counts;
    public long countPairs(int n, int[][] edges) {
        parents = new int[n];
        counts = new int[n];
        for (int i = 0; i < n; i ++) {
            parents[i] = i;
            counts[i] = 1;
        }
        for (int[] edge : edges) {
            union(edge[0], edge[1]);
        }
        List<Long> list = new ArrayList<>();
        for (int i = 0; i < n; i ++) {
            if (find(i) == i) {
                list.add(counts[i] + 0L);
            }
        }
        long total = n * (n - 1L) / 2;
        long connectedPairs = 0;
        for (long i : list) {
            connectedPairs += i * (i - 1) / 2;
        }
        return total - connectedPairs;
    }

    public int find(int x){
        if (x != parents[x])
            parents[x] = find(parents[x]);
        return parents[x];
    }

    public void union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        if (x_parent != y_parent){
            parents[x_parent] = y_parent;
            counts[y_parent] += counts[x_parent];
        }
    }
}
```

python

```python
class Solution:
    def __init__(self):
        self.parents = {}
        self.count = {}
    
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        for i in range(n):
            self.count[i] = 1
        for edge in edges:
            self.union(edge[0], edge[1])
        lst = []
        for i in range(n):
            if self.find(i) == i:
                lst.append(self.count[i])
        total = n * (n - 1) // 2
        connectedPairs = 0
        for i in lst:
            connectedPairs += i * (i - 1) // 2
        return total - connectedPairs

    def find(self, x: int):
        if x != self.parents.get(x, x):
            self.parents[x] = self.find(self.parents.get(x))
        return self.parents.get(x, x)

    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        if a_parent != b_parent:
            self.parents[a_parent] = b_parent
            self.count[b_parent] += self.count.get(a_parent, 0)
```



### [2492. Minimum Score of a Path Between Two Cities](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/)

Java

```java
class Solution {
    HashMap<Integer,Integer> parents;
    HashMap<Integer,Integer> counts;
    public int minScore(int n, int[][] roads) {
        parents = new HashMap<>();
        counts = new HashMap<>();
        for (int[] row : roads){
            union(row[0], row[1]);
        }
        int temp = find(1), res = Integer.MAX_VALUE;
        for (int[] row : roads){
            if (find(row[0]) == temp || find(row[1]) == temp)
                res = Math.min(res, row[2]);
        }
        return res;
    }

    public int find(int x){
        while (x != parents.getOrDefault(x,x))
            x = parents.get(x);
        return x;
    }

    public void union(int x, int y){
        int x_parent = find(x), y_parent = find(y);
        int x_size = counts.getOrDefault(x_parent, 1);
        int y_size = counts.getOrDefault(y_parent, 1);
        if (x_parent != y_parent){
            if (x_size < y_size){
                parents.put(x_parent,y_parent);
                counts.put(y_parent,x_size + y_size);
            }else{
                parents.put(y_parent,x_parent);
                counts.put(x_parent,x_size + y_size);
            }
        }
    }
}
```

python

```python
class Solution:
    def __init__(self):
            self.parents = {}
            self.count = {}

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        for row in roads: self.union(row[0], row[1])
        temp = self.find(1)
        res = 10001
        for row in roads:
            if (self.find(row[0]) == temp or self.find(row[1]) == temp):
                res = min(res, row[2])
        return res

    def find(self, x: int) -> int:
        while(x != self.parents.get(x,x)):
            x = self.parents[x]
        return x

    def union(self, a: int, b: int):
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.count.get(a_parent, 1), self.count.get(b_parent, 1)
        
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.count[b_parent] = a_size + b_size
            else:
                self.parents[b_parent] = a_parent
                self.count[a_parent] = a_size + b_size
```



