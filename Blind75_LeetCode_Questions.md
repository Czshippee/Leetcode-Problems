# LeetCode Blind 75 Questions

[TOC]

A list of Blind 75 Leetcode problems which is very useful, enjoy!



## Array

### [1. Two Sum](https://leetcode.com/problems/two-sum/description/)

java:

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashmap = new HashMap<>();
        for (int i = 0; i< nums.length; i++){
            if (hashmap.containsKey(nums[i]))
                return new int[]{hashmap.get(nums[i]), i};
            hashmap.put(target-nums[i], i);
        }
        return new int[2];
    }
}
```

python3:

``` python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i,num in enumerate(nums):
            if num in dic:
                return [dic[num],i]
            dic[target - num] = i
```

### [15. 3Sum](https://leetcode.com/problems/3sum/description/)

java:

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length-2; i++){
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int left = i+1, right = nums.length-1;
            while (left < right){
                int threesome = nums[i] + nums[left] + nums[right];
                if (threesome == 0){
                    res.add(List.of(nums[i], nums[left], nums[right]));
                    left ++;
                    right --;
                    while (nums[left] == nums[left-1] && left < right)
                        left ++;
                    while (nums[right] == nums[right+1] && left < right)
                        right --;
                }else if (threesome > 0)
                    right --;
                else
                    left ++;
            }
        }
        return res;
    }
}
```

python3:

``` python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)-2):
            firstNum = nums[i]
            if i>0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                threeSome = firstNum + nums[left] + nums[right]
                if threeSome > 0:
                    right -= 1
                elif threeSome < 0:
                    left += 1
                else:
                    result.append([firstNum,nums[left],nums[right]])
                    left += 1
                    right -= 1
                    while nums[left] == nums[left-1] and left < right:
                        left += 1
                    while nums[right] == nums[right+1] and left < right:
                        right -= 1
        return result
```

### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

java:

```java
class Solution {
    public int maxProfit(int[] prices) {
        int minprice = prices[0];
        int res = 0;
        for (int price : prices){
            minprice = Math.min(price, minprice);
            res = Math.max(price-minprice, res);
        }
        return res;
    }
}
```

python3:

``` python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = prices[0]
        res = 0
        for i in range(1, len(prices)):
            if prices[i] < minprice:
                minprice = prices[i]
            res = max(prices[i]-minprice, res)
        return res
```

### [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/description/)

java:

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums){
            if (set.contains(num))
                return true;
            else
                set.add(num);
        }
        return false;
    }
}
```

python3:

``` python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        res = set()
        for num in nums:
            if num in res: return True
            else: res.add(num)
        return False
```

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

java:

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int result [] = new int [nums.length];
        int prefixprod = 1;
        for (int i = 0; i < nums.length; i++){
            result[i] = prefixprod;
            prefixprod *= nums[i];
        }
        prefixprod = 1;
        for (int i = nums.length-1; i > -1; i--){
            result[i] *= prefixprod;
            prefixprod *= nums[i];
        }
        return result;
    }
}
```

python3:

``` python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        result = [1 for _ in range(len(nums))]
        prefixprod = 1
        for i,num in enumerate(nums):
            result[i] = prefixprod
            prefixprod *= num
        prefixprod = 1
        for i in range(len(nums)-1, -1, -1):
            result[i] *= prefixprod
            prefixprod *= nums[i]
        return result
```

### [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/)

java:

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int[] opt = new int[nums.length];
        int maxresult = nums[0];
        opt[0] = nums[0];
        for (int i=1; i<nums.length; i++){
            opt[i] = Math.max(nums[i]+opt[i-1], nums[i]);
            if (opt[i] > maxresult) maxresult = opt[i];
        }
        return maxresult;
    }
}
```

python3:

``` python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        totalSum = 0
        maxSum = nums[0]
        for num in nums:
            totalSum += num
            maxSum = max(totalSum, maxSum)
            if totalSum<0: totalSum = 0
        return maxSum
```

### [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

java:

```java
class Solution {
    public int maxProduct(int[] nums) {
        int maxproduct = Integer.MIN_VALUE;
        int curproduct = 1;
        for (int num : nums){
            curproduct *= num;
            maxproduct = Math.max(maxproduct, curproduct);
            if (num == 0)
                curproduct = 1;
        }
        curproduct = 1;
        for (int i = nums.length-1; i >= 0; i--){
            curproduct *= nums[i];
            maxproduct = Math.max(maxproduct, curproduct);
            if (nums[i] == 0)
                curproduct = 1;
        }
        return maxproduct;
    }
}
```

python3:

``` python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxproduct = nums[0]
        curproduct = 1
        for num in nums:
            curproduct *= num
            maxproduct = max(maxproduct, curproduct)
            if num == 0:
                curproduct = 1
        curproduct = 1
        for i in range(len(nums)-1, -1, -1):
            curproduct *= nums[i]
            maxproduct = max(maxproduct, curproduct)
            if nums[i] == 0:
                curproduct = 1
        return maxproduct
```

### [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/)

java:

```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length-1;
        while (left <= right){
            if (nums[left] < nums[right]) 
                return nums[left];
            int mid = left + (right-left)>>1;
            if (nums[left] < nums[mid])
                left = mid + 1;
            else
                right = mid;
        }
        return nums[left];
    }
}
```

python3:

``` python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left,right = 0,len(nums)-1
        while left < right:
            if nums[left] < nums[right]:
                return nums[left]
            mid = (left+right)//2
            if nums[right] < nums[mid]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

### [33. Search in Rotated Sorted Array](bbb)

java:

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        while (left <= right){
            int mid = left + (right-left)/2;
            if (nums[mid]==target) 
                return mid;
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target <= nums[mid])
                    right = mid - 1;
                else
                    left = mid + 1;
            } else {
                if (nums[mid] <= target && target <= nums[right])
                    left = mid + 1;
                else
                    right = mid - 1;
            }
        }
        return -1;
    }
}
```

python3:

``` python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right-left)//2
            if nums[mid]==target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target and target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] <= target and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/)

java:

```java
class Solution {
    public int maxArea(int[] height) {
        int left = 0, right = height.length-1;
        int maxarea = 0;
        while (left < right){
            maxarea = Math.max(maxarea, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right])
                left ++;
            else
                right --;
        }
        return maxarea;
    }
}
```

python3:

``` python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        resarea = 0
        while left < right:
            resarea = max(resarea, (right-left)*min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return resarea
```



## Binary

### [371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/description/)

java:

```java
class Solution {
    public int getSum(int a, int b) {
        while (b != 0){
            int carry = (a & b) << 1;
            a ^= b;
            b = carry;
        }
        return a;
    }
}
```

python3:

``` python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        while b != 0:
            a, b = (a ^ b) & 0xFFFFFFFF, ((a & b) << 1) & 0xFFFFFFFF
        return a if a <= 0x7FFFFFFF else ~(a^0xFFFFFFFF)
```

### [191. Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/description/)

java:

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        while (n!=0){
            n &= (n-1);
            count ++;
        }
        return count;
    }
}
```

python3:

``` python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= (n-1)
            count += 1
        return count
```

### [338. Counting Bits](https://leetcode.com/problems/counting-bits/description/)

java:

```java
class Solution {
    public int[] countBits(int n) {
        int[] res = new int[n+1];
        for (int i = 1; i < n+1; i++){
            if ((i&1)==1)
                res[i] = res[i-1] + 1;
            else
                res[i] = res[i/2];
        }
        return res;
    }
}
```

python3:

``` python
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0]
        for i in range(1, n+1):
            if (i&1) == 1:
                res.append(res[-1] + 1)
            else:
                res.append(res[i//2])
        return res
```

### [268. Missing Number](https://leetcode.com/problems/missing-number/description/)

java:

```java
class Solution {
    public int missingNumber(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; i ++){
            res ^= i;
            res ^= nums[i];
        }
        res ^= nums.length;
        return res;
    }
}
```

python3:

``` python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = 0
        for i,num in enumerate(nums):
            n ^= i
            n ^= num
        n ^= len(nums)
        return n
```

### [190. Reverse Bits](https://leetcode.com/problems/reverse-bits/description/)

java:

```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++){
            res = (res<<1) | (n&1);
            n >>= 1;
        }
        return res;
    }
}
```

python3:

``` python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res
```

## Dynamic Programming

### [aaa](bbb)

java:

```java

```

python3:

``` python
```

[Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

[Coin Change](https://leetcode.com/problems/coin-change/)

[Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```







## Graph

### [133. Clone Graph](https://leetcode.com/problems/clone-graph/description/)

java:

```java
class Solution {
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        Map<Integer, Node> visited = new HashMap<>();
        visited.put(node.val, new Node(node.val));
        Queue<Node> queue = new LinkedList<>();
        queue.add(node);
        while (! queue.isEmpty()){
            Node orinode = queue.poll();
            Node cloneNode = visited.get(orinode.val);
            for (Node ngbr : orinode.neighbors){
                if (! visited.containsKey(ngbr.val)){
                    visited.put(ngbr.val, new Node(ngbr.val));
                    queue.add(ngbr);
                }
                cloneNode.neighbors.add(visited.get(ngbr.val));
            }
        }
        return visited.get(node.val);
    }
}
```

python3:

``` python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return node
        queue = deque([node])
        dic = {node.val: Node(node.val)}
        while queue:
            orinode = queue.popleft() 
            clonenode = dic[orinode.val]            
            for ngbr in orinode.neighbors:
                if ngbr.val not in dic:
                    dic[ngbr.val] = Node(ngbr.val)
                    queue.append(ngbr)
                clonenode.neighbors.append(dic[ngbr.val])
        return dic[node.val]
```

### [207. Course Schedule](https://leetcode.com/problems/course-schedule/description/)

java:

```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, ArrayList<Integer>> hashmap = new HashMap<>();
        int[] numpreq = new int[numCourses];
        for (int[] prerequisite : prerequisites){
            ArrayList<Integer> temp;
            if (hashmap.containsKey(prerequisite[1]))
                temp = hashmap.get(prerequisite[1]);
            else
                temp = new ArrayList<>();
            temp.add(prerequisite[0]);
            hashmap.put(prerequisite[1], temp);
            numpreq[prerequisite[0]] += 1;
        }
        Queue<Integer> queue = new LinkedList<>();
        int result = 0;
        for (int i = 0; i < numpreq.length; i++){
            if (numpreq[i] == 0)
              queue.add(i);  
        }
        while (queue.size()>0){
            int u = queue.poll();
            result ++;
            for (int v : hashmap.getOrDefault(u, new ArrayList<>())){
                numpreq[v] --;
                if (numpreq[v] == 0)
                    queue.add(v);
            }
        }
        return result == numCourses;
    }
}
```

python3:

``` python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        numpreq = [0] * numCourses
        result = 0;
        for u, v in prerequisites:
            graph[v].append(u)
            numpreq[u] += 1
        queue = Deque([u for u in range(numCourses) if numpreq[u] == 0])
        while queue:
            u = queue.popleft()
            result += 1
            for v in graph[u]:
                numpreq[v] -= 1
                if numpreq[v] == 0:
                    queue.append(v)
        return result == numCourses
```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```

### [aaa](bbb)

java:

```java

```

python3:

``` python

```



## Interval

### [57. Insert Interval](https://leetcode.com/problems/insert-interval/)

java:

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> array = new ArrayList<>();
        int i = 0;
        while (i < intervals.length && intervals[i][1] < newInterval[0]){
            array.add(intervals[i]);
            i ++;
        }
        while (i < intervals.length && intervals[i][0] <= newInterval[1]){
            newInterval[0] = Math.min(intervals[i][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[i][1], newInterval[1]);
            i ++;
        }
        array.add(newInterval);
        while (i < intervals.length){
            array.add(intervals[i]);
            i ++;
        }
        return array.toArray(new int [array.size()][]);
    }
}
```

python3:

``` python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1
        while i < len(intervals) and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i += 1
        result.append(newInterval)

        while i < len(intervals):
            result.append(intervals[i])
            i += 1
        return result
```

### [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/description/)

java:

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new CustomSort());
        List<int[]> result = new ArrayList<>();
        result.add(intervals[0]);
        int[] prev = intervals[0];
        for (int i = 1; i < intervals.length; i++){
            if (intervals[i][0] <= prev[1]){
                prev[1] = Math.max(intervals[i][1], prev[1]);
                result.remove(result.size()-1);
                result.add(prev);
            }else{
                prev = intervals[i];
                result.add(intervals[i]);
            }
        }
        return result.toArray(new int [result.size()][]);
    }
}
class CustomSort implements Comparator<int[]>{
    public int compare(int[] array1, int[] array2)
    {
        return array1[0] - array2[0];
    }
}
```

python3:

``` python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        result = []
        result.append(intervals[0])
        prev = intervals[0]
        for i in range(1,len(intervals)):
            left, right=prev[0], prev[1]
            if right >= intervals[i][0]:
                result[-1] = [left, max(right, intervals[i][1])]
                prev = result[-1]
            else:
                prev = intervals[i]
                result.append(prev)
        return result
```

### [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

java:

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, new CustomSort());
        int res = 0;
        int prev = intervals[0][1];
        for (int i=1; i<intervals.length; i++){
            if (intervals[i][0] >= prev)
                prev = intervals[i][1];
            else
                res ++;
        }
        return res;
    }
}
class CustomSort implements Comparator<int[]>{
    public int compare(int[] array1, int[] array2)
    {
        return array1[1] - array2[1];
    }
}
```

python3:

``` python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        count = 0
        prev = intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] >= prev:
                prev = intervals[i][1]
            else:
                count += 1
        return count
```

### [252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)

(Leetcode Premium)

### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

(Leetcode Premium)

## Linked List

### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

java:

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if ((head == null) || (head.next == null)) return head;
        ListNode prevnode = null, currnode = head, nextnode = head.next;
        while (nextnode!= null){
            currnode.next = prevnode;
            prevnode = currnode;
            currnode = nextnode;
            nextnode = nextnode.next;
        }
        currnode.next = prevnode;
        return currnode;
    }
}
```

python3:

``` python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next == None:
            return head
        prev, curr, next = None, head, head.next
        while next:
            curr.next = prev
            prev, curr, next = curr, next, next.next
        curr.next = prev
        return curr
```

### [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

java:

```java
class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null || head.next.next == null)
            return false;
        ListNode slow = head.next, fast = head.next.next;
        while (slow != fast){
            if (fast.next == null || fast.next.next == null)
                return false;
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```

python3:

``` python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head == None or head.next == None or head.next.next == None:
            return False
        slow, fast = head.next, head.next.next
        while slow != fast:
            if fast.next is None or fast.next.next is None:
                return False
            slow, fast = slow.next, fast.next.next
        return True
```

### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/description/)

java:

```java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummyhead = new ListNode();
        ListNode node = dummyhead;
        while (list1!=null && list2!=null){
            if (list1.val < list2.val){
                node.next = list1;
                list1 = list1.next;
            }else{
                node.next = list2;
                list2 = list2.next;
            }
            node = node.next;
        }
        if (list1==null) node.next = list2;
        else node.next = list1;
        return dummyhead.next;
    }
}
```

python3:

``` python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummyhead = node = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next
        node.next = list1 or list2
        return dummyhead.next
```

### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

java:

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        ListNode dummynode = new ListNode();
        ListNode head = dummynode;
        for (int i = 0; i<lists.length; i++){
            if (lists[i] != null){
                pq.add(new Pair(lists[i].val, i));
            }
        }
        while (! pq.isEmpty()){
            Pair small_pair = pq.poll();
            
            int small_value = small_pair.a;
            int small_index = small_pair.b;
            
            lists[small_index] = lists[small_index].next;
            dummynode.next = new ListNode(small_value);
            dummynode = dummynode.next;

            if (lists[small_index] != null){
                pq.add(new Pair(lists[small_index].val, small_index));
            }
        }
        return head.next;
    }
}
class Pair implements Comparable<Pair> {
    int a, b;
    public Pair(int a, int b) {
        this.a = a;
        this.b = b;
    }
    public int compareTo(Pair pair) {
        return a - pair.a;
    }
}
```

python3:

``` python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        dummynode = head = ListNode()
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i))
        while heap:
            # find the small index lst
            small_index = heapq.heappop(heap)[1]
            smallnode = lists[small_index]
            # update that index lst without head
            lists[small_index] = smallnode.next
            dummynode.next = smallnode
            dummynode = dummynode.next
            if smallnode.next:
                heapq.heappush(heap, (smallnode.next.val, small_index))
        return head.next
```

### [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)

java:

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode slow = dummy, fast = dummy;
        while (fast.next!=null && n>0){
            fast = fast.next;
            n --;
        }
        while (fast.next!=null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```

python3:

``` python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummynode = ListNode(next = head)
        slow = fast = dummynode
        for i in range(n):
            fast = fast.next
        while fast.next is not None:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummynode.next
```

### [143. Reorder List](https://leetcode.com/problems/reorder-list/description/)

java:

```java
class Solution {
    public void reorderList(ListNode head) {
        ListNode left = new ListNode(0, head);
        ListNode right = new ListNode(0, head);
        while (right != null && right.next != null){
            left = left.next;
            right = right.next.next;
        }
        ListNode prev = null, curr = left.next, next;
        left.next = null;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        ListNode curr_forw = head, curr_back = prev;
        while (curr_forw!=null && curr_back!=null){
            ListNode next_forw = curr_forw.next;
            ListNode next_back = curr_back.next;
            curr_forw.next = curr_back;
            curr_back.next = next_forw;
            curr_forw = next_forw;
            curr_back = next_back;
        }



    }
}
```

python3:

``` python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        left = right = ListNode(next = head)
        while right and right.next:
            left = left.next
            right = right.next.next
        prev, curr, next = None, left.next, None
        left.next = None
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        curr_forw, curr_back = head, prev
        while curr_forw and curr_back:
            next_forw, next_back = curr_forw.next, curr_back.next
            curr_forw.next, curr_back.next = curr_back, next_forw
            curr_forw, curr_back = next_forw, next_back
```

## Matrix

### [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

java:

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int[] rowset = new int[matrix.length];
        int[] colset = new int[matrix[0].length];
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0;j < matrix[0].length; j++){
                if (rowset[i]==1 && colset[j]==1)
                    continue;
                if (matrix[i][j] == 0){
                    rowset[i]=1;
                    colset[j]=1;
                }
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (rowset[i]==1 || colset[j]==1)
                    matrix[i][j] = 0;
            }
        }
    }
}
```

python3:

``` python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        rows, cols = set(), set()
        for i in range(m):
            for j in range(n):
                if i in rows and j in cols:
                    continue
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        for i in range(m):
            for j in range(n):
                if i in rows or j in cols:
                    matrix[i][j] = 0
```

### [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

java:

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        List<Integer> lst = new ArrayList<Integer>();
        boolean[][] seen = new boolean[row][col];
        int x=0, y=0, di=0, dj=1;
        while (lst.size() < row*col){
            seen[x][y] = true;
            lst.add(matrix[x][y]);
            int nextx = x+di, nexty = y+dj;
            if (!(0<=nextx && nextx<row && 0<=nexty && nexty<col && (!seen[nextx][nexty]) )){
                int temp = di;
                di = dj;
                dj = -temp;
            }
            x = x+di;
            y = y+dj;
        }
        return lst;
    }
}
```

python3:

``` python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        row, col = len(matrix), len(matrix[0])
        lst = []
        seen = [[False for _ in range(col)] for _ in range(row)]
        x, y, di, dj = 0, 0, 0, 1
        while len(lst) < row*col:
            seen[x][y] = True
            lst.append(matrix[x][y])
            nextx, nexty = x+di, y+dj
            if not (0<= nextx <row and 0<= nexty <col and (not seen[nextx][nexty])):
                di, dj = dj, -di
            x, y = x+di, y+dj
        return lst
```

### [48. Rotate Image](https://leetcode.com/problems/rotate-image/)

java:

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] temp = new int[n][n];
        for (int i = 0; i < n; i ++){
            for (int j = 0; j < n; j ++) {
                temp[j][n-1-i] = matrix[i][j];
            }
        }
        for (int i = 0; i < n; i ++){
            for (int j = 0; j < n; j ++) {
                matrix[i][j] = temp[i][j];
            }
        }
    }
}
```

python3:

``` python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        res = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                res[j][n-i-1] = matrix[i][j]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = res[i][j]
```

### [79. Word Search](https://leetcode.com/problems/word-search/)

java:

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length;
        char[] wordarray = word.toCharArray();
        for (int row = 0; row < m; row ++){
            for (int col = 0; col < n; col ++){
                boolean[][] visited = new boolean[m][n];
                if (board[row][col] == wordarray[0]){
                    if (dfs(board, row, col, visited, wordarray, 0))
                        return true;
                }
            }
        }
        return false;
    }
    public boolean dfs(char[][] board, int row, int col, boolean[][] visited, char[] word, int idx){
        if (word[idx] != board[row][col] || visited[row][col] == true || idx >= word.length)
            return false;
        if (word[idx] == board[row][col] && idx == word.length-1)
            return true;
        visited[row][col] = true;
        if (row > 0 && dfs(board, row-1, col, visited, word, idx + 1))
            return true;
        if (row < board.length-1 && dfs(board, row+1, col, visited, word, idx + 1))
            return true;
        if (col > 0 && dfs(board, row, col-1, visited, word, idx + 1))
            return true;
        if (col < board[0].length-1 && dfs(board, row, col+1, visited, word, idx + 1))
            return true;
        visited[row][col] = false;
        return false;
    }
}
```

python3:

``` python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m,n = len(board),len(board[0])
        for row in range(m):
            for col in range(n):
                if board[row][col] == word[0]:
                    if self.dfs(board, row, col, set(), word, 0): 
                        return True
        return False

    def dfs(self, board, row, col, visited, word, idx):
        if board[row][col] != word[idx] or (row, col) in visited or idx >= len(word):
            return False
        if board[row][col] == word[idx] and idx == len(word) - 1:
            return True
        visited.add((row, col))
        if row > 0 and self.dfs(board, row-1, col, visited, word, idx + 1): 
            return True
        if row < len(board) - 1 and self.dfs(board, row+1, col, visited, word, idx + 1): 
            return True
        if col > 0 and self.dfs(board, row, col-1, visited, word, idx + 1): 
            return True
        if col < len(board[0]) - 1 and self.dfs(board, row, col+1, visited, word, idx + 1): 
            return True
        visited.remove((row, col))
        return False
```

## String

### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

java:

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        char[] charArray = s.toCharArray();
        int[] window = new int[100];
        int left = 0;
        int res = 0;
        for (int right = 0; right < charArray.length; right ++){
            int index = charArray[right] - ' ';
            if (window[index] == 0) {
                window[index] ++;
                res = Math.max(right - left + 1, res);
            }else {
                while (charArray[left] != charArray[right]) {
                    window[charArray[left] - ' '] --;
                    left ++;
                }
                window[index] ++;
                window[charArray[left] - ' '] --;
                left ++;
            }
        }
        return res;
    }
}
```

python3:

``` python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        window = set()
        maxlength = 0
        left = 0
        for right, char in enumerate(s):
            if char not in window:
                window.add(char)
                maxlength = max(maxlength, right-left+1)
            else: 
                while s[left] != char:
                    window.remove(s[left])
                    left += 1
                window.remove(s[left])
                left += 1
                window.add(char)
        return maxlength
```

### [\424. Longest Repeating Character Replacement](bbb)

java:

```java

```

python3:

``` python

```

### [\76. Minimum Window Substring](bbb)

java:

```java

```

python3:

``` python

```

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)

java:

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length())
            return false;
        int[] dic = new int[26];
        for (char c : s.toCharArray())
            dic[c-'a'] ++;
        for (char c : t.toCharArray()){
            dic[c-'a'] --;
            if (dic[c-'a'] < 0) return false;
        }
        for (int i = 0; i < 26; i++) {
            if (dic[i] != 0)
                return false;
        }
        return true;
    }
}
```

python3:

``` python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)
```

### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

java:

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, ArrayList<String>> map = new HashMap<>();
        for (String str : strs){
            char[] ca = str.toCharArray();
            Arrays.sort(ca);
            String keystr = String.valueOf(ca);
            if (! map.containsKey(keystr))
                map.put(keystr, new ArrayList<String>());
            map.get(keystr).add(str);
        }
        return new ArrayList<>(map.values());
    }
}
```

python3:

``` python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strs_table = {}
        for string in strs:
            sorted_string = ''.join(sorted(string))
            if sorted_string not in strs_table:
                strs_table[sorted_string] = []
            strs_table[sorted_string].append(string)
        return list(strs_table.values())
```

### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)

java:

```java
class Solution {
    public boolean isValid(String s) {
        HashMap<Character, Character> map = new HashMap<Character, Character>();
        map.put(')','(');
        map.put('}','{');
        map.put(']','[');
        Stack<Character> stack = new Stack<Character>();
        for (char c : s.toCharArray()){
            if (! map.containsKey(c))
                stack.push(c);
            else if (stack.size() == 0 || map.get(c)!= stack.pop())
                return false;
        }
        if (stack.size() == 0)
            return true;
        return false;
    }
}
```

python3:

``` python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        parenthesis = {')':'(',']':'[','}':'{'}
        for char in s:
            if char not in parenthesis:
                stack.append(char)
            elif len(stack) ==0 or stack.pop() != parenthesis[char]:
                return False
        if len(stack) == 0:
            return True
        return False
```

### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/description/)

java:

```java
class Solution {
    public boolean isPalindrome(String s) {
        char[] charArray = s.toLowerCase().toCharArray();
        int i = 0, j = charArray.length-1;
        while (i < j){
            if (!Character.isLetterOrDigit(charArray[i]))
                i ++;
            else if (!Character.isLetterOrDigit(charArray[j]))
                j --;
            else {
                if (charArray[i] != charArray[j])
                    return false;
                i ++;
                j --;
            }
        }
        return true;
    }
}
```

python3:

``` python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join(filter(str.isalnum, str(s).lower()))
        for i in range(len(s)//2):
            if s[i] != s[len(s)-i-1]:
                return False
        return True
```

### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/)

java:

```java
class Solution {
    public String longestPalindrome(String s) {
        int[][] opt = new int[s.length()][s.length()];
        char[] sArray = s.toCharArray();
        int maxlength = 0;
        int left=0, right=0;
        for (int i=0; i<s.length(); i++){
            opt[i][i] = 1;
        }
        for (int i = sArray.length-1; i > -1; i--){
            for (int j = i+1; j < sArray.length; j++){
                if (sArray[i] == sArray[j]){
                    if (opt[i+1][j-1]>0 || j-i<2)
                        opt[i][j] = opt[i+1][j-1] + 2;
                    if (opt[i][j] > maxlength){
                        maxlength = opt[i][j];
                        left = i;
                        right = j;
                    }
                }
            }
        }
        return s.substring(left, right+1);
    }
}
```

python3:

``` python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        opt = [[0 for _ in range(len(s))] for _ in range(len(s))]
        maxlenth = 0
        left, right = 0, 0
        for i in range(0, len(s)):
            opt[i][i] = 1
        for i in range(len(s)-1, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    if opt[i+1][j-1]>0 or j-i<2:
                        opt[i][j] = opt[i+1][j-1] + 2
                    if opt[i][j] > maxlenth:
                        maxlenth = j - i + 1
                        left, right = i, j
        return s[left:right + 1]
```

### [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

java:

```java
class Solution {
    public int countSubstrings(String s) {
        boolean[][] opt = new boolean[s.length()][s.length()];
        char[] sArray = s.toCharArray();
        int result = 0;
        for (int i = sArray.length-1; i > -1; i--){
            for (int j = i; j < sArray.length; j++){
                if (sArray[i] == sArray[j]){
                    if (j-i <= 1){
                        result ++;
                        opt[i][j] = true;
                    }else if (opt[i+1][j-1]){
                        result ++;
                        opt[i][j] = true;
                    }
                }
            }
        }
        return result;
    }
}
```

python3:

``` python
class Solution:
    def countSubstrings(self, s: str) -> int:
        opt = [[0 for _ in range(len(s))] for _ in range(len(s))]
        result = 0
        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j]:
                    if j-i < 2 or opt[i+1][j-1]:
                        opt[i][j] = 1
                        result += 1
        return result
```

### [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)

(Leetcode Premium)

## Tree

### [100. Same Tree](https://leetcode.com/problems/same-tree/description/)

java:

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p==null && q==null)
            return true;
        else if (p==null || q==null)
            return false;
        else if (p.val == q.val)
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        return false;
    }
}
```

python3:

``` python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False
```

### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

java:

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> results = new ArrayList<>();
        if (root==null) return results;
        Deque<TreeNode> deque = new ArrayDeque<>();
        deque.add(root);
        while (! deque.isEmpty()){
            int size = deque.size();
            List<Integer> result = new ArrayList<>();
            for (int i=0; i<size; i++){
                TreeNode node = deque.poll();
                result.add(node.val);
                if (node.left!=null) deque.add(node.left);
                if (node.right!=null) deque.add(node.right);
            }
            results.add(result);
        }
        return results;
    }
}
```

python3:

``` python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        results = []
        if not root: return results
        queue = deque()
        queue.append(root)
        while queue:
            size = len(queue)
            result = []
            for _ in range(size):
                node = queue.popleft()
                result.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            results.append(result)
        return results
```

### [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

java:

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

python3:

``` python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: 
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

### [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)

java:

```java
class Solution {
    Map<Integer, Integer> map;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++)
            map.put(inorder[i], i);
        return helper(preorder, 0, preorder.length-1, inorder, 0, inorder.length-1);
    }

    public TreeNode helper(int[] preorder, int preleft, int preright, int[] inorder, int inleft, int inright){
        if (preleft>preright || inleft>inright) return null;
        int root_val = preorder[preleft];
        int pos = map.get(root_val);
        TreeNode root = new TreeNode(root_val);
        root.left = helper(preorder, preleft+1, preleft+pos-inleft, inorder, inleft, pos-1);
        root.right = helper(preorder, preleft+pos-inleft+1, preright, inorder, pos+1, inright);
        return root;
    }
}
```

python3:

``` python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        rootValue = preorder[0]
        root = TreeNode(rootValue)
        rootPos = inorder.index(rootValue)
        root.left = self.buildTree(preorder[1:rootPos+1], inorder[:rootPos])
        root.right = self.buildTree(preorder[rootPos+1:], inorder[rootPos+1:])
        return root
```

### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

java:

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root==null) return root;
        TreeNode parent = new TreeNode(root.val);
        parent.left = invertTree(root.right);
        parent.right = invertTree(root.left);
        return parent;
    }
}
```

python3:

``` python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        parent = TreeNode(root.val)
        parent.left, parent.right = self.invertTree(root.right), self.invertTree(root.left)
        return parent
```

### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/)

java:

```java
class Solution {
    int res;
    public int maxPathSum(TreeNode root) {
        res = root.val;
        traversal(root);
        return res;
    }

    public int traversal(TreeNode root) {
        if (root == null) return 0;
        int left = traversal(root.left);
        int right = traversal(root.right);
        res = Math.max(res, left + right + root.val);
        return Math.max(root.val + Math.max(left, right), 0);
    }
}
```

python3:

``` python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.ans = root.val
        self.maxend(root)
        return self.ans
    
    def maxend(self, node):
        if not node: return 0
        left = self.maxend(node.left)
        right = self.maxend(node.right)
        self.ans = max(self.ans, left+right+node.val)
        return max(node.val + max(left, right), 0)
```

### [297. Serialize and Deserialize Binary Tree](bbb)

java recursion:

```java
public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder result = new StringBuilder();
        buildString(root, result);
        return result.toString();
    }

    private void buildString(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append("null").append(",");
        } else {
            sb.append(node.val).append(",");
            buildString(node.left, sb);
            buildString(node.right,sb);
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>();
        queue.addAll(Arrays.asList(data.split(",")));
        return buildTree(queue);
    }

    private TreeNode buildTree(Queue<String> queue) {
        String val = queue.poll();
        if (val.equals("null")) return null;
        else {
            TreeNode node = new TreeNode(Integer.parseInt(val));
            node.left = buildTree(queue);
            node.right = buildTree(queue);
            return node;
        }
    }
}
```

java iteratively:

``` java
public class Codec {
    static Map<String, TreeNode> map = new HashMap<>();

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "";
        StringBuilder result = new StringBuilder();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (node==null){
                result.append("null"+", ");
                continue;
            }
            result.append(node.val+", ");
            queue.add(node.left);
            queue.add(node.right);
        }
        map.put(result.toString(), root);
        return result.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == "") return null;
        String[] values = data.split(", ");
        TreeNode root = new TreeNode(Integer.parseInt(values[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 0;
        while  (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if ((i+1<values.length) && (!values[i+1].equals("null"))){
                node.left = new TreeNode(Integer.parseInt(values[i+1]));
                queue.add(node.left);
            }
            if ((i+2<values.length) && (!values[i+2].equals("null"))){
                node.right = new TreeNode(Integer.parseInt(values[i+2]));
                queue.add(node.right);
            } 
            i += 2;
        }
        return root;
    }
}
```

python3:

``` python
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string."""
        result = []
        self.buildString(root, result)
        return "".join(result)

    def buildString(self, root, result):
        if not root: 
            result.append("#")
            result.append(",")
        else:
            result.append(str(root.val))
            result.append(",")
            self.buildString(root.left, result)
            self.buildString(root.right, result)

    def deserialize(self, data):
        """Decodes your encoded data to tree."""
        queue = deque()
        for str in data.split(","):
            queue.append(str)
        return self.buildTree(queue)
    
    def buildTree(self, queue):
        val = queue.popleft()
        if val == "#": return None
        else:
            node = TreeNode(int(val))
            node.left = self.buildTree(queue)
            node.right = self.buildTree(queue)
            return node
```

### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/description/)

java:

```java
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root != null && subRoot != null){
            if (root.val==subRoot.val){
                if (sameTree(root, subRoot)) return true;
            }
            return (isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot));
        }
        return false;
    }
    
    public boolean sameTree(TreeNode t1, TreeNode t2){
        if (t1 == null && t2 == null)
            return true;
        if (t1 == null || t2 == null || t1.val!=t2.val)
            return false;
        if (! sameTree(t1.left, t2.left))
            return false;
        if (! sameTree(t1.right, t2.right))
            return false;
        return true;
    }
}
```

python3:

``` python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if root is not None:
            if root.val == subRoot.val:
                if self.sameTree(root, subRoot):
                    return True
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        return False

    def sameTree(self, t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return self.sameTree(t1.left, t2.left) and self.sameTree(t1.right, t2.right)
```

### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/description/)

java:

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return trversal(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    public boolean trversal(TreeNode root, long min, long max){
        if (root == null) return true;
        if (root.val <= min || root.val >= max)
            return false;
        return trversal(root.left,min,root.val) && trversal(root.right,root.val,max);
    }
}
```

python3:

``` python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.isValid(root, float('-inf'), float('inf'))

    def isValid(self, root, minVal, maxVal):
        if not root: return True
        if root.val <= minVal or root.val >= maxVal:
            return False
        return self.isValid(root.left, minVal, root.val) and self.isValid(root.right, root.val, maxVal)
```

### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

java:

```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || ! stack.isEmpty()){
            if (curr != null){
                stack.push(curr);
                curr = curr.left;
            } else {
                curr = stack.pop();
                k --;
                if (k == 0) return curr.val;
                curr = curr.right;
            }
        }
        return curr.val;
    }
}
```

python3:

``` python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                k -= 1
                if k == 0: return root.val
                root = root.right
```

### [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)

java:

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root == p || root == q) return root;
        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        else if (root.val < p.val && root.val < q.val) 
            return lowestCommonAncestor(root.right, p, q);
        else return root;
    }
}
```

python3:

``` python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root==None or root==p or root==q:
            return root
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
```

### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/)

java:

```java
class TrieNode {
    boolean isWord;
    TrieNode[] children;
    public TrieNode() {
        isWord = false;
        children = new TrieNode[26];
    }
}
class Trie {
    TrieNode root;
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                node.children[index] = new TrieNode();
            }
            node = node.children[index];
        }
        node.isWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                return false;
            }
            node = node.children[index];
        }
        return node.isWord;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                return false;
            }
            node = node.children[index];
        }
        return true;
    }
}
```

python3:

``` python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

java:

```java
class WordNode {
    boolean isWord;
    WordNode[] children;
    public WordNode() {
        isWord = false;
        children = new WordNode[26];
    }
}
class WordDictionary {
    WordNode root;
    public WordDictionary() {
        root = new WordNode();
    }
    
    public void addWord(String word) {
        WordNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                node.children[index] = new WordNode();
            }
            node = node.children[index];
        }
        node.isWord = true;
    }
    
    public boolean search(String word) {
        WordNode node = root;
        return partialSearch(node, word.toCharArray(), 0);
    }

    public boolean partialSearch(WordNode node, char[] word, int start){
        if (start==word.length) return node.isWord;
        if (word[start] != '.'){
            int index = word[start] - 'a';
            if (node.children[index] != null){
                return partialSearch(node.children[index], word, start + 1);
            }
        } else {
            for (int i = 0; i < 26; i++) {
                if (node.children[i] != null) {
                    if (partialSearch(node.children[i], word, start + 1)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
```

python3:

``` python
class WordNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class WordDictionary:
    def __init__(self):
        self.root = WordNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = WordNode()
            node = node.children[char]
        node.is_word = True
        
    def search(self, word: str) -> bool:
        node = self.root
        return self.partialSearch(node, word, 0)
        
    def partialSearch(self, node: WordNode, word: str, start: int) -> bool:
        if start==len(word): return node.is_word
        char = word[start]
        if char != ".":
            if char in node.children:
                return self.partialSearch(node.children[char], word, start+1)
        else:
            for choice in node.children:
                if self.partialSearch(node.children[choice], word, start+1):
                    return True
        return False
```

### [212. Word Search II](https://leetcode.com/problems/word-search-ii/description/)

java:

```java
class TrieNode {
    TrieNode[] next = new TrieNode[26];
    String word;
}
class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        TrieNode root = buildTrie(words);
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, i, j, root, res);
            }
        }
        return res;
    }
    public TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int i = c - 'a';
                if (node.next[i] == null) 
                    node.next[i] = new TrieNode();
                node = node.next[i];
            }
            node.word = word;
        }
        return root;
    }
    public void dfs(char[][] board, int row, int col, TrieNode root, List<String> res){
        char c = board[row][col];
        if (c == '#' || root.next[c - 'a'] == null)
            return;
        root = root.next[c - 'a'];
        if (root.word != null) {
            res.add(root.word);
            root.word = null;
        }
        board[row][col] = '#';
        if (row > 0)
            dfs(board, row-1, col, root, res);
        if (row < board.length-1)
            dfs(board, row+1, col, root, res);
        if (col > 0)
            dfs(board, row, col-1, root, res);
        if (col < board[0].length-1)
            dfs(board, row, col+1, root, res);
        board[row][col] = c;
    }
}
```

python3:

``` python
class TrieNode:
    def __init__(self):
        self.next = {}
        self.word = None
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        self.res = []
        root = self.buildTrie(words)
        for row in range(len(board)):
            for col in range(len(board[0])):
                self.dfs(board, row, col, root)
        return self.res

    def buildTrie(self, words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.next:
                    node.next[char] = TrieNode()
                node = node.next[char]
            node.word = word
        return root
        
    def dfs(self, board, row, col, root):
        char = board[row][col]
        if char == "#" or char not in root.next:
            return
        root = root.next[char]
        if root.word:
            self.res.append(root.word)
            root.word = None
        board[row][col] = '#'
        if row > 0:
            self.dfs(board, row-1, col, root)
        if row < len(board) - 1:
            self.dfs(board, row+1, col, root)
        if col > 0:
            self.dfs(board, row, col-1, root)
        if col < len(board[0]) - 1:
            self.dfs(board, row, col+1, root)
        board[row][col] = char
```



## Heap

### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

java:

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        ListNode dummynode = new ListNode();
        ListNode head = dummynode;
        for (int i = 0; i<lists.length; i++){
            if (lists[i] != null){
                pq.add(new Pair(lists[i].val, i));
            }
        }
        while (! pq.isEmpty()){
            Pair small_pair = pq.poll();
            
            int small_value = small_pair.a;
            int small_index = small_pair.b;
            
            lists[small_index] = lists[small_index].next;
            dummynode.next = new ListNode(small_value);
            dummynode = dummynode.next;

            if (lists[small_index] != null){
                pq.add(new Pair(lists[small_index].val, small_index));
            }
        }
        return head.next;
    }
}
class Pair implements Comparable<Pair> {
    int a, b;
    public Pair(int a, int b) {
        this.a = a;
        this.b = b;
    }
    public int compareTo(Pair pair) {
        return a - pair.a;
    }
}
```

python3:

``` python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        dummynode = head = ListNode()
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i))
        while heap:
            # find the small index lst
            small_index = heapq.heappop(heap)[1]
            smallnode = lists[small_index]
            # update that index lst without head
            lists[small_index] = smallnode.next
            dummynode.next = smallnode
            dummynode = dummynode.next
            if smallnode.next:
                heapq.heappush(heap, (smallnode.next.val, small_index))
        return head.next
```

### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

java:

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> counter = new HashMap<>();
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        int[] result = new int[k];
        for (int num : nums)
            counter.put(num, counter.getOrDefault(num,0)+1);
        for (int num : counter.keySet())
            pq.add(new Pair(num, counter.get(num)));
        for (int i = 0; i < k; i++)
            result[i] = pq.poll().num;
        return result;
    }
}
class Pair implements Comparable<Pair> {
    int num, times;
    public Pair(int num, int times) {
        this.num = num;
        this.times = times;
    }
    public int compareTo(Pair pair) {
        return pair.times - times;
    }
}
```

python3:

``` python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        frac = collections.Counter(nums)
        priorityQueue = []
        for key,value in frac.items():
            heapq.heappush(priorityQueue, (-value, key))
        result = []
        for _ in range(k):
            result.append(heapq.heappop(priorityQueue)[1])
        return result
```

### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

java:

```java
class MedianFinder {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
    boolean isEven = true;
    
    public MedianFinder() {}
    
    public void addNum(int num) {
        if (isEven){
            minHeap.add(num);
            maxHeap.add(minHeap.poll());
        } else {
            maxHeap.add(num);
            minHeap.add(maxHeap.poll());
        }
        isEven = !isEven;
    }
    
    public double findMedian() {
        if (isEven) 
            return (maxHeap.peek() + minHeap.peek())/2.0d;
        return maxHeap.peek();
    }
}
```

python3:

``` python
class MedianFinder:
    def __init__(self):
        self.maxHeap = []
        self.minHeap = []
        self.isEven = True
        
    def addNum(self, num: int) -> None:
        if self.isEven:
            heappush(self.minHeap, num)
            heappush(self.maxHeap, -heappop(self.minHeap))
        else:
            heappush(self.maxHeap, -num)
            heappush(self.minHeap, -heappop(self.maxHeap))
        self.isEven = not self.isEven

    def findMedian(self) -> float:
        if self.isEven:
            return (-self.maxHeap[0] + self.minHeap[0]) / 2.0
        return -self.maxHeap[0]
```



## Important Link

### [14 Patterns to Ace Any Coding Interview Question](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed)

- One of the most common points of anxiety developers

  - **Have I solved enough practice questions? Could I have done more?**

  If you understand the generic patterns, you can use them as a template to solve a myriad of other problems with slight variations.

  Here, I’ve laid out the top 14 patterns that can be used to solve any coding interview question, as well as how to identify each pattern, and some example questions for each.

1. **Sliding Window**
   - s
2. **Two Pointers or Iterators**
3. **Fast and Slow pointers**
4. **Merge Intervals**
5. **Cyclic sort**
6. **In-place reversal of linked list**
7. **Tree BFS**
8. **Tree DFS**
9. **Two heaps**
10. **Subsets**
11. **Modified binary search**
12. **Top K elements**
13. **K-way Merge**
14. **Topological sort**







