# LeetCode Blind 75 LeetCode Questions

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

- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [Word Break Problem](https://leetcode.com/problems/word-break/)
- [Combination Sum](https://leetcode.com/problems/combination-sum-iv/)
- [House Robber](https://leetcode.com/problems/house-robber/)
- [House Robber II](https://leetcode.com/problems/house-robber-ii/)
- [Decode Ways](https://leetcode.com/problems/decode-ways/)
- [Unique Paths](https://leetcode.com/problems/unique-paths/)
- [Jump Game](https://leetcode.com/problems/jump-game/)





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

### [Meeting Rooms (Leetcode Premium)](https://leetcode.com/problems/meeting-rooms/)

### [Meeting Rooms II (Leetcode Premium)](https://leetcode.com/problems/meeting-rooms-ii/)

## Linked List

## Matrix

## String

## Tree

## Heap

## Important Link













