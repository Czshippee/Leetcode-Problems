# **LeetCode** 刷题攻略

<div align="center">
    <img src='./pics/leetcode.png'>
    </img>
</div> 
[TOC]

# 前序

## 1. 时间复杂度

- 什么是时间复杂度

  **时间复杂度是一个函数，它定性描述该算法的运行时间**。

  我们在软件开发中，时间复杂度就是用来方便开发者估算出程序运行的答题时间。

  如何估计程序运行时间，通常会估算算法的操作单元数量来代表程序消耗的时间，这里默认CPU的每个单元运行消耗的时间都是相同的。

  假设算法的问题规模为n，那么操作单元数量便用函数f(n)来表示，随着数据规模n的增大，算法执行时间的增长率和f(n)的增长率相同，这称作为算法的渐近时间复杂度，简称时间复杂度，记为 O(f(n))。

- 什么是大O

  **大O用来表示上线的**，当用它作为算法的最坏情况运行时间的上界，就是对任意数据输入的运行时间的上界。

  输入数据的形式对程序运算时间是有很大影响的，在数据本来有序的情况下时间复杂度是O(n)，但如果数据是逆序的话，插入排序的时间复杂度就是O(n^2)，也就对于所有输入情况来说，最坏是O(n^2) 的时间复杂度，所以称插入排序的时间复杂度为O(n^2)。

- 不同数据规模的差异

  <div align="center">
      <img src='./pics/qx1_1.png'>
      </img>
  </div> 
  
  
  在决定使用哪些算法的时候，不是时间复杂越低的越好（因为简化后的时间复杂度忽略了常数项等等），要考虑数据规模，如果数据规模很小甚至可以用O(n^2)的算法比O(n)的更合适（在有常数项的时候）。
  
  **因为大O就是数据量级突破一个点且数据量级非常大的情况下所表现出的时间复杂度，这个数据量也就是常数项系数已经不起决定性作用的数据量**。
  
  **所以我们说的时间复杂度都是省略常数项系数的，是因为一般情况下都是默认数据规模足够的大，基于这样的事实，给出的算法时间复杂的的一个排行如下所示**：
  
  $O(1)常数阶 < O(logn)对数阶 < O(n)线性阶 < O(n^2)平方阶 < O(n^3)(立方阶) < O(2^n) (指数阶)$

## 2. 空间复杂度

- 什么是空间复杂度

  对一个算法在运行过程中占用内存空间大小的量度，记做S(n)=O(f(n)。

  空间复杂度(Space Complexity)记作S(n) 依然使用大O来表示。利用程序的空间复杂度，可以对程序运行中需要多少内存有个预先估计。

  **空间复杂度是考虑程序运行时占用内存的大小，而不是可执行文件的大小。**

  空间复杂度不是准确算出程序运行时所占用的内存, 而是预先大体评估程序内存使用的大小。

- 什么时候的空间复杂度是O(1)

  随着n的变化，所需开辟的内存空间并不会随着n的变化而变化。

  ```c
  int j = 0;
  for (int i = 0; i < n; i++) {
      j++;
  }
  ```

- 什么时候的空间复杂度是O(n)

  ```c
  int* a = new int(n);
  for (int i = 0; i < n; i++) {
      a[i] = i;
  }
  ```


## 3. 二分法

- 一个有序数组寻找一个数字是否存在

  不断找中点数，$O(log_2N)$

- 大于等于某个数的最左侧index

  二分找缩小范围

- 无序array，相邻数不等，找局部最小

  二分法随机找局部左右



## 4. 十大排序算法

- 选择排序（Selection Sort）

  - **无论什么数据进去都是O(n2)的时间复杂度**

    - 在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，


    - 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
    
    - 以此类推，直到所有元素均排序完毕。


  - 不稳定排序

    ```python
    def selection_sort(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                if nums[i] > nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
        return nums
    ```


- 插入排序（Insertion Sort）

  - 插入排序是前面**已排序数组**找到插入的位置

    - 0-n，找最大，和第0个交换

    - 1-n找最大，和第1个交换

    - 以此类推


  - 稳定排序，内排序, 时间复杂度 $O(n^2)$

    ```python
    def insertion_sort(nums):
        n = len(nums)
        for i in range(1, n):
            while i > 0 and nums[i - 1] > nums[i]:
                nums[i - 1], nums[i] = nums[i], nums[i - 1]
                i -= 1
        return nums
    ```


- 冒泡排序（Bubble Sort）

  - 冒泡排序时针对**相邻元素之间**的比较，可以将大的数慢慢“沉底”(数组尾部)

  - 每相邻两个，小的放前面

  - 稳定排序，内排序，时间复杂度 $O(n^2)$

    ```python
    def bubble_sort(nums):
        n = len(nums)
        # 进行多次循环
        for c in range(n):
            for i in range(1, n - c):
                if nums[i - 1] > nums[i]:
                    nums[i - 1], nums[i] = nums[i], nums[i - 1]
        return nums
    ```

- 希尔排序（Shell Sort）

  - 插入排序进阶版, 与插入排序的不同之处在于，它会优先比较距离较远的元素。希尔排序又叫缩小增量排序。

    - 整个待排序的记录序列分割成为若干子序列分别进行直接插入排序

    - 选择增量gap=length/k，按增量序列个数k，对序列进行k 趟排序

    - 增量因子为1 时，整个序列作为一个表来处理，表长度即为整个序列的长度。


  - 非稳定排序，内排序, 希尔排序的时间复杂度和增量序列是相关的。

  - `{1,2,4,8,...}`这种序列并不是很好的增量序列，使用这个增量序列的时间复杂度（最坏情形）是$O(n^2)$

  - `1,3,7，...,2^k−1`，这种序列的时间复杂度(最坏情形)为 $O(n^{1.5})$；

    ```python
    def shell_sort(nums):
        n = len(nums)
        gap = n // 2
        while gap:
            for i in range(gap, n):
                while i - gap >= 0 and nums[i - gap] > nums[i]:
                    nums[i - gap], nums[i] = nums[i], nums[i - gap]
                    i -= gap
            gap //= 2
        return nums
    ```

- 归并排序（Merge Sort）

  - 采用是分治法，先将数组分成子序列，让子序列有序，再将子序列间有序，合并成有序数组。

  - 把长度为n的输入序列分成两个长度为n/2的子序列，对这两个子序列分别采用归并排序，将两个排序好的子序列合并成一个最终的排序序列。

  - 稳定排序， 外排序，时间复杂度 $O(nlogN)$， 额外空间复杂度O(N)

    ```python
    def merge_sort(nums):
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        # 分
        left = merge_sort(nums[:mid])
        right = merge_sort(nums[mid:])
        # 合并
        return merge(left, right)
    def merge(left, right):
        res = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        res += left[i:]
        res += right[j:]
        return res
    ```

- 快速排序（Quick Sort）

  - 选取一个 pivot，将小于pivot放在左边，把大于pivot放在右边，分割成两部分，并且可以固定pivot在数组的位置，在对左右两部分继续进行排序。

    - 从数列中挑出一个元素为pivot, 重新排序数列，

    - 所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面, 基准就处于数列的中间 (partition)

    - 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序

  - 不稳定排序，内排序，时间复杂度 $O(nlogN)$, 额外空间复杂度O(logN)

  - 快排2.0 分三个区，等于区 多处理一部分数

    ```python
    def quick_sort(nums):
        n = len(nums)
    
        def quick(left, right):
            if left >= right:
                return nums
            pivot = left
            i = left
            j = right
            while i < j:
                while i < j and nums[j] > nums[pivot]:
                    j -= 1
                while i < j and nums[i] <= nums[pivot]:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[pivot], nums[j] = nums[j], nums[pivot]
            quick(left, j - 1)
            quick(j + 1, right)
            return nums
    
        return quick(0, n - 1)
    ```

  - 荷兰国旗问题：

    - 1）选num=S，  [i] < num，[i] 和 <区下一个交换，<区右扩，i++
    - 2）[i] == num， i++
    - 3）[i] > num，[i] 和 > 区 前一个交换，>区左扩，i不变

- 堆排序（Heap Sort)

  - 堆排序是利用数据结构 堆 设计的排序算法。左child表示2i+1，右child表示2i+2

    - 建堆，从底向上调整堆，使得父亲节点比孩子节点值大，构成Max-Heap；

    - 交换堆顶和最后一个元素(排除)，重新调整堆

    - heapsize控制堆范围，某个index数字往下移动叫heapify，两个child取最大，parent和child取最大

    - 向上调整叫heapinsert

  - 不稳定排序，内排序，时间复杂度为$O(nlogN)$

    ```python
    def heap_sort(nums):
        # 调整堆
        # 迭代写法
        def adjust_heap(nums, startpos, endpos):
            newitem = nums[startpos]
            pos = startpos
            childpos = pos * 2 + 1
            while childpos < endpos:
                rightpos = childpos + 1
                if rightpos < endpos and nums[rightpos] >= nums[childpos]:
                    childpos = rightpos
                if newitem < nums[childpos]:
                    nums[pos] = nums[childpos]
                    pos = childpos
                    childpos = pos * 2 + 1
                else:
                    break
            nums[pos] = newitem
        
        # 递归写法
        def adjust_heap(nums, startpos, endpos):
            pos = startpos
            chilidpos = pos * 2 + 1
            if chilidpos < endpos:
                rightpos = chilidpos + 1
                if rightpos < endpos and nums[rightpos] > nums[chilidpos]:
                    chilidpos = rightpos
                if nums[chilidpos] > nums[pos]:
                    nums[pos], nums[chilidpos] = nums[chilidpos], nums[pos]
                    adjust_heap(nums, pos, endpos)
    
        n = len(nums)
        # 建堆
        for i in reversed(range(n // 2)):
            adjust_heap(nums, i, n)
        # 调整堆
        for i in range(n - 1, -1, -1):
            nums[0], nums[i] = nums[i], nums[0]
            adjust_heap(nums, 0, i)
        return nums
    ```

- 计数排序（Counting Sort)

  - 将输入的数据值转化为键存储在额外开辟的数组空间中，典型的空间换时间算法。

  - 作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。

    - 找出待排序的数组中最大和最小的元素

    - 统计数组中每个值为i的元素出现的次数，存入数组C的第i项

    - 反向填充目标数组

  - 稳定排序，外排序，时间复杂度O(n)，但是对于数据范围很大的数组，需要大量时间和内存。

    ```python
    def counting_sort(nums):
        if not nums: return []
        n = len(nums)
        _min = min(nums)
        _max = max(nums)
        tmp_arr = [0] * (_max - _min + 1)
        for num in nums:
            tmp_arr[num - _min] += 1
        j = 0
        for i in range(n):
            while tmp_arr[j] == 0:
                j += 1
            nums[i] = j + _min
            tmp_arr[j] -= 1
        return nums
    ```

- 桶排序（Bucket Sort）

  - 桶排序是计数排序的升级版, 输入数据服从均匀分布的，将数据分到有限数量的桶里，每个桶再分别排序

    - 人为设置一个桶的`BucketSize`，作为每个桶放置多少个**不同数值**（可以放无数个相同数值）

    - 遍历待排序数据，并且把数据一个一个放到对应的桶里去

    - 对每个不是桶进行排序，可以使用其他排序方法，也递归排序

    - 不是空的桶里数据拼接起来

  - 稳定排序，外排序，时间复杂度O(n + k)，`k`为桶的个数。

    ```python
    def bucket_sort(nums, bucketSize):
        if len(nums) < 2:
            return nums
        _min = min(nums)
        _max = max(nums)
        # 需要桶个数
        bucketNum = (_max - _min) // bucketSize + 1
        buckets = [[] for _ in range(bucketNum)]
        for num in nums:
            # 放入相应的桶中
            buckets[(num - _min) // bucketSize].append(num)
        res = []
    
        for bucket in buckets:
            if not bucket: continue
            if bucketSize == 1:
                res.extend(bucket)
            else:
                # 当都装在一个桶里,说明桶容量大了
                if bucketNum == 1:
                    bucketSize -= 1
                res.extend(bucket_sort(bucket, bucketSize))
    ```

- 基数排序（Radix Sort)

  - 基数排序是对数字每一位进行排序，从最低位开始排序

    - 找到数组最大值，得最大位数，其他低位数用0填充

    - `arr`为原始数组，从最低位开始取每个位组成`radix`数组

    - 对`radix`进行计数排序，按照每个位复原数组

    - 升高位再继续

  - 稳定排序，外排序，时间复杂度 $O(k*n)$, 其中k为最大位数，n为元素个数

    ```python
    def Radix_sort(nums):
        if not nums: 
            return []
        
        _max = max(nums)
        
        maxDigit = len(str(_max)) # 最大位数
        bucketList = [[] for _ in range(10)]
        
        # 从低位开始排序
        div, mod = 1, 10
        for i in range(maxDigit):
            for num in nums:
                bucketList[num % mod // div].append(num)
            div *= 10
            mod *= 10
            idx = 0
            for j in range(10):
                for item in bucketList[j]:
                    nums[idx] = item
                    idx += 1
                bucketList[j] = []
        return nums
    ```

### 总结

|          | 时间复杂度   | 空间复杂度  | 稳定性 |
| -------- | ------------ | ----------- | ------ |
| 选择排序 | $O(N^2)$     | $ O(1)$     | 不稳定 |
| 冒泡排序 | $O(N^2)$     | $ O(1)$     | 稳定   |
| 插入排序 | $O(N^2)$     | $ O(1)$     | 稳定   |
| 归并排序 | $O(Nlog(N))$ | $O(N)$      | 稳定   |
| 快速排序 | $O(Nlog(N))$ | $O(log(N))$ | 不稳定 |
| 堆排序   | $O(Nlog(N))$ | $ O(1)$     | 不稳定 |

- 归并排序空间复杂度高，但是可以稳定

- 快速排序常数项低，空间复杂度没有O(1)，也做不到稳定

- 堆排序空间使用低

- 没有低于$O(Nlog(N))$时间复杂度的排序







# 正文

## 1.数组和字符串

### 1.1 什么是数组

- 数组是非常基础的数据结构, **是存放在连续内存空间上的相同类型数据的集合**

- 数组可以方便的通过下标索引的方式获取到下标下对应的数据

  <img src=".\pics\zw1_1.png" alt="zw1_1" style="zoom:50%;" />

- **数组下标都是从0开始的。**
- **数组内存空间的地址是连续的, 所以我们在删除或者增添元素的时候，就难免要移动其他元素的地址**
- 数组的元素是**不能删的，只能覆盖**(java and c++)

### 1.2 二维数组

- 在C++中二维数组内存的空间地址是连续分布的

  <img src=".\pics\zw1_2.png" alt="zw1_2" style="zoom:50%;" />

### 1.3 数组常见面试问题

【题目】给定一个数组和一个值，找到这个值

【要求】时间复杂度O(logn)

【思路】二分法， 要在二分查找的过程中，保持不变量，就是在while寻找中每一次边界的处理都要坚持根据区间的定义来操作



【题目】给定一个数组和一个值，删掉数组所有等于这个值的元素

【要求】空间复杂度O(1)

【思路】双指针法（快慢指针法）：通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。



【题目】给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组

【要求】时间复杂度O(n)

【思路】滑动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。从而将O(n^2)的暴力解法降为O(n)。



### 1.4 什么是字符串

- 字符串是 Python 中最常用的数据类型。我们可以使用 **'** 或 **"** 来创建字符串
- Python 不支持单字符类型，单字符在 Python 中也是作为一个字符串使用
- Python 访问子字符串，可以使用方括号来截取字符串 ： 'Hello World'[1:5]
- Python 中三引号可以将复杂的字符串进行赋值，允许一个字符串跨多行，字符串中可以包含换行符、制表符以及其他特殊字符。



### 1.5 前缀树

- 构造前缀树：起始一个空节点，没有的字符加一条edge，edge的weight是这个字符，一串字符一是条路
- 但是相同前部分的字符会共享某些边
- 每个点会记录 pass 和 end 值，头节点的pass值就是所有字符串数量
- 单词加入过几次：看最后end值
- 有几个是以这个单词作为前缀： 看最后pass值
- 删除一个词：先search一遍，确实单词存在，然后沿途pass值-1，最后end值-1



### 1.6 KMP算法

- 什么是KMP： 由三位学者Knuth，Morris 和 Pratt发明的，所以取了三位学者名字的首字母，叫做KMP

- KMP有什么用
  - KMP主要应用在字符串匹配上
  - 主要思想是**当出现字符串不匹配时，可以知道一部分之前已经匹配的文本内容，可以利用这些信息避免从头再去做匹配**
  - 所以如何记录已经匹配的文本内容，是KMP的重点，也是next数组肩负的重任
  - **KMP在字符串匹配中极大的提高的搜索的效率**

- 前缀表
  - KMP需要用到next数组，next数组就是一个前缀表（prefix table）
  - **前缀表是用来回退的，它记录了模式串与主串(文本串)不匹配的时候，模式串应该从哪里开始重新匹配**
  - 暴力匹配，如果发现不匹配，就要从头匹配
  - 如果使用前缀表，就不会从头匹配，而是从上次已经匹配的内容开始匹配
  - 前缀表：**记录下标i之前（包括i）的字符串中，有多大长度的相同前缀后缀。**

- 最长公共前后缀

  - 字符串前缀是指**不包含最后一个字符的所有以第一个字符开头的连续子串**

  - 字符串后缀是指**不包含第一个字符的所有以最后一个字符结尾的连续子串**

  - 通过匹配时可以知道， 匹配失败时候，A串与B串有一段相同的字串

  - 例子： AABAABAAF match AABAAF

    AABAAB 和 AABAAF 匹配失败，但是AABAAB 后缀AAB 和 AABAAF前缀相同，所以我们可以直接从AAB开始匹配

- 前缀表与next数组

  - next数组即可以就是前缀表，也可以是前缀表统一减一（右移一位，初始位置为-1）

  - 快速构建next数组, B串与自己匹配，求B[1]到B[i]最长公共前后缀长度

    - next[0] = -1, next[1] = 0，之后依次计算最长公共前后缀
    - 匹配的时候，当遇到匹配不上的时候，就可以寻找最大的前缀和后缀匹配长度
    - ‘aabaabc’ next数组为 [ -1, 0, 1, 0, 1, 2, 3]
  
    ```python
    # needle 为目标字符串
    def getnext(needle):
        if len(needle) == 1:
            return[-1]
        nextArr = [0 for _ in range(len(needle))]
        nextArr[0], nextArr[1] = -1, 0
        i, current = 2, 0
        while i < len(nextArr):
            if needle[i-1] == needle[current]:
                current += 1
                nextArr[i] = current
                i += 1
            elif current > 0:
                current = nextArr[current]
            else:
                nextArr[i] = 0
                i += 1
        return nextArr
    ```
  
  - 有了next数组，就可以根据next数组来 匹配文本串s，和模式串t
  
    ```python
    def KMP(string, needle):
        i, j = 0, 0
        nextArr = getnext(needle)
        while (i<len(string) and j<len(needle)):
            if string[i] == needle[j]: 
                i,j = i+1, j+1
            elif j == 0: i += 1
            else: j = nextArr[j]
        if j == len(needle):
            return i-j
        else:
            return -1
    ```
    
    

## 2. 哈希表和有序表

### 2.1 哈希表（hash table）

- 哈希表（Hash table）是根据关键码的值而直接进行访问的数据结构，在使用层面上可以理解为一种集合结构。

- 哈希表中关键码就是数组的索引下表，然后通过下表直接访问数组中的元素，如下图所示：

  <img src=".\pics\zw2_1.png" alt="zw2_1" style="zoom:50%;" />

- 一般哈希表都是用来快速判断一个元素是否出现集合里。查询一个元素是否存在时间复杂度是O(n)，但如果使用哈希表的话， 只需要O(1) 就可以做到。

- 如果哈希表只有key，没有伴随数据 value，叫**HashSet** (unorderedSet、unsortedSet)

- 如果哈希表有key，也有伴随数据 value，叫**HashMap** (unorderedMap、unsortedMap)

- 使用哈希表增(put)、删(remove)、改(put)和查(get)的操作，可以认为时间复杂度为O(1)，但是常数时间比较大

### 2.2 哈希函数（hash function）

- 哈希函数，将**哈希表**中元素的关键键值映射为元素存储位置的**函数**。
- 输入域无穷，可以接受任意长度，输出域相对有限, MD5输出域是$(0, 2^{64}-1)$, SHA-1输出域是$(0 ,2^{128}-1)$
- 相同输入导致相同输出，不相同输入也会导致相同输出（哈希碰撞，几率很低）
- 哈希函数生成结果具有均匀性和离散性
- 可以把任意长度的输入通过hash算法变换成固定长度的输出，该输出就是hash value。
- 如果hash Code得到的数值大于 哈希表的大小了，也就是大于tableSize了，会发生哈希碰撞（hash collision）

### 2.3 哈希碰撞（hash collision）

- 所有hash 函数都有如下一个基本特性：如果两个hash 值是不相同的（根据同一函数），那么这两个hash 值的原始输入也是不相同的。
- 如果两个hash 值相同，两个输入值很可能是相同的，但也可能不同，这种情况称为“**哈希碰撞**”

- 一般哈希碰撞有两种解决方法， 拉链法和线性探测法。

- **拉链法**

  - 由于不同的Key，通过hash function可能会算的同样的 hash value，所以此时用了拉链法解决冲突，把HashCode相同的Value连成链表. 

  - 拉链法就是要选择适当的哈希表的大小，这样既不会因为数组空值而浪费大量内存，也不会因为链表太长而在查找上浪费太多时间。

- **线性探测法**

  - 使用线性探测法，一定要保证tableSize大于dataSize。 我们需要依靠哈希表中的空位来解决碰撞问题。

  - 当hash function对一个给定值产生一个key，并且这个键指向hash table中某个已经被另一个key所占用的单元时，线性探测用于解决此时产生的冲突：

  - 查找hash table中离冲突单元最近的空闲单元，并且把新的key插入这个空闲单元。同样的，查找也同插入如出一辙：从hash function给出的hash value对应的单元开始查找，直到找到与键对应的值或者是找到空单元。

### 2.4 哈希法

- 当我们要使用集合来解决哈希问题的时候，优先使用unordered_set，因为它的查询和增删效率是最优的
- 如果需要集合是有序的，那么就用set，如果要求不仅有序还要有重复数据的话，那么就用multiset
- 那么再来看一下map ，在map 是一个key value 的数据结构，map中，对key是有限制，对value没有限制的，因为key的存储方式使用红黑树实现的。
- 总结一下，**当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法**。
- 哈希法也是**牺牲了空间换取了时间**，因为我们要使用额外的数组，set或者是map来存放数据，才能实现快速的查找。

### 2.5. 有序表介绍

- 有序表和哈希表的本质区别就是：哈希表的Key是通过hash function组织的，而有序表的Key是顺序组织的，可以理解为一种集合结构。

- 如果有序表有key，没有伴随数据 value，叫TreeSet (OrderedSet、sortedset)

- 如果有序表有key，有伴随数据 value，叫TreeMap (OrderedMap、sortedmap)

- 有序表除了支持哈希表的所有操作之外，还提供了一些由于Key的有序性可以实现的其他操作。

  最大或最小的Key对应的Value、给定一个Key找到比它小且最近的Key对应的Value是多少

- 有序表所有操作的时间复杂度都是O(logN)，非常高效。

- 很多结构都可以实现有序表。例如：红黑树（Red Black tree）、AVL树、SB树（Size Balanced Tree）、跳表（Skip List）。

  这些数据结构实现的有序表的性能指标都一样，都是O(logN)。

- 由于它们各自实现有序表的原理不同，因此即使时间复杂度有区别也只是常数时间的差距，而且常数时间的差距也比较小。

### 2.6 有序表固定操作

- 增加或者更新
- 查询某一个值 get(key): 根据给定的key，查询value并返回
- 移除某个值 remove(key): 移除key的记录
- 查询是否存在 containsKey(K key): 询问是否有关于key的记录
- 返回最大最小key  K firstKey() / K lastKey()
- 返回key附近值 floorKey(key) / ceilingKey(key)
- 所有操作时间复杂度都是O(logN)

### 2.7 布隆过滤器

- 1970年由布隆提出， 它实际上是**一个很长的二进制向量和一系列随机映射函数**。
- 布隆过滤器可以用于检索一个元素是否在一个集合中。 
- 它的优点是空间效率和查询时间都远远超过一般的算法，缺点是有一定的误识别率和删除困难。

- n = 样本量， P=失误率
  - $m=-\frac{n*lnP}{(ln2)^2}$
  - $k=ln2*\frac{m}{n}$ (向上取整)
  - $P_真=(1-e^{-n*k_真/m_真})^{k_真}$

### 2.8 一致性哈希原理

- 哈希就是一个键值对存储，在给定键的情况下，可以非常高效地找到所关联的值

- 当数据太大而无法存储在一个节点或机器上时，系统中需要多个这样的节点或机器来存储它

- **最简单的解决方案是使用哈希取模 **确定哪个 key 存储在哪个节点

- ```ini
  node_number = hash(key) % N # 其中 N 为节点数。
  ```

- 一致性哈希算法对简单哈希算法进行了修正 ：
  - 首先，对存储节点的哈希值进行计算，其将存储空间抽象为一个环，将存储节点配置到环上。环上所有的节点都有一个值。
  - 其次，对数据进行哈希计算，按顺时针方向将其映射到离其最近的节点上去。

- 问题：
  - 机器很少的时候，很难做到环均分
  - 即便可以均分，那加一个机器，很难做到均分
- 解决方法：虚拟节点



## 3. 链表

### 3.1 关于链表

- 链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）

  <img src=".\pics\zw3_1.png" alt="zw3_1" style="zoom:67%;" />

### 3.2 链表的类型

- 单链表

  **单链表**是一种链式存取的数据结构，用一组地址任意的存储单元存放线性表中的数据元素。 **链表**中的数据是以结点来表示的，每个结点的构成：元素(数据元素的映象) + 指针(指示后继元素存储位置)，元素就是存储数据的存储单元，指针就是连接每个结点的地址数据。

- 双链表

  **双链表**的每个数据结点中都有两个指针，分别指向上一个节点和下一个节点。 所以，从**双向链表**中的任意一个结点开始，都可以很方便地访问它的前后节点。

  <img src=".\pics\zw3_2.png" alt="zw3_2" style="zoom:67%;" />

- 循环链表

  **循环链表**是一种链式存储结构，它的最后一个结点指向头结点，形成一个环。因此，从循环链表中的任何一个结点出发都能找到任何其他结点。

  循环链表可以用来解决约瑟夫环问题。

  <img src=".\pics\zw3_3.png" alt="zw3_3" style="zoom:67%;" />



### 3.3 链表的存储方式

- 数组是在内存中是连续分布的，但是链表在内存中可不是连续分布的。
- 链表是通过指针域的指针链接在内存中各个节点。链表中的节点在内存中不是连续分布的 ，而是散乱分布在内存中的某地址上，分配机制取决于操作系统的内存管理。 各个节点分布在内存个不同地址空间上，通过指针串联在一起。

### 3.4 链表的定义

- 单链表

  ```python
  class ListNode(object):
      def __init__(self, data=None):
          self.data = data  # Assign data
          self.next = None  # Initialize next as null
  
  class LinkedList: 
      def __init__(self):
          self.head = None
  ```

- 双链表

  ```python
  class ListNode(object):
      def __init__(self, next=None, prev=None, data=None):
          self.data = data
          self.next = next # reference to next node in DLL
          self.prev = prev # reference to previous node in DLL
  ```


### 3.5 链表的操作

- 删除节点

  删除D节点，只要将C节点的next指针指向E节点就可以

- 添加节点

  添加C节点在AB中间，只要将A节点的next指针指向C节点, C节点的next指针指向B节点就可以

  链表的增添和删除都是O(1)操作，也不会影响到其他节点。

  但是删除某个节点，需要从头节点查找到前一个节点通过next指针进行删除操作，查找的时间复杂度是O(n)

### 3.6 链表常见面试问题

- 【题目】给定一个可能有环也可能无环的单链表，头节点head, 请实现如果有环，返回相交的第一个节点。如果无环，返回null。

  【要求】如果链表长度为N，时问复杂度请达到0(N)，额外空间复杂度请达到0 (1)。

  【思路】用哈希表，但是额外空间大；一直走走到null就是无环，走不到就是有环； 使用快慢指针，相遇后会到开头，每次走一步，再次相遇就是相交节点

  

- 【题目】给定两个可能有也可能相交的单链表，头节点head1和head2。请实现一个如果两个链表相交，返回相交的 第一个节点。如果不相交，返回null

  【要求】如果两个链表长度之和为N，时问复杂度请达到0(N)，额外空间复杂度请达到0 (1)。

  【思路】有相交意味着后面有重合，检查是否tail内存地址是否一样

  

## 4. 栈与队列

### 4.1 什么是栈

- 栈 (Stack)，是限制插入与删除操作只能在末端 (Top) 进行的线性存储结构
- 栈只能从一端存取数据，而另外一端是封闭的
- 栈的开口端被称为栈顶；封口端被称为栈底

- 先进后出 (LIFO: Last In First Out)

  ![zw4_1](.\pics\zw4_1.png)

### 4.2 栈的操作集

- 栈的核心操作有: Push、Pop、Top（peek）

- 进栈 Push：将新元素放到**栈顶**元素的上面，成为新的栈顶元素（入栈或压栈）

- 出栈 Pop：将**栈顶**元素删除掉，使得与其相邻的元素成为新的栈顶元素（退栈或弹栈）

- 顺序结构：使用数组实现 或 链式结构：使用链表实现

- ```python
  # using list
  stack = []
  # append() function to push element in the stack
  stack.append('a')
  # pop() function to pop element from stack in LIFO order
  print(stack.pop())
  ```

  

### 4.3 什么是队列

- 队列(Queue)是一种要求数据只能从一端进，从另一端出且遵循 "先进先出" 原则的线性存储结构
- 进数据的一端为 "队尾"，出数据的一端为 "队头"
- 先进先出 (FIFO: First In First Out)

### 4.4 队列的操作集

- 队列的核心操作集有: Enqueue、Dequeue

- ```python
  from collections import deque
  queue = deque()
  # append() function to push element in the stack
  queue.append('a')
  # pop() function to pop element from stack in LIFO order
  print(queue.popleft())
  ```


### 4.5 Priority Queue (优先队列)

- 优先队列也是一种队列，只不过不同的是，优先队列的出队顺序是按照优先级来的

- 如果最小键值元素拥有最高的优先级，那么这种优先队列叫作**升序优先队列**（即总是先删除最小的元素）

- 如果最大键值元素拥有最高的优先级，那么这种优先队列叫作**降序优先队列**（即总是先删除最大的元素）

- 一个典型的优先级队列支持以下操作:

  - `insert(key, data)`：插入键值为key的数据到优先队列中，元素以其key进行排序； 
  - `deleteMin/deleteMax`：删除并返回最小/最大键值的元素； 
  - `getMinimum/getMaximum`：返回最小/最大剑指的元素，但不删除它；
  - `topkMax/topkMin`：返回优先队列中键值为第k个最小/最大的元素；
  - `大小（size）`：返回优先队列中的元素个数；
  - `堆排序（Heap Sort）`：基于键值的优先级将优先队列中的元素进行排序；

- ```python
  import heapq
  lst = [5, 7, 9, 1, 3]
  # using heapify to convert list into heap
  heapq.heapify(lst)
  # using heappush() to push elements into heap pushes 4
  heapq.heappush(lst,4)
  # using heappop() to pop smallest element
  print (heapq.heappop(lst))
  ```

  

### 4.6 栈与队列常见面试问题

【题目】用栈实现队列

【思路】pop的时候, **输出栈如果为空，就把进栈数据全部导入进来**, 否则直接**从出栈弹出数据**就可以



【题目】用队列实现栈

【思路】只需要一个queue，一直popleft到剩下一个就可以



【题目】前k个元素

【思路】要统计最大前k个元素，只有小顶堆每次将最小的元素弹出，最后小顶堆里积累的才是前k个最大元素**



## 5. 二叉树

### 5.1 二叉树的种类

- 在我们解题过程中二叉树有两种主要的形式：满二叉树和完全二叉树。

- 满二叉树 (Full Binary Tree)

  - 如果一棵二叉树只有度为0的结点和度为2的结点，并且度为0的结点在同一层上，则这棵二叉树为满二叉树。

  - 满二叉树也可以说深度为k，有$2^k-1$个节点的二叉树。

  <img src=".\pics\zw5_1.png" alt="zw5_1" style="zoom: 33%;" />

- 完全二叉树 (Complete Binary Tree)

  - 在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。

  - 若最底层为第 h 层，则该层包含 [1, $2^h -1$]  个节点。

  - 堆就是一棵完全二叉树，同时保证父子节点的顺序关系。

    <img src=".\pics\zw5_2.png" alt="zw5_2" style="zoom:50%;" />

- 二叉搜索树 (Binary Search Tree)
  - 二叉搜索树是有数值的了，**二叉搜索树是一个有序树**
  - 若它的left tree不空，则左子树上所有结点的值均小于它的根结点的值；
  - 若它的right tree不空，则右子树上所有结点的值均大于它的根结点的值；
  - 它的左、右子树也分别为二叉排序树(Binary Sort Tree)

- 平衡二叉搜索树 (AVL / Adelson-Velsky and Landis)

  - 它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

  - **C++中map、set、multimap，multiset的底层实现都是平衡二叉搜索树**，所以map、set的增删操作时间时间复杂度是logn

  - 注意unordered_map、unordered_set，unordered_map、unordered_map底层实现是哈希表。

    <img src=".\pics\zw5_3.png" alt="zw5_3" style="zoom:50%;" />

### 5.2 二叉树的存储方式

- 二叉树可以链式存储，也可以顺序存储。

- 链式存储方式就用指针， 顺序存储的方式就是用数组。

- 顾名思义就是顺序存储的元素在内存是连续分布的，而链式存储则是通过指针把分布在散落在各个地址的节点串联一起。

- 顺序存储原理：**如果父节点的数组下表是i，那么它的左孩子就是i * 2 + 1，右孩子就是 i * 2 + 2**

- 二叉树节点的定义

  ```c++
  struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };
  TreeNode* a = new TreeNode(9);
  ```

### 5.3 二叉树的遍历方式

- 二叉树主要有两种遍历方式：

  - 深度优先遍历：先往深走，遇到叶子节点再往回走。
    - 前序遍历（递归法，迭代法）Preorder Traversal
    
      ```python
      # 递归
      class Solution:
          def preorderTraversal(self, root: TreeNode) -> List[int]:
              result = []
              
              def traversal(root: TreeNode):
                  if root == None:
                      return
                  result.append(root.val) # 前序
                  traversal(root.left)    # 左
                  traversal(root.right)   # 右
      
              traversal(root)
              return result
      # 迭代
      class Solution:
          def preorderTraversal(self, root: TreeNode) -> List[int]:
              # 根结点为空则返回空列表
              if not root:
                  return []
              stack = [root]
              result = []
              while stack:
                  node = stack.pop()
                  # 中结点先处理
                  result.append(node.val)
                  # 右孩子先入栈
                  if node.right:
                      stack.append(node.right)
                  # 左孩子后入栈
                  if node.left:
                      stack.append(node.left)
              return result
      ```
    
    - 中序遍历（递归法，迭代法）Inorder Traversal
    
      ```python
      class Solution:
          def inorderTraversal(self, root: TreeNode) -> List[int]:
              result = []
      
              def traversal(root: TreeNode):
                  if root == None:
                      return
                  traversal(root.left)    # 左
                  result.append(root.val) # 中序
                  traversal(root.right)   # 右
      
              traversal(root)
              return result
      # 迭代
      class Solution:
          def inorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return []
              stack = []  # 不能提前将root结点加入stack中
              result = []
              cur = root
              while cur or stack:
                  # 先迭代访问最底层的左子树结点
                  if cur:     
                      stack.append(cur)
                      cur = cur.left		
                  # 到达最左结点后处理栈顶结点    
                  else:		
                      cur = stack.pop()
                      result.append(cur.val)
                      # 取栈顶元素右结点
                      cur = cur.right	
              return result
      ```
    
    - 后序遍历（递归法，迭代法）Postorder Traversal
    
      ```python
      class Solution:
          def postorderTraversal(self, root: TreeNode) -> List[int]:
              result = []
      
              def traversal(root: TreeNode):
                  if root == None:
                      return
                  traversal(root.left)    # 左
                  traversal(root.right)   # 右
                  result.append(root.val) # 后序
      
              traversal(root)
              return result
      # 迭代
      class Solution:
          def postorderTraversal(self, root: TreeNode) -> List[int]:
              if not root:
                  return []
              stack = [root]
              result = []
              while stack:
                  node = stack.pop()
                  # 中结点先处理
                  result.append(node.val)
                  # 左孩子先入栈
                  if node.left:
                      stack.append(node.left)
                  # 右孩子后入栈
                  if node.right:
                      stack.append(node.right)
              # 将最终的数组翻转
              return result[::-1]
      ```
    
  - 广度优先遍历：一层一层的去遍历。
    
    - 层次遍历（迭代法）Postorder Traversal

- 前中后序遍历指的就是中间节点的遍历顺序

  <img src=".\pics\zw5_4.png" alt="zw5_4" style="zoom:33%;" />

- 二叉树中深度优先和广度优先遍历实现方式，做二叉树相关题目使用递归的方式来实现深度优先遍历是比较方便的

  而广度优先遍历的实现一般使用队列来实现，这也是队列先进先出的特点所决定的，因为需要先进先出的结构，才能一层一层的来遍历二叉树。

- 递归遍历：先print parent， 然后 f（left child）和 f（right child）

- 非递归遍历：前中后序遍历的逻辑其实都是可以借助栈使用非递归的方式来实现的，**栈其实就是递归的一种是实现结构**

  先把head放到stack里面，while stack非空: pop head,  head右非空，先放到stack里面，head左非空，再放到stack里面

  其次，前序方法颠倒’头右左‘然后再放到另一个stack，reverse后是后序

  中序的话，先把最左边边界全放进stack，每次pop一个检查右树，继续放左边

### 5.4 二叉树的定义

- 二叉树有两种存储方式顺序存储，和链式存储，顺序存储就是用数组来存

- 链式存储的二叉树节点的定义:

  Java：

  ```
  public class TreeNode {
      int val;
    	TreeNode left;
    	TreeNode right;
    	TreeNode() {}
    	TreeNode(int val) { this.val = val; }
    	TreeNode(int val, TreeNode left, TreeNode right) {
      		this.val = val;
      		this.left = left;
      		this.right = right;
    	}
  }
  ```

  Python：

  ```
  class TreeNode: 
      def __init__(self, value):
          self.value = value
          self.left = None
          self.right = None
  ```




### 5.5 Morris遍历

- Morris遍历是二叉树遍历算法的超强进阶算法，可以实现时间复杂度为O(N)，而空间复杂度为O(1)
- 时间复杂度O(N)，额外空间复杂度O(1)，通过利用原树中大量空闲指针的方式，达到节省空间的目的
- Morris遍历细节
  - 假设来到当前节点cur，开始时cur来到头节点位置
  - 如果cur没有左孩子，cur向右移动(cur = cur.right)
  - 如果cur有左孩子，找到左子树上最右的节点mostRight：
    - 如果mostRight的右指针指向空，让其指向cur，然后cur向左移动(cur = cur.left)
    - 如果mostRight的右指针指向cur，让其指向null，然后cur向右移动(cur = cur.right)
  - cur为空时遍历



### 5.6 二叉树常见面试问题

【题目】给定一棵二叉树，求二叉树宽度【广度优先遍历】

【思路】使用queue，首先将head放进去；弹出head，将head的leftchild和rightchild放进去；dequque node，然后继续处理node的左右

​				哈希表记录每层node数, 每次进入下一层时候统计上一层的node数

​				如果不用哈希表，用queue，enqueue所有child 并记录个数n，根据n dequeue再dequeue对应child



【题目】给定一棵二叉树，判断是二叉搜索树（左大于中大于右）

【思路】中序遍历，一直升序就是二叉搜索树, 利用prevalue，check left tree， head， right tree

​				或者利用递归，左右树都要return是否是BST以及 min和max



【题目】给定一棵二叉树，判断是完全二叉树

【思路】宽度遍历，如果有right child 没有left child，return False；如果有任何node左右不全，后面node应全是leaf node



【题目】给定一棵二叉树，判断是满二叉树

【思路】深度L，数量N， 如果$N = 2^L-1$



【题目】给定一棵二叉树，判断是平衡二叉树

【思路】如果整个是AVL，左右都是AVL，而且左右高度差小于等于1



【题目】给定一棵二叉树两节点，找到最低公共祖先

【思路】遍历生成左右node parent的hashmap，向上查找

​				或者遇到o1 return o1，遇到o2 return o2，否则return null；



【题目】找一棵二叉树一个节点的后继节点

【思路】X有right tree，找右树最左节点; X无right tree，检查是不是parent的left child，return parent，否则null（右下角）



【题目】二叉树序列化和反序列化，字符串转换成树 以及 树转换成字符串

【思路】先序遍历，#表示null；反序列化就是先建左子树再建右子树



## 6. 回溯算法

### 6.1 什么是回溯法

- 回溯法也可以叫做回溯搜索法，它是一种搜索的方式
- 回溯是递归的副产品，只要有递归就会有回溯
- 回溯函数也就是递归函数，指的都是一个函数

### 6.2 回溯法的效率

- 回溯法的性能如何，虽然回溯法很难，很不好理解，但是回溯法并不是什么高效的算法。
- 因为回溯的本质是穷举，穷举所有可能，然后选出我们想要的答案，如果想让回溯法高效一些，可以加一些剪枝的操作，但也改不了回溯法就是穷举的本质

### 6.3回溯法解决的问题

- 回溯法，一般可以解决如下几种问题：
  - 组合问题：N个数里面按一定规则找出k个数的集合
  - 切割问题：一个字符串按一定规则有几种切割方式
  - 子集问题：一个N个数的集合里有多少符合条件的子集
  - 排列问题：N个数按一定规则全排列，有几种排列方式
  - 棋盘问题：N皇后，解数独等等

- 另外什么是组合，什么是排列？
  - 组合是不强调元素顺序的，排列是强调元素顺序。
  - 组合无序，排列有序

### 6.4 理解回溯法

- **所有**回溯法解决的问题**都可以**抽象为树形结构
- 因为回溯法解决的都是在集合中递归查找子集，集合的大小就构成了树的宽度，递归的深度，都构成的树的深度
- 递归就要有终止条件，所以必然是一棵高度有限的树（N叉树）

### 6.5 回溯法模板

- 回溯三部曲
  - 回溯函数模板返回值以及参数 (backtracking)
    - 返回值一般为void
    - 先写逻辑，然后需要什么参数，就填什么参数
  - 回溯函数终止条件
    - 回溯也有要终止条件
    - 一般来说搜到叶子节点了，也就找到了满足条件的一条答案，把这个答案存放起来，并结束本层递归
  - 回溯搜索的遍历过程
    - for循环可以理解是横向遍历，backtracking（递归）就是纵向遍历，这样就把这棵树全遍历完了，
    - <img src="https://camo.githubusercontent.com/f65ca647f31913496481cd1aff144040bd7ee4f6bc30accd370bc78b4b265d13/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303231303133303137333633313137342e706e67" style="zoom:50%;" />







## 7. 贪心算法

### 7.1 什么是贪心

- 贪心的本质是选择每一阶段的局部最优，从而达到全局最优
- **贪心算法并没有固定的套路**
- 要点就是通过局部最优，推出整体最优
- 如何验证可不可以用贪心算法, 一般数学证明有如下两种方法:
  - 数学归纳法
  - 反证法

### 7.2 贪心一般解题步骤

- 将问题分解为若干个子问题

- 找出适合的贪心策略
- 求解每一个子问题的最优解
- 将局部最优解堆叠成全局最优解



## 8. 动态规划

### 8.1 什么是动态规划(Dynamic Programming)

- 如果某一问题有很多重叠子问题，使用动态规划是最有效的
- 一个背包问题的例子: 有N件物品和一个最多能背重量为W 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大
- 动态规划中dp[j]是由dp[j-weight[i]]推导出来的，然后取max(dp[j], dp[j - weight[i]] + value[i])

### 8.2 动态规划解题步骤

- 动态规划问题五步曲:
  - 确定dp数组（dp table）以及下标的含义
  - 确定递推公式
  - dp数组如何初始化
  - 确定遍历顺序
  - 举例推导dp数组

- **debug找问题的最好方式就是把dp数组打印出来，看看究竟是不是按照自己思路推导的**
- 做动规的题目，写代码之前一定要把状态转移在dp数组的上具体情况模拟一遍，心中有数，确定最后推出的是想要的结果

### 8.3 背包问题

- 对于面试的话，其实掌握01背包，和完全背包，就够用了，最多可以再来一个多重背包。

  <img src=".\pics\zw6_1.png" alt="zw6_1" style="zoom:50%;" />

- 二维数组
  - 初始化dp数组
  - 更新dp数组: 先遍历物品, 再遍历背包
  - 如果物品重量大于背包，说明背包装不下当前物品: $dp[i][j] = dp[i - 1][j]$
  - 否则定义dp数组: $dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - curWeight]+ curVal)$
- 一维滚动数组
  - 在一维dp数组中，dp[j]表示：容量为j的背包，所背的物品价值可以最大为dp[j]
  - $dp[j] = max(dp[j], dp[j - weight[i]] + value[i])$
  - 一维dp遍历的时候，背包是从大到小
- 问有多少种不同的方法：
  - 如果求**组合数**就是外层for循环遍历**物品**，内层for遍历**背包**。
  - 如果求**排列数**就是外层for遍历**背包**，内层for循环遍历**物品**。
  - dp[i] += ...
- 问所需的最少的个数
  - 有顺序和没有顺序都可以，都不影响钱币的最小个数, 所以不用在意遍历顺序
  - dp[i]  = min()
- 物品可以使用多次，便是个完全背包问题

### 8.4 动态规划常见面试问题

【题目】斐波那契数 / 爬楼梯

【思路】opt一维矩阵，opt[i] = opt[i - 1] + opt[i - 2]



【题目】使用最小花费爬楼梯](https://programmercarl.com/0746.使用最小花费爬楼梯.html)

【思路】就是在爬台阶的基础上加了一个花费，opt[i] = min(opt[i - 1] + cost[i - 1], opt[i - 2] + cost[i - 2]);



【题目】给格子大小求路径总数

【思路】使用二维矩阵，第一行和第一列还需要初始化为1，表示只有一条路。$opt[i][j] = opt[i-1][j] + opt[i][j-1]$



【题目】给格子大小和障碍物求路径总数

【思路】依然是 $opt[i][j] = opt[i-1][j] + opt[i][j-1]$，但是要注意起始点有障碍，初始行有障碍的话，init时候break让后面都是0，其他位置每次计算的时候判断有障碍直接赋值0



【题目】整数拆分

【思路】初始化dp[2] = 1， 递推公式 dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j))



【题目】不同的二叉搜索树

【思路】n个node轮流作为root，dp[i] += dp[j - 1] * dp[i - j]



## 9. 图论

### 9.1 图论定义

- 一个图(graph) G = (V, E) 由顶点(vertex)的集合 V 和 边(edge)的集合E组成边也可以称作弧(arc)
- 如果图中的边是带方向的，那么图就是有向图(directed)，否则就是无向图。
- 如果有一条边从顶点v到顶点w，则称顶点v和w邻接(adjacent)。
- 如果图中有一条从一个顶点到它自身的边，那么图是带环的。通常是针对有向图中的环，对于图的边，有时会赋予其权值(weight)，例如用权值表示从从一个城市(顶点v)到另一个城市(顶点w)的距离。
- 如果一个无向图中从每一个顶点到其他每个顶点都存在一条路径，则称该无向图是连通的(connected)。
- 具有这样性质的有向图称为是强连通的(strongly connected)，如果一个有向图不是强连通的，但是去掉其边上的方向后形成的图是连通的，则称该有向图是弱连通的(weakly connected)。
- 完全图(complete graph)是其每一对顶点间都存在一条边的图。

### 9.2 图的表示

- 邻接表表示法

  邻接表在表示稀疏图（边的条数|E|远远小于|V^2| 的图）时非常紧凑而成为通常的选择

- 邻接矩阵表示法

  在表示**稠密图**时，更倾向于使用邻接矩阵

- ```
  Hashmap<Node> nodes
  Hashset <Edge> edges
  ```

### 9.3 图的遍历

- 宽度优先遍历
  - 利用queue实现
  - 从源节点按照宽度依次进队列，
  - 每弹出一个点，把该点所有邻接点放入队列
  - 直到queue空
- 广度优先遍历
  - 利用stack实现
  - 从源节点按照深度依次进栈，
  - 每弹出一个点，把该点下一个临界点放入栈
  - 直到stack空
- 拓扑排序算法
  - 对一个有向无环图(Directed Acyclic Graph简称DAG)G进行拓扑排序，是将G中所有顶点排成一个线性序列
  - 使得图中任意一对顶点u和v，若边<u,v>∈E(G)，则u在线性序列中出现在v之前。

### 9.4 寻找最短路径

- Dijkstra’s Algorithm
  - 适用权值没有负数的树
  - 每一次在表中选距离最短的点，依次更新从原点到当前点的最短距离

### 9.5 生成最小生成树

- 最小生成树其实是**最小权重生成树**的简称，使所有点联通，同时联通后边的权值之和最小

- Prim’s Algorithm
  - 从一个点开始，每次在neighbor中找weight最小的边
  - 总运行时间为 **O(m log n)**。
- Kruskal’s Algorithm
  - 使用优先级队列把所有边排序，依次选择最小的边，判断加上边之后有没有形成环
  - O(m log n)
  - 集合查询的结构 - 并查集
    - 假设每个点是单独一个集合，如果两个点不在一个集合，那个必然不能形成一个环

## 10. 单调栈



## 11. 并查集

- 并查集主要用于解决一些**元素分组**的问题，管理一系列**不相交的集合**，并支持两种操作：

- **合并**（Union）：把两个不相交的集合合并为一个集合。

- **查询**（Find）：查询两个元素是否在同一个集合中。

- 每个节点最开始都指向自己，如果两个节点是同一个union，则将一个节点挂在另一个下面

- Quick Find 可以保证在常数时间复杂度内完成查找，采用的思路是维护一颗扁平的树

- Quick Union 不能保证在常数时间复杂度内完成查找，但是可以快速添加新边。数量少的头节点挂在数量多的头节点下面

- ```python
  class UnionFind(object): 
      def __init__(self, n):
          # it depend on start from 0 or 1
          self.parents = [i for i in range(n)]
          self.count = [1 for _ in range(n)]
          
      def find(self, node):
          """
          :type node: int
          :rtype: int
          """
          while(node != self.parents[node]):
              node = self.parents[node];
          return node
      
      def union(self, a, b):
          """
          :type a: int
          :type b: int
          :rtype: None
          """
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
  
- 并行处理方法：划分成n片，如果边缘有特殊点则记录头节点，合并时候检查



## 12. 位运算

### 12.1 位运算

- 计算机中的数在内存中都是以二进制形式进行存储的，用位运算就是直接对整数在内存中的二进制位进行操作，因此其执行效率非常高
- 位操作符
  - & 运算, 两个位都是 1 时，结果才为 1，否则为 0
  - | 运算, 两个位都是 0 时，结果才为 0，否则为 1
  - ^ 异或运算，两个位相同则为 0，不同则为 1
  - ~ 取反运算，0 则变为 1，1 则变为 0
  - << 左移运算，向左进行移位操作，高位丢弃，低位补 0
  - \>> 右移运算，向右进行移位操作，对无符号数，高位补 0

### 12.2 位运算操作

- 模拟加法

  - ```python
    求异或+和，直到结果不变
    ```

- 模拟减法

  - ```
    a - b = a + -b = a + (flip(b) + 1)
    ```

    

- 数 a 向右移一位，相当于将 a 除以 2；数 a 向左移一位，相当于将 a 乘以 2

  ```python
  a = a<<2 # a*2
  a = a/2 # a/2
  ```

- 位操作交换两数,

  ```python
  a ^= b
  b ^= a
  a ^= b
  ```

- 判断奇偶数, num & 1 == 0 就是偶数

- 交换符号

  ```python
  a = ~a + 1
  ```

- 求绝对值

  ```python
  i = num >> 31
  num = (num^i) - i
  ```

- 统计二进制中 1 的个数

  ```python
  count = 0
  while num:
    num &= (num-1)
    count += 1
  ```


### 12.3 典型面试题

- 100亿url，找到所有重复url，找到top 100
  - 利用hashmap分流，求哈希值然后mod，每个小文件建立一个堆，根据每个堆顶的数建立另外一个堆（二维堆）

-  40亿无符号整数，找出出现两次的数
  - 定义2bit，00/01/10/11 表示出现次数
- 40亿个数中，只有10KB如果找出中位数
  - 用hashmap函数分流思想分段统计词频，可以锁定中位数在哪个hashmap范围中
- 有10GB大文件里面存放着无序有符号的整数，给你5GB空间，如何把这个无序变成有序
  - 先分桶再排序
  - 用小根堆（每一个元素有数值，次数两个属性）来作为媒介遍历每一个文件段（5GB可以申请可以存储多大的小根堆），每一个小段进入小根堆后，输出小根堆就可以了，然后总结果就是依次排序。
- 不用比较 判断a b中大的数字
  - a-b得到差值，右移31位 然后 & 1， 正数得到0，负数得到1
  - 结果flip取相反数，return a * SCa+b * SCb, 小的一项是0 大的是1 乘出来直接return
- 是不是2的幂
  - 二进制中只有一位1， num &  (num-1) = 0



## 13. 大数据算法题目解题技巧

- 哈希函数可以把数据按照种类均匀分流
- 布隆过滤器用于集合的建立与查询，并可以节省大量空间
- 一致性hash解决数据服务器的负载管理问题
- 利用并查集结构做岛问题的并行计算
- 位图解决某一范围上数字的出现情况，并可以节省大量空间
- 利用分段统计思想、并进一步节省空间
- 利用堆、外排序来做多个处理单元的结果合并








- 最小生成树
- 线段树
- 树状数组
- 字典树



# 前序

* 编程语言


* * [C++面试&C++学习指南知识点整理](https://github.com/youngyangyang04/TechCPP)
  
* 项目
    * [基于跳表的轻量级KV存储引擎](https://github.com/youngyangyang04/Skiplist-CPP)
    * [Nosql数据库注入攻击系统](https://github.com/youngyangyang04/NoSQLAttack)

* 编程素养
    * [看了这么多代码，谈一谈代码风格！](./problems/前序/代码风格.md)
    * [力扣上的代码想在本地编译运行？](./problems/前序/力扣上的代码想在本地编译运行？.md)
    * [什么是核心代码模式，什么又是ACM模式？](./problems/前序/什么是核心代码模式，什么又是ACM模式？.md)
    * [ACM模式如何构造二叉树](./problems/前序/ACM模式如何构建二叉树.md)
    * [解密互联网大厂研发流程](./problems/前序/互联网大厂研发流程.md)

* 工具 
    * [一站式vim配置](https://github.com/youngyangyang04/PowerVim)
    * [保姆级Git入门教程，万字详解](https://mp.weixin.qq.com/s/Q_O0ey4C9tryPZaZeJocbA)
    * [程序员应该用什么用具来写文档？](./problems/前序/程序员写文档工具.md)

* 求职 
    * [程序员的简历应该这么写！！（附简历模板）](./problems/前序/程序员简历.md)
    * [BAT级别技术面试流程和注意事项都在这里了](./problems/前序/BAT级别技术面试流程和注意事项都在这里了.md)
    * [北京有这些互联网公司，你都知道么？](./problems/前序/北京互联网公司总结.md)
    * [上海有这些互联网公司，你都知道么？](./problems/前序/上海互联网公司总结.md)
    * [深圳有这些互联网公司，你都知道么？](./problems/前序/深圳互联网公司总结.md)
    * [广州有这些互联网公司，你都知道么？](./problems/前序/广州互联网公司总结.md)
    * [成都有这些互联网公司，你都知道么？](./problems/前序/成都互联网公司总结.md)
    * [杭州有这些互联网公司，你都知道么？](./problems/前序/杭州互联网公司总结.md)
    
* 算法性能分析
    * [O(n)的算法居然超时了，此时的n究竟是多大？](./problems/前序/On的算法居然超时了，此时的n究竟是多大？.md)
    * [通过一道面试题目，讲一讲递归算法的时间复杂度！](./problems/前序/通过一道面试题目，讲一讲递归算法的时间复杂度！.md)
    * [本周小结！（算法性能分析系列一）](./problems/周总结/20201210复杂度分析周末总结.md)
    * [递归算法的时间与空间复杂度分析！](./problems/前序/递归算法的时间与空间复杂度分析.md)
    * [刷了这么多题，你了解自己代码的内存消耗么？](./problems/前序/刷了这么多题，你了解自己代码的内存消耗么？.md)

## 知识星球精选

* [秋招面试，心态很重要！](./problems/知识星球精选/秋招总结3.md)
* [秋招倒霉透顶，触底反弹！](./problems/知识星球精选/秋招总结2.md)
* [无竞赛，无实习，如何秋招？](./problems/知识星球精选/秋招总结1.md)
* [offer总决赛，何去何从！](./problems/知识星球精选/offer总决赛，何去何从.md)
* [入职后担心代码能力跟不上！](./problems/知识星球精选/入职后担心代码能力跟不上.md)
* [秋招进入offer决赛圈！](./problems/知识星球精选/offer对比-决赛圈.md)
* [非科班的困扰](./problems/知识星球精选/非科班的困扰.md)
* [offer的选择-开奖](./problems/知识星球精选/秋招开奖.md)
* [看到代码就抵触！怎么办？](./problems/知识星球精选/不喜欢写代码怎么办.md)
* [遭遇逼签，怎么办？](./problems/知识星球精选/逼签.md)
* [HR特意刁难非科班！](./problems/知识星球精选/HR特意刁难非科班.md)
* [offer的选择](./problems/知识星球精选/offer的选择.md)
* [天下乌鸦一般黑，哪家没有PUA？](./problems/知识星球精选/天下乌鸦一般黑.md)
* [初入大三，考研VS工作](./problems/知识星球精选/初入大三选择考研VS工作.md)
* [非科班2021秋招总结](./problems/知识星球精选/非科班2021秋招总结.md)
* [秋招下半场依然没offer，怎么办？](./problems/知识星球精选/秋招下半场依然没offer.md)
* [合适自己的就是最好的](./problems/知识星球精选/合适自己的就是最好的.md)
* [为什么都说客户端会消失](./problems/知识星球精选/客三消.md)
* [博士转计算机如何找工作](./problems/知识星球精选/博士转行计算机.md)
* [不一样的七夕](./problems/知识星球精选/不一样的七夕.md)
* [HR面注意事项](./problems/知识星球精选/HR面注意事项.md)
* [刷题攻略要刷两遍！](./problems/知识星球精选/刷题攻略要刷两遍.md)
* [秋招进行中的迷茫与焦虑......](./problems/知识星球精选/秋招进行中的迷茫与焦虑.md)
* [大厂新人培养体系应该是什么样的？](./problems/知识星球精选/大厂新人培养体系.md)
* [你的简历里「专业技能」写的够专业么？](./problems/知识星球精选/专业技能可以这么写.md)
* [Carl看了上百份简历，总结了这些！](./problems/知识星球精选/写简历的一些问题.md)
* [备战2022届秋招](./problems/知识星球精选/备战2022届秋招.md)
* [技术不太好，如果选择方向](./problems/知识星球精选/技术不好如何选择技术方向.md)
* [刷题要不要使用库函数](./problems/知识星球精选/刷力扣用不用库函数.md)
* [关于实习的几点问题](./problems/知识星球精选/关于实习大家的疑问.md)
* [面试中遇到了发散性问题，怎么办？](./problems/知识星球精选/面试中发散性问题.md)
* [英语到底重不重要！](./problems/知识星球精选/英语到底重不重要.md)
* [计算机专业要不要读研！](./problems/知识星球精选/要不要考研.md)
* [关于提前批的一些建议](./problems/知识星球精选/关于提前批的一些建议.md)
* [已经在实习的录友要如何准备秋招](./problems/知识星球精选/如何权衡实习与秋招复习.md)
* [华为提前批已经开始了](./problems/知识星球精选/提前批已经开始了.md)



# 正文

## 链表

1. [链表：链表相交](./problems/面试题02.07.链表相交.md)
2. [链表：总结篇！](./problems/链表总结篇.md)


## 字符串

3. [字符串：替换空格](./problems/剑指Offer05.替换空格.md)
5. [字符串：反转个字符串还有这个用处？](./problems/剑指Offer58-II.左旋转字符串.md)

## 双指针法 

3. [字符串：替换空格](./problems/剑指Offer05.替换空格.md)
5. [链表：链表相交](./problems/面试题02.07.链表相交.md)

## 高级数据结构经典题目 

* 最小生成树 
* 线段树 
* 树状数组 
* 字典树 

## 海量数据处理



# 补充题目

## 模拟
* [657.机器人能否返回原点](./problems/0657.机器人能否返回原点.md) 
* [31.下一个排列](./problems/0031.下一个排列.md) 

## 位运算
* [1356.根据数字二进制下1的数目排序](./problems/1356.根据数字二进制下1的数目排序.md) 




# 算法模板 

[各类基础算法模板](https://github.com/youngyangyang04/leetcode/blob/master/problems/算法模板.md)



# 算法题目汇总

## 数组

1. 3.Longest Substring Without Repeating Characters
2. 4.Median of Two Sorted Arrays
3. 11.Container With Most Water
4. 16.3Sum Closest
5. **26.Remove Duplicates from Sorted Array**
6. **27.移除元素**
7. 30.Substring with Concatenation of All Words
8. 33.Search in Rotated Sorted Array
9. **34.Find First and Last Position of Element in Sorted Array**
10. **35.搜索插入位置**
11. 36.Valid Sudoku
12. 41.First Missing Positive
13. 48.Rotate Image
14. **54.螺旋矩阵**
15. 57.Insert Interval
16. **59.螺旋矩阵II**
17. 66.Plus One
18. **69.Sqrt(x)**
19. 73.Set Matrix Zeroes
20. 74.Search a 2D Matrix
21. 75.Sort Colors
22. **76.Minimum Window Substring**
23. 80.Remove Duplicates from Sorted Array II
24. 88.Merge Sorted Array
25. 118.Pascal’s Triangle
26. 119.Pascal’s Triangle II
27. 152.Maximum Product Subarray
28. 164.Maximum Gap
29. 169.Majority Element
30. 172.Factorial Trailing Zeroes
31. 189.旋转数组
32. **209.长度最小的子数组**
33. 217.Contains Duplicate
34. 219.Contains Duplicate II
35. 220.Contains Duplicate III
36. 228.Summary Ranges
37. 229.Majority Element II
38. 238.Product of Array Except Self
39. 278.First Bad Version
40. **283.移动零**
41. 289.Game of Life
42. 334.Increasing Triplet Subsequence
43. 350.Itersection of Two Arrays II
44. 367.Valid Perfect Square
45. 373.Find K Pairs with Smallest Sums
46. 390.Elimination Game
47. 395.Longest Substring with At Least K Repeating Characters
48. 398.Random Pick Index
49. 442.Find All Duplicates in an Array
50. 448.Find All Numbers Disappeared in an Array
51. 451.Sort Characters By Frequency
52. 485.Max Consecutive Ones
53. 496.Next Greater Element I
54. 498.Diagonal Traverse
55. 503.Next Greater Element II
56. 506.Relative Ranks
57. 532.K-diff Pairs in an Array
58. 556.Next Greater Element III
59. 561.Array Partition
60. 566.Reshape the Matrix
61. 575.Distribute Candies
62. 581.Shortest Unsorted Continuous Subarray
63. 599.Minimum Index Sum of Two Lists
64. 605.Can Place Flowers
65. 661.Image Smoother
66. 665.Non-decreasing Array
67. 682.Baseball Game
68. 690.Employee Importance
69. **704.二分查找**
70. 724.寻找数组的中心下标
71. 744.Find Smallest Letter Greater Than Target
72. 747.Largest Number At Least Twice of Others
73. 766.Toeplitz Matrix
74. 805.Split Array With Same Average
75. 867.Transpose Matrix
76. 875.Koko Eating Bananas
77. 885.Spiral Matrix III
78. 904.Fruit Into Baskets
79. 905.Sort Array By Parity
80. 912.Sort an Array
81. 915.Partition Array into Disjoint Intervals
82. 918.Maximum Sum Circular Subarray
83. 922.按奇偶排序数组II
84. 933.Number of Recent Calls
85. 937.Reorder Data in Log Files
86. 941.有效的山脉数组
87. 953.Verifying an Alien Dictionary
88. 961.N-Repeated Element in Size 2N Array
89. 967.Numbers With Same Consecutive Differences
90. **977.有序数组的平方**
91. 985.Sum of Even Numbers After Queries
92. 986.Interval List Intersections
93. 989.Add to Array-Form of Integer
94. 1004.Max Consecutive Ones III
95. 1011.Capacity To Ship Packages Within D Days
96. 1051.Height Checker
97. 1089.Duplicate Zeros
98. 1122.Relative Sort Array
99. 1207.独一无二的出现次数
100. 1232.Check If It Is a Straight Line
101. 1331.Rank Transform of an Array
102. 1365.有多少小于当前数字的数字
103. 1366.Rank Teams by Votes
104. 1395.Count Number of Teams
105. 1431.Kids With the Greatest Number of Candies
106. 1480.Running Sum of 1d Array
107. 1491.Average Salary Excluding the Minimum and Maximum Salary
108. 1493.Longest Subarray of 1's After Deleting One Element
109. 1539.Kth Missing Positive Number
110. 1588.Sum of All Odd Length Subarrays
111. 1636.Sort Array by Increasing Frequency
112. 1675.Minimize Deviation in Array
113. 1706.Where Will the Ball Fall
114. 1732.Find the Highest Altitude
115. 1798.Maximum Number of Consecutive Values You Can Make
116. 1822.Sign of the Product of an Array
117. 1909.Remove One Element to Make the Array Strictly Increasing
118. 2048.Next Greater Numerically Balanced Number
119. 2161.Partition Array According to Given Pivot
120. 2187.Minimum Time to Complete Trips
121. 2200.Find All K-Distant Indices in an Array
122. 2248.Intersection of Multiple Arrays
123. 2300.Successful Pairs of Spells and Potions
124. 2326.Spiral Matrix IV
125. 2348.Number of Zero-Filled Subarrays
126. 2439.Minimize Maximum of Array
127. 2444.Count Subarrays With Fixed Bounds
128. 2454.Next Greater Element IV
129. 2485.Find the Pivot Integer
130. 2560.House Robber IV

## 链表

1. 2.Add Two Numbers
2. **19.删除链表的倒数第N个节点**
3. 21.Merge Two Sorted Lists
4. 23.Merge k Sorted Lists
5. **24.两两交换链表中的节点**
6. 25.Reverse Nodes in k-Group
7. 61.Rotate List
8. 82.Remove Duplicates from Sorted List II
9. 83.Remove Duplicates from Sorted List
10. 86.Partition List
11. 92.Reverse Linked List II
12. 138.Copy List with Random Pointer
13. **141.环形链表**
14. **142.环形链表II**
15. 143.重排链表
16. 148.Sort List
17. 160.链表相交
18. **203.移除链表元素**
19. **206.翻转链表**
20. 234.回文链表
21. 237.Delete Node in a Linked List
22. 328.Odd Even Linked List
23. 382.Linked List Random Node
24. 445.Add Two Numbers II
25. **707.设计链表**
26. 725.Split Linked List in Parts
27. 817.Linked List Components
28. 876.Middle of the Linked List
29. 1019.Next Greater Node In Linked List
30. 1171.Remove Zero Sum Consecutive Nodes from Linked List
31. 1472.Design Browser History
32. 1721.Swapping Nodes in a Linked List
33. 2090.K Radius Subarray Averages
34. 2130.Maximum Twin Sum of a Linked List

## 哈希表

1. **1.两数之和**
2. **15.三数之和**
3. **18.四数之和**
4. 187.Repeated DNA Sequences
5. **202.快乐数**
6. 205.同构字符串
7. 208.Implement Trie (Prefix Tree)
8. 211.Design Add and Search Words Data Structure
9. **242.有效的字母异位词**
10. 287.Find the Duplicate Number
11. 299.Bulls and Cows
12. **349.两个数组的交集**
13. **383.赎金信**
14. 388.Longest Absolute File Path
15. 424.Longest Repeating Character Replacement
16. **454.四数相加II**
17. 460.LFU Cache
18. 524.Longest Word in Dictionary through Deleting
19. 525.Contiguous Array
20. 560.Subarray Sum Equals K
21. 594.Longest Harmonious Subsequence
22. 692.Top K Frequent Words
23. 697.Degree of an Array
24. 705.Design HashSet
25. 706.Design HashMap
26. 720.Longest Word in Dictionary
27. 748.Shortest Completing Word
28. 811.Subdomain Visit Count
29. **884.两句生词**
30. 916.Word Subsets
31. 957.Prison Cells After N Days
32. 974.Subarray Sums Divisible by K
33. **1002.查找常用字符**
34. 1010.Pairs of Songs With Total Durations Divisible by 60
35. 1015.Smallest Integer Divisible by K
36. 1072.Flip Columns For Maximum Number of Equal Rows
37. 1074.Number of Submatrices That Sum to Target
38. 1128.Number of Equivalent Domino Pairs
39. 1138.Alphabet Board Path
40. 1160.Find Words That Can Be Formed by Characters
41. 1224.Maximum Equal Frequency
42. 1311.Get Watched Videos by Your Friends
43. 1394.Find Lucky Integer in an Array
44. 1396.Design Underground System
45. 1399.Count Largest Group
46. 1497.Check If Array Pairs Are Divisible by k
47. 1546.Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
48. 1577.Number of Ways Where Square of Number Is Equal to Product of Two Numbers
49. 1590.Make Sum Divisible by P
50. 1658.Minimum Operations to Reduce X to Zero
51. 1711.Count Good Meals
52. 2114.Maximum Number of Words Found in Sentences
53. 2122.Recover the Original Array
54. 2352.Equal Row and Column Pairs

## 字符串

1. 6.Zigzag Conversion
2. 12.Integer to Roman
3. 13.Roman to Integer
4. 14.Longest Common Prefix
5. **28.实现strStr**
6. 38.Count and Say
7. 43.Multiply Strings
8. 58.Length of Last Word
9. 71.Simplify Path
10. 125.Valid Palindrome
11. **151.翻转字符串里的单词**
12. 165.Compare Version Numbers
13. 212.Word Search II
14. 290.Word Pattern
15. 291.Word Pattern II
16. **344.反转字符串**
17. 345.Reverse Vowels of a String
18. 387.First Unique Character in a String
19. 409.Longest Palindrome
20. 434.Number of Segments in a String
21. 438.Find All Anagrams in a String
22. 443.String Compression
23. **459.重复的子字符串**
24. 482.License Key Formatting
25. 500.Keyboard Row
26. 520.Detect Capital
27. 539.Minimum Time Difference
28. **541.反转字符串II**
29. 551.Student Attendance Record I
30. 557.Reverse Words in a String III
31. 657.Robot Return to Origin
32. 678.Valid Parenthesis String
33. 680.Valid Palindrome II
34. 696.Count Binary Substrings
35. 709.To Lower Case
36. 771.Jewels and Stones
37. 777.Swap Adjacent in LR String
38. 784.Letter Case Permutation
39. 788.Rotated Digits
40. 791.Custom Sort String
41. 796.Rotate String
42. 804.Unique Morse Code Words
43. 806.Number of Lines To Write String
44. 809.Expressive Words
45. 819.Most Common Word
46. 821.Shortest Distance to a Character
47. 824.Goat Latin
48. 833.Find And Replace in String
49. 859.Buddy Strings
50. 890.Find and Replace Pattern
51. 893.Groups of Special-Equivalent Strings
52. 917.Reverse Only Letters
53. 921.Minimum Add to Make Parentheses Valid
54. 925.长按键入
55. 929.Unique Email Addresses
56. 1003.Check If Word Is Valid After Substitutions
57. 1071.Greatest Common Divisor of Strings
58. 1016.Binary String With Substrings Representing 1 To N
59. 1106.Parsing A Boolean Expression
60. 1156.Swap For Longest Repeated Character Substring
61. 1309.Decrypt String from Alphabet to Integer Mapping
62. 1358.Number of Substrings Containing All Three Characters
63. 1408.String Matching in an Array
64. 1419.Minimum Number of Frogs Croaking
65. 1456.Maximum Number of Vowels in a Substring of Given Length
66. 1545.Find Kth Bit in Nth Binary String
67. 1638.Count Substrings That Differ by One Character
68. 1768.Merge Strings Alternately

## 位运算

1. 136.Single Number
2. 137.Single Number II
3. 190.Reverse Bits
4. 191.Number of 1 Bits
5. 260.Single Number III
6. 268.Missing Number
7. 338.Counting Bits
8. 371.Sum of Two Integers
9. 389.Find the Difference
10. 401.Binary Watch
11. 405.Convert a Number to Hexadecimal
12. 442.Find All Duplicates in an Array
13. 448.Find All Numbers Disappeared in an Array
14. 461.Hamming Distance
15. 476.Number Complement
16. 540.Single Element in a Sorted Array
17. 693.Binary Number with Alternating Bits
18. 1318.Minimum Flips to Make a OR b Equal to c
19. 2595.Number of Even and Odd Bits

## 栈与队列

1. **20.有效的括号**
2. **150.逆波兰表达式求值**
3. **225.用队列实现栈**
4. **232.用栈实现队列**
5. **239.滑动窗口最大值**
6. 295.Find Median from Data Stream
7. **347.前K个高频元素**
8. 394.Decode String
9. 662.Maximum Width of Binary Tree
10. 703.Kth Largest Element in a Stream
11. 844.Backspace String Compare
12. **1047.删除字符串中的所有相邻重复项**
13. 2336.Smallest Number in Infinite Set

## 二叉树

1. **94.二叉树的中序遍历**
2. **98.验证二叉搜索树**
3. 100.相同的树
4. **101.对称二叉树**
5. **102.二叉树的层序遍历**
6. **104.二叉树的最大深度**
7. **105.从前序和中序遍历构造二叉树**
8. **106.从中序与后序遍历序列构造二叉树**
9. **107.二叉树的层次遍历II**
10. **108.将有序数组转换为二叉搜索树**
11. 109.Convert Sorted List to Binary Search Tree
12. **110.平衡二叉树**
13. **111.二叉树的最小深度**
14. **112.路径总和**
15. **116.填充每个节点的下一个右侧节点指针**
16. **117.填充每个节点的下一个右侧节点指针II**
17. 124.Binary Tree Maximum Path Sum
18. 129.求根到叶子节点数字之和
19. **144.二叉树的前序遍历**
20. **145.二叉树的后序遍历**
21. **199.二叉树的右视图**
22. **222.完全二叉树的节点个数**
23. **226.翻转二叉树**
24. **235.二叉搜索树的最近公共祖先**
25. **236.二叉树的最近公共祖先**
26. **257.二叉树的所有路径**
27. 297.Serialize and Deserialize Binary Tree
28. **404.左叶子之和**
29. **429.N叉树的层序遍历**
30. 449.Serialize and Deserialize BST
31. **450.删除二叉搜索树中的节点**
32. **501.二叉搜索树中的众数**
33. **513.找树左下角的值**
34. **515.在每个树行中找最大值**
35. **530.二叉搜索树的最小绝对差**
36. **538.把二叉搜索树转换为累加树**
37. 543.Diameter of Binary Tree
38. **559.n叉树的最大深度**
39. **572.另一棵树的子树**
40. **589.N-ary Tree 前序遍历**
41. **590.N-ary 树后序遍历**
42. **617.合并二叉树**
43. **637.二叉树的层平均值**
44. **654.最大二叉树**
45. **669.修剪二叉搜索树**
46. **700.二叉搜索树中的搜索**
47. **701.二叉搜索树中的插入操作**
48. 958.Check Completeness of a Binary Tree
49. 1161.Maximum Level Sum of a Binary Tree
50. 1372.Longest ZigZag Path in a Binary Tree
51. 1382.将二叉搜索树变平衡

## 回溯

1. **17.电话号码的字母组合**
2. **37.解数独**
3. **39.组合总和**
4. **40.组合总和II**
5. **46.全排列**
6. **47.全排列 II**
7. **51.N皇后**
8. **51.N皇后 II**
9. **77.组合**
10. **78.子集**
11. 79.Word Search
12. **90.子集II**
13. **93.复原IP地址**
14. 113.Path Sum II
15. **131.分割回文串**
16. **216.组合总和III**
17. **332.重新安排行程**
18. **406.根据身高重建队列**
19. 437.Path Sum III
20. **491.递增子序列**
21. 2305.Fair Distribution of Cookies

## 贪心

1. **45.跳跃游戏II**
2. **53.最大子序和**
3. **55.跳跃游戏**
4. **56.合并区间**
5. **122.买卖股票的最佳时机II**
6. **134.加油站**
7. **135.分发糖果**
8. **376.摆动序列**
9. **435.无重叠区间**
10. **452.用最少数量的箭引爆气球**
11. **455.分发饼干**
12. 502.IPO
13. 621.Task Scheduler
14. 649.Dota2参议院
15. **714.买卖股票的最佳时机含手续费**
16. **738.单调递增的数字**
17. **763.划分字母区间**
18. 767.Reorganize String
19. **860.柠檬水找零**
20. 881.Boats to Save People
21. **968.监控二叉树**
22. **1005.K次取反后最大化的数组和**
23. 1221.分割平衡字符
24. 1402.Reducing Dishes
25. 1802.Maximum Value at a Given Index in a Bounded Array
26. 2405.Optimal Partition of String

## 动态规划

1. 5.最长回文子串
2. **53.最大子序和**
3. **62.不同路径**
4. **63.不同路径II**
5. **70.爬楼梯**
6. **72.编辑距离**
7. 87.Scramble String
8. **96.不同的二叉搜索树**
9. **115.不同的子序列**
10. 120.Triangle
11. **121.买卖股票的最佳时机**
12. **122.买卖股票的最佳时机II**
13. **123.买卖股票的最佳时机III**
14. 132.分割回文串 II
15. **139.Word Break**
16. 140.Word Break II
17. 174.Dungeon Game
18. **188.买卖股票的最佳时机IV**
19. **198.打家劫舍**
20. **213.打家劫舍II**
21. **279.完全平方数**
22. **300.最长上升子序列**
23. **309.最佳买卖股票时机含冷冻期**
24. **322.零钱兑换**
25. **337.打家劫舍III**
26. **343.整数拆分**
27. **377.组合总和Ⅳ**
28. **392.判断子序列**
29. **416.分割等和子集**
30. **474.一和零**
31. **494.目标和**
32. **509.斐波那契数**
33. **516.最长回文子序列**
34. **518.零钱兑换II**
35. **583.两个字符串的删除操作**
36. **647.回文子串**
37. 673.最长递增子序列的个数
38. **674.最长连续递增序列**
39. **714.买卖股票的最佳时机含手续费**
40. **718.最长重复子数组**
41. **746.使用最小花费爬楼梯**
42. 837.New 21 Game
43. 877.Stone Game
44. 983.Minimum Cost For Tickets
45. 1027.Longest Arithmetic Subsequence
46. **1035.不相交的线**
47. 1046.Last Stone Weight
48. **1049.最后一块石头的重量II**
49. 1140.Stone Game II
50. **1143.最长公共子序列**
51. 1406.Stone Game III
52. 1416.Restore The Array
53. 2218.Maximum Value of K Coins From Piles

## 单调栈

1. **42.接雨水**
2. **84.柱状图中最大的矩形**
3. **496.下一个更大元素I**
4. **503.下一个更大元素II**
5. **739.每日温度**
6. 1964.Find the Longest Valid Obstacle Course at Each Position

## 图论

1. 127.单词接龙
2. 133.Clone Graph
3. 200.Number of Islands
4. 207.Course Schedule
5. 399.Evaluate Division
6. 417.Pacific Atlantic Water Flow
7. 463.岛屿的周长
8. 733.Flood Fill
9. 785.Is Graph Bipartite?
10. 841.钥匙和房间
11. 863.All Nodes Distance K in Binary Tree
12. 1020.Number of Enclaves
13. 1091.Shortest Path in Binary Matrix
14. 1254.Number of Closed Islands
15. 1306.Jump Game III
16. 1376.Time Needed to Inform All Employees
17. 1557.Minimum Number of Vertices to Reach All Nodes
18. 1857.Largest Color Value in a Directed Graph
19. 2101.Detonate the Maximum Bombs
20. 2360.Longest Cycle in a Graph

## 并查集

1. 128.Longest Consecutive Sequence
2. 547.Number of Provinces
3. 684.Redundant Connection
4. 685.Redundant Connection II
5. 1319.Number of Operations to Make Network Connected
6. 1489.Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
7. 1579.Remove Max Number of Edges to Keep Graph Fully Traversable
8. 1697.Checking Existence of Edge Length Limited Paths
9. 2316.Count Unreachable Pairs of Nodes in an Undirected Graph
10. 2492.Minimum Score of a Path Between Two Cities

## 数学

1. 319.Bulb Switcher

## 模拟设计

1. 1146.Snapshot Array
1. 1603.Design Parking System