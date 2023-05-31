## LeetCode Problem - Hashtable

[TOC]

### [1. Two Sum](https://leetcode.com/problems/two-sum/)

Java

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

python

``` python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i,num in enumerate(nums):
            if num in dic:
                return [dic[num],i]
            dic[target - num] = i
```

### [15. 3Sum](https://leetcode.com/problems/3sum/)

Java

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

python

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

### [18. 4Sum](https://leetcode.com/problems/4sum/)

python

``` python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        nums.sort()
        for i in range(len(nums) - 3):
            firstNum = nums[i]
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                secondNum = nums[j]
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left, right = j+1, len(nums)-1
                while left < right:
                    fourSome = firstNum + secondNum + nums[left] + nums[right]
                    if fourSome > target:
                        right -= 1
                    elif fourSome < target:
                        left += 1
                    else:
                        result.append([firstNum,secondNum,nums[left],nums[right]])
                        left += 1
                        right -= 1
                        while nums[left] == nums[left-1] and left < right:
                            left += 1
                        while nums[right] == nums[right+1] and left < right:
                            right -= 1
        return result
```

### [187. Repeated DNA Sequences](https://leetcode-cn.com/problems/repeated-dna-sequences/)

找出出现两次以上的DNA连续序列，序列长度为10。

```
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]
```

方法一：此题简单方法用滑动窗口加counter就行了。

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        c = Counter(s[i:i+10] for i in range(len(s)-9))
        return [k for k, cnt in c.items() if cnt > 1]
```

方法二：此题有个位运算的标签，所以试了一下，将DNA序列编码为二进制的形式。

```python
def findRepeatedDnaSequences(self, s: str) -> List[str]:
    d = dict(zip('ACGT', range(4)))
    dd = {v: k for k, v in d.items()}
    dna = reduce(lambda x, y: x<<2 | d[y], s[:10], 0)
    cnt = Counter([dna])

    for i, c in enumerate(islice(s, 10, None)):
        dna <<= 2
        dna &= (1<<20)-1    # 取出多余的高位
        dna |= d[c]
        cnt[dna] += 1

    def decode(n):
        s = ''
        for _ in range(10):
            s += dd[n % 4]
            n >>= 2
        return s[::-1]

    return [decode(n) for n, c in cnt.items() if c > 1]
```

### [202. Happy Number](https://leetcode.com/problems/happy-number/)

Java

```java
class Solution {
    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<>();
        set.add(n);
        while (n != 1){
            int cur = 0;
            while (n>0){
                cur += Math.pow(n%10, 2);
                n /= 10;
            }
            if (set.contains(cur)) return false;
            n = cur;
            set.add(n);
        }
        return true;
    }
}
```

python

``` python
class Solution:
    def isHappy(self, n: int) -> bool:
        history = set()
        while n != 1:
            cur = 0
            while n > 0:
                cur += (n%10)**2
                n //= 10
            if cur in history:
                return False
            history.add(cur)
            n = cur
        return True
```

### [205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/)

Java

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) return false;
        HashMap<Character,Character> chars = new HashMap<>();
        HashMap<Character,Character> chart = new HashMap<>();
        for (int i = 0; i<s.length(); i++){
            Character charS = s.charAt(i);
            Character charT = t.charAt(i);
            if (chars.get(charS) == null){
                chars.put(charS, charT);
            }
            if (chart.get(charT) == null){
                chart.put(charT, charS);
            }
            if (chars.get(charS) != charT ||chart.get(charT) != charS){
                return false;
            }
        }
        return true;
    }
}
```

python

``` python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        dic_s, dic_t = {}, {}
        
        for i in range(len(s)):
            chars = s[i]
            chart = t[i]
            if chars not in dic_s:
                dic_s[chars] = chart
            if chart not in dic_t:
                dic_t[chart] = chars
            if dic_s[chars] != chart or dic_t[chart] != chars:
                return False
        return True
```

### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

Java

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
            if (node.children[index] == null)
                node.children[index] = new TrieNode();
            node = node.children[index];
        }
        node.isWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null)
                return false;
            node = node.children[index];
        }
        return node.isWord;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null)
                return false;
            node = node.children[index];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

python

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

Java

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
            if (node.children[index] == null)
                node.children[index] = new WordNode();
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
            if (node.children[index] != null)
                return partialSearch(node.children[index], word, start + 1);
        } else {
            for (int i = 0; i < 26; i++) {
                if (node.children[i] != null)
                    if (partialSearch(node.children[i], word, start + 1))
                        return true;
            }
        }
        return false;
    }
}
```

python

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

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

Java

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
        for (int i : dic)
            if (i != 0)
                return false;
        return true;
    }
}
```

python

``` python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)
```

### [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

Java

```java
class Solution {
    public int findDuplicate(int[] nums) {
        // Floyd's Cycle
        int slow = 0, fast = 0, slow2 = 0;
        while(true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if(slow == fast) break;
        }
        while(true) {
            slow = nums[slow];
            slow2 = nums[slow2];
            if(slow==slow2) break;
        }
        return slow;
    }
}
```

python

``` python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow, fast, slow2 = 0, 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast: break
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow==slow2: break
        return slow
```

### [299. Bulls and Cows](https://leetcode.com/problems/bulls-and-cows/description/)

Java

```java
class Solution {
    public String getHint(String secret, String guess) {
        char[] secretArray = secret.toCharArray();
        char[] guessArray = guess.toCharArray();
        int bulls = 0, cows = 0;
        int[] numbers = new int[10];
        for (int i=0; i<secretArray.length; i++){
            if (secretArray[i] == guessArray[i]){
                bulls ++;
            }else{
                if (numbers[secretArray[i]-'0'] < 0) cows ++;
                if (numbers[guessArray[i]-'0'] > 0) cows ++;
                numbers[secretArray[i]-'0'] ++;
                numbers[guessArray[i]-'0'] --;
            }
        }
        return bulls+"A"+cows+"B";
    }
}
```

python

``` python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        numbers = collections.defaultdict(int)
        bulls, cows = 0, 0
        for i, secretchar in enumerate(secret):
            if secretchar == guess[i]:
                bulls += 1
            else:
                if numbers[secretchar] < 0:
                    cows += 1
                if numbers[guess[i]] > 0:
                    cows += 1
                numbers[secretchar] += 1
                numbers[guess[i]] -= 1
        return f"{bulls}A{cows}B"
```

### [349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

Java

```java

```

python

``` python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return set(nums1).intersection(set(nums2))
```

### [383. Ransom Note](https://leetcode.com/problems/ransom-note/)

Java

```java

```

python

``` python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        for char in set(ransomNote):
            if ransomNote.count(char) > magazine.count(char):
                return False
        return True
```

### [388. Longest Absolute File Path](https://leetcode.com/problems/longest-absolute-file-path/)

最长的绝对路径。

Python

方法一：自己用栈实现的。

```python
def lengthLongestPath(self, input: str) -> int:
    stack = []
    path = input.split('\n')
    ans = ['']
    for p in path:
        t = p.count('\t')
        p = p[t:]
        while t < len(stack):
            stack.pop()
        stack.append(p)
        if '.' in p:
            ans.append('/'.join(stack))
    return len(max(ans, key=len))
```

方法二：用字典记录深度。

```python
def lengthLongestPath(self, input: str) -> int:
    ans = 0
    path_len = {0: 0}
    for line in input.splitlines():
        name = line.lstrip('\t')
        depth = len(line) - len(name)
        if '.' in name:
            ans = max(ans, path_len[depth] + len(name))
        else:
            path_len[depth+1] = path_len[depth] + len(name) + 1
    return ans
```

### [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

Java

```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] arr = new int[26];
        char[] charArray = s.toCharArray();
        int maxlen = 0, largestCount = 0;
        for(int i = 0; i < charArray.length; i ++){
            arr[charArray[i] - 'A'] ++;
            largestCount = Math.max(largestCount, arr[charArray[i] - 'A']);
            if (maxlen - largestCount >= k)
                arr[charArray[i - maxlen] - 'A'] --;
            else
                maxlen ++;
        }
        return maxlen;
    }
}
```

python

``` python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        maxlen, largestCount = 0, 0
        arr = collections.Counter()
        for i in range(len(s)):
            arr[s[i]] += 1
            largestCount = max(largestCount, arr[s[i]])
            if maxlen - largestCount >= k:
                arr[s[i - maxlen]] -= 1
            else:
                maxlen += 1
        return maxlen
```

### [454. 4Sum II](https://leetcode.com/problems/4sum-ii/)

Java

```java

```

python

在四个数组中每个各选出一个，使得它们所有的和为0。

```
Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
```

方法一：和18题不同，这里不需要想3Sum之后怎么怎么样，因为3Sum的方法需要排序，而这里没法处理两个数组排序，双指针也比较麻烦了。换一个思路，拆成2个2Sum的子问题。

```python
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        sum_dict = collections.defaultdict(int)
        for i in nums1:
            for j in nums2:
                sum_dict[i+j] += 1        
        count = 0           
        for i in nums3:
            for j in nums4:
                if 0-i-j in sum_dict:
                    count += sum_dict[0-i-j]
        return count
```

方法二：和Stenfan学习。

```python
def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    ab = Counter(-a-b for a in A for b in B)
    return sum(ab[c+d] for c in C for d in D)
```

### [460. LFU Cache](https://leetcode.com/problems/lfu-cache/description/)

Java

```java

```

python

``` python
from collections import deque, defaultdict
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.freq = 1

class LFUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.minFreq = 0
        self.keyToNode = {}
        self.freqToList = defaultdict(deque)
        self.freqToKey = defaultdict(set)
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.keyToNode:
            return -1
        node = self.keyToNode[key]
        self.updateFreq(node)
        return node.value
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if self.capacity == 0:
            return
        if key in self.keyToNode:
            node = self.keyToNode[key]
            node.value = value
            self.updateFreq(node)
            return
        if len(self.keyToNode) == self.capacity:
            minFreqKey = self.freqToList[self.minFreq].pop()
            self.freqToKey[self.minFreq].remove(minFreqKey)
            del self.keyToNode[minFreqKey]
        self.minFreq = 1
        self.freqToList[1].appendleft(key)
        self.freqToKey[1].add(key)
        self.keyToNode[key] = Node(key, value)
        
    def updateFreq(self, node):
        prevFreq = node.freq
        newFreq = node.freq + 1
        self.freqToList[prevFreq].remove(node.key)
        self.freqToKey[prevFreq].remove(node.key)
        if len(self.freqToList[prevFreq]) == 0:
            del self.freqToList[prevFreq]
            if prevFreq == self.minFreq:
                self.minFreq += 1
        if newFreq not in self.freqToList:
            self.freqToList[newFreq] = deque()
            self.freqToKey[newFreq] = set()
        self.freqToList[newFreq].appendleft(node.key)
        self.freqToKey[newFreq].add(node.key)

        node.freq = newFreq
```

### [524. Longest Word in Dictionary through Deleting](bbb)

Java

```java

```

python

``` python
class Solution(object):
    def findLongestWord(self, s, dictionary):
        """
        :type s: str
        :type dictionary: List[str]
        :rtype: str
        """
        res = ''
        for word in dictionary:
            if self.includeword(s, word):
                if len(word)>len(res) or (len(word)==len(res) and word<res):
                    res = word
        return res

    def includeword(self, string, s):
        i, j = 0, 0
        while i<len(string) and j<len(s):
            if string[i] == s[j]:
                j += 1
            i += 1
        return j==len(s)
```

### [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)

Java

```java

```

python

找出二进制数组中拥有相等个数0和1的最长子串的长度。

```
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.
```

方法一：此题解法类似于买卖股票，维护一个`count`如果是0减一，如果是1加一，那么当count值相等的时候，说明这个子串中有相等1和0。使用一个字典来记录每次count的最小索引。需要注意的是索引从1开始。

```python
def findMaxLength(self, nums: List[int]) -> int:
    count = ans = 0
    table = {0: 0}
    for i, num in enumerate(nums, 1):
        if num == 0:
            count -= 1
        else:
            count += 1
        if count in table:
            ans = max(ans, i-table[count])
        else:
            table[count] = i
            
    return ans
```

### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

Java

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int res = 0, sumation = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int num : nums){
            sumation += num;
            res += map.getOrDefault(sumation-k, 0);
            map.put(sumation, map.getOrDefault(sumation, 0)+1);
        }
        return res;
    }
}
```

python

子数组和为k的个数。

```java
Input:nums = [1,1,1], k = 2
Output: 2
```

方法一：累加一开始想到了，补0也想到了，没想到用哈希，而是用循环去迭代，这样时间超时了。

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sumation = res = 0
        dic = collections.defaultdict(int)
        dic[0] = 1
        for num in nums:
            sumation += num
            res += dic[sumation-k]
            dic[sumation] += 1
        return res
```



### [594. Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence/)

Java

```java

```

python

``` python
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = collections.Counter(nums)
        res = 0
        for i, num in count.items():
            sumcount = num + count[i+1] if count[i+1] else 0
            res = max(sumcount, res)
        return res
```

### [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/description/)

Java

```java
class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> counter = new HashMap<>();
        for (String word : words){
            counter.put(word, counter.getOrDefault(word, 0)+1);
        }
        List<String> res = new ArrayList<>();
        PriorityQueue<Pair> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (String x : counter.keySet()){
            pq.add(new Pair(counter.get(x), x));
        }
        for (int i = 0; i < k; i++){
            res.add(pq.poll().b);
        }
        return res;
    }

    class Pair implements Comparable<Pair>{
        int a;
        String b;
        public Pair(int a, String b){
            this.a = a;
            this.b = b;
        }
        public int compareTo(Pair pair){
            if (this.a != pair.a)
                return this.a - pair.a;
            else
                return pair.b.compareTo(this.b);
        }
    }
}
```

python

最高频的K个单词，相同频率，优先返回字符顺序优先。

方法一：记录写法，nsmallest 也能接受key函数。

```python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        freq = collections.Counter(words)
        res, heap = [], []
        for word,times in freq.items():
            heapq.heappush(heap, (-times, word))
        return [heapq.heappop(heap)[1] for _ in range(k)]
```

### [697. Degree of an Array](https://leetcode.com/problems/degree-of-an-array/)

Java

```java

```

python

degree这里表示数组最常见的元素的频率，然后在连续的子数组中寻找同样的degree，求最小子数组的长度。

```
Input: [1, 2, 2, 3, 1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
```

方法一：使用Counter 和index. 600ms有点慢。

```python
def findShortestSubArray(self, nums):
    from collections import Counter
    c = Counter(nums)
    degree = None
    res = n = len(nums)
    for num, count in c.most_common():
        if degree and count != degree:
            break
        min_len = n - nums[::-1].index(num) - 1 - nums.index(num) + 1
        # print(min_len, num)
        res = min(res, min_len)
        degree = count
    return res
```

方法二：使用dict记录索引。

```python
def findShortestSubArray(self, nums):
    c = collections.defaultdict(int)
    left, right = {}, {}
    for i, num in enumerate(nums):
        if num not in left:
            left[num] = i
        right[num] = i
        c[num] += 1
    res = len(nums)
    degree = max(c.values())
    for num, count in c.items():
        if count == degree:
            res = min(res, right[num]-left[num]+1)
    return res
```

方法三：使用Counter + dict.

```python
def findShortestSubArray(self, nums: 'List[int]') -> 'int':
    c = collections.Counter(nums)
    left, right = {}, {}
    for i, num in enumerate(nums):
        if num not in left:
            left[num] = i
        right[num] = i
    degree, res = 0, len(nums)
    for num, count in c.most_common():
        if degree and count != degree:
            break
        res = min(res, right[num]-left[num]+1)
        degree = count
    return res
```

### [705. Design HashSet](https://leetcode.com/problems/design-hashset/)

Java

```java
class MyHashSet {
    int size;
    List<List<Integer>> buckets;

    public MyHashSet() {
        size = 1000;
        buckets = new ArrayList<>(size);
        for (int i = 0; i < size; i++)
            buckets.add(new LinkedList<>());
    }
    
    public void add(int key) {
        int index = key % size;
        List<Integer> bucket = buckets.get(index);
        if (!bucket.contains(key))
            bucket.add(key);
    }
    
    public void remove(int key) {
        int index = key % size;
        List<Integer> bucket = buckets.get(index);
        bucket.remove(Integer.valueOf(key));
    }
    
    public boolean contains(int key) {
        int index = key % size;
        List<Integer> bucket = buckets.get(index);
        return bucket.contains(key);
    }
}
```

python

``` python
class MyHashSet:

    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]

    def add(self, key: int) -> None:
        index = self._hash(key)
        if key not in self.buckets[index]:
            self.buckets[index].append(key)

    def remove(self, key: int) -> None:
        index = self._hash(key)
        if key in self.buckets[index]:
            self.buckets[index].remove(key)

    def contains(self, key: int) -> bool:
        index = self._hash(key)
        return key in self.buckets[index]
        
    def _hash(self, key: int) -> int:
        return key % self.size
```

### [706.Design HashMap](https://leetcode.com/problems/design-hashmap/description/)

Java

```java
class ListNode {
    int key, val;
    ListNode next;
    public ListNode(int key, int val, ListNode next) {
        this.key = key;
        this.val = val;
        this.next = next;
    }
}
class MyHashMap {
    static final int size = 19997;
    static final int mult = 12582917;
    ListNode[] data;

    public MyHashMap() {
        this.data = new ListNode[size];
    }
    
    public void put(int key, int val) {
        remove(key);
        int h = hash(key);
        ListNode node = new ListNode(key, val, data[h]);
        data[h] = node;
    }
    
    public int get(int key) {
        int h = hash(key);
        ListNode node = data[h];
        for (; node != null; node = node.next)
            if (node.key == key) return node.val;
        return -1;
    }
    
    public void remove(int key) {
        int h = hash(key);
        ListNode node = data[h];
        if (node == null) return;
        if (node.key == key) data[h] = node.next;
        else for (; node.next != null; node = node.next)
                if (node.next.key == key){
                    node.next = node.next.next;
                    return;
                }
    }

    private int hash(int key) {
        return (int)((long)key * mult % size);
    }
}
```

python

``` python
class ListNode:
    def __init__(self, key, val, nxt):
        self.key = key
        self.val = val
        self.next = nxt
class MyHashMap:
    def __init__(self):
        self.size = 19997
        self.mult = 12582917
        self.data = [None for _ in range(self.size)]
    def hash(self, key):
        return key * self.mult % self.size
    def put(self, key, val):
        self.remove(key)
        h = self.hash(key)
        node = ListNode(key, val, self.data[h])
        self.data[h] = node
    def get(self, key):
        h = self.hash(key)
        node = self.data[h]
        while node:
            if node.key == key: return node.val
            node = node.next
        return -1
    def remove(self, key: int):
        h = self.hash(key)
        node = self.data[h]
        if not node: return
        if node.key == key:
            self.data[h] = node.next
            return
        while node.next:
            if node.next.key == key:
                node.next = node.next.next
                return
            node = node.next
```

### [720. Longest Word in Dictionary](bbb)

字典中的最长单词，找出一个列表中的一个单词，该单词的子单词也必须在字典中。相同长度的单词，返回字典序最前的一个。

Java

```java

```

python

``` python
class Solution(object):
    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        wordset = set(words)
        queue = collections.deque()
        queue.append("")
        prev = ""
        while queue:
            s = queue.popleft()
            prev = s
            for i in range(25,-1,-1):
                key = s + chr(97 + i)
                if key in wordset:
                    queue.append(key)
        return prev
```

### [748. Shortest Completing Word](https://leetcode.com/problems/shortest-completing-word/)

Java

```java

```

python

最短的完整匹配单词。包含`licensePlate`中的所有字母，大小写不敏感。假设答案一定存在

```
Input: licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]
Output: "steps"
Explanation: The smallest length word that contains the letters "S", "P", "S", and "T".
Note that the answer is not "step", because the letter "s" must occur in the word twice.
Also note that we ignored case for the purposes of comparing whether a letter exists in the word.
```

方法一：先排序。

```python
def shortestCompletingWord(self, licensePlate: 'str', words: 'List[str]') -> 'str':
    lp = ''.join(x for x in licensePlate.lower() if x.isalpha())
    c1 = collections.Counter(lp)
    words.sort(key=len)
    words = map(str.lower, words)
    for word in words:
        diff = c1 - collections.Counter(word)
        if not diff:
            return word
```

方法二：more elegant.

```python
def shortestCompletingWord(self, licensePlate: 'str', words: 'List[str]') -> 'str':
    lp = ''.join(x for x in licensePlate.lower() if x.isalpha())
    c1 = collections.Counter(lp)
    words = map(str.lower, words)
    return min((word for word in words 
                if not c1-collections.Counter(word)), key=len)
```

方法三：most efficient. 认为方法二是在计算`-`的操作时，涉及一些无关的key导致效率过低。

```python
class Solution(object):
    def shortestCompletingWord(self, licensePlate, words):
        res = ''
        licensePlate = ''.join(x for x in licensePlate.lower() if x.isalpha())
        for w in words:
            temp = list(w.lower())
            for char in licensePlate:
                if char in temp:
                    temp.remove(char)
                else: break
            else:
                if len(w) < len(res) or res == '':
                    res = w
        return res
```

### [811. Subdomain Visit Count](https://leetcode.com/problems/subdomain-visit-count/)

Java

```java

```

python

子域名访问量。给定一个三级或二级域名列表，统计所有三级、二级和顶级域名的访问量。

```
Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.
```

方法一：Solution中用了Counter，个人认为defaultdict.

```python
class Solution(object):
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        res = collections.defaultdict(int)
        for domain in cpdomains:
            domain = domain.split()
            times, web = int(domain[0]), domain[1]
            res[web] += times
            while '.' in web:
                web = web[web.index('.')+1:]
                res[web] += times
        return ['{} {}'.format(times, web) for web,times in res.items()]
```

### [884. Uncommon Words from Two Sentences](https://leetcode.com/problems/uncommon-words-from-two-sentences/)

求两句话中的单词，在本句中出现一次，并不在另一句中的单词。也就是在两句中出现一次。

Java

```java

```

python

```
Input: A = "this apple is sweet", B = "this apple is sour"
Output: ["sweet","sour"]
```

方法一：Counter

```python
class Solution(object):
    def uncommonFromSentences(self, A: 'str', B: 'str') -> 'List[str]':
        count = Counter((A + ' ' + B).split())
        return [word for word, c in count.items() if c == 1]
```

### [916. Word Subsets](https://leetcode.com/problems/word-subsets/)

给两个单词列表，返回A中满足这样条件的单词：B中的所有单词都是此单词的子集。

```
Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","oo"]
Output: ["facebook","google"]
```

Java

```java

```

python

方法一：Counter 比较简单。效率不咋高。

```
def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
    c_a = [Counter(w) for w in A]
    c_b = reduce(operator.or_, [Counter(w) for w in B])
    return [word for i, word in enumerate(A) if not c_b-c_a[i]]
```

方法二：一的基础上改进。& 比较貌似快一丢丢？

```
def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
    c_b = reduce(operator.or_, (Counter(w) for w in B))
    return [a for a in A if c_b & Counter(a) == c_b]
```

方法三：直接查字符会比较快。快了一倍左右

```
def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
    subset = {}
    for b in B:
        for char in b:
            subset[char] = max(subset.get(char, 0), b.count(char))
    return [a for a in A if all(a.count(c) >= subset[c] for c in subset.keys())]
```

### [957. Prison Cells After N Days](https://leetcode.com/problems/prison-cells-after-n-days/)

有8个监狱，如果两边的监狱是相同的，那么次日这个监狱会有人，否则为空。求N天之后的监狱

Java

```java

```

python

方法一：看了提示用hash但是还是没想到，总共有8个监狱，首位肯定是0，那么还有6个，6个监狱一共有多少种情况呢，2**6，也就是说最多这些天形成一种循环。

```python
class Solution(object):
    def prisonAfterNDays(self, cells, n):
        allStates = {}
        while n:
            #temp = ''.join(map(str,cells))
            allStates[str(cells)] = n
            n -= 1
            cells = [0] + [cells[i-1]^cells[i+1]^1 for i in range(1,7)]+ [0]
            if str(cells) in allStates:
                n %= allStates[str(cells)]-n
        return cells
```

方法二：Lee发现了规律，有三个情况一种是1，7，14的时候循环。那么，每14次进行一次循环。但是不能直接进行取余，因为当过了一天，才会进入14天循环中的一天，所以如果当N能被14整除时，并且首位不为0，那么实际上他需要进行变换，而不是直接返回。

```python
def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
    N -= max(N - 1, 0) // 14 * 14
    for i in range(N):
        cells = [0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in range(1, 7)] + [0]
    return cells
```

### [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)

连续的子数组的和能被K整除的个数

```
Input: A = [4,5,0,-2,-3,1], K = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by K = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
```

Java

```java

```

python

方法一：这道题没有在规定时间内完成，此答案参考了排名第一的大佬，然后使用`defaultdict`进行了改进。

[这里](https://www.geeksforgeeks.org/count-sub-arrays-sum-divisible-k/)有一个详细的解答。不过那里给出的答案没有这个简单，不过思路大体相同。

假设通过`sum(i, j)`表示切片[i: j]的总和，如果`sum(i, j)`能被K整除，则说明`sum(0, j) - sum(0, i)`也能被K整除，即对`sum(0, j) % K == sum(0, i) % K`。下面的解法使用一个字典记录了余数的个数。**当余数第二次出现的时候，开始计数，但0的时候除外，因为整除了就产生结果了。**

然后再看累加的方法以下文第3行log为例，mod又为4，这时它和之前余数为4的的数组都可以产生一个结果即`[4, 5, 0] - [4] = [5, 0]` 和`[4 , 5, 0] - [4, 5] = [0]`所以要累加原来的和。

```python
class Solution(object):
    def subarraysDivByK(self, nums, k):
        remainderFrq = defaultdict(int)
        remainderFrq[0] = 1
        res = prefixSum = 0
        for n in nums:
            prefixSum += n
            remainder = prefixSum % k
            res += remainderFrq[remainder]
            remainderFrq[remainder] += 1
        return res
```

### [1002. Find Common Characters](bbb)

在给定的单词列表中找到公共字符。

Java

```java

```

python

``` python
class Solution(object):
    def commonChars(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        if not words: 
            return []
        dic = [0] * 26
        for char in words[0]:
            dic[ord(char) - ord('a')] += 1
        
        for word in range(1, len(words)):
            dic_ano = [0] * 26
            for char in words[word]:
                dic_ano[ord(char) - ord('a')] += 1
            for i in range(26):
                dic[i] = min(dic[i], dic_ano[i])
            
        result = []
        for i in range(26):
            result += [chr(i + ord('a'))] * dic[i]
        return result
```

### [1010. Pairs of Songs With Total Durations Divisible by 60](bbb)

和能被60整除的为一对，求有多少对

Java

```java
class Solution {
    public int numPairsDivisibleBy60(int[] time) {
        int[] map = new int[60];
        int result = 0;
        for (int t : time){
            int mod = t%60;
            if (mod==0)
                result += map[0];
            else
                result += map[60-mod];
            map[mod] ++;
        }
        return result;
    }
}
```

python

``` python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        dic = collections.defaultdict(int)
        ans = 0
        for t in time:
            ans += dic[-t % 60] # dic[(60-t%60)%60]
            dic[t%60] += 1
        return ans
```

### [1015. Smallest Integer Divisible by K](https://leetcode.com/problems/smallest-integer-divisible-by-k/)

Java

```java

```

python

``` python
class Solution(object):
    def smallestRepunitDivByK(self, k):
        """
        :type k: int
        :rtype: int
        """
        n, seen, i = 1, set(), 1
        while True:
            reminder = n%k
            if reminder == 0:
                return i
            if reminder in seen:
                return -1
            seen.add(reminder)
            n = reminder * 10 + 1
            i += 1
```

### [1072. Flip Columns For Maximum Number of Equal Rows](https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/)

二维数组，翻转某几列可以最多使多少行内的元素都相同。

Java

```java
class Solution {
    public int maxEqualRowsAfterFlips(int[][] matrix) {
        Map<String, Integer> hashmap = new HashMap<>();
        for (int[] row : matrix){
            int temp = row[0];
            for (int i=0; i<row.length; i++)
                row[i] ^= temp;
            String s = Arrays.toString(row);
            hashmap.put(s, hashmap.getOrDefault(s, 0)+1);
        }
        int result = 0;
        for (int value : hashmap.values())
            result = Math.max(result,value);
        return result;
    }
}
```

python

``` python
class Solution:
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        c = collections.Counter()
        for row in matrix:
            c[tuple([x^row[0] for x in row])] += 1
        return max(c.values())
```

### [1128. Number of Equivalent Domino Pairs](https://leetcode.com/problems/number-of-equivalent-domino-pairs/description/)

相等的多米诺骨牌对数。

Java

```java
class Solution {
    public int numEquivDominoPairs(int[][] dominoes) {
        Map<String, Integer> map = new HashMap<>();
        int result = 0;
        for (int[] domino : dominoes){
            Arrays.sort(domino);
            String key = Arrays.toString(domino);
            if (! map.containsKey(key))
                map.put(key, 0);
            result += map.get(key);
            map.put(key, map.get(key)+1);
        }
        return result;
    }
}
```

python

``` python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        stat = collections.defaultdict(int)
        total = 0
        for domino in dominoes:
            temp = tuple(sorted(domino))
            total += stat[temp]
            stat[temp] += 1
        return total
```

### [1138. Alphabet Board Path](https://leetcode.com/problems/alphabet-board-path/)

小写字母排列的键盘，要打出目标字母需要移动的操作。

Java

```java
class Solution {
    public String alphabetBoardPath(String target) {
        int x = 0, y = 0;
        StringBuilder sb = new StringBuilder();
        for (char c : target.toCharArray()){
            int x1 = (c - 'a') / 5;
            int y1 = (c - 'a') % 5;
            while(x1 < x) {x--; sb.append('U');}
            while(y1 > y) {y++; sb.append('R');}
            while(y1 < y) {y--; sb.append('L');}
            while(x1 > x) {x++; sb.append('D');}
            sb.append('!');
        }
        return sb.toString();
    }
}
```

python

``` python
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        result = ''
        x0 = y0 = 0
        for c in target:
            x, y = (ord(c)-ord('a'))//5, (ord(c)-ord('a'))%5
            if y < y0: result += 'L' * (y0-y)
            if x < x0: result += 'U' * (x0-x)
            if y > y0: result += 'R' * (y-y0)
            if x > x0: result += 'D' * (x-x0)
            x0, y0 = x, y
            result += '!'
        return result
```

### [1160. Find Words That Can Be Formed by Characters](https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/)

找出能被目标字符串组成的子串长度和。

Java

```java
class Solution {
    public int countCharacters(String[] words, String chars) {
        int[] map = new int[26];
        int res = 0;
        for (char c : chars.toCharArray())
            map[c-'a'] ++;
        for (String word : words){
            int[] tempMap = map.clone();
            for (char c : word.toCharArray()){
                tempMap[c-'a'] --;
                if (tempMap[c-'a'] < 0){
                    res -= word.length();
                    break;
                }
            }
            res += word.length();
        }
        return res;
    }
}
```

python

``` python
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        res = 0
        charstat = collections.Counter(chars)
        for word in words:
            wordstat = collections.Counter(word)
            for key, value in wordstat.items():
                if (key not in charstat) or (charstat[key]<value):
                    break
            else:res += len(word)
        return res
```

### [1311. Get Watched Videos by Your Friends](https://leetcode.com/problems/get-watched-videos-by-your-friends/)

找到你的level级别的朋友看的电影，按照频率字母排序。

Java

```java

```

python

``` python
class Solution:
    def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]], id: int, level: int) -> List[str]:
        bfs, seen = {id}, {id}
        for _ in range(level):
            tempset = set()
            for i in bfs:
                for j in friends[i]:
                    if j not in seen:
                        tempset.add(j)
            bfs = tempset
            seen |= bfs
        videos=collections.Counter([v for idx in bfs for v in watchedVideos[idx]])
        return sorted(videos, key=lambda x: (videos[x], x))
```

### [1394. Find Lucky Integer in an Array](https://leetcode.com/problems/find-lucky-integer-in-an-array/)

找到数组中数字和出现次数一致的最大的数

Java

```java
class Solution {
    public int findLucky(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        int result = -1;
        for (int num : arr)
            map.put(num, map.getOrDefault(num, 0)+1);
        for (int key : map.keySet())
            if (key == map.get(key))
                result = Math.max(result, key);
        return result;
    }
}
```

python

``` python
class Solution:
    def findLucky(self, arr: List[int]) -> int:
        dic = collections.Counter(arr)
        result = -1
        for i, num in dic.items():
            if i == num: result = max(result, num)
        return result
```

### [1396. Design Underground System](https://leetcode.com/problems/design-underground-system/description/)

找到数组中数字和出现次数一致的最大的数

Java

```java
class UndergroundSystem {
    Map<Integer, Pair<String, Integer>> checkIns = new HashMap<>();
    Map<Pair<String, String>, int[]> times = new HashMap<>();
    public UndergroundSystem() {
        
    }
    
    public void checkIn(int id, String stationName, int t) {
        checkIns.put(id, new Pair(stationName, t));
    }
    
    public void checkOut(int id, String stationName, int t) {
        var startStation = checkIns.get(id).getKey();
        var startTime = checkIns.get(id).getValue();
        checkIns.remove(id);
        var pair = new Pair(startStation, stationName);
        int totalTime = times.containsKey(pair) ? times.get(pair)[0] : 0;
        int dataPoints = times.containsKey(pair) ? times.get(pair)[1] : 0;

        times.put(pair, new int[] {totalTime + t - startTime, dataPoints + 1});
    }
    
    public double getAverageTime(String startStation, String endStation) {
        var pair = new Pair(startStation, endStation);
        return (double) times.get(pair)[0] / times.get(pair)[1];
    }
}
```

python

``` python
class UndergroundSystem:

    def __init__(self):
        self.journey = {}
        self.history = {} # (startStation, endStation) => (allTime, allCount)

    def checkIn(self, id: int, startStation: str, t: int) -> None:
        self.journey[id] = (startStation, t)
        

    def checkOut(self, id: int, endStation: str, endTime: int) -> None:
        startStation, startTime = self.journey.pop(id)
        key = (startStation, endStation)
        allTime, allCount = self.history.get(key, (0, 0))
        self.history[key] = (allTime + (endTime - startTime), allCount + 1)
        

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        key = (startStation, endStation)
        allTime, allCount = self.history.get(key, (0, 0))
        return allTime / allCount
```

### [1399. Count Largest Group](https://leetcode.com/problems/count-largest-group/)

以数字和为分组，求最大组的个数。

Java

```java
class Solution {
    public int countLargestGroup(int n) {
        int[] array = new int[37];
        for (int i = 1; i <= n; i++){
            array[countDigits(i)-1] ++;
        }
        int result = 0;
        int max = array[0];
        for (int num : array){
            if (num > max){
                result = 1;
                max = num;
            }else if (num == max)
                result ++;
        }
        return result;
    }

    private int countDigits(int n) {
        if (n == 0) return 0;
        return n%10 + countDigits(n/10);
    }
}
```

python

方法一：Counter

``` python
class Solution:
    def countLargestGroup(self, n: int) -> int:
        stat = collections.Counter()
        for i in range(1,n+1):
            stat[self.countDigits(i)] += 1
        result = 0
        max = 1
        for num in stat.values():
            if num > max:
                result = 1
                max = num
            elif num == max:
                result += 1
        return result

    def countDigits(self, n):
        if n == 0: return 0
        return n%10 + self.countDigits(n//10)
```

方法二：标准库。`statistics.multimode`返回一个可迭代对象中出现次数最多的元素。

```
def countLargestGroup(self, n: int) -> int:
    import statistics
    return len(statistics.multimode(sum(map(int, str(d))) for d in range(1, n+1)))
```

### [1497. Check If Array Pairs Are Divisible by k](https://leetcode.com/problems/check-if-array-pairs-are-divisible-by-k/)

Java

```java

```

python

``` python
class Solution(object):
    def canArrange(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: bool
        """
        dic = defaultdict(int)
        for num in arr:
            dic[(num%k+k)%k] += 1
        for key,value in dic.items():
            if key==0 and value%2!=0: 
                return False
            if key and dic[key] != dic[k-key]: 
                return False
        return True
```

### [1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target](https://leetcode.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/description/)

和为目标值的子数组最多有多少个，子数组不能重复

Java

```java
class Solution {
    public int maxNonOverlapping(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        int result = 0;
        int total = 0;
        for (int num : nums){
            total += num;
            if (map.containsKey(total - target))
                result = Math.max(result, map.get(total-target)+1);
            map.put(total, result);
        }
        return result;
    }
}
```

python

``` python
class Solution:
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        result, tempsum = 0, 0
        dic = {0:0}
        for num in nums:
            tempsum += num
            if tempsum-target in dic:
                result = max(result, dic[tempsum-target]+1)
            dic[tempsum] = result
        return result
```

### [1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers](https://leetcode.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers/)

找到一个数组中两个元素乘积等于另一个数组的平方，求总共的个数

Java

```java
class Solution {
    public int numTriplets(int[] nums1, int[] nums2) {
        Map<Long, Integer> map1 = new HashMap<>();
        Map<Long, Integer> map2 = new HashMap<>();
        for (int num : nums1)
            map1.put((long)num*num, map1.getOrDefault((long)num*num, 0) + 1);
        for (int num : nums2)
            map2.put((long)num*num, map2.getOrDefault((long)num*num, 0) + 1);
        int result = 0;
        for (int i=0; i<nums1.length; i++)
            for (int j=i+1; j<nums1.length; j++)
                result += map2.getOrDefault((long)nums1[i]*nums1[j], 0);
        for (int i=0; i<nums2.length; i++)
            for (int j=i+1; j<nums2.length; j++)
                result += map1.getOrDefault((long)nums2[i]*nums2[j], 0);
        return result;
    }
}
```

python

``` python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        map1 = Counter(n**2 for n in nums1)
        map2 = Counter(n**2 for n in nums2)
        result = 0
        for i in range(0, len(nums1)):
            for j in range(i+1, len(nums1)):
                result += map2[nums1[i]*nums1[j]]
        for i in range(0, len(nums2)):
            for j in range(i+1, len(nums2)):
                result += map1[nums2[i]*nums2[j]]
        return result
```

### [1590. Make Sum Divisible by P](https://leetcode.com/problems/make-sum-divisible-by-p/)

删除最小的连续子数组，使得整个数组和能被P整除

Java

```java
class Solution {
    public int minSubarray(int[] nums, int p) {
        int sum = 0;
        for (int num : nums) 
            sum = (sum + num) % p;
        if(sum == 0) return 0;
        
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int prefixSum = 0, result = nums.length;
        for (int i = 0; i < nums.length; i++){
            prefixSum = (prefixSum + nums[i]) % p;
            int difference = (prefixSum - sum + p) % p;
            if (map.containsKey(difference))
                result = Math.min(result, i-map.get(difference));
            map.put(prefixSum, i);
        }
        return result == nums.length ? -1 : result;
    }
}   
```

python

``` python
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        sum = 0
        for num in nums:
            sum = (sum + num) % p
        if sum == 0: return 0
        dic = {0 : -1}
        prefixSum, result = 0, len(nums)
        for i in range(len(nums)):
            prefixSum = (prefixSum + nums[i]) % p
            difference = (prefixSum - sum + p) % p
            if difference in dic:
                result = min(result, i-dic[difference])
            dic[prefixSum] = i
        return -1 if result == len(nums) else result
```

### [1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/)

给定一个数组和一个目标数x，每次可以将x-数组两边的某个数，最少需要多少步可以将x变为0

Java

```java
class Solution {
    public int minOperations(int[] nums, int x) {
        int target = -x;
        for (int num : nums) target += num;
        if (target == 0) return nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int sum = 0;
        int res = -1;
        for (int i = 0; i < nums.length; ++i) {
            sum += nums[i];
            if (map.containsKey(sum - target))
                res = Math.max(res, i - map.get(sum - target));
            map.put(sum, i);
        }
        return res == -1 ? -1 : nums.length - res;
    }
}
```

python

方法一：变长滑动窗口。比赛时没有做出来，想成BFS了，结果超时。这题和1423很像，那题是定长的滑动窗口，此题可以转化为，找到一个最长的窗口使得窗口值的和等于总和-x。1200ms.

```python
def minOperations(self, nums: List[int], x: int) -> int:
    target = sum(nums) - x
    lo, N = 0, len(nums)
    cur_sum, res = 0, -1
    for hi, num in enumerate(nums):
        cur_sum += num
        while lo+1<N and cur_sum>target:
            cur_sum -= nums[lo]
            lo += 1
        if cur_sum == target:
            res = max(res, hi-lo+1)
    return res if res<0 else N-res
```

方法二：前缀和+字典，这个写法比较优雅，但是比较难想。`i`表示从右边减去的数字，`mp[x]`表示从左边减去的数字。

```python
def minOperations(self, nums: List[int], x: int) -> int:
    mp = {0: 0}
    prefix = 0
    for i, num in enumerate(nums, 1): 
        prefix += num
        mp[prefix] = i 
    
    ans = mp.get(x, inf)
    for i, num in enumerate(reversed(nums), 1): 
        x -= num
        if x in mp and mp[x] + i <= len(nums): 
            ans = min(ans, i + mp[x])
    return ans if ans < inf else -1
```

### [1711. Count Good Meals](https://leetcode.com/problems/count-good-meals/)

给你一个整数数组 deliciousness ，其中 deliciousness[i] 是第 i 道餐品的美味程度，返回你可以用数组中的餐品做出的不同 大餐 的数量。结果需要对 109 + 7 取余

Java

```java
class Solution {
    int mod = 1000000007;
    public int countPairs(int[] deliciousness) {
        int[] power = new int[22];
        int p = 1;
        Map<Integer, Integer> map = new HashMap<>();
        long res = 0;
        for (int i = 0; i < 22; i++){
            power[i] = p;
            p *= 2;
        }
        for (int num : deliciousness){
            for (int pow : power){
                if (map.containsKey(pow - num)){
                    res += map.get(pow - num);
                    res %= mod;
                }
            }
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        return (int) res;
    }
}
```

python

``` python
class Solution:
    def countPairs(self, deliciousness: List[int]) -> int:
        refer = [2**i for i in range(22)]
        result = 0
        m = Counter()
        for num in deliciousness:
            for power in refer:
                if power-num in m:
                    result += m[power-num]
            m[num] += 1
        return result % (10**9 + 7)
```

### [2114. Maximum Number of Words Found in Sentences](https://leetcode.com/problems/maximum-number-of-words-found-in-sentences/)

Java

```java

```

python

``` python
class Solution(object):
    def mostWordsFound(self, sentences):
        """
        :type sentences: List[str]
        :rtype: int
        """
        return max([len(x.split()) for x in sentences])
```



## 待解决

### [1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

和为目标值的子矩阵个数

```
Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.
```

方法一：560的升级版。需要先求出每行的前缀和。然后选定两个列（可以相同），以这两个列为宽度，高度逐渐递增，寻找这个宽度的子矩阵的和。

```
def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
    M, N = len(matrix), len(matrix[0])
    for i, row in enumerate(matrix):
        matrix[i][:] = accumulate(row)
    res = 0
    for i in range(N):
        for j in range(i, N):
            cur, d = 0, defaultdict(int)
            d[0] = 1
            for k in range(M):
                cur += matrix[k][j] - (matrix[k][i-1] if i>0 else 0)
                res += d[cur-target]
                d[cur] += 1
    return res
```

### [1224. Maximum Equal Frequency](https://leetcode.com/problems/maximum-equal-frequency/description/)

给定一个数组，返回这个数组最长的前缀，前缀中刚好有删除一个元素使其它元素的频率相等。

```
Input: nums = [2,2,1,1,5,3,3,5]
Output: 7
Explanation: For the subarray [2,2,1,1,5,3,3] of length 7, if we remove nums[4]=5, we will get [2,2,1,1,3,3], so that each number will appear exactly twice.
```

方法一：Lee215的答案，一共分为两种情况，一种情况是将当前元素删除，那么前面的元素具有相同的频率。如果不删除当前的元素，那么这个元素出现了c次，我们用`freq`来记录出现i次的有多少个数。那么删除的元素只能是出现c+1次或者1次，并且这个数只有一个。

```
def maxEqualFreq(self, A: List[int]) -> int:
    count = collections.defaultdict(int)
    freq = [0] * (len(A)+1)
    res = 0
    for n, a in enumerate(A, 1):
        freq[count[a]+1] += 1
        freq[count[a]] -= 1
        c = count[a] = count[a] + 1
        if freq[c]*c==n and n < len(A):
            res = n + 1
        d = n - freq[c]*c
        if d in (c+1, 1) and freq[d]==1:
            res = n
    return res
```

### [2122. Recover the Original Array](https://leetcode.com/problems/recover-the-original-array/)

恢复原始的数组，给你一个合并后的数组，由原数组每个元素-k和+k形成两个数组后合并。

```
Input: nums = [2,10,6,4,8,12]
Output: [3,7,11]
Explanation:
If arr = [3,7,11] and k = 1, we get lower = [2,6,10] and higher = [4,8,12].
Combining lower and higher gives us [2,6,10,4,8,12], which is a permutation of nums.
Another valid possibility is that arr = [5,7,9] and k = 3. In that case, lower = [2,4,6] and higher = [8,10,12].
```

方法一：竞赛时险过，有两个地方想复杂了，其实没必要用堆，本来复制数组就用了O(N)的复杂度。思路是先找`2k`，然后判断这个是否可行。找的过程也是需要O(N)。

```
def recoverArray(self, nums: List[int]) -> List[int]:

    def check(d):
        c, res = Counter(nums), []
        for num in nums:
            if c[num] == 0:
                continue
            elif c[num+d] == 0:
                return False, []
            c[num] -= 1
            c[num+d] -= 1
            res.append(num+d//2)
        return True, res

    N = len(nums)
    nums.sort()
    for i in range(1, N):
        k = nums[i]-nums[0]  # 对比最小元素
        if k!=0 and k&1==0:
            valid, res = check(k)
            if valid: return res
```

### [2488. Count Subarrays With Median K](https://leetcode.com/problems/count-subarrays-with-median-k/)

#### 统计子数组中位数为k的个数。

给你一个长度为 `n` 的数组 `nums` ，该数组由从 `1` 到 `n` 的 **不同** 整数组成。另给你一个正整数 `k` 。

统计并返回 `num` 中的 **中位数** 等于 `k` 的非空子数组的数目

```
输入：nums = [3,2,1,4,5], k = 4
输出：3
解释：中位数等于 4 的子数组有：[4]、[4,5] 和 [1,4,5] 。
```

方法一：比赛时没有时间做，瞄了一眼以为用堆，结果不是。首先很重要得一个条件是，所有的数不同，那么能使中位数为K，一定会包含k，所以可以围绕k向两侧拓展。

对于一个合理的子数组，一定保证小于k的和大于k的相等（奇数），小于k的+1等于大于k的（偶数）。

用 `l1`, `s1` 表示k左侧比k大，比k小的个数。`l2`,`s2`表示k右侧比k大，比k小的个数。

得出`l1+l2=s1+s2`或者`l1+l2=s1+s2+1`，推论出`l1-s1=s2-l2`和`l1-s1=s2-l2+1`。这样可以维护一个map来计数。注意，需要考虑没有左或者右的情况。

```
def countSubarrays(self, nums: List[int], k: int) -> int:
    index = nums.index(k)

    cnt = defaultdict(int)
    res = l1 = s1 = 0
    for i in reversed(range(index)):  #注意这里要倒序
        if nums[i] < k:
            s1 += 1
        else:
            l1 += 1
        cnt[l1-s1] += 1
        if l1==s1 or l1==s1+1:
            res += 1
    l2 = s2 = 0
    for i in range(index+1, len(nums)):
        if nums[i] < k:
            s2 += 1
        else:
            l2 += 1
        res += cnt[s2-l2] + cnt[s2-l2+1]
        if l2==s2 or l2==s2+1:
            res += 1
    return res + 1
```

