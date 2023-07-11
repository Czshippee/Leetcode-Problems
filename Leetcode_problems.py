# Utils
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
# import utility of queue
from collections import deque
# import upriority queue
import heapq
from heapq import heappush, heappop


'''
0001. Two Sum
https://leetcode.com/problems/two-sum/
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
'''
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        records = {}
        for index, value in enumerate(nums):
            if target-value in records:
                return [records[target-value], index]
            else:
                records[value] = index

'''
2. Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. 
The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Example:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
'''
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = dummy = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            v1,v2 = 0,0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            dummy.next = ListNode(val)
            dummy = dummy.next
        return head.next

'''
3. Longest Substring Without Repeating Characters
Given a string s, find the length of the longest substring without repeating characters.
Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
'''
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

'''
4. Median of Two Sorted Arrays
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
The overall run time complexity should be O(log (m+n)).
Example:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
'''
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        m, n = len(nums1), len(nums2)
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        imin, imax, half_len = 0, m, (m + n + 1) // 2
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and nums2[j-1] > nums1[i]:
                imin = i + 1
            elif i > 0 and nums1[i-1] > nums2[j]:
                imax = i - 1
            else:
                if i == 0: max_of_left = nums2[j-1]
                elif j == 0: max_of_left = nums1[i-1]
                else: max_of_left = max(nums1[i-1], nums2[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left

                if i == m: min_of_right = nums2[j]
                elif j == n: min_of_right = nums1[i]
                else: min_of_right = min(nums1[i], nums2[j])

                return (max_of_left + min_of_right) / 2.0

'''
5. Longest Palindromic Substring
Given a string s, return the longest palindromic substring in s.
Example:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
'''
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

'''
6. Zigzag Conversion
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"
Write the code that will take a string and make this conversion given a number of rows:
string convert(string s, int numRows);
Example:
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
'''
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        res = ["" for _ in range(numRows)]
        index = 0
        while index < len(s):
            for i in range(numRows):
                if index >= len(s):
                    break
                res[i] += s[index]
                index += 1
            for j in range(numRows-2, 0, -1):
                if index >= len(s):
                    break
                res[j] += s[index]
                index += 1
        return ''.join(res)

'''
10. Regular Expression Matching
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
Example:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
'''
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not p: return not s
        matchStr = bool(s) and p[0] in {s[0], '.'}
        if len(p)>=2 and p[1]=='*':
            return (self.isMatch(s,p[2:]) or (matchStr and self.isMatch(s[1:],p)))
        else:
            return matchStr and self.isMatch(s[1:], p[1:])

'''
11. Container With Most Water
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.
Notice that you may not slant the container.
Example:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
'''
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left, right = 0, len(height)-1
        resarea = 0
        while left < right:
            resarea = max(resarea, (right-left)*min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return resarea

'''
12. Integer to Roman
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.
Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:
I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given an integer, convert it to a roman numeral.
Example:
Input: num = 3
Output: "III"
'''
class Solution:
    def intToRoman(self, num: int) -> str:
        Roman = ""
        storeIntRoman = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
                        [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], 
                        [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]]
        for i in range(len(storeIntRoman)):
            while num >= storeIntRoman[i][0]:
                Roman += storeIntRoman[i][1]
                num -= storeIntRoman[i][0]
        return Roman

'''
13. Roman to Integer
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.
Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:
I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.
Example:
Input: s = "III"
Output: 3
'''
class Solution:
    def romanToInt(self, s: str) -> int:
        res, num = 0, 0
        for i in range(len(s)-1, -1, -1):
            if s[i]=='I': num = 1
            elif s[i] =='V': num = 5
            elif s[i] =='X': num = 10
            elif s[i] =='L': num = 50
            elif s[i] =='C': num = 100
            elif s[i] =='D': num = 500
            elif s[i] =='M': num = 1000
            if 4*num < res: 
                res -= num
            else: res += num
        return res

'''
14. Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".
Example:
Input: strs = ["flower","flow","flight"]
Output: "fl"
'''
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        prefix = ''
        shortestleng = min([len(i) for i in strs])
        for i in range(shortestleng):
            temp = ''
            for word in strs:
                if temp == '':
                    temp = word[i]
                elif temp != word[i]:
                    return prefix
            prefix += temp
        return prefix

'''
0015. 3Sum
https://leetcode.com/problems/3sum/
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
Notice that the solution set must not contain duplicate triplets.
Example :
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
'''
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
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

'''
16. 3Sum Closest
Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.
Return the sum of the three integers.
You may assume that each input would have exactly one solution.
Example:
Input: nums = [-1,2,1,-4], target = 1
Output: 2
'''
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        mindiff = float('inf')
        res = 0
        for i, firstNum in enumerate(nums):
            if i>0 and firstNum==nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                threeSome = firstNum + nums[left] + nums[right]
                if threeSome == target:
                    return target
                elif abs(threeSome - target) < mindiff:
                    mindiff = abs(threeSome - target)
                    res = threeSome
                elif threeSome < target:
                     left += 1
                elif threeSome > target:
                    right -= 1
        return res

'''
17. Letter Combinations of a Phone Number
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
Example:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
'''
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        dic = {'2':"abc",'3':"def",'4':"ghi",'5':"jkl",'6':"mno",'7':"pqrs",'8':"tuv",'9':"wxyz"}
        results = []
        if digits == "": 
            return []
        def backtracking(digit, result):
            if digit == "":
                results.append(result)
                return
            for char in dic[digit[0]]:
                backtracking(digit[1:], result+char)
        backtracking(digits, "")
        return results

'''
0018. 4Sum
https://leetcode.com/problems/4sum/
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.
Example:
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
'''
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

'''
0019. Remove Nth Node From End of List
https://leetcode.com/problems/remove-nth-node-from-end-of-list/
Given the head of a linked list, remove the nth node from the end of the list and return its head.
Example :
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
'''
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummynode = ListNode(next = head)
        slow = fast = dummynode
        for i in range(n):
            fast = fast.next
        while fast.next is not None:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummynode.next

'''
0020. Valid Parentheses
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Example:
Input: s = "()"
Output: true
'''
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

'''
21. Merge Two Sorted Lists
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
Example:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
'''
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

'''
23. Merge k Sorted Lists
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.
Example:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
'''
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


'''
24. Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
Example:
Input: head = [1,2,3,4]
Output: [2,1,4,3]
'''
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummyNode = ListNode(next = head)
        current = dummyNode
        while current.next and current.next.next:
            former = current.next
            later = current.next.next
            former.val, later.val = later.val, former.val
            current = current.next.next
        return dummyNode.next

'''
25. Reverse Nodes in k-Group
Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
You may not alter the values in the list's nodes, only nodes themselves may be changed.
Example:
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
'''
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = jump =  ListNode(next = head)
        left = right = head
        while True:
            length = 0
            while right and length < k:
                right = right.next
                length += 1
            if length == k:
                prev, cur = right, left
                for _ in range(k):
                    cur.next, prev, cur = prev, cur, cur.next
                jump.next, jump, left = prev, left, right
            else:
                return dummy.next

'''
26. Remove Duplicates from Sorted Array
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. 
The relative order of the elements should be kept the same.
Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. 
More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.
Return k after placing the final result in the first k slots of nums.
Example :
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
'''
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = 0
        fast = 0
        dic = set()
        while fast != len(nums):
            if nums[fast] not in dic:
                dic.add(nums[fast])
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        for i in range(slow, len(nums)):
            nums[i] = '_'
        return slow

'''
27. Remove Element
Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.
Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. 
More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result.
It does not matter what you leave beyond the first k elements.
Return k after placing the final result in the first k slots of nums.
Example:
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
'''
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        slow = 0
        fast = 0
        while fast != len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        for i in range(slow, len(nums)):
            nums[i] = '_'
        return slow

'''
28. Implement strStr()
Implement strStr().
Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
Clarification:
What should we return when needle is an empty string? This is a great question to ask during an interview.
For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().
Example:
Input: haystack = "hello", needle = "ll"
Output: 2
'''
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        i, j = 0, 0
        nextArr = self.getnext(needle)
        while (i<len(haystack) and j<len(needle)):
            if haystack[i] == needle[j]:
                i,j = i+1,j+1
            elif j==0:
                i += 1
            else: j = nextArr[j]
        if j==len(needle):return i-j
        else:return -1
        
    def getnext(self, needle):
        if len(needle) == 1:return[-1]
        nextArr = [0 for _ in range(len(needle))]
        nextArr[0], nextArr[1] = -1, 0
        i, current=2, 0
        while i < len(nextArr):
            if needle[i-1]==needle[current]:
                current += 1
                nextArr[i] = current
                i += 1
            elif current > 0:
                current = nextArr[current]
            else:
                nextArr[i] = 0
                i += 1
        return nextArr

'''
30. Substring with Concatenation of All Words
You are given a string s and an array of strings words. All the strings of words are of the same length.
A concatenated substring in s is a substring that contains all the strings of any permutation of words concatenated.
For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", and "efcdab" are all concatenated strings. "acdbef" is not a concatenated substring because it is not the concatenation of any permutation of words.
Return the starting indices of all the concatenated substrings in s. You can return the answer in any order.
Example:
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Since words.length == 2 and words[i].length == 3, the concatenated substring has to be of length 6.
The substring starting at 0 is "barfoo". It is the concatenation of ["bar","foo"] which is a permutation of words.
The substring starting at 9 is "foobar". It is the concatenation of ["foo","bar"] which is a permutation of words.
The output order does not matter. Returning [9,0] is fine too.
'''
class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        res = []
        wordlength = len(words[0])
    
        for i in range(wordlength):
            dic = collections.Counter(words)
            for j in range(i, len(s) + 1 - wordlength, wordlength):
                substring = s[j : j + wordlength]
                dic[substring] -= 1
                while dic[substring] < 0:
                    dic[s[i : i + wordlength]] += 1
                    i += wordlength
                if i + len(words) * len(words[0]) == j + wordlength:
                    res.append(i)
        return res

'''
33. Search in Rotated Sorted Array
There is an integer array nums sorted in ascending order (with distinct values).
Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
You must write an algorithm with O(log n) runtime complexity.
Example:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
'''
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

'''
34. Find First and Last Position of Element in Sorted Array
Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
If target is not found in the array, return [-1, -1].
You must write an algorithm with O(log n) runtime complexity.
Example:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Runtime: 56 ms, faster than 99.64% of Python online submissions
'''
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def binarysearch(nums, target):
            left, right = 0, len(nums)-1
            while left <= right:
                mid = (left + right) // 2
                # 寻找target的左边界，要在nums[mid] == target的时候更新right
                # 寻找target的右边界, 要在nums[mid] == target的时候更新left
                if nums[mid] >= target:
                    right = mid-1
                else:
                    left = mid+1
            return right+1
        leftboundary = binarysearch(nums, target)
        rightboundary = binarysearch(nums, target+1) - 1
        if leftboundary >= len(nums) or nums[leftboundary] != target:
            return [-1,-1]
        else:
            return [leftboundary,rightboundary]

'''
35. Search Insert Position
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
You must write an algorithm with O(log n) runtime complexity.
Example :
Input: nums = [1,3,5,6], target = 5
Output: 2
'''
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        a, b = 0, len(nums)-1
        while a <= b:
            mid = (a+b) // 2
            if target < nums[mid]:
                b = mid-1
            elif target > nums[mid]:
                a = mid+1
            else:
                return mid
        return b+1

'''
36. Valid Sudoku
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
'''
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        for row in board:
            s = set()
            for ele in row:
                if ele != '.':
                    if ele not in s: 
                        s.add(ele)
                    else: return False
        for col in range(9):
            s = set()
            num = 0
            for row in range(9):
                if board[row][col] != '.':
                    if board[row][col] not in s:
                        s.add(board[row][col])
                    else: return False
        start = [(0,0),(3,0),(6,0),(0,3),(3,3),(6,3),(0,6),(3,6),(6,6)]
        for sr,sc in start:
            s = set()
            num = 0
            for i in range(3):
                for j in range(3):
                    if board[sr+i][sc+j] != '.':
                        s.add(board[sr+i][sc+j])
                        num += 1
            if len(s) != num: return False
        return True

'''
37. Sudoku Solver
Write a program to solve a Sudoku puzzle by filling the empty cells.
A sudoku solution must satisfy all of the following rules:
Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.
Example:
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
'''
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def backtracking():
            for i in range(9):
                for j in range(9):
                    if (board[i][j] != '.'): 
                        continue
                    for k in range(1,10):
                        if self.check(board, i, j, str(k)):
                            board[i][j] = str(k)
                            if backtracking(): return True
                            board[i][j] = '.'
                    return False
            return True
        backtracking()
    
    def check(self, board, row, col, val):
        for i in range(9):
            if board[row][i] == val: return False
        for j in range(9):
            if board[j][col] == val: return False
        startRow = (row // 3) * 3
        startCol = (col // 3) * 3
        for i in range(startRow, startRow+3):
            for j in range(startCol, startCol+3):
                if board[i][j] == val: return False 
        return True

'''
38. Count and Say
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:
countAndSay(1) = "1"
countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.
To determine how you "say" a digit string, split it into the minimal number of substrings such that each substring contains exactly one unique digit. Then for each substring, say the number of digits, then say the digit. Finally, concatenate every said digit.
For example, the saying and conversion for digit string "3322251":
Given a positive integer n, return the nth term of the count-and-say sequence.
Example:
Input: n = 1
Output: "1"
Explanation: This is the base case.
'''
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        result = '1'
        for _ in range(n-1): 
            temp = ''
            for key, nums in itertools.groupby(result):
                temp+=str(len(list(nums)))+key
            result = temp
        return result

'''
39. Combination Sum
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target.
You may return the combinations in any order.
The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.
It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
Example:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
'''
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        results = []
        result = []
        def backtracking(totalSum, startInd):
            if totalSum == target:
                results.append(result[:])
                return
            for i in range(startInd, len(candidates)):
                candidate = candidates[i]
                if totalSum+candidate>target:return
                totalSum += candidate
                result.append(candidate)
                backtracking(totalSum, i)
                totalSum -= candidate
                result.pop()
        backtracking(0, 0)
        return results

'''
40. Combination Sum II
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.
Note: The solution set must not contain duplicate combinations.
Example:
Input: candidates = [10,1,2,7,6,1,5], target = 8
'''
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        results = []
        result = []
        def backtracking(totalSum, startInd):
            if totalSum == target:
                results.append(result[:])
                return
            for i in range(startInd, len(candidates)):
                candidate = candidates[i]
                if totalSum+candidate>target:
                    return
                if i>startInd and candidates[i] == candidates[i - 1]:
                    continue
                totalSum += candidate
                result.append(candidate)
                backtracking(totalSum, i+1)
                totalSum -= candidate
                result.pop()
        backtracking(0, 0)
        return results

'''
41. First Missing Positive
Given an unsorted integer array nums, return the smallest missing positive integer.
You must implement an algorithm that runs in O(n) time and uses constant extra space.
Example:
Input: nums = [1,2,0]
Output: 3
'''
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(len(nums)):
            while 0 <= nums[i]-1 < len(nums) and nums[nums[i]-1] != nums[i]:
                j = nums[i] - 1
                nums[i], nums[j] = nums[j], nums[i]
        for i,num in enumerate(nums):
            if num != i+1:
                return i+1
        return len(nums)+1

'''
42. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
Example:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
'''
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        stack = []
        totalsum = 0
        for i in range(0,len(height)):
            while stack and height[i]>height[stack[-1]]:
                mid = stack[-1]
                stack.pop()
                if stack:
                    h = min(height[stack[-1]], height[i]) - height[mid]
                    w = i - stack[-1] -1
                    totalsum += h * w
            stack.append(i)
        return totalsum

'''
43. Multiply Strings
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
Example:
Input: num1 = "2", num2 = "3"
Output: "6"
'''
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        dic={'1':1,'2':2,'3':3,'4':4,'5':5,
            '6':6,'7':7,'8':8,'9':9,'0':0}
        m, n = len(num1), len(num2)
        res = [0 for _ in range(m + n)]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                p1, p2 = i+j, i+j+1
                sum = (dic[num1[i]] * dic[num2[j]]) + res[p2]
                res[p1] += sum // 10
                res[p2] = (sum) % 10
                print(sum)
        result = []
        for digit in res:
            if not (len(result) == 0 and digit == 0):
                result.append(digit)
        if len(result) == 0: return "0"
        return ''.join(map(str, result))

'''
45. Jump Game II
Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.
You can assume that you can always reach the last index.
Example:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
'''
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cur,nxt,res = 0,0,0
        for i in range(len(nums)-1):
            nxt = max(nxt, i + nums[i])
            if i == cur:
                res += 1
                cur = nxt
                if cur >= len(nums)-1:
                    break
        return res

'''
46. Permutations
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
Example:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
'''
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        results = []
        result = []
        used = [0 for i in range(len(nums))]
        def backtracking():
            if len(result) == len(nums):
                results.append(result[:])
                return
            for i in range(len(nums)):
                if used[i]==1:
                    continue
                result.append(nums[i])
                used[i]=1
                backtracking()
                result.pop()
                used[i]=0
        backtracking()
        return results

'''
47. Permutations II
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
Example:
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
'''
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        results = []
        used = [0 for i in range(len(nums))]
        def backtracking(result):
            if len(result) == len(nums):
                results.append(result[:])
                return
            for i in range(len(nums)):
                if used[i]==1:
                    continue
                if i>0 and nums[i-1]==nums[i] and used[i-1]==0:
                    continue
                result.append(nums[i])
                used[i]=1
                backtracking(result)
                result.pop()
                used[i]=0
        backtracking([])
        return results

'''
48. Rotate Image
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
Example:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
'''
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

'''
49. Group Anagrams
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
Example:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
'''
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strs_table = {}
        for string in strs:
            sorted_string = ''.join(sorted(string))
            if sorted_string not in strs_table:
                strs_table[sorted_string] = []
            strs_table[sorted_string].append(string)
        return list(strs_table.values())

'''
50. Pow(x, n)
Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
Example:
Input: x = 2.00000, n = 10
Output: 1024.00000
'''
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        def help(x, n):
            if x == 0: return 0
            if n == 0: return 1
            res = help(x, n//2)
            res = res*res
            return res if n%2 ==0 else x*res
        res = help(x, abs(n))
        return res if n>=0 else 1/res

'''
51. N-Queens
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.
Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.
Example:
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
'''
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        boards = []
        board = [['.'] * n for _ in range(n)]
        def backtracking(row):
            if row == n:
                temp_res = []
                for temp in board:
                    temp_str = "".join(temp)
                    temp_res.append(temp_str)
                boards.append(temp_res)
                return
            for col in range(n):
                if not self.isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(row+1)
                board[row][col] = '.'
        backtracking(0)
        return boards
    
    def isVaild(self, board, row, col):
        for i in range(len(board)):
            if board[i][col] == 'Q':
                return False
        i, j = row - 1, col - 1
        while i>=0 and j>=0:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j - 1
        i, j = row - 1, col + 1
        while i>=0 and j < len(board):
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j + 1
        return True

'''
52. N-Queens II
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
Given an integer n, return the number of distinct solutions to the n-queens puzzle.
Example:
Input: n = 4
Output: 2
Explanation: There are two distinct solutions to the 4-queens puzzle as shown.
'''
class Solution(object):
    def __init__(self):
        self.result = 0
        
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        board = [['.'] * n for _ in range(n)]
        def backtracking(row):
            if row == n:
                self.result += 1
                return
            for col in range(n):
                if not self.isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(row+1)
                board[row][col] = '.'
        backtracking(0)
        return self.result
    
    def isVaild(self, board, row, col):
        for i in range(len(board)):
            if board[i][col] == 'Q':
                return False
        i, j = row - 1, col - 1
        while i>=0 and j>=0:
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j - 1
        i, j = row - 1, col + 1
        while i>=0 and j < len(board):
            if board[i][j] == 'Q':
                return False
            i, j = i - 1, j + 1
        return True

'''
53. Maximum Subarray
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
A subarray is a contiguous part of an array.
Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
'''
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        opt = [0 for _ in range(len(nums))]
        for i in range(len(nums)):
            opt[i] = max(opt[i-1]+nums[i],nums[i])
        return max(opt)

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        totalSum = 0
        maxSum = nums[0]
        for num in nums:
            totalSum += num
            maxSum = max(totalSum, maxSum)
            if totalSum<0: totalSum = 0
        return maxSum

'''
54. Spiral Matrix
Given an m x n matrix, return all elements of the matrix in spiral order.
Example:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
'''
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

'''
55. Jump Game
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
Return true if you can reach the last index, or false otherwise.
Example:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums)==1: return True
        steps = nums[0]
        for i in range(1, len(nums)-1):
            if steps == 0: return False
            steps -= 1
            steps = max(steps, nums[i])
        return steps > 0

'''
56. Merge Intervals
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
Example:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
'''
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

'''
57. Insert Interval
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti.
You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
Return intervals after the insertion.
Example:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
'''
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
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

'''
58. Length of Last Word
Given a string s consisting of words and spaces, return the length of the last word in the string.
A word is a maximal substring consisting of non-space characters only.
Example:
Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.
'''
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        return len(s.split()[-1])

'''
59. Spiral Matrix II
Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.
Example:
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
'''
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        loop = n // 2
        mid = n // 2
        number = 1
        h1, h2, v1, v2 = 0, n-1, 0, n-1
        for _ in range(loop):
            for i in range(h1,h2):
                matrix[v1][i] = number
                number += 1
            for i in range(v1,v2):
                matrix[i][h2] = number
                number += 1
            for i in range(h2,h1,-1):
                matrix[v2][i] = number
                number += 1
            for i in range(v2,v1,-1):
                matrix[i][h1] = number
                number += 1
            h1, h2, v1, v2 = h1+1, h2-1, v1+1, v2-1
        
        if n % 2 != 0: matrix[mid][mid] = number
        return matrix

'''
61. Rotate List
Given the head of a linked list, rotate the list to the right by k places.
Example:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
'''

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        length = 0
        cur, prev = head, None
        while cur:
            prev, cur = cur, cur.next
            length += 1
        if length==0 or k%length==0:
            return head
        k = k % length
        tail = head
        for _ in range(length-k-1):
            tail = tail.next
        result, tail.next, prev.next = tail.next, None, head
        return result

'''
62. Unique Paths
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). 
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.
Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
The test cases are generated so that the answer will be less than or equal to 2 * 10^9.
Example:
Input: m = 3, n = 7
Output: 28
'''
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        opt = [1 for _ in range(n)]
        for i in range(1,m):
            for j in range(1,n):
                opt[j] = opt[j] + opt[j-1]
        return opt[n-1]

'''
63. Unique Paths II
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and space is marked as 1 and 0 respectively in the grid.
Example:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
'''
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if obstacleGrid[0][0] == 1: 
            return 0
        opt = []
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        opt = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            if obstacleGrid[0][i] == 0:
                opt[0][i] = 1
            else:break
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                opt[i][0] = 1
            else:break
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    opt[i][j] = opt[i-1][j] + opt[i][j-1]
                else: opt[i][j] = 0
        return opt[m-1][n-1]

'''
64. Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.
Example:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
'''
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        opt = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        opt[0][0] = grid[0][0]
        for i in range(1,len(grid)):
            opt[i][0] = opt[i-1][0] + grid[i][0]
        for j in range(1,len(grid[0])):
            opt[0][j] = opt[0][j-1] + grid[0][j]
        for i in range(1,len(grid)):
            for j in range(1, len(grid[0])):
                opt[i][j] = min(opt[i-1][j], opt[i][j-1]) + grid[i][j]
        return opt[-1][-1]



    
'''
66. Plus One
You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer.
The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
Increment the large integer by one and return the resulting array of digits.
Example:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].
'''
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        digits[-1] += 1
        for i in range(len(digits)-1, -1, -1):
            carry, digits[i] = divmod(digits[i], 10)
            if i: digits[i-1] += carry
        if carry:
            digits.insert(0,1)
        return digits

'''
69. Sqrt(x)
Given a non-negative integer x, compute and return the square root of x.
Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.
Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.
Example:
Input: x = 4
Output: 2
'''
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0: return 0
        if x < 4: return 1
        a, b = 0, x//2
        while a < b:
            mid = a + (b-a)//2
            if mid*mid == x:
                return mid
            elif mid*mid > x:
                b = mid
            else:
                a = mid + 1
        if a * a > x: 
            return a-1
        else: return a

'''
70. Climbing Stairs
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
Example:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
'''
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<1: return 0
        if n<2: return 1
        opt = [0 for _ in range(n+1)]
        opt[0],opt[1] = 1,1
        for i in range(2,n+1):
            opt[i] = opt[i-1] + opt[i-2]
        return opt[-1]

'''
71. Simplify Path
Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.
In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.
The canonical path should have the following format:
The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
Return the simplified canonical path.
Example:
Input: path = "/home/"
Output: "/home"
'''
class Solution:
    def simplifyPath(self, path: str) -> str:
        res = []
        for level in path.split('/'):
            if not level:
                continue
            if level == "..":
                if res: res.pop()
                if res: res.pop()
            elif level != ".":
                res.append('/')
                res.append(level)
        if res:
            return ''.join(res)
        return "/"

'''
72. Edit Distance
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
You have the following three operations permitted on a word:
Insert a character
Delete a character
Replace a character
Example:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
'''
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        opt = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(m+1):
            opt[i][0] = i
        for j in range(n+1):
            opt[0][j] = j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1] == word2[j-1]:
                    opt[i][j] = opt[i-1][j-1]
                else:
                    opt[i][j] = min(opt[i-1][j-1],opt[i][j-1],opt[i-1][j]) + 1
        return opt[m][n]

'''
73. Set Matrix Zeroes
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
You must do it in place.
Example:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
'''
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

'''
74. Search a 2D Matrix
You are given an m x n integer matrix matrix with the following two properties:
Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.
You must write a solution in O(log(m * n)) time complexity.
Example 1:
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
'''
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row = left = 0
        col = right = len(matrix[0])-1
        while row < len(matrix) and col < len(matrix[0]):
            if matrix[row][col] < target:
                row += 1
            else:
                while left <= right:
                    mid = left + (right-left) // 2
                    if matrix[row][mid] == target:
                        return True
                    elif matrix[row][mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return False
        return False


'''
75. Sort Colors
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
You must solve this problem without using the library's sort function.
Example:
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
'''
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        left,mid,right = 0,0,len(nums)-1
        while mid<=right:
            if nums[mid] == 0:
                nums[left], nums[mid] = nums[mid], nums[left]
                left += 1
                mid += 1
            elif nums[mid] == 2:
                nums[mid], nums[right] = nums[right], nums[mid]
                right -= 1
            else:
                mid += 1

'''
76. Minimum Window Substring
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. 
If there is no such substring, return the empty string "".
The testcases will be generated such that the answer is unique.
A substring is a contiguous sequence of characters within the string.
Example:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
'''
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t or len(s)<len(t):
            return ""
        if s == t: return s
        dic_t = {}
        for char in s: dic_t[char] = 0
        for char in t:
            if char in dic_t:
                dic_t[char] += 1
            else: return ""
        counter = len(t)
        left,right, minleft, minlength = 0, 0, 0, float('inf')
        
        while right < len(s):
            if dic_t[s[right]] > 0: 
                counter -= 1
            dic_t[s[right]] -= 1
            right+=1
            while counter == 0:
                if right-left < minlength:
                    minlength = right - left
                    minleft = left
                dic_t[s[left]] += 1
                if dic_t[s[left]] > 0:
                    counter += 1
                left += 1
        
        if minlength != float('inf'):
            return s[minleft:minleft+minlength]
        return ""

'''
77. Combinations
Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].
You may return the answer in any order.
Example:
Input: n = 4, k = 2
'''
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []
        path = []
        def backtracking(n, k, StartIndex):
            if len(path) == k:
                result.append(path[:])
                return
            for i in range(StartIndex, n-(k-len(path))+2):
                path.append(i)
                backtracking(n, k, i+1)
                path.pop()
        backtracking(n, k, 1)
        return result

'''
78. Subsets
Given an integer array nums of unique elements, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
Example:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
'''
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        results = []
        result = []
        def backtracking(startIndex):
            results.append(result[:])
            if startIndex == len(nums):
                return
            for i in range(startIndex, len(nums)):
                result.append(nums[i])
                backtracking(i+1)
                result.pop()
        backtracking(0)
        return results

'''
79. Word Search
Given an m x n grid of characters board and a string word, return true if word exists in the grid.
The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
Example:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
'''
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

'''
80. Remove Duplicates from Sorted Array II
Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. 
The relative order of the elements should be kept the same.
Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. 
It does not matter what you leave beyond the first k elements.
Return k after placing the final result in the first k slots of nums.
Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
Example:
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
'''
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        index = 0
        for num in nums:
            if index < 2 or num > nums[index-2]:
                nums[index] = num
                index += 1
        return index

'''
82. Remove Duplicates from Sorted List II
Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. 
Return the linked list sorted as well.
Example:
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]
'''
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dupset = set()
        prev = dummy = ListNode()
        prev.next = head
        while prev.next and prev.next.next:
            if prev.next.val in dupset:
                prev.next = prev.next.next
            elif prev.next.val == prev.next.next.val:
                dupset.add(prev.next.val)
                prev.next = prev.next.next
            else: prev = prev.next
        if prev.next and prev.next.val in dupset:
            prev.next = None
        return dummy.next

'''
83. Remove Duplicates from Sorted List
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
Example:
Input: head = [1,1,2]
Output: [1,2]
'''
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node = head
        while head and head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        return node

'''
84. Largest Rectangle in Histogram
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
Example:
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
'''
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.insert(0, 0)
        heights.append(0)
        result = 0
        stack = []
        for i in range(len(heights)):
            while stack and heights[i]<heights[stack[-1]]:
                mid = stack[-1]
                stack.pop()
                if stack:
                    w = i - stack[-1] - 1
                    h = heights[mid]
                    result = max(result, w * h)
            stack.append(i)
        return result

'''
86. Partition List
Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.
Example:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]
'''
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        left = slow = ListNode(next = head)
        right = dummy = ListNode()
        fast = head
        while fast:
            if fast.val < x:
                slow.next.val = fast.val
                slow = slow.next
            else:
                dummy.next = ListNode(fast.val)
                dummy = dummy.next
            fast = fast.next
        slow.next = right.next
        return left.next

'''
87. Scramble String
We can scramble a string s to get a string t using the following algorithm:
If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
Apply step 1 recursively on each of the two substrings x and y.
Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.
Example:
Input: s1 = "great", s2 = "rgeat"
Output: true
'''
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        if s1 == s2: return True
        if sorted(s1) != sorted(s2):
            return False
        n = len(s1)
        dp = [[[False] * (n+1) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dp[i][j][1] = (s1[i] == s2[j])
        for length in range(2, n+1):
            for i in range(n-length+1):
                for j in range(n-length+1):
                    for k in range(1, length):
                        if (dp[i][j][k] and dp[i+k][j+k][length-k]) or (dp[i][j+length-k][k] and dp[i+k][j][length-k]):
                            dp[i][j][length] = True
                            break
        return dp[0][0][n]

'''
88. Merge Sorted Array
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
Merge nums1 and nums2 into a single array sorted in non-decreasing order.
The final sorted array should not be returned by the function, but instead be stored inside the array nums1.
To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
Example :
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
'''
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while m>0 and n>0:
            if nums1[m-1] < nums2[n-1]:
                nums1[n+m-1] = nums2[n-1]
                n -= 1
            else:
                nums1[n+m-1] = nums1[m-1]
                m -= 1
        while n>0:
            nums1[n+m-1] = nums2[n-1]
            n -= 1

'''
90. Subsets II
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
Example:
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
'''
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        results = []
        result = []
        used = [0 for i in range(len(nums))]
        def backtracking(startIndex):
            results.append(result[:])
            if startIndex == len(nums):
                return
            for i in range(startIndex, len(nums)):
                if i>0 and nums[i]==nums[i-1] and used[i-1]==0:
                    continue
                result.append(nums[i])
                used[i]=1
                backtracking(i+1)
                result.pop()
                used[i]=0
        backtracking(0)
        return results

'''
91. Decode Ways
A message containing letters from A-Z can be encoded into numbers using the following mapping:
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".
Given a string s containing only digits, return the number of ways to decode it.
The test cases are generated so that the answer fits in a 32-bit integer.
Example:
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
'''
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        opt = [0 for _ in range(len(s) + 1)]
        opt[0] = 1
        for i in range(1, len(s)+1):
            if s[i-1] != "0":
                opt[i] = opt[i-1]
            if i > 1:
                num = int(s[i-2: i])
                if 10 <= num <= 26:
                    opt[i] += opt[i-2]
        return opt[-1]
    
'''
92. Reverse Linked List II
Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.
Example:
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
'''
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy = p = ListNode()
        dummy.next = head
        for _ in range(left-1): 
            p = p.next
        tail = p.next

        for _ in range(right-left):
            tmp = p.next
            p.next = tail.next
            tail.next = tail.next.next
            p.next.next = tmp

        return dummy.next

'''
93. Restore IP Addresses
A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.
For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.
Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s.
You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.
Example:
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
'''
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if len(s) > 12: return []
        results = []
        def backtracking(s, start_index, point_num):
            if point_num == 3:
                if self.is_valid(s, start_index, len(s)-1):
                    results.append(s[:])
                return
            for i in range(start_index, len(s)):
                if self.is_valid(s, start_index, i):
                    s = s[:i+1] + '.' + s[i+1:]
                    backtracking(s, i+2, point_num+1)
                    s = s[:i+1] + s[i+2:]
                else:
                    break
        backtracking(s, 0, 0)
        return results
    
    def is_valid(self, s, start, end):
        if start > end: 
            return False
        if s[start] == '0' and start != end:
            return False
        if not 0 <= int(s[start:end+1]) <= 255:
            return False
        return True

'''
94. Binary Tree Inorder Traversal
Given the root of a binary tree, return the inorder traversal of its nodes' values.
Example:
Input: root = [1,null,2,3]
Output: [1,3,2]
'''
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stack = []
        result = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                result.append(cur.val)
                cur = cur.right	
        return result

'''
96. Unique Binary Search Trees
Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.
Example:
Input: n = 3
Output: 5
'''
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        opt = [0 for _ in range(n+1)]
        opt[0] = 1
        for i in range(1,n+1):
            for j in range(1,n+1):
                opt[i] += opt[j-1] * opt[i-j] 
        return opt[-1]

'''
98. Validate Binary Search Tree
Given the root of a binary tree, determine if it is a valid binary search tree (BST).
A valid BST is defined as follows:
The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example:
Input: root = [2,1,3]
Output: true
'''
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.isValid(root, float('-inf'), float('inf'))

    def isValid(self, root, minVal, maxVal):
        if not root: return True
        if root.val <= minVal or root.val >= maxVal:
            return False
        return self.isValid(root.left, minVal, root.val) and self.isValid(root.right, root.val, maxVal)

'''
100. Same Tree
Given the roots of two binary trees p and q, write a function to check if they are the same or not.
Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
Example:
Input: p = [1,2,3], q = [1,2,3]
Output: true
'''
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False

'''
101. Symmetric Tree
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
Example:
Input: root = [1,2,2,3,4,4,3]
Output: true
'''
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        queue = collections.deque()
        queue.append(root.left)
        queue.append(root.right)
        while queue:
            leftnode = queue.popleft()
            rightnode = queue.popleft()
            if not leftnode and not rightnode:
                return True
            if not leftnode or not rightnode or leftnode.val != rightnode.val:
                return False
            queue.append(leftnode.left)
            queue.append(rightnode.right)
            queue.append(leftnode.right)
            queue.append(rightnode.left)
        return True

'''
102. Binary Tree Level Order Traversal
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
Example:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
'''
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        results = []
        queue = deque()
        queue.append(root)
        while queue:
            nums = len(queue)
            result = []
            for _ in range(nums):
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            results.append(result)
        return results

'''
103. Binary Tree Zigzag Level Order Traversal
Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).
Example:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]
'''
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        results = []
        direction = 1
        queue = deque()
        queue.append(root)
        while queue:
            nums = len(queue)
            result = []
            for _ in range(nums):
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            results.append(result[::direction])
            direction *= -1
        return results
    
'''
104. Maximum Depth of Binary Tree
Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
Example:
Input: root = [3,9,20,null,null,15,7]
Output: 3
'''
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        queue = collections.deque()
        queue.append(root)
        depth = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            depth += 1
        return depth

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: 
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

'''
105. Construct Binary Tree from Preorder and Inorder Traversal
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, 
construct and return the binary tree.
Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
'''
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder:
            return None
        root_value = preorder[0]
        root = TreeNode(root_value)
        pos = inorder.index(root_value)
        root.left = self.buildTree(preorder[1:pos+1], inorder[:pos])
        root.right = self.buildTree(preorder[pos+1:], inorder[pos+1:])
        return root

'''
106. Construct Binary Tree from Inorder and Postorder Traversal
Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, 
construct and return the binary tree.
Example:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]
'''
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder:
            return None
        rootValue = postorder[-1]
        root = TreeNode(rootValue)
        pos = inorder.index(rootValue)
        root.left = self.buildTree(inorder[:pos], postorder[:pos])
        root.right = self.buildTree(inorder[pos+1:], postorder[pos:len(postorder)-1])
        return root

'''
107. Binary Tree Level Order Traversal II
Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).
Example:
Input: root = [3,9,20,null,null,15,7]
Output: [[15,7],[9,20],[3]]
'''
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        results = []
        queue = deque()
        queue.append(root)
        while queue:
            nums = len(queue)
            result = []
            for _ in range(nums):
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            results.append(result)
        return results[::-1]

'''
108. Convert Sorted Array to Binary Search Tree
Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.
A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.
Example:
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
'''
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        mid = len(nums)//2
        left = nums[:mid]
        right = nums[mid+1:]
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

'''
109. Convert Sorted List to Binary Search Tree
Given the head of a singly linked list where elements are sorted in ascending order, convert it to a 
height-balanced binary search tree.
Input: head = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
'''
class Solution:
    def __init__(self):
        self.cur = None

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        size = 0
        self.cur = head
        while head:
            size += 1
            head = head.next
        return self.convertToBst(0, size-1)
    
    def convertToBst(self,start,end):
        if start>end:return None
        mid = (start+end)//2
        left = self.convertToBst(start, mid-1)
        root = TreeNode(self.cur.val)
        self.cur = self.cur.next
        root.left = left
        root.right = self.convertToBst(mid+1, end)
        return root

'''
110. Balanced Binary Tree
Given a binary tree, determine if it is height-balanced.
For this problem, a height-balanced binary tree is defined as:
a binary tree in which the left and right subtrees of every node differ in height by no more than 1.
Example:
Input: root = [3,9,20,null,null,15,7]
Output: true
'''
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.isBls(root) >= 0

    def isBls(self, root):
        if not root: return 0
        left = self.isBls(root.left)
        right = self.isBls(root.right)
        #print(left, right)
        if left==-1 or right==-1 or abs(right-left) > 1:
            print(left, right)
            return -1
        return 1 + max(left, right)

'''
111. Minimum Depth of Binary Tree
Given a binary tree, find its minimum depth.
The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
Note: A leaf is a node with no children.
Example:
Input: root = [3,9,20,null,null,15,7]
Output: 2
'''
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        result = 1
        queue = deque()
        queue.append(root)
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                if not node.left and not node.right:
                    return result
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result += 1
        return result

'''
112. Path Sum
Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.
A leaf is a node with no children.
Example:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.
faster than 98.65% of Python online submissions for Subtree of Another Tree.
'''
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        elif not root.left and not root.right and root.val==targetSum:
            return True
        else:
            return (self.hasPathSum(root.left, targetSum-root.val) or 
                    self.hasPathSum(root.right, targetSum-root.val))

'''
113. Path Sum II
Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.
A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.
Example:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
'''
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        self.results = []
        self.result = []
        self.traverse(root, targetSum)
        return self.results
    
    def traverse(self, root, targetSum):
        if not root: return
        self.result.append(root.val)
        targetSum -= root.val
        if targetSum == 0 and not root.left and not root.right:
            self.results.append(self.result[:])
        self.traverse(root.left, targetSum)
        self.traverse(root.right, targetSum)
        self.result.pop()
    
'''
115. Distinct Subsequences
Given two strings s and t, return the number of distinct subsequences of s which equals t.
A string's subsequence is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. 
(i.e., "ACE" is a subsequence of "ABCDE" while "AEC" is not).
The test cases are generated so that the answer fits on a 32-bit signed integer.
Example:
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.
rabbbit
rabbbit
rabbbit
'''
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        if len(s) == 0 or len(t) == 0:
            return max(len(s), len(t))
        opt = [[0 for _ in range(len(t)+1)] for _ in range(len(s)+1)]
        for i in range(len(s)+1):
            opt[i][0] = 1
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    opt[i][j] = opt[i-1][j-1] + opt[i-1][j]
                else:
                    opt[i][j] = opt[i-1][j]
        return opt[-1][-1]

'''
116. Populating Next Right Pointers in Each Node
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
Initially, all next pointers are set to NULL.
Example:
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
'''
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
from collections import deque
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        # if not root:
        #     return None
        # queue = deque()
        # queue.append(root)
        # while queue:
        #     n = len(queue)
        #     for i in range(n):
        #         node = queue.popleft()
        #         if node.left:
        #             queue.append(node.left)
        #         if node.right:
        #             queue.append(node.right)
        #         if i == n - 1:
        #             break
        #         node.next = queue[0]
        # return root
        
        def dfs(node, nxt):
            if not node:
                return None
            node.next = nxt
            dfs(node.left, node.right)
            dfs(node.right, nxt.left if nxt else None)
            
        dfs(root,None)
        return root

'''
117. Populating Next Right Pointers in Each Node II
Given a binary tree
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
Initially, all next pointers are set to NULL.
Example:
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
'''
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None
        stack = [root]
        while stack:
            nextstack = []
            for i in range(len(stack)):
                if i < len(stack) - 1:
                    stack[i].next = stack[i + 1]
                else:
                    stack[i].next = None
                
                if stack[i].left:
                    nextstack.append(stack[i].left)
                if stack[i].right:
                    nextstack.append(stack[i].right)
            stack = nextstack
        return root

'''
118. Pascal's Triangle
Given an integer numRows, return the first numRows of Pascal's triangle.
In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:
Example:
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
'''
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        results = [[1]]
        for i in range(numRows-1):
            result = [1]
            for j in range(1,len(results[-1])):
                result.append(results[-1][j]+results[-1][j-1])
            result.append(1)
            results.append(result)
        return results

'''
119. Pascal's Triangle II
Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.
In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:
Example:
Input: rowIndex = 3
Output: [1,3,3,1]
'''
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if not rowIndex: return [1]
        results = [[1]]
        for i in range(rowIndex):
            result = [1]
            for j in range(1,len(results)):
                result.append(results[j]+results[j-1])
            result.append(1)
            results=result
        return results

'''
120. Triangle
Given a triangle array, return the minimum path sum from top to bottom.
For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.
Example:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
'''
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        opt = [0 for _ in range(n+1)]
        for level in range(n-1, -1, -1):
            for i in range(level+1):
                opt[i] = triangle[level][i] + min(opt[i], opt[i+1])
        return opt[0]

'''
121. Best Time to Buy and Sell Stock
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
Example:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = prices[0]
        res = 0
        for i in range(1, len(prices)):
            if prices[i] < minprice:
                minprice = prices[i]
            res = max(prices[i]-minprice, res)
        return res

'''
122. Best Time to Buy and Sell Stock II
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.
Example:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        opt = 0
        for i in range(1, len(prices)):
            opt = max(prices[i] - prices[i-1], 0) + opt
        return opt

'''
123. Best Time to Buy and Sell Stock III
You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete at most two transactions.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Example:
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        opt = [[0 for _ in range(len(prices))] for _ in range(4)]
        opt[0][0], opt[2][0] = -prices[0], -prices[0]
        for i in range(1, len(prices)):
            opt[0][i] = max(opt[0][i-1], 0-prices[i])
            opt[1][i] = max(opt[1][i-1], opt[0][i-1]+prices[i])
            opt[2][i] = max(opt[2][i-1], opt[1][i-1]-prices[i], )
            opt[3][i] = max(opt[3][i-1], opt[2][i-1]+prices[i])
        return opt[3][-1]
    
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        opt = [0]*4
        opt[0], opt[2] = -prices[0], -prices[0]
        for i in range(1, len(prices)):
            opt[0] = max(opt[0], 0-prices[i])
            opt[1] = max(opt[1], opt[0]+prices[i])
            opt[2] = max(opt[2], opt[1]-prices[i], )
            opt[3] = max(opt[3], opt[2]+prices[i])
        return opt[3]

'''
124. Binary Tree Maximum Path Sum
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
The path sum of a path is the sum of the node's values in the path.
Given the root of a binary tree, return the maximum path sum of any non-empty path.
Example:
Input: root = [1,2,3]
Output: 6
'''
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


'''
125. Valid Palindrome
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
Given a string s, return true if it is a palindrome, or false otherwise.
Example:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
'''
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = ''.join(filter(str.isalnum, str(s).lower()))
        for i in range(len(s)//2):
            if s[i] != s[len(s)-i-1]:
                return False
        return True

'''
127. Word Ladder
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
Every adjacent pair of words differs by a single letter.
Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
sk == endWord
Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
Example:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
'''
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        wordSet = set(wordList)
        if len(wordSet)== 0:
            return 0
        if endWord not in wordSet:
            return 0
        visitmap = {beginWord:1}
        queue = deque([beginWord]) 
        while queue:
            word = queue.popleft()
            path = visitmap[word]
            for i in range(len(word)):
                word_list = list(word)
                for j in range(26):
                    word_list[i] = chr(ord('a')+j)
                    newWord = "".join(word_list)
                    if newWord == endWord:
                        return path+1
                    if newWord in wordSet and newWord not in visitmap:
                        visitmap[newWord] = path+1
                        queue.append(newWord) 
        return 0

'''
128. Longest Consecutive Sequence
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
You must write an algorithm that runs in O(n) time.
Example:
Input: nums = [100,4,200,1,3,2]
Output: 4
'''
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
        if x != self.parents.get(x, x):
            self.parents[x] = self.find(self.parents.get(x))
        return self.parents.get(x, x)

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

'''
129. Sum Root to Leaf Numbers
You are given the root of a binary tree containing digits from 0 to 9 only.
Each root-to-leaf path in the tree represents a number.
For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.
A leaf node is a node with no children.
Example:
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
'''
class Solution:
    def __init__(self):
        self.total = 0

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        self.trvarsal(root, "")
        return self.total
    def trvarsal(self, root, path):
        if not root: return
        path+=str(root.val)
        if not root.left and not root.right:
            self.total += int(path)
        if root.left:
            self.trvarsal(root.left, path)
        if root.right:
            self.trvarsal(root.right, path)  

'''
131. Palindrome Partitioning
Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
A palindrome string is a string that reads the same backward as forward.
Example:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
'''
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        results = []
        result = []
        def backtracking(string):
            if string == '':
                results.append(result[:])
            for i in range(len(string)):
                temp = string[:i+1]
                if temp == temp[::-1]:
                    result.append(temp)
                    backtracking(string[i+1:])
                    result.pop()
        backtracking(s)
        return results

'''
132. Palindrome Partitioning II
Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.
Example:
Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
'''
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        opt = [[0 for _ in range(len(s))] for _ in range(len(s))]
        for i in range(len(s)-1,-1,-1):
            for j in range(i, len(s)):
                if s[i]!=s[j]:
                    opt[i][j] = 0
                elif j-i<=1 or opt[i+1][j-1]:
                    opt[i][j] = 1
        
        dp = [len(s)] * len(s)
        dp[0] = 0
        for i in range(1, len(s)):
            if opt[0][i]:
                dp[i]=0
            for j in range(0,i):
                if opt[j+1][i]:
                    dp[i]=min(dp[i], dp[j]+1)
        return dp[-1]

'''
133. Clone Graph
Given a reference of a node in a connected undirected graph.
Return a deep copy (clone) of the graph.
Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
class Node {
    public int val;
    public List<Node> neighbors;
}
Test case format:
For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.
An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.
The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.
Example:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
'''
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

'''
134. Gas Station
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station.
You begin the journey with an empty tank at one of the gas stations.
Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.
If there exists a solution, it is guaranteed to be unique
Example:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
'''
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        startInd = 0
        curSum, totalSum = 0, 0
        for i in range(len(gas)):
            curSum += gas[i] - cost[i]
            totalSum += gas[i] - cost[i]
            if curSum < 0:
                startInd = i +1
                curSum = 0
        if totalSum < 0:
            return -1
        return startInd

'''
135. Candy
There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.
You are giving candies to these children subjected to the following requirements:
Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.
Example:
Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
'''
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        result = [1 for _ in range(len(ratings))]
        for i in range(len(ratings)-1):
            if ratings[i] < ratings[i+1]:
                result[i+1] = result[i]+1
        for i in range(len(ratings)-1, 0, -1):
            if ratings[i-1] > ratings[i]:
                result[i-1] = max(result[i]+1,result[i-1])
        return sum(result)

'''
136. Single Number
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
You must implement a solution with a linear runtime complexity and use only constant extra space.
Example:
Input: nums = [2,2,1]
Output: 1
'''
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = nums[0]
        for x in range(1,len(nums)):
            result^=nums[x] 
        return result

'''
137. Single Number II
Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it.
You must implement a solution with a linear runtime complexity and use only constant extra space.
Example:
Input: nums = [2,2,3,2]
Output: 3
'''
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for i in range(32):
            count = 0
            for num in nums:
                count += (num >> i) & 1
            result |= (count % 3 != 0) << i
        if result >= 2**31:
            result -= 2**32
        return result

'''
138. Copy List with Random Pointer
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state.
None of the pointers in the new list should point to nodes in the original list.
For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.
Return the head of the copied linked list.
The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:
val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.
Example:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
'''
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        dic = {}
        m = n = head
        while m:
            dic[m] = Node(m.val)
            m = m.next
        while n:
            dic[n].next = dic.get(n.next)
            dic[n].random = dic.get(n.random)
            n = n.next
        return dic.get(head)

'''
139. Word Break
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
Note that the same word in the dictionary may be reused multiple times in the segmentation.
Example:
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
'''
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        opt = [0 for _ in range((len(s) + 1))]
        opt[0] = 1
        for j in range(1, len(s) + 1):
            for word in wordDict:
                if j >= len(word):
                    opt[j] = opt[j] or (opt[j - len(word)] and word == s[j - len(word):j])
        return opt[len(s)]

'''
140. Word Break II
Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
Note that the same word in the dictionary may be reused multiple times in the segmentation.
Example:
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
'''
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        result = []
        def backtracking(s: str, wordset: Set[str], startIndex: int, cur: str) -> None:
            if startIndex == len(s):
                result.append(cur[:-1])
            for i in range(startIndex, len(s)):
                substring = s[startIndex:i+1]
                if substring in wordset:
                    backtracking(s, wordset, i+1, cur + substring + " ")
        wordset = set(wordDict)
        backtracking(s, wordset, 0, "")
        return result


'''
0141. Linked List Cycle
Given head, the head of a linked list, determine if the linked list has a cycle in it.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer.
Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
Return true if there is a cycle in the linked list. Otherwise, return false.
Example:
Input: head = [3,2,0,-4], pos = 1
Output: true
'''
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None or head.next == None or head.next.next == None:
            return False
        slow, fast = head.next, head.next.next
        while slow != fast:
            if fast.next is None or fast.next.next is None:
                return False
            slow, fast = slow.next, fast.next.next
        return True

'''
142. Linked List Cycle II
Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. 
Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. 
Note that pos is not passed as a parameter.
Example:
Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
'''
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slowNode = fastNode = head
        while fastNode and fastNode.next:
            slowNode = slowNode.next
            fastNode = fastNode.next.next
            if slowNode == fastNode:
                slowNode = head
                while slowNode != fastNode:
                    slowNode = slowNode.next
                    fastNode = fastNode.next
                return slowNode
        return None

'''
143. Reorder List
You are given the head of a singly linked-list. The list can be represented as:
L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.
Example:
Input: head = [1,2,3,4]
Output: [1,4,2,3]
'''
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        lst = []
        cur = head
        while cur is not None:
            lst.append(cur.val)
            cur = cur.next
        cur = head
        for i in range(len(lst)//2):
            cur.val, cur.next.val = lst[i], lst[len(lst)-1-i]
            if cur.next is not None and cur.next.next is not None:
                cur = cur.next.next
        if len(lst)%2 == 1:
            cur.val = lst[len(lst)//2]

'''
144. Binary Tree Preorder Traversal
Given the root of a binary tree, return the preorder traversal of its nodes' values.
Example:
Input: root = [1,null,2,3]
Output: [1,2,3]
'''
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stack = []
        result = []
        stack.append(root)
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result

'''
145. Binary Tree Postorder Traversal
Given the root of a binary tree, return the postorder traversal of its nodes' values.
Example:
Input: root = [1,null,2,3]
Output: [3,2,1]
'''
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stack = []
        result = []
        stack.append(root)
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return result[::-1] 

'''
148. Sort List
Given the head of a linked list, return the list after sorting it in ascending order.
Example:
Input: head = [4,2,1,3]
Output: [1,2,3,4]
'''
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def cut_list(head, size):
            for i in range(size-1):
                if not head: break
                head = head.next
            if not head: return None
            next_head, head.next = head.next, None  # cut
            return next_head
        def merge_two(l1, l2, pre):
            while l1 and l2:
                if l1.val <= l2.val:
                    pre.next, l1 = l1, l1.next
                else:
                    pre.next, l2 = l2, l2.next
                pre = pre.next
            pre.next = l1 or l2
            while pre.next: pre = pre.next
            return pre
        
        dummy = ListNode(next = head)
        h = head
        length, d = 0, 1
        while h:
            h = h.next
            length += 1
        while d < length:
            pre, h = dummy, dummy.next
            while h:
                left = h
                right = cut_list(left, d)
                h = cut_list(right, d)
                pre = merge_two(left, right, pre)
            d *= 2
        return dummy.next

'''
150. Evaluate Reverse Polish Notation
Evaluate the value of an arithmetic expression in Reverse Polish Notation.
Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
Note that division between two integers should truncate toward zero.
It is guaranteed that the given RPN expression is always valid. 
That means the expression would always evaluate to a result, and there will not be any division by zero operation.
Example:
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
'''
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for element in tokens:
            if not stack:
                stack.append(element)
                continue
            if element in {'+', '-', '*', '/'}:
                right = int(stack.pop())
                left = int(stack.pop())
                if element == '+':stack.append(left+right)
                elif element == '-':stack.append(left-right)
                elif element == '*':stack.append(left*right)
                elif element == '/':stack.append(int(float(left) / right))
            else: stack.append(element)
        return stack.pop()

'''
151. Reverse Words in a String
Given an input string s, reverse the order of the words.
A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.
Return a string of the words in reverse order concatenated by a single space.
Note that s may contain leading or trailing spaces or multiple spaces between two words. 
The returned string should only have a single space separating the words. Do not include any extra spaces.
Example:
Input: s = "the sky is blue"
Output: "blue is sky the"
'''
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = s.split()
        length= len(s)
        for i in range (length//2):
            s[i], s[length-i-1] = s[length-i-1], s[i]
        return ' '.join(s)

'''
152. Maximum Product Subarray
Given an integer array nums, find a 
subarray that has the largest product, and return the product.
The test cases are generated so that the answer will fit in a 32-bit integer.
Example:
Input: nums = [2,3,-2,4]
Output: 6
'''
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

'''
153. Find Minimum in Rotated Sorted Array
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:
[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
Given the sorted rotated array nums of unique elements, return the minimum element of this array.
You must write an algorithm that runs in O(log n) time.
Example:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
'''
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
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

'''
154. Find Minimum in Rotated Sorted Array II
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,4,4,5,6,7] might become:
[4,5,6,7,0,1,4] if it was rotated 4 times.
[0,1,4,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
Given the sorted rotated array nums that may contain duplicates, return the minimum element of this array.
You must decrease the overall operation steps as much as possible.
Example:
Input: nums = [1,3,5]
Output: 1
'''
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left,right = 0,len(nums)-1
        while left < right:
            if nums[left] < nums[right]:
                return nums[left]
            mid = (left+right)//2
            if nums[right] < nums[mid]:
                left = mid + 1
            elif nums[right] > nums[mid]:
                right = mid
            else: right -= 1
        return nums[left]

'''
155. Min Stack
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
Implement the MinStack class:
MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.
Example:
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
Output
[null,null,null,null,-3,null,0,-2]
'''
class MinStack(object):

    def __init__(self):
        self.stack = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        if len(self.stack) == 0:
            self.stack.append((val, val))
            return
        min_num = self.stack[-1][1]
        if val < min_num:
            self.stack.append((val, val))
        else:
            self.stack.append((val, min_num))
        return
        
    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        
    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]
        
    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]
    
'''
160. Intersection of Two Linked Lists
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.
For example, the following two linked lists begin to intersect at node c1:
The test cases are generated such that there are no cycles anywhere in the entire linked structure.
Note that the linked lists must retain their original structure after the function returns.
Custom Judge:
The inputs to the judge are given as follows (your program is not given these inputs):
intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
listA - The first linked list.
listB - The second linked list.
skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.
The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. 
If you correctly return the intersected node, then your solution will be accepted.
Example:
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
'''
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        
        nodeA,nodeB = headA,headB
        while nodeA != nodeB:
            nodeA = nodeA.next if nodeA else headB
            nodeB = nodeB.next if nodeB else headA
        return nodeA

'''
164. Maximum Gap
Given an integer array nums, return the maximum difference between two successive elements in its sorted form. If the array contains less than two elements, return 0.
You must write an algorithm that runs in linear time and uses linear extra space.
Example:
Input: nums = [3,6,9,1]
Output: 3
'''
class Solution(object):
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2: return 0
        minv, maxv = min(nums), max(nums)
        diffbucket = max(1,(maxv-minv)//(len(nums)-1))
        numbucket = (maxv-minv)//diffbucket + 1
        used = [False for _ in range(numbucket)]
        buckets = [[maxv, minv] for _ in range(numbucket)]
        for num in nums:
            idx = (num-minv)/diffbucket
            used[idx] = True
            buckets[idx][0] = min(num, buckets[idx][0])
            buckets[idx][1] = max(num, buckets[idx][1])
        res = 0
        prevmin = minv
        for i,bucket in enumerate(buckets):
            if used[i]:
                res = max(res, bucket[0]-prevmin)
                prevmin = bucket[1]
        return res

'''
165. Compare Version Numbers
Given two version numbers, version1 and version2, compare them.
Version numbers consist of one or more revisions joined by a dot '.'. Each revision consists of digits and may contain leading zeros. 
Every revision contains at least one character. Revisions are 0-indexed from left to right, with the leftmost revision being revision 0, the next revision being revision 1, and so on. 
For example 2.5.33 and 0.1 are valid version numbers.
To compare version numbers, compare their revisions in left-to-right order. Revisions are compared using their integer value ignoring any leading zeros. 
This means that revisions 1 and 001 are considered equal. If a version number does not specify a revision at an index, then treat the revision as 0. 
For example, version 1.0 is less than version 1.1 because their revision 0s are the same, but their revision 1s are 0 and 1 respectively, and 0 < 1.
Return the following:
If version1 < version2, return -1.
If version1 > version2, return 1.
Otherwise, return 0.
Example:
Input: version1 = "1.01", version2 = "1.001"
Output: 0
'''
class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        version1, version2 = map(int, version1.split('.')), map(int, version2.split('.'))
        for v1,v2 in itertools.izip_longest(version1, version2, fillvalue=0):
            if v1>v2:return 1
            elif v1<v2:return -1
        return 0

'''
169. Majority Element
Given an array nums of size n, return the majority element.
The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
Example:
Input: nums = [3,2,3]
Output: 3
'''
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # Boyer-Moore Voting Algorithm
        count = 0
        candidate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        return candidate
    
        # another method
        if len(nums) == 1: return nums[0]
        dic = {}
        throuhoud = len(nums)//2
        for num in nums:
            if num in dic:
                dic[num] += 1
                if dic[num]>throuhoud:
                    return num
            else: dic[num] = 1

'''
172. Factorial Trailing Zeroes
Given an integer n, return the number of trailing zeroes in n!.
Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.
Example:
Input: n = 3
Output: 0
'''
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        return 0 if n == 0 else n / 5 + self.trailingZeroes(n / 5)

'''
174. Dungeon Game
The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. 
The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight was initially positioned in the top-left room and must fight his way through dungeon to rescue the princess.
The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).
To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
Return the knight's minimum initial health so that he can rescue the princess.
Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.
Example:
Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
Output: 7
'''
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        row, col = len(dungeon), len(dungeon[0])
        opt = [[0 for _ in range(col)] for _ in range(row)]
        for i in range(row-1, -1, -1):
            for j in range(col-1, -1, -1):
                if i==row-1 and j==col-1:
                    opt[i][j] = max(1, 1-dungeon[i][j])
                elif i==row-1:
                    opt[i][j] = max(1, opt[i][j+1]-dungeon[i][j])
                elif j==col-1:
                    opt[i][j] = max(1, opt[i+1][j]-dungeon[i][j])
                else:
                    opt[i][j] = max(1, min(opt[i][j+1],opt[i+1][j])-dungeon[i][j])
        return opt[0][0]

'''
179. Largest Number
Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.
Since the result may be very large, so you need to return a string instead of an integer.
Example:
Input: nums = [10,2]
Output: "210"
'''
class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if not nums:
            return "0"
        heap = []
        for i in nums:
            heapq.heappush(heap,key(str(i)))
        res = ""
        while heap:
            res += heapq.heappop(heap).num
        return res
class key:
    def __init__(self,num):
        self.num = num
    def __lt__(self,other):
        return self.num+other.num > other.num+self.num
            
'''
187. Repeated DNA Sequences
The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.
For example, "ACGAATTCCG" is a DNA sequence.
When studying DNA, it is useful to identify repeated sequences within the DNA.
Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.
Example:
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]
'''
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        subseq, repeated = set(),set()
        for i in range(len(s)-9):
            substr = s[i:i+10]
            if substr in subseq:
                repeated.add(substr)
            else:
                subseq.add(substr)
        return repeated

'''
188. Best Time to Buy and Sell Stock IV
You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
Find the maximum profit you can achieve. You may complete at most k transactions.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Example:
Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
'''
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        opt = [0 for _ in range(k*2+1)]
        for i in range(1, 2*k, 2):
            opt[i] = -prices[0]
        for i in range(1,len(prices)):
            for j in range(1, k*2, 2):
                opt[j] = max(opt[j], opt[j-1]-prices[i])
                opt[j+1] = max(opt[j+1], opt[j]+prices[i])
        return opt[k*2]

'''
189. Rotate Array
Given an array, rotate the array to the right by k steps, where k is non-negative.
Example:
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
'''
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        res = nums[-k:]+nums[:-k]
        for i in range(len(nums)):
            nums[i] = res[i]

'''
190. Reverse Bits
Reverse bits of a given 32 bits unsigned integer.
Note:
Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.
Example:
Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
'''
class Solution:
    def reverseBits(self, n):
        return int(str(bin(n))[2:][::-1].ljust(32, '0'),2)

'''
191. Number of 1 Bits
Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).
Note:
Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.
Example:
Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
'''
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n:
            n &= (n-1)
            count += 1
        return count

'''
198. House Robber
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, 
the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police 
if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
Example:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
'''
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        opt = [0 for _ in range(len(nums))]
        opt[0] = nums[0]
        opt[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            opt[i] = max(opt[i-2] + nums[i], opt[i-1])
        return opt[-1]

'''
199. Binary Tree Right Side View
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
Example :
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]
'''
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        result = []
        layer = deque()
        layer.append(root)
        while layer:
            result.append(layer[-1].val)
            for i in range(len(layer)):
                cur = layer.popleft()
                if cur.left:
                    layer.append(cur.left)
                if cur.right:
                    layer.append(cur.right)
        return result

'''
200. Number of Islands
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
Example:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
'''
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if grid[i][j] == '0': return
        grid[i][j] = '0'
        if i > 0: 
            self.dfs(grid, i-1, j)
        if i < len(grid) - 1: 
            self.dfs(grid, i+1, j)
        if j > 0:
            self.dfs(grid, i, j-1)
        if j < len(grid[0]) - 1: 
            self.dfs(grid, i, j+1)

'''
0202. Happy Number
Write an algorithm to determine if a number n is happy.
A happy number is a number defined by the following process:
Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.
Example:
Input: n = 19
Output: true
'''
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

'''
203. Remove Linked List Elements
Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.
Example:
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummyNode = ListNode()
        dummyNode.next = head
        if head is not None:
            if head.val == val:
                dummyNode.next = head.next
        else:
            return head
        
        prev = dummyNode
        current = dummyNode.next 
        while current is not None:
            if current.val == val:
                current = current.next
                prev.next = current
            else:
                prev, current = current, current.next
        return dummyNode.next

'''
205. Isomorphic Strings
Given two strings s and t, determine if they are isomorphic.
Two strings s and t are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with another character while preserving the order of characters.
No two characters may map to the same character, but a character may map to itself.
Example:
Input: s = "egg", t = "add"
Output: true
'''
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        dic_s = {}
        dic_t = {}
        
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

'''
206. Reverse Linked List
Given the head of a singly linked list, reverse the list, and return the reversed list.
Example:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next == None:
            return head
        prevnode = None
        currentnode = head
        nextnode = head.next
        
        while nextnode is not None:
            currentnode.next = prevnode
            prevnode,currentnode,nextnode = currentnode,nextnode,nextnode.next
        currentnode.next = prevnode
        return currentnode

'''
207. Course Schedule
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.
Example:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
'''
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        numpreq = [0] * numCourses
        result = 0
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

'''
208. Implement Trie (Prefix Tree)
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings.
There are various applications of this data structure, such as autocomplete and spellchecker.
Implement the Trie class:
Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
Example:
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]
'''
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

'''
209. Minimum Size Subarray Sum
Given an array of positive integers nums and a positive integer target, 
return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. 
If there is no such subarray, return 0 instead.
Example:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
'''
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left, result = 0, len(nums)+1
        for right in range(len(nums)):
            target -= nums[right]
            while target <= 0:
                result = min(result, right-left+1)
                target += nums[left]
                left += 1
        return result % (len(nums)+1)

'''
211. Design Add and Search Words Data Structure
Design a data structure that supports adding new words and finding if a string matches any previously added string.
Implement the WordDictionary class:
WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
Example:
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]
'''
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

'''
212. Word Search II
Given an m x n board of characters and a list of strings words, return all words on the board.
Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.
The same letter cell may not be used more than once in a word.
Example:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
'''
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

'''
213. House Robber II
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. 
That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, 
and it will automatically contact the police if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
Example:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
'''
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 2: return max(nums)
        def sub_rob(sub_nums, start, end):
            if end == start: return nums[start]
            prev, curr = 0, 0
            for i in range(start, end + 1):
                prev, curr = curr, max(prev+nums[i], curr)
            return curr
        return max(sub_rob(nums, 0, len(nums)-2), sub_rob(nums, 1, len(nums)-1))

'''
215. Kth Largest Element in an Array
Given an integer array nums and an integer k, return the kth largest element in the array.
Note that it is the kth largest element in the sorted order, not the kth distinct element.
You must solve it in O(n) time complexity.
Example:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
'''
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = nums[:k]
        heapq.heapify(heap)
        print(heap)
        for i in range(k,len(nums)):
            heapq.heappush(heap, nums[i])
            heapq.heappop(heap)
        return heapq.heappop(heap)

'''
216. Combination Sum III
Find all valid combinations of k numbers that sum up to n such that the following conditions are true:
Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.
Example:
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
'''
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        results = []
        result = []
        def backtracking(totalSum, startInd):
            if totalSum > n:
                return
            if len(result) == k:
                if totalSum == n:
                    results.append(result[:])
                return
            for i in range(startInd, 10 - (k - len(result)) + 1):
                totalSum += i
                result.append(i)
                backtracking(totalSum, i+1)
                totalSum -= i
                result.pop()
        backtracking(0, 1)
        return results

'''
217. Contains Duplicate
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
Example:
Input: nums = [1,2,3,1]
Output: true
'''
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        res = set()
        for num in nums:
            if num in res: return True
            else: res.add(num)
        return False

'''
219. Contains Duplicate II
Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
Example:
Input: nums = [1,2,3,1], k = 3
Output: true
'''
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dic = {}
        for i, num in enumerate(nums):
            if num in dic:
                if i-dic[num]<=k:
                    return True
            dic[num]=i
        return False

'''
220. Contains Duplicate III
Given an integer array nums and two integers k and t, return true if there are two distinct indices i and j in the array such that abs(nums[i] - nums[j]) <= t and abs(i - j) <= k.
Example:
Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
'''
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if k < 1 or t < 0: return False
        dic = collections.OrderedDict()
        for n in nums:
            key = n//t if t else n
            for m in (dic.get(key - 1), dic.get(key), dic.get(key + 1)):
                if m is not None and abs(n - m) <= t:
                    return True
            if len(dic) == k:
                dic.popitem(False)
            dic[key] = n
        return False


'''
222. Count Complete Tree Nodes
Given the root of a complete binary tree, return the number of the nodes in the tree.
According to Wikipedia, every level, except possibly the last, 
is completely filled in a complete binary tree, 
and all nodes in the last level are as far left as possible. 
It can have between 1 and 2h nodes inclusive at the last level h.
Design an algorithm that runs in less than O(n) time complexity.
Example:
Input: root = [1,2,3,4,5,6]
Output: 6
'''
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def trverse(node):
            if root is None:
                return 0
            leftnode = node.left
            rightnode = node.right
            if leftnode is None and rightnode is None:
                return 1
            elif rightnode is None:
                return 1 + trverse(leftnode)
            else:
                return 1 + trverse(leftnode) + trverse(rightnode)
            
        return trverse(root)

'''
225. Implement Stack using Queues
Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).
Implement the MyStack class:
void push(int x) Pushes element x to the top of the stack.
int pop() Removes the element on the top of the stack and returns it.
int top() Returns the element on the top of the stack.
boolean empty() Returns true if the stack is empty, false otherwise.
Notes:
You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
Depending on your language, the queue may not be supported natively. 
You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.
Example:
Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]
'''
class MyStack(object):

    def __init__(self):
        self.queue = deque()

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.queue.append(x)
        
    def pop(self):
        """
        :rtype: int
        """
        if self.empty():
            return None
        for i in range(len(self.queue)-1):
            self.queue.append(self.queue.popleft())
        return self.queue.popleft()
        

    def top(self):
        """
        :rtype: int
        """
        popValue = self.pop()
        self.queue.append(popValue)
        return popValue

    def empty(self):
        """
        :rtype: bool
        """
        return not self.queue

'''
226. Invert Binary Tree
Given the root of a binary tree, invert the tree, and return its root.
Example:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
'''
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        parent = TreeNode(root.val)
        parent.left, parent.right = self.invertTree(root.right), self.invertTree(root.left)
        return parent

'''
228. Summary Ranges
You are given a sorted unique integer array nums.
A range [a,b] is the set of all integers from a to b (inclusive).
Return the smallest sorted list of ranges that cover all the numbers in the array exactly. 
That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.
Each range [a,b] in the list should be output as:
"a->b" if a != b
"a" if a == b
Example:
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
'''
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        stack = []
        result = []
        for num in nums:
            if not stack or stack[-1][-1]+1<num:
                stack.append([num,num])
            else:
                stack[-1][1] = num
        for p in stack:
            if p[0]==p[1]: result.append(str(p[0]))
            else: result.append('->'.join(map(str, p)))
        return result

'''
229. Majority Element II
Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
Example:
Input: nums = [3,2,3]
Output: [3]
'''
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) == 1: return nums
        result = []
        dic = {}
        threshold = len(nums)//3
        for num in nums:
            if num in dic:
                dic[num] += 1
            else: dic[num] = 1
        for key,value in dic.items():
            if value > threshold:
                result.append(key)
        return result

'''
230. Kth Smallest Element in a BST
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
Example:
Input: root = [3,1,4,null,2], k = 1
Output: 1
'''
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

'''
232. Implement Queue using Stacks
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).
Implement the MyQueue class:
void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.
Notes:
You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively. 
You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.
Example:
Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]
'''
class MyQueue(object):

    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack1.append(x)

    def pop(self):
        """
        :rtype: int
        """
        if self.empty():
            return None
        if self.stack2:
            return self.stack2.pop()
        else:            
            for i in range(len(self.stack1)-1):
                self.stack2.append(self.stack1.pop())
            return self.stack1.pop()
        
    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        peekValue = self.pop()
        self.stack2.append(peekValue)
        return peekValue   

    def empty(self):
        """
        :rtype: bool
        """
        return not (self.stack1 or self.stack2)

'''
234. Palindrome Linked List
Given the head of a singly linked list, return true if it is a palindrome.
Example:
Input: head = [1,2,2,1]
Output: true
'''
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        lst = []
        cur = head
        while cur is not None:
            lst.append(cur.val)
            cur = cur.next
        for i in range(len(lst)//2):
            if lst[i] != lst[len(lst)-1-i]:
                return False
        return True

'''
235. Lowest Common Ancestor of a Binary Search Tree
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
According to the definition of LCA on Wikipedia: 
“The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
Example:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
'''
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
'''
236. Lowest Common Ancestor of a Binary Tree
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
According to the definition of LCA on Wikipedia: 
“The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
Example:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
'''
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root==None or root==p or root==q:
            return root
        leftnode = self.lowestCommonAncestor(root.left, p, q)
        rightnode = self.lowestCommonAncestor(root.right, p, q)
        if leftnode != None and rightnode != None:
            return root
        elif leftnode:
            return leftnode
        return rightnode

'''
237. Delete Node in a Linked List
Write a function to delete a node in a singly-linked list. 
You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.
It is guaranteed that the node to be deleted is not a tail node in the list.
Example:
Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.
'''
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

'''
238. Product of Array Except Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.
Example:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
'''
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


'''
239. Sliding Window Maximum
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
You can only see the k numbers in the window. Each time the sliding window moves right by one position.
Return the max sliding window.
Example:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
'''
class Solution(object):
    def __init__(self):
        self.window = deque()
    def push(self, number):
        while self.window and self.window[-1] < number:
            self.window.pop()
        self.window.append(number)
    def pop(self, val):
        if self.window[0] == val:
            self.window.popleft()
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        result = []
        for i in range(len(nums)):
            self.push(nums[i])
            if i < k-1: continue
            result.append(self.window[0])
            self.pop(nums[i-k+1])
        return result

'''
240. Search a 2D Matrix II
Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:
Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true
'''
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row, col = len(matrix), len(matrix[0])
        i, j = 0, col-1
        while i<row and j>=0 : 
            if matrix[i][j]==target:
                return True
            elif matrix[i][j]>target:
                j -= 1
            else:
                i += 1
        return False

'''
0242. Valid Anagram
https://leetcode.com/problems/valid-anagram/
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
Example:
Input: s = "anagram", t = "nagaram"
Output: true
'''
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

'''
257. Binary Tree Paths
Given the root of a binary tree, return all root-to-leaf paths in any order.
A leaf is a node with no children.
Example:
Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]
'''
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        stack = deque([root]) 
        path_lst = deque()
        result = []
        path_lst.append(str(root.val))
        while stack:
            current_node = stack.pop()
            path = path_lst.pop()
            if not(current_node.left or current_node.right):
                result.append(path)
            if current_node.right:
                stack.append(current_node.right)
                path_lst.append(path + '->' + str(current_node.right.val))
            if current_node.left:
                stack.append(current_node.left)
                path_lst.append(path + '->' + str(current_node.left.val))
        return result

'''
260. Single Number III
Given an integer array nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once. You can return the answer in any order.
You must write an algorithm that runs in linear runtime complexity and uses only constant extra space.
Example:
Input: nums = [1,2,1,3,2,5]
Output: [3,5]
Explanation:  [5, 3] is also a valid answer.
'''
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        xortotal = 0
        bitpos = 0
        xor1, xor2 = 0, 0
        for num in nums:
            xortotal ^= num
        while not xortotal&1:
            bitpos += 1
            xortotal >>= 1
        for num in nums:
            if num>>bitpos&1:
                xor1 ^= num
            else:
                xor2 ^= num
        return xor1, xor2

'''
268. Missing Number
Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
Example:
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
'''
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(range(len(nums)+1)) - sum(nums)

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = 0
        for i,num in enumerate(nums):
            n ^= i
            n ^= num
        n ^= len(nums)
        return n

'''
278. First Bad Version
You are a product manager and currently leading a team to develop a new product.
Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.
Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.
You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.
Example:
Input: n = 5, bad = 4
Output: 4
'''
class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 0, n
        while left < right:
            mid = left + (right - left)//2
            if isBadVersion(mid): right = mid
            else: left = mid+1
        return left

'''
279. Perfect Squares
Given an integer n, return the least number of perfect square numbers that sum to n.
A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.
Example:
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
'''
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        nums = [i**2 for i in range(1, n + 1) if i**2 <= n]
        opt = [10**4]*(n + 1)
        opt[0] = 0
        for num in nums:
            for j in range(num, n + 1):
                opt[j] = min(opt[j], opt[j - num] + 1)
        return opt[n]


'''
283. Move Zeroes
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
Note that you must do this in-place without making a copy of the array.
Example:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
'''
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        slow = 0
        fast = 0
        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1
        return nums

'''
287. Find the Duplicate Number
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
There is only one repeated number in nums, return this repeated number.
You must solve the problem without modifying the array nums and uses only constant extra space.
Example:
Input: nums = [1,3,4,2,2]
Output: 2
'''
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

'''
289. Game of Life
According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."
The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0).
Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
Any live cell with fewer than two live neighbors dies as if caused by under-population.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by over-population.
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.
Example:
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
'''

class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        m,n = len(board), len(board[0])
        tempboard = [[0 for _ in range(n)] for _ in range(m)]
        for row in range(m):
            for col in range(n):
                tempboard[row][col] = self.checkNeighbor(board, row, col)
        for row in range(m):
            for col in range(n):
                if tempboard[row][col]<2:
                    board[row][col] = 0
                elif board[row][col] and tempboard[row][col]>3:
                    board[row][col] = 0
                elif board[row][col]==0 and tempboard[row][col]==3:
                    board[row][col] = 1

    def checkNeighbor(self, board, row, col):
        '''return #live'''
        m,n = len(board), len(board[0])
        num = 0
        if row > 0:
            if board[row-1][col]: num += 1
            if col > 0 and board[row-1][col-1]: num += 1
            if col < n-1 and board[row-1][col+1]: num += 1
        if row < m-1:
            if board[row+1][col]: num += 1
            if col > 0 and board[row+1][col-1]: num += 1
            if col < n-1 and board[row+1][col+1]: num += 1
        if col > 0 and board[row][col-1]: num += 1
        if col < n-1 and board[row][col+1]: num += 1
        return num


'''
290. Word Pattern
Given a pattern and a string s, find if s follows the same pattern.
Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.
Example:
Input: pattern = "abba", s = "dog cat cat dog"
Output: true
'''
class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        from collections import OrderedDict
        d1,d2 = OrderedDict(), OrderedDict()
        for i, char in enumerate(pattern):
            d1[char] = d1.get(char, []) + [i]
        for i, word in enumerate(s.split()):
            d2[word] = d2.get(word, []) + [i]
        return list(d1.values()) == list(d2.values())

'''
291. Word Pattern II
Given a pattern and a string s, return true if s matches the pattern.
A string s matches a pattern if there is some bijective mapping of single characters to strings such that if each character in pattern is replaced by the string it maps to, then the resulting string is s. 
A bijective mapping means that no two characters map to the same string, and no character maps to two different strings.
Example:
Input: pattern = "abab", s = "redblueredblue"
Output: true
'''
class Solution(object):
    def wordPatternMatch(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        return self.dfs(pattern, s, {})

    def dfs(self, pattern, str, dict):
        if len(pattern) == len(str) == 0:
            return True
        if len(pattern) == 0 and len(str) > 0:
            return False
        
        for end in range(1, len(str)-len(pattern)+2):
            if pattern[0] not in dict and str[:end] not in dict.values():
                dict[pattern[0]] = str[:end]
                if self.dfs(pattern[1:], str[end:], dict):
                    return True
                del dict[pattern[0]]
            elif pattern[0] in dict and dict[pattern[0]] == str[:end]:
                if self.dfs(pattern[1:], str[end:], dict):
                    return True
        return False

'''
295. Find Median from Data Stream
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:
MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
Example:
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]
'''
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
    
'''
297. Serialize and Deserialize Binary Tree
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
Example:
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
'''
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        result = []
        queue = deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            if not node: 
                result.append('None')
                continue
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        return ','.join(result)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(',')
        if data[0] == 'None': return None
        root = TreeNode(int(data[0]))
        queue = deque([root])
        i = 0
        while queue:
            node = queue.popleft()
            if i + 1 < len(data) and data[i + 1] != 'None':
                node.left = TreeNode(int(data[i + 1]))
                queue.append(node.left)
            if i + 2 < len(data) and data[i + 2] != 'None':
                node.right = TreeNode(int(data[i + 2]))
                queue.append(node.right)
            i += 2
            
        return root

'''
299. Bulls and Cows
You are playing the Bulls and Cows game with your friend.
You write down a secret number and ask your friend to guess what the number is. When your friend makes a guess, you provide a hint with the following info:
The number of "bulls", which are digits in the guess that are in the correct position.
The number of "cows", which are digits in the guess that are in your secret number but are located in the wrong position. Specifically, the non-bull digits in the guess that could be rearranged such that they become bulls.
Given the secret number secret and your friend's guess guess, return the hint for your friend's guess.
The hint should be formatted as "xAyB", where x is the number of bulls and y is the number of cows. Note that both secret and guess may contain duplicate digits.
Example:
Input: secret = "1807", guess = "7810"
Output: "1A3B"
'''
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

##300##
'''
300. Longest Increasing Subsequence
Given an integer array nums, return the length of the longest strictly increasing subsequence.
A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. 
For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].
Example:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
'''
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        opt = [1 for _ in range(len(nums))]
        result = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    opt[i] = max(opt[i], opt[j]+1)
            result = max(result, opt[i])
        return result

'''
309. Best Time to Buy and Sell Stock with Cooldown
You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Example:
Input: prices = [1,2,3,0,2]
Output: 3
Explanation: transactions = [buy, sell, cooldown, buy, sell]
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n == 0:
            return 0
        dp = [[0] * 4 for _ in range(n)]
        dp[0][0] = -prices[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], max(dp[i-1][3], dp[i-1][1]) - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][3])
            dp[i][2] = dp[i-1][0] + prices[i]
            dp[i][3] = dp[i-1][2]
        return max(dp[n-1][3], dp[n-1][1], dp[n-1][2])

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices: return 0
        opt = [0] * 4
        opt[0] = -prices[0]
        for i in range(1, len(prices)):
            temp0, temp2 = opt[0], opt[2]
            opt[0] = max(opt[0], max(opt[1], opt[3]) - prices[i])
            opt[1] = max(opt[1], opt[3])
            opt[2] = temp0 + prices[i]
            opt[3] = temp2
        return max(opt[3], opt[1], opt[2])

'''
319. Bulb Switcher
There are n bulbs that are initially off. You first turn on all the bulbs, then you turn off every second bulb.
On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb.
Return the number of bulbs that are on after n rounds.
Example:
Input: n = 3
Output: 1
'''
class Solution:
    def bulbSwitch(self, n: int) -> int:
        return floor(sqrt(n))


'''
322. Coin Change
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
Example:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
'''
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        maxNum = amount+1
        opt = [maxNum for _ in range(amount+1)]
        opt[0] = 0
        for coin in coins:
            for j in range(coin, maxNum):
                opt[j] = min(opt[j], opt[j-coin]+1)
        if opt[-1] < maxNum:
            return opt[-1]
        return -1

'''
328. Odd Even Linked List
Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.
The first node is considered odd, and the second node is even, and so on.
Note that the relative order inside both the even and odd groups should remain as it was in the input.
You must solve the problem in O(1) extra space complexity and O(n) time complexity.
Example:
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]
'''
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        odd = oddhead = ListNode(next = head)
        even = evenhead = ListNode()
        index = 1
        while head:
            if index % 2 == 1:
                odd.next.val = head.val
                odd = odd.next
            else:
                even.next = ListNode(head.val)
                even = even.next   
            head = head.next
            index += 1
        odd.next = evenhead.next
        return oddhead.next

'''
332. Reconstruct Itinerary
You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival airports of one flight.
Reconstruct the itinerary in order and return it.
All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". 
If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.
For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.
Example:
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
'''
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        ticketSet = defaultdict(list)
        for key, value in tickets:
            ticketSet[key].append(value)
        result = ["JFK"]
        def backtracking(start_point):
            if len(result) == len(tickets)+1:
                return True
            ticketSet[start_point].sort()
            for _ in ticketSet[start_point]:
                end_point = ticketSet[start_point].pop(0)
                result.append(end_point)
                if backtracking(end_point):
                    return True
                result.pop()
                ticketSet[start_point].append(end_point)
        backtracking("JFK")
        return result

'''
334. Increasing Triplet Subsequence
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.
Example:
Input: nums = [1,2,3,4,5]
Output: true
'''
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        first = second = float('inf')
        for num in nums:
            if num > second: return True
            first = min(first, num)
            if first < num < second: second = num
        return False

'''
337. House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.
Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. 
It will automatically contact the police if two directly-linked houses were broken into on the same night.
Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.
Example:
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
'''
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def bfs(node):
            if not node: return (0,0)
            leftNode = bfs(node.left)
            rightNode = bfs(node.right)
            withoutNode = max(leftNode[0], leftNode[1]) + max(rightNode[0], rightNode[1])
            withNode = node.val + leftNode[0] + rightNode[0]
            return (withoutNode, withNode)
        return max(bfs(root))

'''
338. Counting Bits
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
Example 1:
Input: n = 2
Output: [0,1,1]
'''
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0]
        for i in range(1, n+1):
            if (i&1) == 1:
                res.append(res[-1] + 1)
            else:
                res.append(res[i//2])
        return res

'''
343. Integer Break
Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.
Return the maximum product you can get.
Example:
Input: n = 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
'''
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        opt = [0 for _ in range(n+1)]
        opt[0] = 0
        opt[1] = 1
        for i in range(1, n+1):
            for j in range(1, n+1):
                opt[i] = max(opt[i], max((i-j)*j, j*opt[i-j]) )
        return opt[-1]

'''
347. Top K Frequent Elements
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
Example:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
'''
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        frac = collections.Counter(nums)
        priorityQueue = []
        for key,value in frac.items():
            heapq.heappush(priorityQueue, (value, key))
            if len(priorityQueue) > k:
                heapq.heappop(priorityQueue)
        result = []
        for ele in priorityQueue:
            result.append(ele[1])
        return result

'''
344. Reverse String
Write a function that reverses a string. The input string is given as an array of characters s.
You must do this by modifying the input array in-place with O(1) extra memory.
Example:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]
'''
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        length= len(s)
        for i in range (length//2):
            s[i], s[length-i-1] = s[length-i-1], s[i]

'''
345. Reverse Vowels of a String
Given a string s, reverse only all the vowels in the string and return it.
The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both cases.
Example:
Input: s = "hello"
Output: "holle"
'''
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = set(list('aeiouAEIOU'))
        s = list(s)
        left,right = 0, len(s)-1
        while left < right:
            while left < right and s[left] not in vowels:
                left += 1
            while left < right and s[right] not in vowels:
                right -= 1
            s[left], s[right] = s[right], s[left]
            left,right = left+1,right-1
        return ''.join(s)

'''
0349. Intersection of Two Arrays
Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.
Example:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]
'''
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return set(nums1).intersection(set(nums2))

'''
350. Intersection of Two Arrays II
Given two integer arrays nums1 and nums2, return an array of their intersection.
Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.
Example:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
'''
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        result = []
        dic = collections.Counter(nums1)
        for num in nums2:
            if num in dic and dic[num]>0:
                result.append(num)
                dic[num] -= 1
        return result

'''
367. Valid Perfect Square
Given a positive integer num, return true if num is a perfect square or false otherwise.
A perfect square is an integer that is the square of an integer. In other words, it is the product of some integer with itself.
You must not use any built-in library function, such as sqrt.
Example:
Input: num = 16
Output: true
'''
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num == 1: return True
        i,l = 2, num//2
        while i<=l:
            mid = (i+l)//2
            if mid*mid == num:
                return True
            elif mid*mid < num:
                i = mid+1
            else:
                l = mid-1
        return False


'''
371. Sum of Two Integers
Given two integers a and b, return the sum of the two integers without using the operators + and -.
Example:
Input: a = 1, b = 2
Output: 3
'''
class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        mask = 0xFFFFFFFF
        while b!=0:
            a,b = (a^b)&mask, ((a&b)<<1)&mask
        return a if a <= 0x7FFFFFFF else ~(a^mask)

'''
373. Find K Pairs with Smallest Sums
You are given two integer arrays nums1 and nums2 sorted in non-decreasing order and an integer k.
Define a pair (u, v) which consists of one element from the first array and one element from the second array.
Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.
Example:
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
'''
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        result, heap = [], []
        for num in nums1:
            heappush(heap, [num + nums2[0], 0])
        while k > 0 and heap:
            pairsum, pos = heappop(heap)
            result.append([pairsum-nums2[pos], nums2[pos]])
            if pos + 1 < len(nums2):
                heappush(heap, [pairsum-nums2[pos]+nums2[pos+1], pos+1])
            k -= 1
        return result

'''
376. Wiggle Subsequence
A wiggle sequence is a sequence where the differences between successive numbers strictly alternate between positive and negative.
The first difference (if one exists) may be either positive or negative. 
A sequence with one element and a sequence with two non-equal elements are trivially wiggle sequences.
For example, [1, 7, 4, 9, 2, 5] is a wiggle sequence because the differences (6, -3, 5, -7, 3) alternate between positive and negative.
In contrast, [1, 4, 7, 2, 5] and [1, 7, 4, 5, 5] are not wiggle sequences. 
The first is not because its first two differences are positive, and the second is not because its last difference is zero.
A subsequence is obtained by deleting some elements (possibly zero) from the original sequence, leaving the remaining elements in their original order.
Given an integer array nums, return the length of the longest wiggle subsequence of nums.
Example:
Input: nums = [1,7,4,9,2,5]
Output: 6
Explanation: The entire sequence is a wiggle sequence with differences (6, -3, 5, -7, 3).
'''
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return len(nums)
        prediff, curdiff, result = 0,0,1
        for i in range(1, len(nums)):
            curdiff = nums[i] - nums[i-1]
            if prediff * curdiff <= 0 and curdiff !=0:
                result += 1
                prediff = curdiff
        return result

'''
377. Combination Sum IV
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
The test cases are generated so that the answer can fit in a 32-bit integer.
Example :
Input: nums = [1,2,3], target = 4
Output: 7
'''
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        dp = [0] * (target + 1)
        dp[0] = 1

        for i in range(1, target+1):
            for j in nums:
                if i >= j:
                    dp[i] += dp[i - j]
        return dp[-1]

'''
382. Linked List Random Node
Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.
Implement the Solution class:
Solution(ListNode head) Initializes the object with the head of the singly-linked list head.
int getRandom() Chooses a node randomly from the list and returns its value. All the nodes of the list should be equally likely to be chosen.
Example:
Input
["Solution", "getRandom", "getRandom", "getRandom", "getRandom", "getRandom"]
[[[1, 2, 3]], [], [], [], [], []]
Output
[null, 1, 3, 2, 2, 3]
'''
class Solution(object):

    def __init__(self, head):
        """
        :type head: Optional[ListNode]
        """
        self.head = head

    def getRandom(self):
        """
        :rtype: int
        """
        res = 0
        node = self.head
        count = 0
        while node:
            if random.randint(0, count) == 0:
                res = node.val
            node = node.next
            count += 1
        return res

'''
0383. Ransom Note
Given two strings ransomNote and magazine, return true if ransomNote can be constructed from magazine and false otherwise.
Each letter in magazine can only be used once in ransomNote.
Example:
Input: ransomNote = "a", magazine = "b"
Output: false
'''
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

'''
387. First Unique Character in a String
Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
Example:
Input: s = "leetcode"
Output: 0
'''
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic = collections.Counter(s)
        for i, char in enumerate(s):
            if dic[char] == 1:
                return i
        return -1

'''
388. Longest Absolute File Path
Suppose we have a file system that stores both files and directories. An example of one system is represented in the following picture:
Here, we have dir as the only directory in the root. dir contains two subdirectories, subdir1 and subdir2. subdir1 contains a file file1.ext and subdirectory subsubdir1. subdir2 contains a subdirectory subsubdir2, which contains a file file2.ext.
In text form, it looks like this (with ⟶ representing the tab character):
dir
⟶ subdir1
⟶ ⟶ file1.ext
⟶ ⟶ subsubdir1
⟶ subdir2
⟶ ⟶ subsubdir2
⟶ ⟶ ⟶ file2.ext
If we were to write this representation in code, it will look like this: "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext". Note that the '\n' and '\t' are the new-line and tab characters.
Every file and directory has a unique absolute path in the file system, which is the order of directories that must be opened to reach the file/directory itself, all concatenated by '/'s. Using the above example, the absolute path to file2.ext is "dir/subdir2/subsubdir2/file2.ext". Each directory name consists of letters, digits, and/or spaces. Each file name is of the form name.extension, where name and extension consist of letters, digits, and/or spaces.
Given a string input representing the file system in the explained format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.
Note that the testcases are generated such that the file system is valid and no file or directory name has length 0.
Example 1:
Input: input = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
Output: 20
Explanation: We have only one file, and the absolute path is "dir/subdir2/file.ext" of length 20.
'''
class Solution(object):
    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """
        ans = 0
        path_len = {0: 0}
        for line in input.splitlines():
            filename = line.lstrip('\t')
            depth  = len(line)-len(filename)
            if '.' in filename:
                ans = max(ans, path_len[depth] + len(filename))
            else:
                path_len[depth+1] = path_len[depth] + len(filename) + 1
        return ans

'''
389. Find the Difference
You are given two strings s and t.
String t is generated by random shuffling string s and then add one more letter at a random position.
Return the letter that was added to t.
Example:
Input: s = "abcd", t = "abcde"
Output: "e"
Explanation: 'e' is the letter that was added.
'''
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        return chr(reduce(xor, map(ord, s+t)))

'''
390. Elimination Game
You have a list arr of all integers in the range [1, n] sorted in a strictly increasing order. Apply the following algorithm on arr:
Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.
Repeat the previous step again, but this time from right to left, remove the rightmost number and every other number from the remaining numbers.
Keep repeating the steps again, alternating left to right and right to left, until a single number remains.
Given the integer n, return the last number that remains in arr.
Example:
Input: n = 9
Output: 6
'''
class Solution(object):
    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """
        start, end = 1, n
        diff = 1
        direction = 1
        while start < end:
            mid = (end-start)//diff + 1
            if direction==1:
                start += diff
                end -= (mid&1)*diff
            else:
                start += (mid&1)*diff
                end -= diff
            direction *= -1
            diff *= 2
        return start

'''
392. Is Subsequence
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
A subsequence of a string is a new string that is formed from the original string by deleting 
some (can be none) of the characters without disturbing the relative positions of the remaining characters.
(i.e., "ace" is a subsequence of "abcde" while "aec" is not).
Example:
Input: s = "abc", t = "ahbgdc"
Output: true
'''
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if len(s) > len(t): return False
        if len(s) == 0: return True
        i = 0
        for j in range(len(t)):
            if s[i] == t[j]: i += 1
            if i == len(s): return True
        return False

'''
394. Decode String
Given an encoded string, return its decoded string.
The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.
You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].
The test cases are generated so that the length of the output will never exceed 105.
Example:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"
'''
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        curr_num = 0
        curr_str = ""
        for c in s:
            if c.isdigit():
                curr_num  = curr_num*10 + int(c)
            elif c == "[":
                stack.append(curr_num)
                stack.append(curr_str)
                curr_num = 0
                curr_str = ""
            elif c == "]":
                curr_str = stack.pop() + curr_str*stack.pop()
            else:
                curr_str += c
        return curr_str

'''
395. Longest Substring with At Least K Repeating Characters
Given a string s and an integer k, return the length of the longest substring of s such that the frequency of each character in this substring is greater than or equal to k.
Example:
Input: s = "aaabb", k = 3
Output: 3
'''
class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        if len(s) < k:
            return 0
        for char in set(s):
            if s.count(char) < k:
                return max(self.longestSubstring(t, k) for t in s.split(char))
        return len(s)

'''
398. Random Pick Index
Given an integer array nums with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.
Implement the Solution class:
Solution(int[] nums) Initializes the object with the array nums.
int pick(int target) Picks a random index i from nums where nums[i] == target. If there are multiple valid i's, then each index should have an equal probability of returning.
Example:
Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]
'''
class Solution(object):
    # defaultdict(list) 可以解决
    # Reservoir Sampling 可以解决
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        idx = random.randint(0,len(self.nums)-1)
        while self.nums[idx] != target:
            idx = random.randint(0,len(self.nums)-1)
        return idx

'''
399. Evaluate Division
You are given an array of variable pairs equations and an array of real numbers values, where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents a single variable.
You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.
Return the answers to all queries. If a single answer cannot be determined, return -1.0.
Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.
Example:
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
'''
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(dict)
        res = []
        for i, equation in enumerate(equations):
            src, des = equation[0], equation[1]
            graph[src][des] = values[i]
            graph[des][src] = 1/values[i]
        for query in queries:
            res.append(self.dfs(query[0], query[1], graph, set()))
        return res

    def dfs(self, src, des, graph, visited):
        if src not in graph or des not in graph:
            return -1.0
        if src == des:
            return 1.0
        visited.add(src)
        for nbr,weight in graph[src].items():
            if nbr in visited: continue
            ans = self.dfs(nbr, des, graph, visited)
            if ans != -1:
                return ans * weight
        return -1.0

##400##
'''
400. Nth Digit
Given an integer n, return the nth digit of the infinite integer sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...].
Example:
Input: n = 3
Output: 3
'''
class Solution(object):
    def findNthDigit(self, n):
        """
        :type n: int
        :rtype: int
        """
        digitnum, digit = 9, 1
        while n>digitnum:
            n-=digitnum
            digit+=1
            digitnum=9*10**(digit-1)*digit
        n-=1
        return str(10**(digit-1) + n/digit)[n%digit]
    
'''
401. Binary Watch
A binary watch has 4 LEDs on the top to represent the hours (0-11), and 6 LEDs on the bottom to represent the minutes (0-59). Each LED represents a zero or one, with the least significant bit on the right.
For example, the below binary watch reads "4:51".
Given an integer turnedOn which represents the number of LEDs that are currently on (ignoring the PM), return all possible times the watch could represent. You may return the answer in any order.
The hour must not contain a leading zero.
For example, "01:00" is not valid. It should be "1:00".
The minute must be consist of two digits and may contain a leading zero.
For example, "10:2" is not valid. It should be "10:02".
Example:
Input: turnedOn = 1
Output: ["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]
'''
class Solution(object):
    def readBinaryWatch(self, turnedOn):
        """
        :type turnedOn: int
        :rtype: List[str]
        """
        return ['{:d}:{:0>2d}'.format(h, m)
            for h in range(12) for m in range(60)
            if (bin(h)+bin(m)).count('1') == turnedOn]

'''
404. Sum of Left Leaves
Given the root of a binary tree, return the sum of all left leaves.
A leaf is a node with no children. A left leaf is a leaf that is the left child of another node.
Example:
Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.
'''
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = deque()
        if root:
            queue.append(root)
        leftnodeSum = 0
        while queue:
            numNode = len(queue)
            for _ in range(numNode):
                cur = queue.popleft()
                if cur.left:
                    queue.append(cur.left)
                    if cur.left.left is None and cur.left.right is None:
                        leftnodeSum += cur.left.val
                if cur.right: 
                    queue.append(cur.right)
        return leftnodeSum

'''
405. Convert a Number to Hexadecimal
Given an integer num, return a string representing its hexadecimal representation. For negative integers, two’s complement method is used.
All the letters in the answer string should be lowercase characters, and there should not be any leading zeros in the answer except for the zero itself.
Note: You are not allowed to use any built-in library method to directly solve this problem.
Example:
Input: num = 26
Output: "1a"
'''
class Solution(object):
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num==0: return '0'
        mp = '0123456789abcdef'  # like a map
        ans = ''
        for i in range(8):
            n = num & 15       # this means num & 1111b
            c = mp[n]          # get the hex char 
            ans = c + ans
            num = num >> 4
        return ans.lstrip('0')  #strip leading zeroes

'''
406. Queue Reconstruction by Height
You are given an array of people, people, which are the attributes of some people in a queue (not necessarily in order). 
Each people[i] = [hi, ki] represents the ith person of height hi with exactly ki other people in front who have a height greater than or equal to hi.
Reconstruct and return the queue that is represented by the input array people. 
The returned queue should be formatted as an array queue, where queue[j] = [hj, kj] is the attributes of the jth person in the queue (queue[0] is the person at the front of the queue).
Example:
Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
Explanation:
Person 0 has height 5 with no other people taller or the same height in front.
Person 1 has height 7 with no other people taller or the same height in front.
Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
Person 3 has height 6 with one person taller or the same height in front, which is person 1.
Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.
Person 5 has height 7 with one person taller or the same height in front, which is person 1.
Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
'''
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda x: (-x[0], x[1]))
        result = []
        for p in people:
            result.insert(p[1], p)
        return result

'''
409. Longest Palindrome
Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.
Letters are case sensitive, for example, "Aa" is not considered a palindrome here.
Example:
Input: s = "abccccdd"
Output: 7
Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.
'''
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        dic = collections.Counter(s)
        for count in dic.values():
            result += count // 2
        return result*2 + (len(s) > result*2)

'''
416. Partition Equal Subset Sum
Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.
Example:
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
'''
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        totalSum = sum(nums)
        if totalSum % 2 == 1: 
            return False
        totalSum //= 2
        opt = [[0 for _ in range(totalSum+1)] for _ in range(len(nums))]
        
        for i in range(1, len(nums)):
            for j in range(totalSum+1):
                if j >= nums[i]:
                    opt[i][j] = max(opt[i-1][j], opt[i-1][j-nums[i]] + nums[i])
                else:
                    opt[i][j] = opt[i-1][j]
        return opt[-1][totalSum] == totalSum    

'''
417. Pacific Atlantic Water Flow
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.
The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).
The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.
Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
Example:
Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
'''
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
            rows, cols = len(heights), len(heights[0])
            pacific = [[False for _ in range(cols)] for _ in range(rows)]
            atlantic = [[False for _ in range(cols)] for _ in range(rows)]
            res = []
            for i in range(rows):
                self.dfs(heights, i, 0, rows, cols, pacific, heights[i][0])
                self.dfs(heights, i, cols-1, rows, cols, atlantic, heights[i][cols-1])
            for j in range(cols):
                self.dfs(heights, 0, j, rows, cols, pacific, heights[0][j])
                self.dfs(heights, rows-1, j ,rows, cols, atlantic, heights[rows-1][j])
            for i in range(rows):
                for j in range(cols):
                    if pacific[i][j] & atlantic[i][j]:
                        res.append([i,j])
            return res
    
    def dfs(self, heights, i, j, rows, cols, visited, prevHeight):
        if i < 0 or i >= rows or j < 0 or j >= cols:
            return
        if visited[i][j] or prevHeight > heights[i][j]:
            return
        visited[i][j] = True
        self.dfs(heights, i-1, j, rows, cols, visited, heights[i][j])
        self.dfs(heights, i+1, j, rows, cols, visited, heights[i][j])
        self.dfs(heights, i, j-1, rows, cols, visited, heights[i][j])
        self.dfs(heights, i, j+1, rows, cols, visited, heights[i][j])

'''
424. Longest Repeating Character Replacement
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character.
You can perform this operation at most k times.
Return the length of the longest substring containing the same letter you can get after performing the above operations.
Example:
Input: s = "ABAB", k = 2
Output: 4
'''
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

'''
429. N-ary Tree Level Order Traversal
Given an n-ary tree, return the level order traversal of its nodes' values.
Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).
Example:
Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]
'''
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return []
        results = []
        layer = deque()
        layer.append(root)
        while layer:
            result = []
            for i in range(len(layer)):
                cur = layer.popleft()
                result.append(cur.val)
                for child in cur.children:
                    layer.append(child)
            results.append(result)
        return results

'''
434. Number of Segments in a String
Given a string s, return the number of segments in the string.
A segment is defined to be a contiguous sequence of non-space characters.
Example:
Input: s = "Hello, my name is John"
Output: 5
Explanation: The five segments are ["Hello,", "my", "name", "is", "John"]
'''
class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        segment_count = 0
        for i in range(len(s)):
            if (i == 0 or s[i-1] == ' ') and s[i] != ' ':
                segment_count += 1
        return segment_count

'''
435. Non-overlapping Intervals
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
Example:
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
'''
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if len(intervals)==1:return 0
        intervals.sort(key=lambda x: x[1])
        count = 1
        pos = intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] < pos:
                continue
            else:
                pos = intervals[i][1]
                count += 1
        return len(intervals)-count

'''
437. Path Sum III
Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along the path equals targetSum.
The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).
Example:
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3
'''
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.hashmap = {0:1}
        return self.dfs(root, targetSum, 0)

    def dfs(self, node, targetSum, preSum):
        if not node: return 0
        preSum += node.val
        res = self.hashmap.get(preSum - targetSum, 0)
        self.hashmap[preSum] = self.hashmap.get(preSum, 0) + 1
        res += self.dfs(node.left, targetSum, preSum)
        res += self.dfs(node.right, targetSum, preSum)
        self.hashmap[preSum] -= 1
        return res

'''
438. Find All Anagrams in a String
Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
Example:
Input: s = "cbaebabacd", p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
'''
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        result = []
        ls,lp = len(s), len(p)
        if ls < lp: return []
        window = [0 for _ in range(26)]
        target =  [0 for _ in range(26)]
        for i in range(lp-1):
            window[ord(s[i])-ord('a')] += 1
        for i in range(lp):
            target[ord(p[i])-ord('a')] += 1
        for i in range(lp-1,ls):
            window[ord(s[i])-ord('a')] += 1
            if window == target:
                result.append(i-lp+1)
            window[ord(s[i-lp+1])-ord('a')] -=1
        return result

'''
442. Find All Duplicates in an Array
Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.
You must write an algorithm that runs in O(n) time and uses only constant extra space.
Example:
Input: nums = [4,3,2,7,8,2,3,1]
Output: [2,3]
'''
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        dic = collections.defaultdict(int)
        for num in nums:
            dic[num] += 1
            if dic[num] == 2: result.append(num)
        return result

'''
443. String Compression
Given an array of characters chars, compress it using the following algorithm:
Begin with an empty string s. For each group of consecutive repeating characters in chars:
If the group's length is 1, append the character to s.
Otherwise, append the character followed by the group's length.
The compressed string s should not be returned separately, but instead, be stored in the input character array chars. Note that group lengths that are 10 or longer will be split into multiple characters in chars.
After you are done modifying the input array, return the new length of the array.
You must write an algorithm that uses only constant extra space.
Example:
Input: chars = ["a","a","b","b","c","c","c"]
Output: Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".
'''
class Solution(object):
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        index = 0
        for key, nums in itertools.groupby(chars):
            c = len(list(nums))
            if c == 1:
                chars[index] = key
                index += 1
            else:
                chars[index] = key
                if c < 10:
                    chars[index+1] = str(c)
                else: 
                    for i, digit in enumerate(str(c), index+1):
                        chars[i] = digit
                index += 1+len(str(c))
        return index

'''
445. Add Two Numbers II
You are given two non-empty linked lists representing two non-negative integers.
The most significant digit comes first and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Example:
Input: l1 = [7,2,4,3], l2 = [5,6,4]
Output: [7,8,0,7]
'''
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        v1 = v2 = 0
        while l1:
            v1 = v1*10 + l1.val
            l1 = l1.next
        while l2:
            v2 = v2*10 + l2.val
            l2 = l2.next
        val = v1+v2
        tail, head = None, None
        while val > 0:
            head = ListNode(val % 10)
            head.next = tail
            tail = head
            val //= 10
        return head if head else ListNode(0)

'''
448. Find All Numbers Disappeared in an Array
Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.
Example:
Input: nums = [4,3,2,7,8,2,3,1]
Output: [5,6]
'''
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in xrange(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        uniqueSet = set(nums)
        result = []
        for i in xrange(1,len(nums)+1):
            if i not in uniqueSet:
                result.append(i)
        return result

'''
449. Serialize and Deserialize BST
Serialization is converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work. You need to ensure that a binary search tree can be serialized to a string, and this string can be deserialized to the original tree structure.
The encoded string should be as compact as possible.
Example:
Input: root = [2,1,3]
Output: [2,1,3]
'''
class Codec:

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single string.
        """
        self.result = ""
        self.helpserialize(root)
        return self.result

    def helpserialize(self, root):
        if not root:
            self.result += "x,"
            return
        self.result += str(root.val) + ","
        self.helpserialize(root.left)
        self.helpserialize(root.right)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree.
        """
        queue = deque(data.split(","))
        return self.helpdeserialize(queue)

    def helpdeserialize(self, queue): 
        s = queue.popleft()
        if s=="x": return None
        root = TreeNode(int(s))
        root.left = self.helpdeserialize(queue)
        root.right = self.helpdeserialize(queue)
        return root

'''
450. Delete Node in a BST
Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.
Basically, the deletion can be divided into two stages:
Search for a node to remove.
If the node is found, delete the node.
Example:
Input: root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
'''
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if root.left is None and root.right is None:
                return None
            elif root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                temp = root.right
                while temp.left:
                    temp = temp.left
                temp.left = root.left
                root = root.right
        return root

'''
451. Sort Characters By Frequency
Given a string s, sort it in decreasing order based on the frequency of the characters. The frequency of a character is the number of times it appears in the string.
Return the sorted string. If there are multiple answers, return any of them.
Example:
Input: s = "tree"
Output: "eert"
'''
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        dic = collections.defaultdict(int)
        for char in s:
            dic[char] += 1
        d = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        final = ''
        for i in d:
            final += i[0]*i[1]
        return final

'''
452. Minimum Number of Arrows to Burst Balloons
There are some spherical balloons taped onto a flat wall that represents the XY-plane. 
The balloons are represented as a 2D integer array points where points[i] = [xstart, xend] denotes a balloon whose horizontal diameter stretches between xstart and xend. 
You do not know the exact y-coordinates of the balloons.
Arrows can be shot up directly vertically (in the positive y-direction) from different points along the x-axis.
A balloon with xstart and xend is burst by an arrow shot at x if xstart <= x <= xend. 
There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.
Given the array points, return the minimum number of arrows that must be shot to burst all balloons.
Example:
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: The balloons can be burst by 2 arrows:
- Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
- Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].
'''
class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points) == 0:
            return 0
        points.sort(key=lambda x: x[1])
        result = 1
        arrow = points[0][1]
        for i in range(len(points)):
            if points[i][0] <= arrow and points[i][1] >= arrow:
                continue
            else:
                arrow = points[i][1]
                result += 1
        return result
    
        ##if len(points) == 0:
        ##    return 0
        ##points.sort(key=lambda x: x[0])
        ##result = 1
        ##for i in range(1, len(points)):
        ##    if points[i-1][1] < points[i][0]:
        ##        result += 1
        ##    else:
        ##        points[i][1] = min(points[i - 1][1], points[i][1])
        ##return result

'''
454. 4Sum II
Given four integer arrays nums1, nums2, nums3, and nums4 all of length n, return the number of tuples (i, j, k, l) such that:
0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
Example:
Input: nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
Output: 2
'''
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
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


'''
455. Assign Cookies
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.
Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; 
and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. 
Your goal is to maximize the number of your content children and output the maximum number.
Example:
Input: g = [1,2,3], s = [1,1]
Output: 1
'''
class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort()
        s.sort()
        result = 0
        cookie = len(s)-1
        for i in range(len(g)-1,-1,-1):
            if cookie >= 0 and s[cookie] >= g[i]:
                cookie -= 1
                result += 1
        return result

'''
459. Repeated Substring Pattern
Given a string s, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.
Example:
Input: s = "abab"
Output: true
Explanation: It is the substring "ab" twice.
'''
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0:
            return False
        nextArr = self.getNext(s)
        print(nextArr)
        if nextArr[-1] != -1 and len(s) % (len(s)-nextArr[-1]-1) == 0:
            return True
        return False   
    def getNext(self, s):
        nxt = [0 for _ in range(len(s))]
        nxt[0] = -1
        j = -1
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j+1]:
                j = nxt[j]
            if s[i] == s[j+1]:
                j += 1
            nxt[i] = j
        return nxt

'''
460. LFU Cache
Design and implement a data structure for a Least Frequently Used (LFU) cache.
Implement the LFUCache class:
LFUCache(int capacity) Initializes the object with the capacity of the data structure.
int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
void put(int key, int value) Update the value of the key if present, or inserts the key if not already present. When the cache reaches its capacity, it should invalidate and remove the least frequently used key before inserting a new item. For this problem, when there is a tie (i.e., two or more keys with the same frequency), the least recently used key would be invalidated.
To determine the least frequently used key, a use counter is maintained for each key in the cache. The key with the smallest use counter is the least frequently used key.
When a key is first inserted into the cache, its use counter is set to 1 (due to the put operation). The use counter for a key in the cache is incremented either a get or put operation is called on it.
The functions get and put must each run in O(1) average time complexity.
Example:
Input
["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, 3, null, -1, 3, 4]
'''
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


'''
461. Hamming Distance
The Hamming distance between two integers is the number of positions at which the corresponding bits are different.
Given two integers x and y, return the Hamming distance between them.
Example:
Input: x = 1, y = 4
Output: 2
'''
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return (bin(x^y)).count('1')

'''
463. Island Perimeter
You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.
Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).
The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. 
The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.
Example:
Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.
'''
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        print(len(grid), len(grid[0]))
        direction = [[-1, 0], [1, 0],[0, 1], [0, -1]]
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    for k in range(4):
                        x, y = i + direction[k][0], j + direction[k][1]
                        if (x<0 or x>=len(grid) or y<0 or y>=len(grid[0]) or grid[x][y]==0):
                            result += 1
        return result

'''
474. Ones and Zeroes
You are given an array of binary strings strs and two integers m and n.
Return the size of the largest subset of strs such that there are at most m 0's and n 1's in the subset.
A set x is a subset of a set y if all elements of x are also elements of y.
Example:
Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
Output: 4
Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.
Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.
{"111001"} is an invalid subset because it contains 4 1's, greater than the maximum of 3.
'''
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        opt = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for str in strs:
            ones = str.count('1')
            zeros = str.count('0')
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    opt[i][j] = max(opt[i][j], opt[i - zeros][j - ones] + 1)
        return opt[m][n]

'''
476. Number Complement
The complement of an integer is the integer you get when you flip all the 0's to 1's and all the 1's to 0's in its binary representation.
For example, The integer 5 is "101" in binary and its complement is "010" which is the integer 2.
Given an integer num, return its complement.
Example:
Input: num = 5
Output: 2
'''
class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        i = 1
        while i <= num:
            i <<= 1
        return (i-1)^num

'''
480.Sliding Window Median
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle values.
For examples, if arr = [2,3,4], the median is 3.
For examples, if arr = [1,2,3,4], the median is (2 + 3) / 2 = 2.5.
You are given an integer array nums and an integer k. There is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.
Return the median array for each window in the original array. Answers within 10-5 of the actual value will be accepted.
Example:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [1.00000,-1.00000,-1.00000,3.00000,5.00000,6.00000]
'''

'''
482. License Key Formatting
You are given a license key represented as a string s that consists of only alphanumeric characters and dashes. The string is separated into n + 1 groups by n dashes. You are also given an integer k.
We want to reformat the string s such that each group contains exactly k characters, except for the first group, which could be shorter than k but still must contain at least one character.
Furthermore, there must be a dash inserted between two groups, and you should convert all lowercase letters to uppercase.
Return the reformatted license key.
Example:
Input: s = "5F3Z-2e-9-w", k = 4
Output: "5F3Z-2E9W"
Explanation: The string s has been split into two parts, each part has 4 characters.
Note that the two extra dashes are not needed and can be removed.
'''
class Solution(object):
    def licenseKeyFormatting(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        result = []
        s = s.replace('-', '').upper()
        mod = len(s)%k
        if mod:
            result.append(s[0:mod])
        return '-'.join(result + [s[i:i+k] for i in range(mod,len(s),k)])

'''
485. Max Consecutive Ones
Given a binary array nums, return the maximum number of consecutive 1's in the array.
Example:
Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
'''
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = [0 for _ in range(len(nums))]
        result[0] = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == 1:
                result[i] = result[i-1] + 1
        print(result)
        return max(result)

'''
491. Increasing Subsequences
Given an integer array nums, return all the different possible increasing subsequences of the given array with at least two elements. 
You may return the answer in any order.
The given array may contain duplicates, and two equal integers should also be considered a special case of increasing sequence.
Example:
Input: nums = [4,6,7,7]
Output: [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
'''
class Solution(object):
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        results = []
        result = []
        def backtracking(startInd):
            if len(result) > 1:
                results.append(result[:])
            used = set()
            for i in range(startInd, len(nums)):
                if result and nums[i] < result[-1] or nums[i] in used:
                    continue
                used.add(nums[i])
                result.append(nums[i])
                backtracking(i+1)
                result.pop()
        backtracking(0)
        return results

'''
494. Target Sum
You are given an integer array nums and an integer target.
You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.
Example:
Input: nums = [1,1,1,1,1], target = 3
Output: 5
'''
class Solution(object):
    def findTargetSumWays(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        sumValue = sum(nums)
        if abs(target) > sumValue or (sumValue + target) % 2 == 1: return 0
        bagSize = (sumValue + target) // 2
        dp = [0] * (bagSize + 1)
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(bagSize, nums[i] - 1, -1):
                dp[j] += dp[j - nums[i]]
        return dp[bagSize]

'''
496. Next Greater Element I
The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.
You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.
For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2.
If there is no next greater element, then the answer for this query is -1.
Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.
Example:
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
'''
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        nummap = {}
        result = [-1 for _ in range(len(nums1))]
        for i, num in enumerate(nums1):
            nummap[num] = i
        stack = []
        for j, num in enumerate(nums2):
            while len(stack)>0 and num>nums2[stack[-1]]:
                if nums2[stack[-1]] in nummap:
                    result[nummap[nums2[stack[-1]]]] = num
                stack.pop()
            stack.append(j)
        return result

'''
498. Diagonal Traverse
Given an m x n matrix mat, return an array of all the elements of the array in a diagonal order.
Example:
Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,4,7,5,3,6,8,9]
'''
class Solution(object):
    def findDiagonalOrder(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[int]
        """
        if not mat: return []
        result = []
        dd = collections.defaultdict(list)
        m, n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(n):
                dd[i+j].append(mat[i][j])
        for k in range(m+n-1):
            if k % 2 == 0:
                result.extend(dd[k][::-1])
            else: result.extend(dd[k])
        return result

##500##

'''
500. Keyboard Row
Given an array of strings words, return the words that can be typed using letters of the alphabet on only one row of American keyboard like the image below.
In the American keyboard:
the first row consists of the characters "qwertyuiop",
the second row consists of the characters "asdfghjkl", and
the third row consists of the characters "zxcvbnm".
Example:
Input: words = ["Hello","Alaska","Dad","Peace"]
Output: ["Alaska","Dad"]
'''
class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        rowsa = set(list("qwertyuiop"))
        rowsb = set(list("asdfghjkl"))
        rowsc = set(list("zxcvbnm"))
        result = []
        for word in words:
            w = set(word.lower())
            if w <= rowsa or w <= rowsb or w <= rowsc:
                result.append(word)
        return result

'''
501. Find Mode in Binary Search Tree
Given the root of a binary search tree (BST) with duplicates, return all the mode(s) (i.e., the most frequently occurred element) in it.
If the tree has more than one mode, return them in any order.
Assume a BST is defined as follows:
The left subtree of a node contains only nodes with keys less than or equal to the node's key.
The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
Both the left and right subtrees must also be binary search trees.
Example:
Input: root = [1,null,2,2]
Output: [2]
'''
class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        maxCount = 0
        count = 0
        stack = []
        result = []
        cur = root
        pre = None
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre == None:
                    count = 1
                elif pre.val == cur.val:
                    count += 1
                else:
                    count = 1
                if count == maxCount:
                    result.append(cur.val)
                elif count > maxCount:
                    result = []
                    maxCount = count
                    result.append(cur.val)
                pre = cur
                cur = cur.right
        return result

'''
502. IPO
Suppose LeetCode will start its IPO soon.
In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO.
Since it has limited resources, it can only finish at most k distinct projects before the IPO.
Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects.
You are given n projects where the ith project has a pure profit profits[i] and a minimum capital of capital[i] is needed to start it.
Initially, you have w capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.
Pick a list of at most k distinct projects from given projects to maximize your final capital, and return the final maximized capital.
The answer is guaranteed to fit in a 32-bit signed integer.
Example:
Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
Output: 4
'''
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        n = len(profits)
        paires = list(zip(capital, profits))
        paires.sort()
        q = []
        ptr = 0
        for i in range(k):
            while ptr < n and paires[ptr][0] <= w:
                heappush(q, -paires[ptr][1])
                ptr += 1
            if not q:
                break
            w += -heappop(q)
        return w

'''
503. Next Greater Element II
Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums.
The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number.
If it doesn't exist, return -1 for this number.
Example:
Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number. 
The second 1's next greater number needs to search circularly, which is also 2.
'''
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = [-1 for _ in range(len(nums))]
        stack =[]
        for i in range(len(nums)*2):
            while(len(stack)>0 and nums[i%len(nums)] > nums[stack[-1]]):
                result[stack.pop()] = nums[i%len(nums)]
            stack.append(i%len(nums))
        return result

'''
506. Relative Ranks
You are given an integer array score of size n, where score[i] is the score of the ith athlete in a competition.
All the scores are guaranteed to be unique.
The athletes are placed based on their scores, where the 1st place athlete has the highest score, the 2nd place athlete has the 2nd highest score, and so on.
The placement of each athlete determines their rank:
The 1st place athlete's rank is "Gold Medal".
The 2nd place athlete's rank is "Silver Medal".
The 3rd place athlete's rank is "Bronze Medal".
For the 4th place to the nth place athlete, their rank is their placement number (i.e., the xth place athlete's rank is "x").
Return an array answer of size n where answer[i] is the rank of the ith athlete.
Example:
Input: score = [5,4,3,2,1]
Output: ["Gold Medal","Silver Medal","Bronze Medal","4","5"]
Explanation: The placements are [1st, 2nd, 3rd, 4th, 5th].
'''
class Solution(object):
    def findRelativeRanks(self, score):
        """
        :type score: List[int]
        :rtype: List[str]
        """
        _, order = zip(*sorted(zip(score, range(len(score))))[::-1])
        rank = ["Gold Medal","Silver Medal","Bronze Medal"] + map(str, range(4, len(score)+1))
        _, rank = zip(*sorted(zip(order, rank)))
        return rank

'''
509. Fibonacci Number
The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,
F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
Given n, calculate F(n).
Example:
Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.
'''
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0: return 0
        if n == 1: return 1
        opt = [0 for _ in range(n+1)]
        opt[0], opt[1] = 0, 1
        for i in range(2, n+1):
            opt[i] = opt[i-1] + opt[i-2]
        return opt[-1]

'''
513. Find Bottom Left Tree Value
Given the root of a binary tree, return the leftmost value in the last row of the tree.
Example:
Input: root = [2,1,3]
Output: 1
'''
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = deque()
        if root:
            queue.append(root)
        result = 0
        while queue:
            numNode = len(queue)
            for i in range(numNode):
                cur = queue.popleft()
                if i == 0:
                    result = cur.val
                if cur.left:
                    queue.append(cur.left)
                if cur.right: 
                    queue.append(cur.right)
        return result

'''
515. Find Largest Value in Each Tree Row
Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).
Example:
Input: root = [1,3,2,5,3,null,9]
Output: [1,3,9]
'''
class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        result = []
        layer = deque()
        layer.append(root)
        while layer:
            layer_max = layer[0].val
            for i in range(len(layer)):
                cur = layer.popleft()
                if cur.val > layer_max:
                    layer_max = cur.val
                if cur.left:
                    layer.append(cur.left)
                if cur.right:
                    layer.append(cur.right)
            result.append(layer_max)
        return result

'''
516. Longest Palindromic Subsequence
Given a string s, find the longest palindromic subsequence's length in s.
A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.
Example:
Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".
'''
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        opt = [[0 for _ in range(len(s))] for _ in range(len(s))]
        for i in range(len(s)):
            opt[i][i] = 1
        for i in range(len(s)-1, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    opt[i][j] = opt[i+1][j-1] + 2
                else: 
                    opt[i][j] = max(opt[i+1][j], opt[i][j-1])
        return opt[0][-1]

'''
518. Coin Change 2
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.
The answer is guaranteed to fit into a signed 32-bit integer.
Example:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
'''
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        opt = [0 for _ in range(amount+1)]
        opt[0] = 1
        for coin in coins:
            for i in range(amount+1):
                if i-coin >= 0:
                    opt[i] += opt[i-coin]
        print(opt)
        return opt[-1]

'''
520. Detect Capital
We define the usage of capitals in a word to be right when one of the following cases holds:
All letters in this word are capitals, like "USA".
All letters in this word are not capitals, like "leetcode".
Only the first letter in this word is capital, like "Google".
Given a string word, return true if the usage of capitals in it is right.
Example:
Input: word = "USA"
Output: true
'''
class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        return word.islower() or word.isupper() or word.istitle()

'''
524. Longest Word in Dictionary through Deleting
Given a string s and a string array dictionary, return the longest string in the dictionary that can be formed by deleting some of the given string characters.
If there is more than one possible result, return the longest word with the smallest lexicographical order.
If there is no possible result, return the empty string.
Example:
Input: s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
Output: "apple"
'''
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

'''
525. Contiguous Array
Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.
Example:
Input: nums = [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.
'''
class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = res = 0
        table = {0: 0}
        for i, num in enumerate(nums, 1):
            if num == 0:
                count -= 1
            else:
                count += 1
            if count in table:
                res = max(res, i-table[count])
            else:
                table[count] = i
        return res

'''
530. Minimum Absolute Difference in BST
Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.
Example:
Input: root = [4,2,6,1,3]
Output: 1
'''
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        stack = []
        pre = None
        cur = root
        result = float('inf')
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                cur = stack.pop()
                if pre:
                    result = min(result, cur.val - pre.val)
                pre = cur
                cur = cur.right
        return result

'''
532. K-diff Pairs in an Array
Given an array of integers nums and an integer k, return the number of unique k-diff pairs in the array.
A k-diff pair is an integer pair (nums[i], nums[j]), where the following are true:
0 <= i, j < nums.length
i != j
nums[i] - nums[j] == k
Notice that |val| denotes the absolute value of val.
Example:
Input: nums = [3,1,4,1,5], k = 2
Output: 2
Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
Although we have two 1s in the input, we should only return the number of unique pairs.
'''
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        c = collections.Counter(nums)
        result = 0
        for key, value in c.items():
            if (k==0 and value>1) or (k>0 and key+k in c):
                result += 1
        return result

'''
538. Convert BST to Greater Tree
Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.
As a reminder, a binary search tree is a tree that satisfies these constraints:
The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example:
Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
'''
class Solution(object):  
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        total = 0
        node = root
        stack = []
        while stack or node:
            while node is not None:
                stack.append(node)
                node = node.right
            node = stack.pop()
            total += node.val
            node.val = total
            node = node.left
        return root

'''
539. Minimum Time Difference
Given a list of 24-hour clock time points in "HH:MM" format, return the minimum minutes difference between any two time-points in the list.
Example:
Input: timePoints = ["23:59","00:00"]
Output: 1
'''
class Solution(object):
    def findMinDifference(self, timePoints):
        """
        :type timePoints: List[str]
        :rtype: int
        """
        timePoints = sorted([int(i[:2])*60+int(i[3:]) for i in timePoints])
        timePoints.append(timePoints[0]+60*24)
        return min([b-a for a,b in zip(timePoints, timePoints[1:])])

'''
540. Single Element in a Sorted Array
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.
Return the single element that appears only once.
Your solution must run in O(log n) time and O(1) space.
Example:
Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2
'''
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        res = 0
        for num in nums: res^=num
        return res

'''
541. Reverse String II
Given a string s and an integer k, reverse the first k characters for every 2k characters counting from the start of the string.
If there are fewer than k characters left, reverse all of them. 
If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and leave the other as original.
Example:
Input: s = "abcdefg", k = 2
Output: "bacdfeg"
'''
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        def reverseStr(s):
            right = len(s)-1
            for left in range(0, len(s)//2):
                s[left], s[right] = s[right], s[left]
                right -= 1
            return s
        s = list(s)
        for i in range(0, len(s), 2*k):
            s[i:i+k] = reverseStr(s[i:i+k])
        return ''.join(s)

'''
543. Diameter of Binary Tree
Given the root of a binary tree, return the length of the diameter of the tree.
The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
The length of a path between two nodes is represented by the number of edges between them.
Example:
Input: root = [1,2,3,4,5]
Output: 3
'''
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0

        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.diameter = max(self.diameter, left+right)
            return max(left, right) + 1

        dfs(root)
        return self.diameter

'''
547. Number of Provinces
There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.
A province is a group of directly or indirectly connected cities and no other cities outside of the group.
You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.
Return the total number of provinces.
Example:
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2
'''
class Solution:
    def __init__(self):
        self.parents = []
        self.count = []

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        self.parents = [i for i in range(n)]
        self.count = [1 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if isConnected[i][j] and i!=j:
                    self.union(i, j)
        return len({self.find(i) for i in range(n)})

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

'''
551. Student Attendance Record I
You are given a string s representing an attendance record for a student where each character signifies whether the student was absent, late, or present on that day. The record only contains the following three characters:
'A': Absent.
'L': Late.
'P': Present.
The student is eligible for an attendance award if they meet both of the following criteria:
The student was absent ('A') for strictly fewer than 2 days total.
The student was never late ('L') for 3 or more consecutive days.
Return true if the student is eligible for an attendance award, or false otherwise.
Example:
Input: s = "PPALLP"
Output: true
Explanation: The student has fewer than 2 absences and was never late 3 or more consecutive days.
'''
class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        dic = collections.Counter(s)
        if dic['A']>1:
            return False
        dic = itertools.groupby(s)
        for key, value in dic:
            if key=='L' and len(list(value))>2:
                return False
        return True

'''
556. Next Greater Element III
Given a positive integer n, find the smallest integer which has exactly the same digits existing in the integer n and is greater in value than n.
If no such positive integer exists, return -1.
Note that the returned integer should fit in 32-bit integer, if there is a valid answer but it does not fit in 32-bit integer, return -1.
Example:
Input: n = 12
Output: 21
'''
class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        digits = list(str(n))
        i = len(digits) - 1
        while i-1 >= 0 and digits[i] <= digits[i-1]:
            i -= 1
        if i == 0: return -1
        j = i
        while j+1 < len(digits) and digits[j+1] > digits[i-1]:
            j += 1
        digits[i-1], digits[j] = digits[j], digits[i-1]
        digits[i:] = digits[i:][::-1]
        ret = int(''.join(digits))
        
        return ret if ret < 1<<31 else -1

'''
557. Reverse Words in a String III
Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.
Example:
Input: s = "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
'''
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ' '.join(word[::-1] for word in s.split())

'''
559. Maximum Depth of N-ary Tree
Given a n-ary tree, find its maximum depth.
The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).
Example:
Input: root = [1,null,3,2,4,null,5,6]
Output: 3
'''
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root: return 0
        depth = 0
        for child in root.children:
            depth = max(depth, self.maxDepth(child))
        return depth + 1

'''
560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
A subarray is a contiguous non-empty sequence of elements within an array.
Example:
Input: nums = [1,1,1], k = 2
Output: 2
'''
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        res = totalsum = 0
        dic = {0:1}
        for num in nums:
            totalsum += num
            if totalsum-k in dic:
                res = res + dic[totalsum-k]
            if totalsum not in dic:
                dic[totalsum] = 1
            else:
                dic[totalsum] += 1 
        return res

'''
561. Array Partition
Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2), ..., (an, bn) such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.
Example:
Input: nums = [1,4,3,2]
Output: 4
Explanation: All possible pairings (ignoring the ordering of elements) are:
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
So the maximum possible sum is 4.
'''
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(sorted(nums)[::2])

'''
566. Reshape the Matrix
In MATLAB, there is a handy function called reshape which can reshape an m x n matrix into a new one with a different size r x c keeping its original data.
You are given an m x n matrix mat and two integers r and c representing the number of rows and the number of columns of the wanted reshaped matrix.
The reshaped matrix should be filled with all the elements of the original matrix in the same row-traversing order as they were.
If the reshape operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.
Example:
Input: mat = [[1,2],[3,4]], r = 1, c = 4
Output: [[1,2,3,4]]
'''
class Solution(object):
    def matrixReshape(self, mat, r, c):
        """
        :type mat: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if r*c != len(mat)*len(mat[0]):
            return mat
        mat = [x for row in mat for x in row]
        result = []
        index = 0
        for _ in range(r):
            temp = []
            for _ in range(c):
                temp.append(mat[index])
                index +=1
            result.append(temp)
        return result

'''
572. Subtree of Another Tree
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.
A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.
Example:
Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
'''
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

'''
575. Distribute Candies
Alice has n candies, where the ith candy is of type candyType[i]. 
Alice noticed that she started to gain weight, so she visited a doctor.
The doctor advised Alice to only eat n / 2 of the candies she has (n is always even).
Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice.
Given the integer array candyType of length n, return the maximum number of different types of candies she can eat if she only eats n / 2 of them.
Example:
Input: candyType = [1,1,2,2,3,3]
Output: 3
Explanation: Alice can only eat 6 / 2 = 3 candies. Since there are only 3 types, she can eat one of each type.
'''
class Solution(object):
    def distributeCandies(self, candyType):
        """
        :type candyType: List[int]
        :rtype: int
        """
        return min(len(candyType)//2, len(set(candyType)))

'''
581. Shortest Unsorted Continuous Subarray
Given an integer array nums, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order.
Return the shortest such subarray and output its length.
Example:
Input: nums = [2,6,4,8,10,9,15]
Output: 5
'''
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2: return 0
        leftMax, leftInd = float('-inf'), -1
        rightMin, rightInd = float('inf'),-1
        for i in range(len(nums)):
            if nums[i] >= leftMax:
                leftMax = nums[i]
            else: leftInd = i
        if leftInd == -1: return 0
        for j in range(len(nums)-1,-1,-1):
            if nums[j] <= rightMin:
                rightMin = nums[j]
            else: rightInd = j
        return leftInd-rightInd+1


'''
583. Delete Operation for Two Strings
Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.
In one step, you can delete exactly one character in either string.
Example :
Input: word1 = "sea", word2 = "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
'''
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        opt = [[0 for _ in range(len(word2)+1)] for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            opt[i][0] = i
        for j in range(len(word2)+1):
            opt[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    opt[i][j] = opt[i-1][j-1]
                else:
                    opt[i][j] = min(opt[i-1][j] + 1, opt[i][j-1] + 1)
        return opt[-1][-1]

'''
589. N-ary Tree Preorder Traversal
Given the root of an n-ary tree, return the preorder traversal of its nodes' values.
Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)
Example:
Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]
'''
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        result = []
        def trverse(node):
            if root == None:
                return
            result.append(node.val)
            for ele in node.children:
                trverse(ele)
        trverse(root)
        return result

'''
590. N-ary Tree Postorder Traversal
Given the root of an n-ary tree, return the postorder traversal of its nodes' values.
Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)
Example:
Input: root = [1,null,3,2,4,null,5,6]
Output: [5,6,3,2,4,1]
'''
class Solution:
    def __init__(self):
        self.result = []
        
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return self.result
        for child in root.children:
            if child:
                self.postorder(child)
        self.result.append(root.val)
        return self.result

'''
594. Longest Harmonious Subsequence
We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly 1.
Given an integer array nums, return the length of its longest harmonious subsequence among all its possible subsequences.
A subsequence of array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.
Example:
Input: nums = [1,3,2,2,5,2,3,7]
Output: 5
'''
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

'''
599. Minimum Index Sum of Two Lists
Given two arrays of strings list1 and list2, find the common strings with the least index sum.
A common string is a string that appeared in both list1 and list2.
A common string with the least index sum is a common string such that if it appeared at list1[i] and list2[j] then i + j should be the minimum value among all the other common strings.
Return all the common strings with the least index sum. Return the answer in any order.
Example:
Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["Piatti","The Grill at Torrey Pines","Hungry Hunter Steakhouse","Shogun"]
Output: ["Shogun"]
'''
class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        minsum = len(list1) + len(list2)
        minword = []
        dic1, dic2 = collections.defaultdict(int), collections.defaultdict(int)
        for i, word in enumerate(list1):
            dic1[word] = i
        for i, word in enumerate(list2):
            dic2[word] = i
        intersec = set(dic1.keys()) & set(dic2.keys())
        for word in intersec:
            if dic1[word]>minsum or dic2[word]>minsum:
                continue
            if dic1[word]+dic2[word] < minsum:
                minsum = dic1[word]+dic2[word]
                minword = [word]
            elif dic1[word]+dic2[word] == minsum:
                minword.append(word)
        return minword

'''
605. Can Place Flowers
You have a long flowerbed in which some of the plots are planted, and some are not. 
However, flowers cannot be planted in adjacent plots.
Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule.
Example:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
'''
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        prev = 0
        for flower in flowerbed:
            if flower == 1:
                if prev == 1:
                    n += 1
                prev = 1
            else:
                if prev == 1:
                    prev = 0
                else:
                    n -= 1
                    prev = 1
        return n <= 0

'''
617. Merge Two Binary Trees
You are given two binary trees root1 and root2.
Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. 
You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. 
Otherwise, the NOT null node will be used as the node of the new tree.
Return the merged tree.
Note: The merging process must start from the root nodes of both trees.
Example:
Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
'''
class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        if not root1:
            return root2
        if not root2: 
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1

'''
621. Task Scheduler
Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.
However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.
Return the least number of units of times that the CPU will take to finish all the given tasks.
Example:
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
'''
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = collections.Counter(tasks)
        freq = sorted(list(freq.items()), key = lambda x: x[1])
        maxFreq = freq[-1][1] - 1
        idleSlots = maxFreq * n
        for i in range(len(freq)-2, -1, -1):
            if freq[i][1] > 0:
                idleSlots -= min(maxFreq, freq[i][1])
        return idleSlots + len(tasks) if idleSlots > 0 else len(tasks)

'''
637. Average of Levels in Binary Tree
Given the root of a binary tree, return the average value of the nodes on each level in the form of an array.
Answers within 10-5 of the actual answer will be accepted.
Example:
Input: root = [3,9,20,null,null,15,7]
Output: [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].
'''
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        result = []
        layer = deque()
        layer.append(root)
        while layer:
            lst = []
            layerSum = 0.0
            layerNum = len(layer)
            for i in range(len(layer)):
                cur = layer.popleft()
                layerSum += cur.val
                lst.append(cur.val)
                if cur.left:
                    layer.append(cur.left)
                if cur.right:
                    layer.append(cur.right)
            result.append(layerSum/layerNum)
        return result

'''
647. Palindromic Substrings
Given a string s, return the number of palindromic substrings in it.
A string is a palindrome when it reads the same backward as forward.
A substring is a contiguous sequence of characters within the string.
Example:
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
'''
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        opt = [[0 for _ in range(len(s))] for _ in range(len(s))]
        result = 0
        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j]:
                    if j-i <2:
                        opt[i][j] = 1
                        result += 1
                    elif (opt[i+1][j-1]):
                        opt[i][j] = 1
                        result += 1
        return result

'''
649. Dota2 Senate
In the world of Dota2, there are two parties: the Radiant and the Dire.
The Dota2 senate consists of senators coming from two parties. Now the Senate wants to decide on a change in the Dota2 game. The voting for this change is a round-based procedure. In each round, each senator can exercise one of the two rights:
Ban one senator's right: A senator can make another senator lose all his rights in this and all the following rounds.
Announce the victory: If this senator found the senators who still have rights to vote are all from the same party, he can announce the victory and decide on the change in the game.
Given a string senate representing each senator's party belonging. The character 'R' and 'D' represent the Radiant party and the Dire party. Then if there are n senators, the size of the given string will be n.
The round-based procedure starts from the first senator to the last senator in the given order. This procedure will last until the end of voting. All the senators who have lost their rights will be skipped during the procedure.
Suppose every senator is smart enough and will play the best strategy for his own party. Predict which party will finally announce the victory and change the Dota2 game. The output should be "Radiant" or "Dire".
Example:
Input: senate = "RD"
Output: "Radiant"
Explanation: 
The first senator comes from Radiant and he can just ban the next senator's right in round 1. 
And the second senator can't exercise any rights anymore since his right has been banned. 
And in round 2, the first senator can just announce the victory since he is the only guy in the senate who can vote.
'''
class Solution(object):
    def predictPartyVictory(self, senate):
        """
        :type senate: str
        :rtype: str
        """
        Radiant, Dire = True, True
        flag = 0
        senate = list(senate)
        while Radiant and Dire:
            Radiant, Dire = False, False
            for i in range(len(senate)):
                if senate[i] == 'R':
                    if flag<0:
                        senate[i] = '0'
                    else: 
                        Radiant = True
                    flag+=1
                if senate[i] == 'D':
                    if flag>0:
                        senate[i] = '0'
                    else:
                        Dire = True
                    flag-=1
        if Radiant:
            return 'Radiant'
        else:
            return 'Dire'


'''
654. Maximum Binary Tree
You are given an integer array nums with no duplicates. A maximum binary tree can be built recursively from nums using the following algorithm:
Create a root node whose value is the maximum value in nums.
Recursively build the left subtree on the subarray prefix to the left of the maximum value.
Recursively build the right subtree on the subarray suffix to the right of the maximum value.
Return the maximum binary tree built from nums.
Example:
Input: nums = [3,2,1,6,0,5]
Output: [6,3,5,null,2,0,null,null,1]
'''
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        maxvalue = max(nums)
        index = nums.index(maxvalue)
        root = TreeNode(maxvalue)
        left = nums[:index]
        right = nums[index + 1:]
        root.left = self.constructMaximumBinaryTree(left)
        root.right = self.constructMaximumBinaryTree(right)
        return root

'''
657. Robot Return to Origin
There is a robot starting at the position (0, 0), the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves.
You are given a string moves that represents the move sequence of the robot where moves[i] represents its ith move. Valid moves are 'R' (right), 'L' (left), 'U' (up), and 'D' (down).
Return true if the robot returns to the origin after it finishes all of its moves, or false otherwise.
Note: The way that the robot is "facing" is irrelevant. 'R' will always make the robot move to the right once, 'L' will always make it move left, etc. Also, assume that the magnitude of the robot's movement is the same for each move.
Example:
Input: moves = "UD"
Output: true
Explanation: The robot moves up once, and then down once. All moves have the same magnitude, so it ended up at the origin where it started. Therefore, we return true.
'''
class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        count = collections.Counter(moves)
        return count['U'] == count['D'] and count['L'] == count['R']

'''
661. Image Smoother
An image smoother is a filter of the size 3 x 3 that can be applied to each cell of an image by rounding down the average of the cell and the eight surrounding cells (i.e., the average of the nine cells in the blue smoother).
If one or more of the surrounding cells of a cell is not present, we do not consider it in the average (i.e., the average of the four cells in the red smoother).
Given an m x n integer matrix img representing the grayscale of an image, return the image after applying the smoother on each cell of it.
Example:
Input: img = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[0,0,0],[0,0,0],[0,0,0]]
'''
class Solution(object):
    def imageSmoother(self, img):
        """
        :type img: List[List[int]]
        :rtype: List[List[int]]
        """
        dirs = [[0,1,],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]
        m, n = len(img), len(img[0])
        if m == 1 and n==1: return img
        result = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                weight, count = img[i][j], 1
                for d in dirs:
                    x,y = i+d[0],j+d[1]
                    if x >= 0 and x < m and y >= 0 and y < n:
                        weight += img[x][y]
                        count += 1
                result[i][j] = int(weight / count)
        return result

'''
662. Maximum Width of Binary Tree
Given the root of a binary tree, return the maximum width of the given tree.
The maximum width of a tree is the maximum width among all levels.
The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that level are also counted into the length calculation.
It is guaranteed that the answer will in the range of a 32-bit signed integer.
Example:
Input: root = [1,3,2,5,3,null,9]
Output: 4
'''
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        queue = deque([(root, 0)])
        max_width = 0
        
        while queue:
            level_length = len(queue)
            _, level_start = queue[0]
            for i in range(level_length):
                node, index = queue.popleft()
                if node.left:
                    queue.append((node.left, 2*index))
                if node.right:
                    queue.append((node.right, 2*index+1))
            max_width = max(max_width, index - level_start + 1)
        return max_width

'''
665. Non-decreasing Array
Given an array nums with n integers, your task is to check if it could become non-decreasing by modifying at most one element.
We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for every i (0-based) such that (0 <= i <= n - 2).
Example:
Input: nums = [4,2,3]
Output: true
'''
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        count = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                if count or (1<i<len(nums)-1 and nums[i-2]>nums[i] and nums[i-1]>nums[i+1]):
                    return False
                count = 1
        return True

'''
669. Trim a Binary Search Tree
Given the root of a binary search tree and the lowest and highest boundaries as low and high, trim the tree so that all its elements lies in [low, high]. 
Trimming the tree should not change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). 
It can be proven that there is a unique answer.
Return the root of the trimmed binary search tree. Note that the root may change depending on the given bounds.
Example:
Input: root = [1,0,2], low = 1, high = 2
Output: [1,null,2]
'''
class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: TreeNode
        """
        if not root:
            return None
        if root.val < low:
            return self.trimBST(root.right, low, high)
        elif root.val > high:
            return self.trimBST(root.left, low, high)
        else:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
        return root

'''
673. Number of Longest Increasing Subsequence
Given an integer array nums, return the number of longest increasing subsequences.
Notice that the sequence has to be strictly increasing.
Example:
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].
'''
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        maxlength, res = 0, 0
        opt = [1 for _ in range(len(nums))]
        count = [1 for _ in range(len(nums))]
        for i, num in enumerate(nums):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    if opt[j]+1 > opt[i]:
                        count[i]=count[j]
                        opt[i] = opt[j]+1
                    elif opt[j]+1 == opt[i]:
                        count[i]+=count[j]
            if maxlength == opt[i]:
                res += count[i]
            elif maxlength < opt[i]:
                maxlength = opt[i]
                res = count[i]
        return res

'''
674. Longest Continuous Increasing Subsequence
Given an unsorted array of integers nums, return the length of the longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.
A continuous increasing subsequence is defined by two indices l and r (l < r) such that it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].
Example:
Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element 4.
'''
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        opt = [1 for _ in range(len(nums))]
        result = 0
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                opt[i] = opt[i-1] + 1
            result = max(result, opt[i]) 
        return result

'''
678. Valid Parenthesis String
Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
The following rules define a valid string:
Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
Example:
Input: s = "()"
Output: true
'''
class Solution(object):
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for char in s:
            if char=='(' or char=='*':
                stack.append(char)
            else:
                if len(stack)>0:
                    stack.pop()
                else:
                    return False
        stack = []
        for char in s[::-1]:
            if char==')' or char=='*':
                stack.append(char)
            else:
                if len(stack)>0:
                    stack.pop()
                else:
                    return False     
                
        return True

'''
680. Valid Palindrome II
Given a string s, return true if the s can be palindrome after deleting at most one character from it.
Example:
Input: s = "aba"
Output: true
'''
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i, j = 0, len(s)-1
        while i<j:
            if s[i] != s[j]:
                case1, case2 = s[i:j], s[i+1:j+1]
                return case1==case1[::-1] or case2==case2[::-1] 
            i,j = i+1,j-1
        return True

'''
682. Baseball Game
You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.
You are given a list of strings operations, where operations[i] is the ith operation you must apply to the record and is one of the following:
An integer x.
Record a new score of x.
'+'.
Record a new score that is the sum of the previous two scores.
'D'.
Record a new score that is the double of the previous score.
'C'.
Invalidate the previous score, removing it from the record.
Return the sum of all the scores on the record after applying all the operations.
The test cases are generated such that the answer and all intermediate calculations fit in a 32-bit integer and that all operations are valid.
Example:
Input: ops = ["5","2","C","D","+"]
Output: 30
'''
class Solution(object):
    def calPoints(self, operations):
        """
        :type operations: List[str]
        :rtype: int
        """
        stack = []
        for oper in operations:
            if oper == '+':
                stack.append(stack[-1]+stack[-2])
            elif oper == 'D':
                stack.append(2*stack[-1])
            elif oper == 'C':
                stack.pop()
            else:
                stack.append(int(oper))
        return sum(stack)

'''
684. Redundant Connection
In this problem, a tree is an undirected graph that is connected and has no cycles.
You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. 
The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.
Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.
Example:
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]
'''
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

'''
685. Redundant Connection II
In this problem, a rooted tree is a directed graph such that, there is exactly one node (the root) for which all other nodes are descendants of this node, plus every node has exactly one parent, except for the root node which has no parents.
The given input is a directed graph that started as a rooted tree with n nodes (with distinct values from 1 to n), with one additional directed edge added. 
The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed.
The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [ui, vi] that represents a directed edge connecting nodes ui and vi, where ui is a parent of child vi.
Return an edge that can be removed so that the resulting graph is a rooted tree of n nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.
Example:
Input: edges = [[1,2]
'''
class Solution(object):
    def __init__(self):
        self.parents = None
        self.count = None
        self.numNodes = 0
        self.edgeRemoved = -1
        self.edgeMakesCycle = -1
        
    def findRedundantDirectedConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        self.numNodes = len(edges)
        self.parents = [i for i in range(self.numNodes+1)]
        self.count = [1 for i in range(self.numNodes+1)]
        p = [0 for i in range(self.numNodes+1)]
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
    
    def find(self, node):
        """
        :type node: int
        :rtype: int
        """
        while(node != self.parents[node]):
            node = self.parents[node]
        return node
    
    def union(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: boolean
        """
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.count[a_parent], self.count[b_parent]
        if a_parent == b_parent:
            return False
        if a_size < b_size:
            self.parents[a_parent] = b_parent
            self.count[b] += a_size
        else:
            self.parents[b_parent] = a_parent
            self.count[a] += b_size
        return True

'''
690. Employee Importance
You have a data structure of employee information, including the employee's unique ID, importance value, and direct subordinates' IDs.
You are given an array of employees employees where:
employees[i].id is the ID of the ith employee.
employees[i].importance is the importance value of the ith employee.
employees[i].subordinates is a list of the IDs of the direct subordinates of the ith employee.
Given an integer id that represents an employee's ID, return the total importance value of this employee and all their direct and indirect subordinates.
Example:
Input: employees = [[1,5,[2,3]],[2,3,[]],[3,3,[]]], id = 1
Output: 11
'''
"""
# Definition for Employee.
class Employee(object):
    def __init__(self, id, importance, subordinates):
    	#################
        :type id: int
        :type importance: int
        :type subordinates: List[int]
        #################
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""
class Solution(object):
    def getImportance(self, employees, id):
        """
        :type employees: List[Employee]
        :type id: int
        :rtype: int
        """
        
        totalimp = 0
        empdic = {}
        for employee in employees:
            empdic[employee.id] = employee
        stack = [id]
        while stack:
            empid = stack.pop()
            totalimp += empdic[empid].importance
            stack += empdic[empid].subordinates
        return totalimp

'''
692. Top K Frequent Words
Given an array of strings words and an integer k, return the k most frequent strings.
Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.
Example:
Input: words = ["i","love","leetcode","i","love","coding"], k = 2
Output: ["i","love"]
Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.
'''
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        freq = collections.Counter(words)
        res = sorted(freq, key=lambda x: (-freq[x], x))
        return res[:k]

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        freq = collections.Counter(words)
        res, heap = [], []
        for word,times in freq.items():
            heapq.heappush(heap, (-times, word))
        return [heapq.heappop(heap)[1] for _ in range(k)]

'''
693. Binary Number with Alternating Bits
Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have different values.
Example:
Input: n = 5
Output: true
'''
class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        num = n^(n>>1)
        return not (num&(num+1))

'''
696. Count Binary Substrings
Given a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.
Substrings that occur multiple times are counted the number of times they occur.
Example:
Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
'''
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        groups = list(map(len, re.findall('0+|1+', s)))
        return sum(min(a, b) for a, b in zip(groups, groups[1:]))

'''
697. Degree of an Array
Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency of any one of its elements.
Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.
Example:
Input: nums = [1,2,2,3,1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
'''
class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        freq = collections.Counter(nums)
        left, right = {}, {}
        for i, num in enumerate(nums):
            if num not in left:
                left[num] = i
            right[num] = i
        degree, res = max(freq.values()), len(nums)
        for num, count in freq.items():
            if count == degree:
                res = min(res, right[num]-left[num]+1)
        return res

##700##       

'''
700. Search in a Binary Search Tree
You are given the root of a binary search tree (BST) and an integer val.
Find the node in the BST that the node's value equals val and return the subtree rooted with that node. If such a node does not exist, return null.
Example:
Input: root = [4,2,7,1,3], val = 2
Output: [2,1,3]
'''
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root: return None
        if root.val == val:
            return root
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

'''
701. Insert into a Binary Search Tree
You are given the root node of a binary search tree (BST) and a value to insert into the tree. 
Return the root node of the BST after the insertion. 
It is guaranteed that the new value does not exist in the original BST.
Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. 
You can return any of them.
Example:
Input: root = [4,2,7,1,3], val = 5
Output: [4,2,7,1,3,5]
'''
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root:
            return TreeNode(val)
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        elif root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        return root

'''
703. Kth Largest Element in a Stream
Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.
Implement KthLargest class:
KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of integers nums.
int add(int val) Appends the integer val to the stream and returns the element representing the kth largest element in the stream.
Example:
Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]
'''
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = []
        for num in nums:
            self.add(num)
        
    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

'''
704. Binary Search
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums.
If target exists, then return its index. Otherwise, return -1.
You must write an algorithm with O(log n) runtime complexity.
Example:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
'''
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        a, b = 0, len(nums)-1
        while a <= b:
            mid = (a+b) // 2
            if target < nums[mid]:
                b = mid-1
            elif target > nums[mid]:
                a = mid+1
            else:
                return mid
        return -1

'''
705. Design HashSet
Design a HashSet without using any built-in hash table libraries.
Implement MyHashSet class:
void add(key) Inserts the value key into the HashSet.
bool contains(key) Returns whether the value key exists in the HashSet or not.
void remove(key) Removes the value key in the HashSet. If key does not exist in the HashSet, do nothing.
Example:
Input
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
Output
[null, null, null, true, false, null, true, null, false]
'''
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


'''
706. Design HashMap
Design a HashMap without using any built-in hash table libraries.
Implement the MyHashMap class:
MyHashMap() initializes the object with an empty map.
void put(int key, int value) inserts a (key, value) pair into the HashMap. If the key already exists in the map, update the corresponding value.
int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.
Example:
Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]
'''
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

'''
707. Design Linked List
Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node.
If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement the MyLinkedList class:

MyLinkedList() Initializes the MyLinkedList object.
int get(int index) Get the value of the indexth node in the linked list. If the index is invalid, return -1.
void addAtHead(int val) Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
void addAtTail(int val) Append a node of value val as the last element of the linked list.
void addAtIndex(int index, int val) Add a node of value val before the indexth node in the linked list. If index equals the length of the linked list, the node will be appended to the end of the linked list. If index is greater than the length, the node will not be inserted.
void deleteAtIndex(int index) Delete the indexth node in the linked list, if the index is valid.

Example:
Input
["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
[[], [1], [3], [1, 2], [1], [1], [1]]
Output
[null, null, null, null, 2, null, 3]
'''
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class MyLinkedList(object):

    def __init__(self):
        self.head = Node(0)
        self.count = 0

    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if 0 <= index < self.count:
            node = self.head
            for _ in range(index):
                node = node.next
            return node.val
        else:
            return -1

    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.addAtIndex(0, val)
        

    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.addAtIndex(self.count, val)

    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        if index < 0: 
            index = 0
        if index > self.count: 
            return
        if index == 0:
            new_node = Node(val)
            if self.count == 0:
                self.head = new_node
            else:
                new_node.next = self.head
                self.head = new_node
            self.count += 1
        else:
            self.count += 1
            new_node = Node(val)
            previous_node = None
            current_node = self.head
            
            while index != 0:
                previous_node = current_node
                current_node = current_node.next
                index -= 1
            previous_node.next, new_node.next = new_node, current_node
   
    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if 0 < index < self.count:
            self.count -= 1
            previous_node, current_node = None, self.head
            while index != 0:
                previous_node, current_node = current_node, current_node.next
                index -= 1
            previous_node.next = current_node.next
        elif index == 0:
            self.head = self.head.next
            self.count -= 1

    def print_all(self):
        current_node = self.head
        while current_node is not None:
            print(current_node.val, end = ' ')
            current_node = current_node.next

'''
709. To Lower Case
Given a string s, return the string after replacing every uppercase letter with the same lowercase letter.
Example:
Input: s = "Hello"
Output: "hello"
'''
class Solution(object):
    def toLowerCase(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ''.join(chr(ord(char)+32) if 65<=ord(char)<=90 else char for char in s)

'''
714. Best Time to Buy and Sell Stock with Transaction Fee
You are given an array prices where prices[i] is the price of a given stock on the ith day, and an integer fee representing a transaction fee.
Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Example:
Input: prices = [1,3,2,8,4,9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
- Buying at prices[0] = 1
- Selling at prices[3] = 8
- Buying at prices[4] = 4
- Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
'''
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = -prices[0] #持股票
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee)
        return max(dp[-1][0], dp[-1][1])

class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        opt = [-prices[0],0]
        for i in range(1,len(prices)):
            temp0 = opt[0]
            opt[0] = max(opt[0], opt[1] - prices[i])
            opt[1] = max(opt[1], temp0 + prices[i] - fee)
        return max(opt)

'''
718. Maximum Length of Repeated Subarray
Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears in both arrays.
Example:
Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
Output: 3
Explanation: The repeated subarray with maximum length is [3,2,1].
'''
class Solution(object):
    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        opt = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]
        result = 0
        for i in range(1, len(nums1)+1):
            for j in range(1, len(nums2)+1):
                if nums1[i-1] == nums2[j-1]:
                    opt[i][j] = opt[i-1][j-1] + 1
                result = max(result, opt[i][j]) 
        return result

'''
720. Longest Word in Dictionary
Given an array of strings words representing an English Dictionary, return the longest word in words that can be built one character at a time by other words in words.
If there is more than one possible answer, return the longest word with the smallest lexicographical order. If there is no answer, return the empty string.
Note that the word should be built from left to right with each additional character being added to the end of a previous word. 
Example:
Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
'''
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

'''
724. Find Pivot Index
Given an array of integers nums, calculate the pivot index of this array.
The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.
If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left. This also applies to the right edge of the array.
Return the leftmost pivot index. If no such index exists, return -1.
Example:
Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The pivot index is 3.
Left sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11
Right sum = nums[4] + nums[5] = 5 + 6 = 11
'''
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        leftsum, rightsum = 0, sum(nums)
        for i, num in enumerate(nums):
            rightsum -= num
            if leftsum == rightsum:
                return i
            leftsum += num
        return -1

'''
725. Split Linked List in Parts
Given the head of a singly linked list and an integer k, split the linked list into k consecutive linked list parts.
The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.
The parts should be in the order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal to parts occurring later.
Return an array of the k parts.
Example:
Input: head = [1,2,3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The first element output[0] has output[0].val = 1, output[0].next = null.
The last element output[4] is null, but its string representation as a ListNode is [].
'''
class Solution(object):
    def splitListToParts(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: List[ListNode]
        """
        length, current = 0, head
        while current:
            current, length = current.next, length+1
        size, remain = divmod(length, k)
        result = [size + 1] * remain + [size] * (k - remain)
        pre, cur = None, head
        for index, num in enumerate(result):
            if pre:
                pre.next = None
            result[index] = cur
            for i in range(num):
                pre, cur = cur, cur.next
        return result

'''
738. Monotone Increasing Digits
An integer has monotone increasing digits if and only if each pair of adjacent digits x and y satisfy x <= y.
Given an integer n, return the largest number that is less than or equal to n with monotone increasing digits.
Example:
Input: n = 10
Output: 9
'''
class Solution(object):
    def monotoneIncreasingDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        listNum = list(str(n))
        for i in range(len(listNum)-1, 0,-1):
            if int(listNum[i]) < int(listNum[i-1]):
                listNum[i-1] = str(int(listNum[i-1])-1)
                listNum[i:] = '9' * (len(listNum) - i)
        return int("".join(listNum)) 

'''
739. Daily Temperatures
Given an array of integers temperatures represents the daily temperatures, 
return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.
If there is no future day for which this is possible, keep answer[i] == 0 instead.
Example:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
'''
class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        result = [0 for _ in range(len(temperatures))]
        stack = []
        for i,temp in enumerate(temperatures):
            while(len(stack)):
                if temp > stack[-1][0]:
                    result[stack[-1][1]] = i-stack[-1][1]
                    stack.pop()
                else:
                    break
            stack.append([temp,i])
        return result

'''
744. Find Smallest Letter Greater Than Target
You are given an array of characters letters that is sorted in non-decreasing order, and a character target. There are at least two different characters in letters.
Return the smallest character in letters that is lexicographically greater than target. If such a character does not exist, return the first character in letters.
Example:
Input: letters = ["c","f","j"], target = "a"
Output: "c"
'''
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        left, right = 0, len(letters)-1
        while left <= right:
            mid = left+(right-left)//2
            if letters[mid] <= target:
                left = mid+1
            else:
                right = mid-1
        return letters[left] if left < len(letters) else letters[0]

'''
746. Min Cost Climbing Stairs
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.
You can either start from the step with index 0, or the step with index 1.
Return the minimum cost to reach the top of the floor.
Example:
Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.
'''
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        opt = [0 for _ in range(len(cost) + 1)]
        for i in range(2, len(cost) + 1):
            opt[i] = min(opt[i-1] + cost[i-1], opt[i-2] + cost[i-2])
            #print('opt{}:{}+{},{}+{}'.format(i, opt[i-1], cost[i-1], opt[i-2], cost[i-2]))
        return opt[-1]

'''
747. Largest Number At Least Twice of Others
You are given an integer array nums where the largest integer is unique.
Determine whether the largest element in the array is at least twice as much as every other number in the array. If it is, return the index of the largest element, or return -1 otherwise.
Example:
Input: nums = [3,6,1,0]
Output: 1
'''
class Solution(object):
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max1, max2 = -float('inf'), -float('inf')
        index = None
        for i in range(len(nums)):
            if nums[i] > max1:
                max1, max2 = nums[i], max1
                index = i
            elif nums[i] < max2:
                continue
            else:
                max2 = nums[i]
        if max1 >= 2*max2: return index
        return -1

'''
748. Shortest Completing Word
Given a string licensePlate and an array of strings words, find the shortest completing word in words.
A completing word is a word that contains all the letters in licensePlate. Ignore numbers and spaces in licensePlate, and treat letters as case insensitive.
If a letter appears more than once in licensePlate, then it must appear in the word the same number of times or more.
For example, if licensePlate = "aBc 12c", then it contains letters 'a', 'b' (ignoring case), and 'c' twice. Possible completing words are "abccdef", "caaacab", and "cbca".
Return the shortest completing word in words. It is guaranteed an answer exists. If there are multiple shortest completing words, return the first one that occurs in words.
Example:
Input: licensePlate = "1s3 PSt", words = ["step","steps","stripe","stepple"]
Output: "steps"
Explanation: licensePlate contains letters 's', 'p', 's' (ignoring case), and 't'.
"step" contains 't' and 'p', but only contains 1 's'.
"steps" contains 't', 'p', and both 's' characters.
"stripe" is missing an 's'.
"stepple" is missing an 's'.
Since "steps" is the only word containing all the letters, that is the answer.
'''
class Solution(object):
    def shortestCompletingWord(self, licensePlate, words):
        """
        :type licensePlate: str
        :type words: List[str]
        :rtype: str
        """
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

'''
733. Flood Fill
An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.
You are also given three integers sr, sc, and color. You should perform a flood fill on the image starting from the pixel image[sr][sc].
To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with color.
Return the modified image after performing the flood fill.
Example:
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
'''
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        visited = [[False for _ in range(len(image[0]))] for _ in range(len(image))]
        self.dfs(image,sr,sc,color,image[sr][sc],visited)
        return image

    def dfs(self, image, sr, sc, color, ori, visited):
        if sr>=len(image) or sc>=len(image[0]): return
        if image[sr][sc] != ori: return
        image[sr][sc] = color
        visited[sr][sc] = True
        if sr-1>=0 and not visited[sr-1][sc]:
            self.dfs(image, sr-1, sc, color, ori, visited)
        if sr+1<len(image) and not visited[sr+1][sc]:
            self.dfs(image, sr+1, sc, color, ori, visited)
        if sc-1>=0 and not visited[sr][sc-1]:
            self.dfs(image, sr, sc-1, color, ori, visited)
        if sc+1<len(image[0]) and not visited[sr][sc+1]:
            self.dfs(image, sr, sc+1, color, ori, visited)

'''
763. Partition Labels
You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.
Note that the partition is done so that after concatenating all the parts in order, the resultant string should be s.
Return a list of integers representing the size of these parts.
Example:
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
'''
class Solution(object):
    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        dic = [0 for i in range(26)]
        for i in range(len(s)):
            dic[ord(s[i]) - ord('a')] = i
        result = []
        left = 0
        right = 0
        for i in range(len(s)):
            right = max(right, dic[ord(s[i]) - ord('a')])
            if i == right:
                result.append(right - left + 1)
                left = i + 1
        return result

'''
766. Toeplitz Matrix
Given an m x n matrix, return true if the matrix is Toeplitz. Otherwise, return false.
A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.
Example:
Input: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
Output: true
'''
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        for i in range(1,len(matrix)):
            for j in range(1,len(matrix[0])):
                if matrix[i][j] != matrix[i-1][j-1]:
                    return False
        return True

'''
767. Reorganize String
Given a string s, rearrange the characters of s so that any two adjacent characters are not the same.
Return any possible rearrangement of s or return "" if not possible.
Example 1:
Input: s = "aab"
Output: "aba"
'''
class Solution:
    def reorganizeString(self, s: str) -> str:
        counter = collections.Counter(s)
        pq = []
        for char, count in counter.items():
            heapq.heappush(pq, (-count, char))
        res = ""
        pre = None
        while pq:
            count, char = heapq.heappop(pq)
            count += 1
            res += char
            if (pre and pre[0] < 0):
                heapq.heappush(pq, pre)
            pre = (count, char)
        return res if len(res)==len(s) else ""

'''
771. Jewels and Stones
You're given strings jewels representing the types of stones that are jewels, and stones representing the stones you have. Each character in stones is a type of stone you have. You want to know how many of the stones you have are also jewels.
Letters are case sensitive, so "a" is considered a different type of stone from "A".
Example:
Input: jewels = "aA", stones = "aAAbbbb"
Output: 3
'''
class Solution(object):
    def numJewelsInStones(self, jewels, stones):
        """
        :type jewels: str
        :type stones: str
        :rtype: int
        """
        jewels = set(jewels)
        return sum(stone in jewels for stone in stones)

'''
777. Swap Adjacent in LR String
In a string composed of 'L', 'R', and 'X' characters, like "RXXLRXRXL", a move consists of either replacing one occurrence of "XL" with "LX", or replacing one occurrence of "RX" with "XR". Given the starting string start and the ending string end, return True if and only if there exists a sequence of moves to transform one string to the other.
Example:
Input: start = "RXXLRXRXL", end = "XRLXXRRLX"
Output: true
'''
class Solution(object):
    def canTransform(self, start, end):
        """
        :type start: str
        :type end: str
        :rtype: bool
        """
        start = [(c, i) for i, c in enumerate(start) if c!='X']
        end = [(c, i) for i, c in enumerate(end) if c!='X']
        if len(start) != len(end):
            return False
        for (c1, i), (c2, j) in zip(start, end):
            if c1 != c2: return False
            if c1 == 'L' and i < j: return False
            if c1 == 'R' and i > j: return False
        return True

'''
784. Letter Case Permutation
Given a string s, you can transform every letter individually to be lowercase or uppercase to create another string.
Return a list of all possible strings we could create. Return the output in any order.
Example:
Input: s = "a1b2"
Output: ["a1b2","a1B2","A1b2","A1B2"]
'''
class Solution(object):
    def __init__(self):
        self.result = []
    
    def letterCasePermutation(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def traverse(lst, ind):
            if ind >= len(lst):
                self.result.append(''.join(lst))
            elif lst[ind].isalpha():
                lst[ind] = lst[ind].lower()
                traverse(lst, ind+1)
                lst[ind] = lst[ind].upper()
                traverse(lst, ind+1)
            else:
                traverse(lst, ind+1)
        traverse(list(s), 0)
        return self.result

'''
785. Is Graph Bipartite?
There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1. You are given a 2D array graph, where graph[u] is an array of nodes that node u is adjacent to. More formally, for each v in graph[u], there is an undirected edge between node u and node v. The graph has the following properties:
There are no self-edges (graph[u] does not contain u).
There are no parallel edges (graph[u] does not contain duplicate values).
If v is in graph[u], then u is in graph[v] (the graph is undirected).
The graph may not be connected, meaning there may be two nodes u and v such that there is no path between them.
A graph is bipartite if the nodes can be partitioned into two independent sets A and B such that every edge in the graph connects a node in set A and a node in set B.
Return true if and only if it is bipartite.
Example:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
'''
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        col = [-1] * (n+1)
        for i in range(n):
            if col[i] == -1:
                if not self.bfs(i, 0, col, graph):
                    return False
        return True

    def bfs(self, node: int, c: int, col: List[int], graph: List[List[int]]) -> bool:
        q = deque()
        q.append(node)
        col[node] = c
        while q:
            parent = q.popleft()
            for child in graph[parent]:
                if col[child] == -1:
                    col[child] = 1 - col[parent]
                    q.append(child)
                elif col[child] == col[parent]:
                    return False
        return True

'''
788. Rotated Digits
An integer x is a good if after rotating each digit individually by 180 degrees, we get a valid number that is different from x. Each digit must be rotated - we cannot choose to leave it alone.
A number is valid if each digit remains a digit after rotation. For example:
0, 1, and 8 rotate to themselves,
2 and 5 rotate to each other (in this case they are rotated in a different direction, in other words, 2 or 5 gets mirrored),
6 and 9 rotate to each other, and
the rest of the numbers do not rotate to any other number and become invalid.
Given an integer n, return the number of good integers in the range [1, n].
Example:
Input: n = 10
Output: 4
Explanation: There are four good numbers in the range [1, 10] : 2, 5, 6, 9.
Note that 1 and 10 are not good numbers, since they remain unchanged after rotating.
'''
class Solution(object):
    def rotatedDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        validdic = {'2','5','6','9', '0', '1', '8'}
        unchange = {'0', '1', '8'}
        result = 0
        for i in range(1,n+1):
            temp = set(list(str(i)))
            if temp <= validdic and not temp <= unchange:
                result += 1
        return result

'''
791. Custom Sort String
You are given two strings order and s. All the characters of order are unique and were sorted in some custom order previously.
Permute the characters of s so that they match the order that order was sorted. More specifically, if a character x occurs before a character y in order, then x should occur before y in the permuted string.
Return any permutation of s that satisfies this property.
Example:
Input: order = "cba", s = "abcd"
Output: "cbad"
'''
class Solution(object):
    def customSortString(self, order, s):
        """
        :type order: str
        :type s: str
        :rtype: str
        """
        dic = collections.Counter(s)
        return ''.join([dic[char]*char for char in order if char in dic]) + ''.join([dic[char]*char for char in dic.keys() if char not in order])

'''
796. Rotate String
Given two strings s and goal, return true if and only if s can become goal after some number of shifts on s.
A shift on s consists of moving the leftmost character of s to the rightmost position.
For example, if s = "abcde", then it will be "bcdea" after one shift.
Example:
Input: s = "abcde", goal = "cdeab"
Output: true
'''
class Solution(object):
    def rotateString(self, s, goal):
        """
        :type s: str
        :type goal: str
        :rtype: bool
        """
        if not len(s)==len(goal):
            return False
        if s in goal+goal:
            return True
        return False

##800##
'''
804. Unique Morse Code Words
International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows:
'a' maps to ".-",
'b' maps to "-...",
'c' maps to "-.-.", and so on.
For convenience, the full table for the 26 letters of the English alphabet is given below:
[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
Given an array of strings words where each word can be written as a concatenation of the Morse code of each letter.
For example, "cab" can be written as "-.-..--...", which is the concatenation of "-.-.", ".-", and "-...". We will call such a concatenation the transformation of a word.
Return the number of different transformations among all words we have.
Example:
Input: words = ["gin","zen","gig","msg"]
Output: 2
'''
class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        transformations = []
        for word in words:
            transformations.append(''.join([morse[ord(char)-ord('a')] for char in word]))
        return len(set(transformations))

'''
805. Split Array With Same Average
You are given an integer array nums.
You should move each element of nums into one of the two arrays A and B such that A and B are non-empty, and average(A) == average(B).
Return true if it is possible to achieve that and false otherwise.
Note that for an array arr, average(arr) is the sum of all the elements of arr over the length of arr.
Example:
Input: nums = [1,2,3,4,5,6,7,8]
Output: true
'''
class Solution(object):
    def splitArraySameAverage(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        lengthArr, totalSum = len(nums), sum(nums)
        midian = totalSum / lengthArr
        arr = [num*lengthArr-totalSum for num in nums]
        left, right = arr[:lengthArr//2], arr[lengthArr//2:]
        innerset = set()
        for i in range(1, lengthArr//2+1):
            for inner in combinations(left, i):
                innersum = sum(inner)
                if innersum == 0: return True
                else: innerset.add(innersum)
        for i in range(1, lengthArr-len(left)):
            for inner in combinations(right, i):
                innersum = sum(inner)
                if innersum == 0: return True
                elif -innersum in innerset:
                    return True
        return False

'''
806. Number of Lines To Write String
You are given a string s of lowercase English letters and an array widths denoting how many pixels wide each lowercase English letter is. Specifically, widths[0] is the width of 'a', widths[1] is the width of 'b', and so on.
You are trying to write s across several lines, where each line is no longer than 100 pixels. Starting at the beginning of s, write as many letters on the first line such that the total width does not exceed 100 pixels. Then, from where you stopped in s, continue writing as many letters as you can on the second line. Continue this process until you have written all of s.
Return an array result of length 2 where:
result[0] is the total number of lines.
result[1] is the width of the last line in pixels.
Example:
Input: widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], s = "abcdefghijklmnopqrstuvwxyz"
Output: [3,60]
Explanation: You can write s as follows:
abcdefghij  // 100 pixels wide
klmnopqrst  // 100 pixels wide
uvwxyz      // 60 pixels wide
There are a total of 3 lines, and the last line is 60 pixels wide.
'''
class Solution(object):
    def numberOfLines(self, widths, s):
        """
        :type widths: List[int]
        :type s: str
        :rtype: List[int]
        """
        line,width=0,0
        ind=0
        while ind<len(s):
            w = widths[ord(s[ind])-ord('a')]
            if width+w>100:
                line+=1
                width=0
            else:
                width+=w
                ind+=1
        return [line+1,width]

'''
809. Expressive Words
Sometimes people repeat letters to represent extra feeling. For example:
"hello" -> "heeellooo"
"hi" -> "hiiii"
In these strings like "heeellooo", we have groups of adjacent letters that are all the same: "h", "eee", "ll", "ooo".
You are given a string s and an array of query strings words. A query word is stretchy if it can be made to be equal to s by any number of applications of the following extension operation: choose a group consisting of characters c, and add some number of characters c to the group so that the size of the group is three or more.
For example, starting with "hello", we could do an extension on the group "o" to get "hellooo", but we cannot get "helloo" since the group "oo" has a size less than three. Also, we could do another extension like "ll" -> "lllll" to get "helllllooo". If s = "helllllooo", then the query word "hello" would be stretchy because of these two extension operations: query = "hello" -> "hellooo" -> "helllllooo" = s.
Return the number of query strings that are stretchy.
Example:
Input: s = "heeellooo", words = ["hello", "hi", "helo"]
Output: 1
'''
class Solution(object):
    def expressiveWords(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: int
        """
        def statcharCount(s):
            return zip(*[(num, len(list(grp))) for num, grp in itertools.groupby(s)])
        chars, count = statcharCount(s)
        result = 0
        for word in words:
            wordchars, wordcount = statcharCount(word)
            if chars != wordchars: continue
            result += all(c1>=max(c2, 3) or c1==c2 for c1, c2 in zip(count, wordcount))
        return result

'''
811. Subdomain Visit Count
A website domain "discuss.leetcode.com" consists of various subdomains.
At the top level, we have "com", at the next level, we have "leetcode.com" and at the lowest level, "discuss.leetcode.com". When we visit a domain like "discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.
A count-paired domain is a domain that has one of the two formats "rep d1.d2.d3" or "rep d1.d2" where rep is the number of visits to the domain and d1.d2.d3 is the domain itself.
For example, "9001 discuss.leetcode.com" is a count-paired domain that indicates that discuss.leetcode.com was visited 9001 times.
Given an array of count-paired domains cpdomains, return an array of the count-paired domains of each subdomain in the input. You may return the answer in any order.
Example:
Input: cpdomains = ["9001 discuss.leetcode.com"]
Output: ["9001 leetcode.com","9001 discuss.leetcode.com","9001 com"]
Explanation: We only have one website domain: "discuss.leetcode.com".
As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.
'''
class Solution(object):
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        res = collections.defaultdict(int)
        for dmn in cpdomains:
            dmn = dmn.split()
            times, web = int(dmn[0]), dmn[1]
            res[web] += times
            while '.' in web:   
                web = web[web.index('.')+1:]
                res[web] += times
        return ['{} {}'.format(times, web) for web,times in res.items()]


'''
817. Linked List Components
You are given the head of a linked list containing unique integer values and an integer array nums that is a subset of the linked list values.
Return the number of connected components in nums where two values are connected if they appear consecutively in the linked list.
Example:
Input: head = [0,1,2,3], nums = [0,1,3]
Output: 2
Explanation: 0 and 1 are connected, so [0, 1] and [3] are the two connected components.
'''
class Solution(object):
    def numComponents(self, head, nums):
        """
        :type head: ListNode
        :type nums: List[int]
        :rtype: int
        """
        result = 0
        nums = set(nums)
        while head.next:
            if head.val in nums and head.next.val not in nums:
                result += 1
            head = head.next
        if head.val in nums: result += 1
        return result

'''
819. Most Common Word
Given a string paragraph and a string array of the banned words banned, return the most frequent word that is not banned. It is guaranteed there is at least one word that is not banned, and that the answer is unique.
The words in paragraph are case-insensitive and the answer should be returned in lowercase.
Example:
Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph. 
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"), 
and that "hit" isn't the answer even though it occurs more because it is banned.
'''
class Solution(object):
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        result = ''
        count = -1
        for key,value in collections.Counter(re.findall(r'\w+', paragraph.lower())).items():
            if value > count and key not in banned:
                result, count = key, value
        return result

'''
821. Shortest Distance to a Character
Given a string s and a character c that occurs in s, return an array of integers answer where answer.length == s.length and answer[i] is the distance from index i to the closest occurrence of character c in s.
The distance between two indices i and j is abs(i - j), where abs is the absolute value function.
Example:
Input: s = "loveleetcode", c = "e"
Output: [3,2,1,0,1,0,0,1,2,2,1,0]
'''
class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        pos = float('-inf')
        result = []
        for i, char in enumerate(s):
            if char == c: pos = i
            result.append(i- pos)
        
        pos = float('inf')
        for i in range(len(s)-1,-1,-1):
            if s[i]==c: pos = i
            result[i] = min(result[i], pos-i)
        return result

'''
824. Goat Latin
You are given a string sentence that consist of words separated by spaces. Each word consists of lowercase and uppercase letters only.
We would like to convert the sentence to "Goat Latin" (a made-up language similar to Pig Latin.) The rules of Goat Latin are as follows:
If a word begins with a vowel ('a', 'e', 'i', 'o', or 'u'), append "ma" to the end of the word.
For example, the word "apple" becomes "applema".
If a word begins with a consonant (i.e., not a vowel), remove the first letter and append it to the end, then add "ma".
For example, the word "goat" becomes "oatgma".
Add one letter 'a' to the end of each word per its word index in the sentence, starting with 1.
For example, the first word gets "a" added to the end, the second word gets "aa" added to the end, and so on.
Return the final sentence representing the conversion from sentence to Goat Latin.
Example:
Input: sentence = "I speak Goat Latin"
Output: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
'''
class Solution(object):
    def toGoatLatin(self, sentence):
        """
        :type sentence: str
        :rtype: str
        """
        vowels = set('aeiouAEIOU')
        words = []
        for i, word in enumerate(sentence.split(), start=1):
            if word[0] not in vowels:
                word = word[1:] + word[0]
            word += 'ma'+'a'*i
            words.append(word)
        return ' '.join(words)

'''
833. Find And Replace in String
You are given a 0-indexed string s that you must perform k replacement operations on. The replacement operations are given as three 0-indexed parallel arrays, indices, sources, and targets, all of length k.
To complete the ith replacement operation:
Check if the substring sources[i] occurs at index indices[i] in the original string s.
If it does not occur, do nothing.
Otherwise if it does occur, replace that substring with targets[i].
For example, if s = "abcd", indices[i] = 0, sources[i] = "ab", and targets[i] = "eee", then the result of this replacement will be "eeecd".
All replacement operations must occur simultaneously, meaning the replacement operations should not affect the indexing of each other. The testcases will be generated such that the replacements will not overlap.
For example, a testcase with s = "abc", indices = [0, 1], and sources = ["ab","bc"] will not be generated because the "ab" and "bc" replacements overlap.
Return the resulting string after performing all replacement operations on s.
A substring is a contiguous sequence of characters in a string.
Example:
Input: s = "abcd", indices = [0, 2], sources = ["a", "cd"], targets = ["eee", "ffff"]
Output: "eeebffff"
'''
class Solution(object):
    def findReplaceString(self, s, indices, sources, targets):
        """
        :type s: str
        :type indices: List[int]
        :type sources: List[str]
        :type targets: List[str]
        :rtype: str
        """
        for ind,sor,tag in sorted(zip(indices, sources, targets), reverse=True):
            if s[ind:ind+len(sor)] == sor:
                s = s[:ind] + tag + s[ind+len(sor):]
        return s

'''
837. New 21 Game
Alice plays the following game, loosely based on the card game "21".
Alice starts with 0 points and draws numbers while she has less than k points. During each draw, she gains an integer number of points randomly from the range [1, maxPts], where maxPts is an integer. Each draw is independent and the outcomes have equal probabilities.
Alice stops drawing numbers when she gets k or more points.
Return the probability that Alice has n or fewer points.
Answers within 10-5 of the actual answer are considered accepted.
Example:
Input: n = 10, k = 1, maxPts = 10
Output: 1.00000
'''
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        if k == 0 or n >= k + maxPts:
            return 1.0
        windowSum = 1.0
        probability = 0.0
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1.0
        for i in range(1, n+1):
            dp[i] = windowSum / maxPts
            if i < k:
                windowSum += dp[i]
            else:
                probability += dp[i]
            if i >= maxPts:
                windowSum -= dp[i - maxPts]
        return probability

'''
841. Keys and Rooms
There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0. Your goal is to visit all the rooms. However, you cannot enter a locked room without having its key.
When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.
Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited room i, return true if you can visit all the rooms, or false otherwise.
Example:
Input: rooms = [[1],[2],[3],[]]
Output: true
Explanation: 
We visit room 0 and pick up key 1.
We then visit room 1 and pick up key 2.
We then visit room 2 and pick up key 3.
We then visit room 3.
Since we were able to visit every room, we return true.
'''
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        visit = [False for i in range(len(rooms))]
        self.dfs(0, rooms, visit)
        for room in visit:
            if not room: return False
        return True
        
    def dfs(self, key, rooms, visit):
        if visit[key]:return
        visit[key] = True
        keys = rooms[key]
        for k in keys:
            self.dfs(k, rooms, visit)

'''
844. Backspace String Compare
Given two strings s and t, return true if they are equal when both are typed into empty text editors. 
'#' means a backspace character.
Note that after backspacing an empty text, the text will continue empty.
Example:
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
'''
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        stack_s, stack_t = [], []
        for char in s:
            if char != '#':
                stack_s.append(char)
            elif stack_s:
                stack_s.pop()
        for char in t:
            if char != '#':
                stack_t.append(char)
            elif stack_t:
                stack_t.pop()
        return stack_s == stack_t

'''
859. Buddy Strings
Given two strings s and goal, return true if you can swap two letters in s so the result is equal to goal, otherwise, return false.
Swapping letters is defined as taking two indices i and j (0-indexed) such that i != j and swapping the characters at s[i] and s[j].
For example, swapping at indices 0 and 2 in "abcd" results in "cbad".
Example:
Input: s = "ab", goal = "ba"
Output: true
Explanation: You can swap s[0] = 'a' and s[1] = 'b' to get "ba", which is equal to goal.
'''
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s)!=len(goal):return False
        if s == goal:
            return len(s) != len(set(goal))
        pairs = []
        for i in range(len(s)):
            if s[i] != goal[i]:
                pairs.append((s[i],goal[i]))
            if len(pairs) > 2:
                return False
        return len(pairs)==2 and pairs[0] == pairs[1][::-1]

'''
860. Lemonade Change
At a lemonade stand, each lemonade costs $5. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill. 
You must provide the correct change to each customer so that the net transaction is that the customer pays $5.
Note that you do not have any change in hand at first.
Given an integer array bills where bills[i] is the bill the ith customer pays, return true if you can provide every customer with the correct change, or false otherwise.
Example:
Input: bills = [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we give a $10 bill and a $5 bill.
Since all customers got correct change, we output true.
'''
class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        dic = {5:0,10:0}
        for bill in bills:
            if bill == 5:
                dic[5] += 1
            elif bill == 10:
                dic[10] += 1
                if dic[5]>0:
                    dic[5] -= 1
                else: return False
            elif bill == 20:
                if dic[5]>0 and dic[10]>0:
                    dic[5] -= 1
                    dic[10] -= 1
                elif dic[5]>2:
                    dic[5] -= 3
                else: return False
        return True

'''
863. All Nodes Distance K in Binary Tree
Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node.
You can return the answer in any order.
Example:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
Output: [7,4,1]
'''
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        graph = {}
        self.buildGraph(root, None, graph)
        queue = deque()
        queue.append((target, 0))
        visited = set([target])
        result = []
        while queue:
            node, distance = queue.popleft()
            if distance == k:
                result.append(node.val)
            if distance > k:
                break
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return result

    def buildGraph(self, node, parent, graph):
        if not node: return
        if node not in graph:
            graph[node] = []
        if parent:
            graph[parent].append(node)
            graph[node].append(parent)
        self.buildGraph(node.left, node, graph)
        self.buildGraph(node.right, node, graph)

'''
867. Transpose Matrix
Given a 2D integer array matrix, return the transpose of matrix.
The transpose of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.
Example:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]
'''
class Solution(object):
    def transpose(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[j][i] = matrix[i][j]
        return result

'''
875. Koko Eating Bananas
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.
Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.
Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.
Return the minimum integer k such that she can eat all the bananas within h hours.
Input: piles = [3,6,7,11], h = 8
Output: 4
'''
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        right = max(piles)
        left = 1
        while left < right:
            mid = (left+right)//2
            hourneed = 0
            for i in piles:
                hourneed += i//mid
                if i%mid:
                    hourneed += 1
            if hourneed > h:
                left = mid + 1
            else:
                right = mid
        return left

'''
876. Middle of the Linked List
Given the head of a singly linked list, return the middle node of the linked list.
If there are two middle nodes, return the second middle node.
Example:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
'''
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = slow = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        return slow

'''
877. Stone Game
Alice and Bob play a game with piles of stones. There are an even number of piles arranged in a row, and each pile has a positive integer number of stones piles[i].
The objective of the game is to end with the most stones. The total number of stones across all the piles is odd, so there are no ties.
Alice and Bob take turns, with Alice starting first. Each turn, a player takes the entire pile of stones either from the beginning or from the end of the row. This continues until there are no more piles left, at which point the person with the most stones wins.
Assuming Alice and Bob play optimally, return true if Alice wins the game, or false if Bob wins.
Example:
Input: piles = [5,3,4,5]
Output: true
'''
class Solution(object):
    def stoneGame(self, piles):
        """
        :type piles: List[int]
        :rtype: bool
        """
        n = len(piles)
        opt = [[0]*n for _ in range(n)]
        for i in range(n):
            opt[i][i] = piles[i]
        for d in range(1, n):
            for i in range(n - d):
                opt[i][i+d] = max(piles[i]-opt[i+1][i+d], piles[i+d]-opt[i][i+d-1])
        return opt[0][-1] > 0

'''
881. Boats to Save People
You are given an array people where people[i] is the weight of the ith person, and an infinite number of boats where each boat can carry a maximum weight of limit.
Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most limit.
Return the minimum number of boats to carry every given person.
Example:
Input: people = [1,2], limit = 3
Output: 1
'''
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        boats = 0
        left, right = 0, len(people)-1
        while right > -1 and left <= right:
            if people[left] + people[right] <= limit:
                left += 1
            boats += 1
            right -= 1
        return boats

'''
884. Uncommon Words from Two Sentences
A sentence is a string of single-space separated words where each word consists only of lowercase letters.
A word is uncommon if it appears exactly once in one of the sentences, and does not appear in the other sentence.
Given two sentences s1 and s2, return a list of all the uncommon words. You may return the answer in any order.
Example:
Input: s1 = "this apple is sweet", s2 = "this apple is sour"
Output: ["sweet","sour"]
'''
class Solution(object):
    def uncommonFromSentences(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: List[str]
        """
        dic = collections.Counter(s1.split() + s2.split())
        return [key for key,value in dic.items() if value == 1]

'''
885. Spiral Matrix III
You start at the cell (rStart, cStart) of an rows x cols grid facing east. 
The northwest corner is at the first row and column in the grid, and the southeast corner is at the last row and column.
You will walk in a clockwise spiral shape to visit every position in this grid. Whenever you move outside the grid's boundary, we continue our walk outside the grid (but may return to the grid boundary later.). 
Eventually, we reach all rows * cols spaces of the grid.
Return an array of coordinates representing the positions of the grid in the order you visited them.
Example:
Input: rows = 1, cols = 4, rStart = 0, cStart = 0
Output: [[0,0],[0,1],[0,2],[0,3]]
'''
class Solution(object):
    def spiralMatrixIII(self, rows, cols, rStart, cStart):
        """
        :type rows: int
        :type cols: int
        :type rStart: int
        :type cStart: int
        :rtype: List[List[int]]
        """
        result = []
        di, dj, step = 0, 1, 0
        while len(result) < rows*cols:
            for s in range(step//2+1):
                if 0<=rStart<rows and 0<=cStart<cols:
                    result.append((rStart, cStart))
                rStart, cStart = rStart+di, cStart+dj
            di, dj, step = dj, -di, step+1
        return result

'''
890. Find and Replace Pattern
Given a list of strings words and a string pattern, return a list of words[i] that match pattern. You may return the answer in any order.
A word matches the pattern if there exists a permutation of letters p so that after replacing every letter x in the pattern with p(x), we get the desired word.
Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and no two letters map to the same letter.
Example:
Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
Output: ["mee","aqq"]
Explanation: "mee" matches the pattern because there is a permutation {a -> m, b -> e, ...}. 
"ccc" does not match the pattern because {a -> c, b -> c, ...} is not a permutation, since a and b map to the same letter.
'''
class Solution(object):
    def findAndReplacePattern(self, words, pattern):
        """
        :type words: List[str]
        :type pattern: str
        :rtype: List[str]
        """
        def map(word):
            m = {}
            return [m.setdefault(char, len(m)) for char in word]
        return [w for w in words if map(w) == map(pattern)]

'''
893. Groups of Special-Equivalent Strings
You are given an array of strings of the same length words.
In one move, you can swap any two even indexed characters or any two odd indexed characters of a string words[i].
Two strings words[i] and words[j] are special-equivalent if after any number of moves, words[i] == words[j].
For example, words[i] = "zzxy" and words[j] = "xyzz" are special-equivalent because we may make the moves "zzxy" -> "xzzy" -> "xyzz".
A group of special-equivalent strings from words is a non-empty subset of words such that:
Every pair of strings in the group are special equivalent, and
The group is the largest size possible (i.e., there is not a string words[i] not in the group such that words[i] is special-equivalent to every string in the group).
Return the number of groups of special-equivalent strings from words.
Example:
Input: words = ["abcd","cdab","cbad","xyzz","zzxy","zzyx"]
Output: 3
Explanation: 
One group is ["abcd", "cdab", "cbad"], since they are all pairwise special equivalent, and none of the other strings is all pairwise special equivalent to these.
The other two groups are ["xyzz", "zzxy"] and ["zzyx"].
Note that in particular, "zzxy" is not special equivalent to "zzyx".
'''
class Solution(object):
    def numSpecialEquivGroups(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        encode = {''.join(sorted(word[::2])) + ''.join(sorted(word[1::2])) for word in words}
        return len(set(encode))

'''
904. Fruit Into Baskets
You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.
You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:
You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
Given the integer array fruits, return the maximum number of fruits you can pick.
Example:
Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.
'''
class Solution(object):
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        basket= {}
        left = 0
        maxsize = 1
        fruitnum = 0
        for right in range(len(fruits)):
            if fruits[right] in basket:
                basket[fruits[right]] += 1  
            else: 
                basket[fruits[right]] = 1
            if basket[fruits[right]] == 1:    
                fruitnum += 1
            if fruitnum < 3:
                maxsize = max(maxsize, right-left+1)
            else:
                basket[fruits[left]] -= 1
                if basket[fruits[left]] == 0:
                    fruitnum -= 1
                left += 1
        return maxsize

'''
905. Sort Array By Parity
Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.
Return any array that satisfies this condition.
Example:
Input: nums = [3,1,2,4]
Output: [2,4,3,1]
Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.
'''
class Solution(object):
    def sortArrayByParity(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        left, right = 0, 0
        while right < len(nums):
            if not nums[right] % 2:
                nums[left],nums[right]=nums[right],nums[left]
                left +=1
            right += 1
        return nums

'''
912. Sort an Array
Given an array of integers nums, sort the array in ascending order and return it.
You must solve the problem without using any built-in functions in O(nlog(n)) time complexity and with the smallest space complexity possible.
Example:
Input: nums = [5,2,3,1]
Output: [1,2,3,5]
'''
class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums)==1:return nums
        mid = len(nums)//2
        left = self.sortArray(nums[0:mid])
        right = self.sortArray(nums[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        res = []
        i,j = 0,0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        if i < len(left):
            res += left[i:]
        else:
            res += right[j:]
        return res

'''
915. Partition Array into Disjoint Intervals
Given an integer array nums, partition it into two (contiguous) subarrays left and right so that:
Every element in left is less than or equal to every element in right.
left and right are non-empty.
left has the smallest possible size.
Return the length of left after such a partitioning.
Test cases are generated such that partitioning exists.
Example:
Input: nums = [5,0,3,8,6]
Output: 3
'''
class Solution(object):
    def partitionDisjoint(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        localmax, globalmax = nums[0], nums[0]
        partition = 0
        for i in range(1,len(nums)):
            num = nums[i]
            if localmax > num:
                localmax = globalmax
                partition = i
            else:
                globalmax = max(globalmax, num)
        return partition+1

'''
916. Word Subsets
You are given two string arrays words1 and words2.
A string b is a subset of string a if every letter in b occurs in a including multiplicity.
For example, "wrr" is a subset of "warrior" but is not a subset of "world".
A string a from words1 is universal if for every string b in words2, b is a subset of a.
Return an array of all the universal strings in words1. You may return the answer in any order.
Example:
Input: words1 = ["amazon","apple","facebook","google","leetcode"], words2 = ["e","o"]
Output: ["facebook","google","leetcode"]
'''
class Solution(object):
    def wordSubsets(self, words1, words2):
        """
        :type words1: List[str]
        :type words2: List[str]
        :rtype: List[str]
        """
        subset = {}
        for chars in words2:
            for char in chars:
                subset[char] = max(subset.get(char, 0), chars.count(char))
        return [word for word in words1 if all(word.count(c) >= subset[c] for c in subset.keys())]

'''
917. Reverse Only Letters
Given a string s, reverse the string according to the following rules:
All the characters that are not English letters remain in the same position.
All the English letters (lowercase or uppercase) should be reversed.
Return s after reversing it.
Example:
Input: s = "ab-cd"
Output: "dc-ba"
'''
class Solution(object):
    def reverseOnlyLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        left,right = 0, len(s)-1
        while left < right:
            while left < right and not s[left].isalpha():
                left += 1
            while left < right and not s[right].isalpha():
                right -= 1
            s[left], s[right] = s[right], s[left]
            left,right = left+1,right-1
        return ''.join(s)

'''
918. Maximum Sum Circular Subarray
Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
A circular array means the end of the array connects to the beginning of the array. 
Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].
A subarray may only include each element of the fixed buffer nums at most once. Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.
Example:
Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.
'''
class Solution(object):
    def maxSubarraySumCircular(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        total, maxSum, curMax, minSum, curMin = 0, nums[0], 0, nums[0], 0
        for num in nums:
            curMax = max(curMax + num, num)
            curMin = min(curMin + num, num)
            maxSum = max(maxSum, curMax)
            minSum = min(minSum, curMin)
            total += num
        return max(maxSum, total-minSum) if maxSum > 0 else maxSum

'''
921. Minimum Add to Make Parentheses Valid
A parentheses string is valid if and only if:
It is the empty string,
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
You are given a parentheses string s. In one move, you can insert a parenthesis at any position of the string.
For example, if s = "()))", you can insert an opening parenthesis to be "(()))" or a closing parenthesis to be "())))".
Return the minimum number of moves required to make s valid.
Example:
Input: s = "())"
Output: 1
'''
class Solution(object):
    def minAddToMakeValid(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s: return True
        stack = []
        result = 0
        for par in s:
            if par == '(':
                stack.append(par)
            elif not stack:
                result += 1
            else: stack.pop()
        return result+len(stack)

'''
922. Sort Array By Parity II
Given an array of integers nums, half of the integers in nums are odd, and the other half are even.
Sort the array so that whenever nums[i] is odd, i is odd, and whenever nums[i] is even, i is even.
Return any answer array that satisfies this condition.
Example:
Input: nums = [4,2,5,7]
Output: [4,5,2,7]
Explanation: [4,7,2,5], [2,5,4,7], [2,7,4,5] would also have been accepted.
'''
class Solution(object):
    def sortArrayByParityII(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        oddindex = 1
        for i in range(0, len(nums), 2):
            if nums[i]%2:
                while nums[oddindex]%2:
                    oddindex+=2
                nums[i],nums[oddindex] = nums[oddindex],nums[i]
        return nums

'''
925. Long Pressed Name
Your friend is typing his name into a keyboard. 
Sometimes, when typing a character c, the key might get long pressed, and the character will be typed 1 or more times.
You examine the typed characters of the keyboard. 
Return True if it is possible that it was your friends name, with some characters (possibly none) being long pressed.
Example 1:
Input: name = "alex", typed = "aaleex"
Output: true
Explanation: 'a' and 'e' in 'alex' were long pressed.
'''
class Solution(object):
    def isLongPressedName(self, name, typed):
        """
        :type name: str
        :type typed: str
        :rtype: bool
        """
        if name[0] != typed[0]: return False
        nameindex,typedindex = 1,1
        while nameindex<len(name) and typedindex<len(typed):
            if typed[typedindex] == name[nameindex]:
                nameindex,typedindex = nameindex+1,typedindex+1
            elif typed[typedindex] == name[nameindex-1]:
                typedindex+=1
            else:
                return False
        if nameindex<len(name): return False
        while typedindex<len(typed):
            if typed[typedindex] == typed[typedindex-1]:
                typedindex+=1
            else: return False
        return True

'''
929. Unique Email Addresses
Every valid email consists of a local name and a domain name, separated by the '@' sign. Besides lowercase letters, the email may contain one or more '.' or '+'.
For example, in "alice@leetcode.com", "alice" is the local name, and "leetcode.com" is the domain name.
If you add periods '.' between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name. Note that this rule does not apply to domain names.
For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.
If you add a plus '+' in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered. Note that this rule does not apply to domain names.
For example, "m.y+name@email.com" will be forwarded to "my@email.com".
It is possible to use both of these rules at the same time.
Given an array of strings emails where we send one email to each emails[i], return the number of different addresses that actually receive mails.
Example:
Input: emails = ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails.
'''
class Solution(object):
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        res = set()
        for email in emails:
            local, domain = email.split('@')
            if '+' in local:
                local = local[:local.index('+')]
            local = local.replace('.', '')
            res.add((local, domain))
        return len(res)

'''
933. Number of Recent Calls
You have a RecentCounter class which counts the number of recent requests within a certain time frame.
Implement the RecentCounter class:
RecentCounter() Initializes the counter with zero recent requests.
int ping(int t) Adds a new request at time t, where t represents some time in milliseconds, and returns the number of requests that has happened in the past 3000 milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range [t - 3000, t].
It is guaranteed that every call to ping uses a strictly larger value of t than the previous call.
Example:
Input
["RecentCounter", "ping", "ping", "ping", "ping"]
[[], [1], [100], [3001], [3002]]
Output
[null, 1, 2, 3, 3]
'''
class RecentCounter(object):

    def __init__(self):
        self.queue = collections.deque()

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)

'''
937. Reorder Data in Log Files
You are given an array of logs. Each log is a space-delimited string of words, where the first word is the identifier.
There are two types of logs:
Letter-logs: All words (except the identifier) consist of lowercase English letters.
Digit-logs: All words (except the identifier) consist of digits.
Reorder these logs so that:
The letter-logs come before all digit-logs.
The letter-logs are sorted lexicographically by their contents. If their contents are the same, then sort them lexicographically by their identifiers.
The digit-logs maintain their relative ordering.
Return the final order of the logs.
Example:
Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
Explanation:
The letter-log contents are all different, so their ordering is "art can", "art zero", "own kit dig".
The digit-logs have a relative order of "dig1 8 1 5 1", "dig2 3 6".
'''
class Solution(object):
    def reorderLogFiles(self, logs):
        """
        :type logs: List[str]
        :rtype: List[str]
        """
        letter_logs = [l for l in logs if l[-1].isalpha()]
        digit_logs = [l for l in logs if not l[-1].isalpha()]
        letter_logs.sort(key=lambda x: x.split()[0])
        letter_logs.sort(key=lambda x: x.split()[1:])
        return letter_logs + digit_logs

'''
941. Valid Mountain Array
Given an array of integers arr, return true if and only if it is a valid mountain array.
Recall that arr is a mountain array if and only if:
arr.length >= 3
There exists some i with 0 < i < arr.length - 1 such that:
arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
arr[i] > arr[i + 1] > ... > arr[arr.length - 1]
Example:
Input: arr = [2,1]
Output: false
'''
class Solution(object):
    def validMountainArray(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        left,right = 0,len(arr)-1
        while left<len(arr)-1 and arr[left+1] > arr[left]:
            left+=1
        while right>0 and arr[right-1] > arr[right]:
            right-=1
        return left==right and left!=len(arr)-1 and right!=0

'''
946. Validate Stack Sequences
Given two integer arrays pushed and popped each with distinct values, return true if this could have been the result of a sequence of push and pop operations on an initially empty stack, or false otherwise.
Example:
Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
Output: true
Explanation: We might do the following sequence:
push(1), push(2), push(3), push(4),
pop() -> 4,
push(5),
pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
'''
class Solution(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        stack = []
        popindex = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[popindex]:
                stack.pop()
                popindex += 1
        if not stack: return True
        else: return False
    
'''
953. Verifying an Alien Dictionary
In an alien language, surprisingly, they also use English lowercase letters, but possibly in a different order.
The order of the alphabet is some permutation of lowercase letters.
Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographically in this alien language.
Example:
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
'''
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        dic = dict(zip(order, range(26)))
        newwords = sorted(words, key=lambda x: [dic[char] for char in x])
        return newwords == words 

'''
957. Prison Cells After N Days
There are 8 prison cells in a row and each cell is either occupied or vacant.
Each day, whether the cell is occupied or vacant changes according to the following rules:
If a cell has two adjacent neighbors that are both occupied or both vacant, then the cell becomes occupied.
Otherwise, it becomes vacant.
Note that because the prison is a row, the first and the last cells in the row can't have two adjacent neighbors.
You are given an integer array cells where cells[i] == 1 if the ith cell is occupied and cells[i] == 0 if the ith cell is vacant, and you are given an integer n.
Return the state of the prison after n days (i.e., n such changes described above).
Example:
Input: cells = [0,1,0,1,1,0,0,1], n = 7
Output: [0,0,1,1,0,0,0,0]
Explanation: The following table summarizes the state of the prison on each day:
Day 0: [0, 1, 0, 1, 1, 0, 0, 1]
Day 1: [0, 1, 1, 0, 0, 0, 0, 0]
Day 2: [0, 0, 0, 0, 1, 1, 1, 0]
Day 3: [0, 1, 1, 0, 0, 1, 0, 0]
Day 4: [0, 0, 0, 0, 0, 1, 0, 0]
Day 5: [0, 1, 1, 1, 0, 1, 0, 0]
Day 6: [0, 0, 1, 0, 1, 1, 0, 0]
Day 7: [0, 0, 1, 1, 0, 0, 0, 0]
'''
class Solution(object):
    def prisonAfterNDays(self, cells, n):
        """
        :type cells: List[int]
        :type n: int
        :rtype: List[int]
        """
        transfer = {}
        while n:
            transfer[str(cells)] = n
            n -= 1
            cells = [0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in range(1, 7)] + [0]
            if str(cells) in transfer:
                n %= transfer[str(cells)]-n
        return cells

'''
958. Check Completeness of a Binary Tree
Given the root of a binary tree, determine if it is a complete binary tree.
In a complete binary tree, every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
Example:
Input: root = [1,2,3,4,5,6]
Output: true
'''
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root: 
            return True
        queue = deque([root])
        nomore = False
        while queue[0]:
            node = queue.popleft()
            queue.append(node.left)
            queue.append(node.right)
        while queue and not queue[0]:
            queue.popleft()
        return not bool(queue)

# method 2
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        queue = collections.deque()
        queue.append(root)
        nomore = False
        while queue:
            node = queue.popleft()
            if nomore and (node.left or node.right):
                return False
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if not node.left and node.right:
                return False
            if self.hasEmpty(node):
                nomore = True
        return True

    def hasEmpty(self, node: Optional[TreeNode]) -> bool:
        return not node.left or not node.right

'''
961. N-Repeated Element in Size 2N Array
You are given an integer array nums with the following properties:
nums.length == 2 * n.
nums contains n + 1 unique elements.
Exactly one element of nums is repeated n times.
Return the element that is repeated n times.
Example:
Input: nums = [1,2,3,3]
Output: 3
'''
class Solution(object):
    def repeatedNTimes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dic = {}
        threshold = len(nums)//2
        for num in nums:
            if num in dic:
                dic[num] += 1
                if dic[num] == threshold:return num
            else:
                dic[num] = 1

'''
967. Numbers With Same Consecutive Differences
Return all non-negative integers of length n such that the absolute difference between every two consecutive digits is k.
Note that every number in the answer must not have leading zeros. For example, 01 has one leading zero and is invalid.
You may return the answer in any order.
Example:
Input: n = 3, k = 7
Output: [181,292,707,818,929]
Explanation: Note that 070 is not a valid number, because it has leading zeroes.
'''
class Solution(object):
    def numsSameConsecDiff(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        digits = set(range(1,10))
        for _ in range(n-1):
            temp = set()
            for digit in digits:
                d = digit % 10
                if d-k >= 0:
                    temp.add(digit*10 + d - k)
                if d+k <= 9:
                    temp.add(digit*10 + d + k)
            digits = temp
        if n == 0: digits.add(0)
        return list(digits)

'''
968. Binary Tree Cameras
You are given the root of a binary tree. We install cameras on the tree nodes where each camera at a node can monitor its parent, itself, and its immediate children.
Return the minimum number of cameras needed to monitor all nodes of the tree.
Example:
Input: root = [0,0,null,0,0]
Output: 1
Explanation: One camera is enough to monitor all nodes if placed as shown.
'''
class Solution(object):
    def __init__(self):
        self.total = 0
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        0: no cover
        1: 放camera
        2: 已经被cover
        """
        if self.traversal(root)==0:
            self.total += 1
        return self.total
    
    def traversal(self, root):
        if root == None:
            return 2
        left = self.traversal(root.left)
        right = self.traversal(root.right)
        if left==2 and right==2:
            return 0
        elif left==0 or right==0:
            self.total += 1
            return 1
        else: return 2

'''
974. Subarray Sums Divisible by K
Given an integer array nums and an integer k, return the number of non-empty subarrays that have a sum divisible by k.
A subarray is a contiguous part of an array.
Example:
Input: nums = [4,5,0,-2,-3,1], k = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by k = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
'''
class Solution(object):
    def subarraysDivByK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        remainderFrq = defaultdict(int)
        remainderFrq[0] = 1
        res = prefixSum = 0
        for n in nums:
            prefixSum += n
            remainder = prefixSum % k
            res += remainderFrq[remainder]
            remainderFrq[remainder] += 1
        return res

'''
977. Squares of a Sorted Array
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
Example:
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
'''
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        left,right, index = 0, len(nums)-1, len(nums)-1
        result = [0 for _ in range(len(nums))]
        while left <= right:
            if -nums[left] >= nums[right]:
                result[index] = nums[left]*nums[left]
                left += 1
            else:
                result[index] = nums[right]*nums[right]
                right -= 1
            index -= 1
        return result

'''
983. Minimum Cost For Tickets
You have planned some train traveling one year in advance. The days of the year in which you will travel are given as an integer array days. Each day is an integer from 1 to 365.
Train tickets are sold in three different ways:
a 1-day pass is sold for costs[0] dollars,
a 7-day pass is sold for costs[1] dollars, and
a 30-day pass is sold for costs[2] dollars.
The passes allow that many days of consecutive travel.
For example, if we get a 7-day pass on day 2, then we can travel for 7 days: 2, 3, 4, 5, 6, 7, and 8.
Return the minimum number of dollars you need to travel every day in the given list of days.
Example :
Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
'''
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        opt = [0 for _ in range(366)]
        travelDays = set(days)
        for i in range(1, 366):
            if i in travelDays:
                opt[i] = min(opt[i-1] + costs[0], opt[max(0, i-7)] + costs[1], opt[max(0, i-30)] + costs[2])
            else:
                opt[i] = opt[i-1]
        return opt[-1]

'''
985. Sum of Even Numbers After Queries
You are given an integer array nums and an array queries where queries[i] = [vali, indexi].
For each query i, first, apply nums[indexi] = nums[indexi] + vali, then print the sum of the even values of nums.
Return an integer array answer where answer[i] is the answer to the ith query.
Example:
Input: nums = [1,2,3,4], queries = [[1,0],[-3,1],[-4,0],[2,3]]
Output: [8,6,2,4]
'''
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        res = []
        evenSum = 0
        for num in nums:
            if num%2==0: evenSum += num
        for queries in queries:
            val, index = queries[0], queries[1]
            num = nums[index]
            if num%2==0 and (num+val)%2==0:
                evenSum += val
            elif num%2==0 and (num+val)%2!=0:
                evenSum -= num
            elif num%2!=0 and (num+val)%2==0:
                evenSum += num+val

            nums[index] += val
            res.append(evenSum)
        return res

'''
986. Interval List Intersections
You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj].
Each list of intervals is pairwise disjoint and in sorted order.
Return the intersection of these two interval lists.
A closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.
The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].
Example:
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
'''
class Solution(object):
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        lst = sorted(firstList + secondList, key = lambda x: x[0])
        res = []
        for i in range(1,len(lst)):
            left,right = lst[i][0], lst[i][1]
            if left <= lst[i-1][1]:
                res.append([max(lst[i-1][0], left),min(lst[i-1][1], right)])
                lst[i][1] = max(lst[i][1], lst[i-1][1])
        return res

'''
989. Add to Array-Form of Integer
The array-form of an integer num is an array representing its digits in left to right order.
For example, for num = 1321, the array form is [1,3,2,1].
Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.
Example:
Input: num = [1,2,0,0], k = 34
Output: [1,2,3,4]
Explanation: 1200 + 34 = 1234
'''
class Solution(object):
    def addToArrayForm(self, num, k):
        """
        :type num: List[int]
        :type k: int
        :rtype: List[int]
        """
        num[-1] += k
        for i in range(len(num)-1, -1, -1):
            carry, num[i] = divmod(num[i], 10)
            if i: num[i-1] += carry
        if carry:
            num = list(map(int, str(carry))) + num
        return num

##1000##

'''
1002. Find Common Characters
Given a string array words, return an array of all characters that show up in all strings within the words (including duplicates). You may return the answer in any order.
Example:
Input: words = ["bella","label","roller"]
Output: ["e","l","l"]
'''
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

'''
1003. Check If Word Is Valid After Substitutions
Given a string s, determine if it is valid.
A string s is valid if, starting with an empty string t = "", you can transform t into s after performing the following operation any number of times:
Insert string "abc" into any position in t. More formally, t becomes tleft + "abc" + tright, where t == tleft + tright. Note that tleft and tright may be empty.
Return true if s is a valid string, otherwise, return false.
Example:
Input: s = "aabcbc"
Output: true
Explanation:
"" -> "abc" -> "aabcbc"
Thus, "aabcbc" is valid.
'''
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        while s:
            if 'abc' in s:
                s = ''.join(s.split('abc'))
            else: return False
        return True

'''
1004. Max Consecutive Ones III
Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can flip at most k 0's.
Example:
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
'''
class Solution(object):
    def longestOnes(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        left = 0
        for right in range(len(nums)):
            if nums[right] ==0:
                k-=1
            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1
        return right - left + 1

'''
1005. Maximize Sum Of Array After K Negations
Given an integer array nums and an integer k, modify the array in the following way:
choose an index i and replace nums[i] with -nums[i].
You should apply this process exactly k times. You may choose the same index i multiple times.
Return the largest possible sum of the array after modifying it in this way.
Example:
Input: nums = [4,2,3], k = 1
Output: 5
Explanation: Choose index 1 and nums becomes [4,-2,3].
'''
class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums = sorted(nums, key=abs, reverse=True)
        for i in range(len(nums)):
            if k > 0 and nums[i] < 0:
                nums[i] = -nums[i]
                k -= 1
        if k > 0:
            nums[-1] *= (-1)**k
        return sum(nums)

'''
1010. Pairs of Songs With Total Durations Divisible by 60
You are given a list of songs where the ith song has a duration of time[i] seconds.
Return the number of pairs of songs for which their total duration in seconds is divisible by 60. Formally, we want the number of indices i, j such that i < j with (time[i] + time[j]) % 60 == 0.
Example:
Input: time = [30,20,150,100,40]
Output: 3
Explanation: Three pairs have a total duration divisible by 60:
(time[0] = 30, time[2] = 150): total duration 180
(time[1] = 20, time[3] = 100): total duration 120
(time[1] = 20, time[4] = 40): total duration 60
'''
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        dic = collections.defaultdict(int)
        ans = 0
        for t in time:
            ans += dic[-t % 60] # dic[(60-t%60)%60]
            dic[t%60] += 1
        return ans

'''
1011. Capacity To Ship Packages Within D Days
A conveyor belt has packages that must be shipped from one port to another within days days.
The ith package on the conveyor belt has a weight of weights[i].
Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.
Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.
Example:
Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
'''
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        left,right = 0,0
        for weight in weights:
            left = max(left, weight)
            right += weight
        result = right
        while left<=right:
            mid = (left+right)//2
            if self.check(weights,days,mid):
                result = mid
                right = mid-1
            else: left = mid+1
        return result

    def check(self,weights,days,capacity):
        requiredDays = 1
        currWeight = 0
        for weight in weights:
            if currWeight+weight>capacity:
                requiredDays += 1
                currWeight = 0
            currWeight += weight
        if requiredDays > days:
            return False
        return True

'''
1015. Smallest Integer Divisible by K
Given a positive integer k, you need to find the length of the smallest positive integer n such that n is divisible by k, and n only contains the digit 1.
Return the length of n. If there is no such n, return -1.
Note: n may not fit in a 64-bit signed integer.
Example:
Input: k = 1
Output: 1
'''
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

'''
1016. Binary String With Substrings Representing 1 To N
Given a binary string s and a positive integer n, return true if the binary representation of all the integers in the range [1, n] are substrings of s, or false otherwise.
A substring is a contiguous sequence of characters within a string.
Example:
Input: s = "0110", n = 3
Output: true
'''
class Solution(object):
    def queryString(self, s, n):
        """
        :type s: str
        :type n: int
        :rtype: bool
        """
        for num in xrange(n,n//2,-1):
            if bin(num)[2:] not in s:
                return False
        return True

'''
1019. Next Greater Node In Linked List
You are given the head of a linked list with n nodes.
For each node in the list, find the value of the next greater node. That is, for each node, find the value of the first node that is next to it and has a strictly larger value than it.
Return an integer array answer where answer[i] is the value of the next greater node of the ith node (1-indexed). If the ith node does not have a next greater node, set answer[i] = 0.
Example:
Input: head = [2,1,5]
Output: [5,5,0]
'''
class Solution(object):
    def nextLargerNodes(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        #vals, dummyhead, result = [], head, []
        #while dummyhead:
        #    vals.append(dummyhead.val)
        #    dummyhead = dummyhead.next
        #for index, x in enumerate(vals):
        #    result.append(next((y for y in vals[index:] if y > x), 0))
        #return result
        stack, vals = [], []
        i, ans = 0, []
        while head:
            num = head.val
            vals.append(num)
            while stack and num > vals[stack[-1]]:
                ans[stack.pop()] = num
            stack.append(i)
            ans.append(0)
            i += 1
            head = head.next
        return ans

'''
1020. Number of Enclaves
You are given an m x n binary matrix grid, where 0 represents a sea cell and 1 represents a land cell.
A move consists of walking from one land cell to another adjacent (4-directionally) land cell or walking off the boundary of the grid.
Return the number of land cells in grid for which we cannot walk off the boundary of the grid in any number of moves.
Example:
Input: grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
Output: 3
'''
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        for i in [0, len(grid)-1]:
            for j in range(len(grid[0])):
                if grid[i][j] == 1: self.dfs(i, j, grid)
        for j in [0, len(grid[0])-1]:
            for i in range(len(grid)):
                if grid[i][j] == 1: self.dfs(i, j, grid)
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1: res += 1
        return res

    def dfs(self, i, j, grid):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]):
            return
        if grid[i][j] == 1:
            grid[i][j] = 0
            self.dfs(i+1, j, grid)
            self.dfs(i-1, j, grid)
            self.dfs(i, j+1, grid)
            self.dfs(i, j-1, grid)

'''
1027. Longest Arithmetic Subsequence
Given an array nums of integers, return the length of the longest arithmetic subsequence in nums.
Note that:
A subsequence is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.
A sequence seq is arithmetic if seq[i + 1] - seq[i] are all the same value (for 0 <= i < seq.length - 1).
Example:
Input: nums = [3,6,9,12]
Output: 4
'''
class Solution(object):
    def longestArithSeqLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = {}
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                dp[(j, nums[j]-nums[i])] = dp.get((i, nums[j]-nums[i]), 1) + 1
        return max(dp.values())

'''
1035. Uncrossed Lines
You are given two integer arrays nums1 and nums2. We write the integers of nums1 and nums2 (in the order they are given) on two separate horizontal lines.
We may draw connecting lines: a straight line connecting two numbers nums1[i] and nums2[j] such that:
nums1[i] == nums2[j], and
the line we draw does not intersect any other connecting (non-horizontal) line.
Note that a connecting line cannot intersect even at the endpoints (i.e., each number can only belong to one connecting line).
Return the maximum number of connecting lines we can draw in this way.
Example 1:
Input: nums1 = [1,4,2], nums2 = [1,2,4]
Output: 2
Explanation: We can draw 2 uncrossed lines as in the diagram.
We cannot draw 3 uncrossed lines, because the line from nums1[1] = 4 to nums2[2] = 4 will intersect the line from nums1[2]=2 to nums2[1]=2.
'''
class Solution(object):
    def maxUncrossedLines(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        opt = [[0 for _ in range(len(nums2)+1)] for _ in range(len(nums1)+1)]
        for i in range(1, len(nums1)+1):
            for j in range(1, len(nums2)+1):
                if nums1[i-1] == nums2[j-1]:
                    opt[i][j] = opt[i-1][j-1] + 1
                else:
                    opt[i][j] = max(opt[i-1][j], opt[i][j-1])
        return opt[-1][-1]

'''
1046. Last Stone Weight
You are given an array of integers stones where stones[i] is the weight of the ith stone.
We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together.
Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:
If x == y, both stones are destroyed, and
If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
At the end of the game, there is at most one stone left.
Return the weight of the last remaining stone. If there are no stones left, return 0.
Example:
Input: stones = [2,7,4,1,8,1]
Output: 1
'''
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        while len(stones) > 1:
            stones.sort()
            newstone = stones.pop()-stones.pop()
            if newstone:
                stones.append(newstone)
        return stones[0] if stones else 0

'''
1047. Remove All Adjacent Duplicates In String
You are given a string s consisting of lowercase English letters. A duplicate removal consists of choosing two adjacent and equal letters and removing them.
We repeatedly make duplicate removals on s until we no longer can.
Return the final string after all such duplicate removals have been made. It can be proven that the answer is unique.
Example:
Input: s = "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, and this is the only possible move.
The result of this move is that the string is "aaca", of which only "aa" is possible, so the final string is "ca".
'''
class Solution(object):
    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        for char in s:
            if not stack:
                stack.append(char)
            elif char == stack[-1]:
                stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)

'''
1049. Last Stone Weight II
You are given an array of integers stones where stones[i] is the weight of the ith stone.
We are playing a game with the stones. On each turn, we choose any two stones and smash them together. Suppose the stones have weights x and y with x <= y. The result of this smash is:
If x == y, both stones are destroyed, and
If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
At the end of the game, there is at most one stone left.
Return the smallest possible weight of the left stone. If there are no stones left, return 0.
Example:
Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation:
We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
we can combine 1 and 1 to get 0, so the array converts to [1], then that's the optimal value.
'''
class Solution(object):
    def lastStoneWeightII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        if len(stones) == 1:
            return stones[0]
        if len(stones) == 2:
            return abs(stones[1] - stones[0])
        target = sum(stones)
        totalSum = target//2
        opt = [[0 for _ in range(totalSum+1)] for _ in range(len(stones))]
        
        for i in range(len(stones)):
            for j in range(totalSum+1):
                if j >= stones[i]:
                    opt[i][j] = max(opt[i-1][j], opt[i-1][j-stones[i]] + stones[i])
                else:
                    opt[i][j] = opt[i-1][j]
        return target - (opt[-1][totalSum] * 2)

'''
1051. Height Checker
A school is trying to take an annual photo of all the students. The students are asked to stand in a single file line in non-decreasing order by height.
Let this ordering be represented by the integer array expected where expected[i] is the expected height of the ith student in line.
You are given an integer array heights representing the current order that the students are standing in. Each heights[i] is the height of the ith student in line (0-indexed).
Return the number of indices where heights[i] != expected[i].
Example:
Input: heights = [1,1,4,2,1,3]
Output: 3
'''
class Solution(object):
    def heightChecker(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        dic = collections.Counter(heights)
        pos = 1
        for i in range(len(heights)):
            while dic[pos] == 0:
                pos += 1
            if heights[i] != pos:
                res += 1
            dic[pos] -= 1
        return res

'''
1071. Greatest Common Divisor of Strings
For two strings s and t, we say "t divides s" if and only if s = t + ... + t (i.e., t is concatenated with itself one or more times).
Given two strings str1 and str2, return the largest string x such that x divides both str1 and str2.
Example:
Input: str1 = "ABCABC", str2 = "ABC"
Output: "ABC"
'''
class Solution(object):
    def gcdOfStrings(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: str
        """
        l1,l2 = len(str1), len(str2)
        p1, p2 = [], []
        for i in range(1,l1+1):
            if l1%i==0 and str1[0:i]*(l1//i)==str1:
                p1.append(str1[0:i])
        for i in range(1,l2+1):
            if l2%i==0 and str2[0:i]*(l2//i)==str2:
                p2.append(str2[0:i])
                
        inter = set(p1).intersection(p2)
        if inter:
            return max(inter, key=len)
        else:return ""

'''
1072. Flip Columns For Maximum Number of Equal Rows
You are given an m x n binary matrix matrix.
You can choose any number of columns in the matrix and flip every cell in that column (i.e., Change the value of the cell from 0 to 1 or vice versa).
Return the maximum number of rows that have all values equal after some number of flips.
Example:
Input: matrix = [[0,1],[1,1]]
Output: 1
Explanation: After flipping no values, 1 row has all values equal.
'''
class Solution:
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        c = collections.Counter()
        for row in matrix:
            c[tuple([x^row[0] for x in row])] += 1
        return max(c.values())

'''
1074. Number of Submatrices That Sum to Target
Given a matrix and a target, return the number of non-empty submatrices that sum to target.
A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2.
Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different: for example, if x1 != x1'.
Example:
Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.
'''
class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: int
        """
        row, col = len(matrix), len(matrix[0])
        for r in matrix:
            for c in range(col-1):
                r[c+1] += r[c]
        result = 0
        for i in range(col):
            for j in range(i,col):
                c = collections.defaultdict(int)
                cur, c[0] = 0, 1
                for k in range(row):
                    cur += matrix[k][j] - (matrix[k][i - 1] if i > 0 else 0)
                    result += c[cur - target]
                    c[cur] += 1
        return result

'''
1089. Duplicate Zeros
Given a fixed-length integer array arr, duplicate each occurrence of zero, shifting the remaining elements to the right.
Note that elements beyond the length of the original array are not written. Do the above modifications to the input array in place and do not return anything.
Example:
Input: arr = [1,0,2,3,0,4,5,0]
Output: [1,0,0,2,3,0,0,4]
'''
class Solution(object):
    def duplicateZeros(self, arr):
        """
        :type arr: List[int]
        :rtype: None Do not return anything, modify arr in-place instead.
        """
        length = len(arr)
        left = length - 1
        right = length + arr.count(0) - 1

        while left >= 0:
            if arr[left] != 0:
                if right < length:
                    arr[right] = arr[left]
            else:
                if right < length:
                    arr[right] = arr[left]
                right -= 1
                if right < length:
                    arr[right] = arr[left]
            left -= 1
            right -= 1

'''
1091. Shortest Path in Binary Matrix
Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.
A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:
All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.
Example:
Input: grid = [[0,1],[1,0]]
Output: 2
'''
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1: return -1;
        n = len(grid)
        directions = [[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]]
        queue = collections.deque()
        queue.append((0, 0))
        visit = [[False for _ in range(n)] for _ in range(n)]
        visit[0][0] = True
        result = 1
        while len(queue):
            for i in range(len(queue)):
                x, y = queue.popleft()
                if x==n-1 and y==n-1:
                    return result
                for stepx, stepy in directions:
                    nx, ny = x+stepx, y+stepy
                    if 0 <= nx < n and 0 <= ny < n and not visit[nx][ny] and grid[nx][ny] == 0:
                        queue.append((nx, ny))
                        visit[nx][ny] = True
            result += 1
        return -1

##1100##

'''
1106. Parsing A Boolean Expression
Return the result of evaluating a given boolean expression, represented as a string.
An expression can either be:

"t", evaluating to True;
"f", evaluating to False;
"!(expr)", evaluating to the logical NOT of the inner expression expr;
"&(expr1,expr2,...)", evaluating to the logical AND of 2 or more inner expressions expr1, expr2, ...;
"|(expr1,expr2,...)", evaluating to the logical OR of 2 or more inner expressions expr1, expr2, ...
Example:
Input: expression = "!(f)"
Output: true
'''
class Solution(object):
    def parseBoolExpr(self, expression):
        """
        :type expression: str
        :rtype: bool
        """
        #t, f = True, False
        #expression = expression.replace('!', 'not |').replace('&(', 'all([').replace('|(', 'any([').replace(')', '])')
        # eval('all([True, False])')
        # eval('not any([True])')
        # eval('any([True, False])')
        #return eval(expression)
        stack = []
        for char in expression:
            if char == ')':
                seen = set()
                while stack[-1] != '(':
                    seen.add(stack.pop())
                stack.pop()
                operator = stack.pop()
                stack.append(all(seen) if operator == '&' else any(seen) if operator == '|' else not seen.pop())
            elif char != ',':
                stack.append(True if char == 't' else False if char == 'f' else char)
        return stack.pop()

'''
1122. Relative Sort Array
Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.
Sort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2. Elements that do not appear in arr2 should be placed at the end of arr1 in ascending order.
Example:
Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
Output: [2,2,2,1,4,3,3,9,6,7,19]
'''
class Solution(object):
    def relativeSortArray(self, arr1, arr2):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :rtype: List[int]
        """
        boolset = set(arr2)
        leftlst, rightlst = [], []
        for num in arr1:
            if num in boolset:
                leftlst.append(num)
            else:
                rightlst.append(num)
        dic = collections.Counter(leftlst)
        temp = []
        for num in arr2:
            for _ in range(dic[num]):
                temp.append(num)
        return temp + sorted(rightlst)

'''
1128. Number of Equivalent Domino Pairs
Given a list of dominoes, dominoes[i] = [a, b] is equivalent to dominoes[j] = [c, d] if and only if either (a == c and b == d), or (a == d and b == c) - that is, one domino can be rotated to be equal to another domino.
Return the number of pairs (i, j) for which 0 <= i < j < dominoes.length, and dominoes[i] is equivalent to dominoes[j].
Example:
Input: dominoes = [[1,2],[2,1],[3,4],[5,6]]
Output: 1
'''
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        stat = collections.defaultdict(int)
        total = 0
        for domino in dominoes:
            temp = tuple(sorted(domino))
            total += stat[temp]
            stat[temp] += 1
        return total

'''
1138. Alphabet Board Path
On an alphabet board, we start at position (0, 0), corresponding to character board[0][0].
Here, board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"], as shown in the diagram below.
We may make the following moves:
'U' moves our position up one row, if the position exists on the board;
'D' moves our position down one row, if the position exists on the board;
'L' moves our position left one column, if the position exists on the board;
'R' moves our position right one column, if the position exists on the board;
'!' adds the character board[r][c] at our current position (r, c) to the answer.
(Here, the only positions that exist on the board are positions with letters on them.)
Return a sequence of moves that makes our answer equal to target in the minimum number of moves.  You may return any path that does so.
Example:
Input: target = "leet"
Output: "DDR!UURRR!!DDD!"
'''
class Solution(object):
    def alphabetBoardPath(self, target):
        """
        :type target: str
        :rtype: str
        """
        posdic = {}
        for i, c in enumerate(string.ascii_lowercase):
            posdic[c] = [i//5, i%5]
        result = ''
        x0 = y0 = 0
        for t in target:
            x, y = posdic[t]
            if y < y0: result += 'L' * (y0-y)
            if x < x0: result += 'U' * (x0-x)
            if y > y0: result += 'R' * (y-y0)
            if x > x0: result += 'D' * (x-x0)
            result += '!'
            x0, y0 = x, y
        return result

'''
1140. Stone Game II
Alice and Bob continue their games with piles of stones.
There are a number of piles arranged in a row, and each pile has a positive integer number of stones piles[i].
The objective of the game is to end with the most stones. 
Alice and Bob take turns, with Alice starting first.  Initially, M = 1.
On each player's turn, that player can take all the stones in the first X remaining piles, where 1 <= X <= 2M.  Then, we set M = max(M, X).
The game continues until all the stones have been taken.
Assuming Alice and Bob play optimally, return the maximum number of stones Alice can get.
Example:
Input: piles = [2,7,9,4,4]
Output: 10
'''
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        n = len(piles)
        @lru_cache(maxsize=None)
        def minimax(st, m, player):
            if st >= n: return 0
            if player:
                return max([sum(piles[st:st+x]) + minimax(st+x, max(m,x), player^1) for x in range(1, 2*m+1)])
            else:
                return min([minimax(st+x, max(m,x), player^1) for x in range(1, 2*m+1)])
        return minimax(0, 1, 1)

'''
1143. Longest Common Subsequence
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.
Example:
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
'''
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        opt = [[0 for _ in range(len(text2)+1)] for _ in range(len(text1)+1)]
        for i in range(1, len(text1)+1):
            for j in range(1, len(text2)+1):
                if text1[i-1] == text2[j-1]:
                    opt[i][j] = opt[i-1][j-1] + 1
                else:
                    opt[i][j] = max(opt[i-1][j], opt[i][j-1])
        return opt[-1][-1]

'''
1146. Snapshot Array
Implement a SnapshotArray that supports the following interface:
SnapshotArray(int length) initializes an array-like data structure with the given length. Initially, each element equals 0.
void set(index, val) sets the element at the given index to be equal to val.
int snap() takes a snapshot of the array and returns the snap_id: the total number of times we called snap() minus 1.
int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id
Example:
Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
'''
class SnapshotArray:

    def __init__(self, length: int):
        self.cur_id = 0
        self.curVals = [0] * length
        self.snapIdArr = [[-1] for _ in range(length)]
        self.arrVal = [[0] for _ in range(length)]
        self.modified = set()

    def set(self, index: int, val: int) -> None:
        if val == self.arrVal[index][-1]:
            if index in self.modified: self.modified.remove(index)
            return
        self.curVals[index] = val
        if index not in self.modified: self.modified.add(index)

    def snap(self) -> int:
        for idx in self.modified:
            self.snapIdArr[idx].append(self.cur_id)
            self.arrVal[idx].append(self.curVals[idx])
        self.modified.clear()
        self.cur_id += 1
        return self.cur_id - 1

    def get(self, index: int, snap_id: int) -> int:
        arr = self.snapIdArr[index]
        l, r = 0, len(arr)
        while l < r:
            m = (l + r) // 2
            if arr[m] <= snap_id:
                l = m + 1
            else: r = m
        return self.arrVal[index][l-1]

'''
1156. Swap For Longest Repeated Character Substring
You are given a string text. You can swap two of the characters in the text.
Return the length of the longest substring with repeated characters.
Example:
Input: text = "ababa"
Output: 3
Explanation: We can swap the first 'b' with the last 'a', or the last 'b' with the first 'a'. Then, the longest repeated character substring is "aaa" with length 3.
'''
class Solution(object):
    def maxRepOpt1(self, text):
        """
        :type text: str
        :rtype: int
        """
        distribute = [[char, len(list(count))] for char, count in itertools.groupby(text)]
        freq = collections.Counter(text)
        res = max(min(count + 1, freq[char]) for char, count in distribute)
        for i in xrange(1, len(distribute) - 1):
            if distribute[i - 1][0] == distribute[i + 1][0] and distribute[i][1] == 1:
                res = max(res, min(distribute[i - 1][1] + distribute[i + 1][1] + 1, freq[distribute[i + 1][0]]))
        return res

'''
1160. Find Words That Can Be Formed by Characters
You are given an array of strings words and a string chars.
A string is good if it can be formed by characters from chars (each character can only be used once).
Return the sum of lengths of all good strings in words.
Example:
Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.
'''
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

'''
1161. Maximum Level Sum of a Binary Tree
Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.
Return the smallest level x such that the sum of all the values of nodes at level x is maximal.
Example:
Input: root = [1,7,0,7,-8,null,null]
Output: 2
'''
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> List[float]:
        queue = [root]
        max_level = 1
        max_sum = float('-inf')
        level = 1
        while queue:
            level_sum = 0
            next_level = []
            for node in queue:
                level_sum += node.val
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            if level_sum > max_sum:
                max_sum = level_sum
                max_level = level
            queue = next_level
            level += 1
        return max_level

'''
1171. Remove Zero Sum Consecutive Nodes from Linked List
Given the head of a linked list, we repeatedly delete consecutive sequences of nodes that sum to 0 until there are no such sequences.
After doing so, return the head of the final linked list.  You may return any such answer.
(Note that in the examples below, all sequences are serializations of ListNode objects.)
Example:
Input: head = [1,2,-3,3,1]
Output: [3,1]
Note: The answer [1,2,1] would also be accepted.
'''
class Solution(object):
    def removeZeroSumSublists(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(next = head)
        prefix = 0
        d = {0:dummy}
        while head:
            prefix += head.val
            d[prefix] = head
            head = head.next
        head = dummy
        prefix = 0
        while head:
            prefix += head.val
            head.next = d[prefix].next
            head = head.next
        return dummy.next

'''
1232. Check If It Is a Straight Line
You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.
Example:
Input: coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
Output: true
'''
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        deltaX = coordinates[1][0]-coordinates[0][0];
        deltaY = coordinates[1][1]-coordinates[0][1];
        for i in range(2, len(coordinates)):
            newDeltaX = coordinates[i][0]-coordinates[0][0];
            newDeltaY = coordinates[i][1]-coordinates[0][1];
            if newDeltaY*deltaX!=newDeltaX*deltaY:
                return False
        return True

'''
1207. Unique Number of Occurrences
Given an array of integers arr, return true if the number of occurrences of each value in the array is unique, or false otherwise.
Example:
Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.
'''
class Solution(object):
    def uniqueOccurrences(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        dic = {}
        for num in arr:
            if num not in dic:
                dic[num]=1
            else: 
                dic[num]+=1
        return len(dic.values())==len(set(dic.values()))

'''
1221. Split a String in Balanced Strings
Balanced strings are those that have an equal quantity of 'L' and 'R' characters.
Given a balanced string s, split it in the maximum amount of balanced strings.
Return the maximum amount of split balanced strings.
Example:
Input: s = "RLRRLLRLRL"
Output: 4
Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains same number of 'L' and 'R'.
'''
class Solution(object):
    def balancedStringSplit(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        count = 0
        for i in range(len(s)):
            if s[i] == 'L':
                count += 1 
            else:
                count -= 1
            if count == 0:
                result += 1
        return result

'''
1224. Maximum Equal Frequency
Given an array nums of positive integers, return the longest possible length of an array prefix of nums, 
such that it is possible to remove exactly one element from this prefix so that every number that has appeared in it will have the same number of occurrences.
If after removing one element there are no remaining elements, it's still considered that every appeared number has the same number of ocurrences (0).
Example:
Input: nums = [2,2,1,1,5,3,3,5]
Output: 7
Explanation: For the subarray [2,2,1,1,5,3,3] of length 7, if we remove nums[4] = 5, we will get [2,2,1,1,3,3], so that each number will appear exactly twice.
'''
class Solution(object):
    def maxEqualFreq(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        freq = collections.defaultdict(int)
        count = [0] * (len(nums)+1)
        result = 0
        for index, num in enumerate(nums, 1):
            count[freq[num]] -= 1
            freq[num] += 1
            count[freq[num]] += 1
            len_eqs = count[freq[num]] * freq[num]
            if len_eqs == index and index < len(nums):
                result = index + 1
            remainder = index - len_eqs
            if count[remainder] == 1 and remainder in [1, freq[num] + 1]:
                result = index
        return result

'''
1254. Number of Closed Islands
Given a 2D grid consists of 0s (land) and 1s (water).  An island is a maximal 4-directionally connected group of 0s and a closed island is an island totally (all left, top, right, bottom) surrounded by 1s.
Return the number of closed islands.
Example:
Input: grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
Output: 2
'''
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        def dfs(i, j):
            if i < 0 or j < 0 or i >= rows or j >= cols:
                return False
            if grid[i][j] == 1:
                return True
            grid[i][j] = 1 # mark as visited
            left = dfs(i, j-1)
            right = dfs(i, j+1)
            up = dfs(i-1, j)
            down = dfs(i+1, j)
            return left and right and up and down
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0 and dfs(i, j):
                    count += 1
        
        return count

##1300##

'''
1306. Jump Game III
Given an array of non-negative integers arr, you are initially positioned at start index of the array.
When you are at index i, you can jump to i + arr[i] or i - arr[i], check if you can reach to any index with value 0.
Notice that you can not jump outside of the array at any time.
Example:
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation: 
All possible ways to reach at index 3 with value 0 are: 
index 5 -> index 4 -> index 1 -> index 3 
index 5 -> index 6 -> index 4 -> index 1 -> index 3 
'''
class Solution(object):
    def canReach(self, arr, start):
        """
        :type arr: List[int]
        :type start: int
        :rtype: bool
        """
        queue, seen = collections.deque([start]), {start}
        while queue:
            ele = queue.popleft()
            if arr[ele] == 0:
                return True
            for nxt in [ele+arr[ele], ele-arr[ele]]:
                if nxt not in seen and 0 <= nxt < len(arr):
                    queue.append(nxt)
                    seen.add(nxt)
        return False

'''
1309. Decrypt String from Alphabet to Integer Mapping
You are given a string s formed by digits and '#'. We want to map s to English lowercase characters as follows:
Characters ('a' to 'i') are represented by ('1' to '9') respectively.
Characters ('j' to 'z') are represented by ('10#' to '26#') respectively.
Return the string formed after mapping.
The test cases are generated so that a unique mapping will always exist.
Example:
Input: s = "10#11#12"
Output: "jkab"
Explanation: "j" -> "10#" , "k" -> "11#" , "a" -> "1" , "b" -> "2".
'''
class Solution(object):
    def freqAlphabets(self, s):
        """
        :type s: str
        :rtype: str
        """
        lst,i = [],0
        dic = {i:chr(i+96) for i in range(1,27)}
        while i<len(s):
            if i+2<len(s) and s[i+2]=='#':
                lst.append(s[i:i+2])
                i+=3
            else:
                lst.append(s[i])
                i+=1
        return ''.join([dic[int(string)] for string in lst])

'''
1311. Get Watched Videos by Your Friends
There are n people, each person has a unique id between 0 and n-1. 
Given the arrays watchedVideos and friends, where watchedVideos[i] and friends[i] contain the list of watched videos and the list of friends respectively for the person with id = i.
Level 1 of videos are all watched videos by your friends, level 2 of videos are all watched videos by the friends of your friends and so on.
In general, the level k of videos are all watched videos by people with the shortest path exactly equal to k with you. 
Given your id and the level of videos, return the list of videos ordered by their frequencies (increasing). For videos with the same frequency order them alphabetically from least to greatest. 
Example:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 1
Output: ["B","C"] 
Explanation: 
You have id = 0 (green color in the figure) and your friends are (yellow color in the figure):
Person with id = 1 -> watchedVideos = ["C"] 
Person with id = 2 -> watchedVideos = ["B","C"] 
The frequencies of watchedVideos by your friends are: 
B -> 1 
C -> 2
'''
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

'''
1318. Minimum Flips to Make a OR b Equal to c
Given 3 positives numbers a, b and c. Return the minimum flips required in some bits of a and b to make ( a OR b == c ). (bitwise OR operation).
Flip operation consists of change any single bit 1 to 0 or change the bit 0 to 1 in their binary representation.
Example:
Input: a = 2, b = 6, c = 5
Output: 3
'''
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        result = 0
        while a or b or c:
            x1, x2, x3 = a&1, b&1, c&1
            if (x1 | x2) != x3:
                if x1 & x2:
                    result += 2
                else:
                    result += 1
            a, b, c = a>>1, b>>1, c>>1
        return result

'''
1319. Number of Operations to Make Network Connected
There are n computers numbered from 0 to n - 1 connected by ethernet cables connections forming a network where connections[i] = [ai, bi] represents a connection between computers ai and bi.
Any computer can reach any other computer directly or indirectly through the network.
You are given an initial computer network connections. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected.
Return the minimum number of times you need to do this in order to make all the computers connected. If it is not possible, return -1.
Example:
Input: n = 4, connections = [[0,1],[0,2],[1,2]]
Output: 1
Explanation: Remove cable between computer 1 and 2 and place between computers 1 and 3.
'''
class Solution(object):
    def __init__(self):
        self.parents = []
        self.count = []
        
    def makeConnected(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        if len(connections) < n-1:
            return -1  
        self.parents = [i for i in range(n)]
        self.count = [1 for _ in range(n)]
        for connection in connections:
            a, b = connection[0], connection[1]
            self.union(a, b)
        return len({self.find(i) for i in range(n)}) - 1
            
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

'''
1331. Rank Transform of an Array
Given an array of integers arr, replace each element with its rank.
The rank represents how large the element is. The rank has the following rules:
Rank is an integer starting from 1.
The larger the element, the larger the rank. If two elements are equal, their rank must be the same.
Rank should be as small as possible.
Example:
Input: arr = [40,10,20,30]
Output: [4,1,2,3]
'''
class Solution(object):
    def arrayRankTransform(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        rank = {}
        for a in sorted(arr):
            rank.setdefault(a, len(rank)+1)
        return [rank[i] for i in arr]

'''
1358. Number of Substrings Containing All Three Characters
Given a string s consisting only of characters a, b and c.
Return the number of substrings containing at least one occurrence of all these characters a, b and c.
Example:
Input: s = "abcabc"
Output: 10
Explanation: The substrings containing at least one occurrence of the characters a, b and c are "abc", "abca", "abcab", "abcabc", "bca", "bcab", "bcabc", "cab", "cabc" and "abc" (again). 
'''
class Solution(object):
    def numberOfSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = i = 0
        count = {c: 0 for c in 'abc'}
        for j in range(len(s)):
            count[s[j]] += 1
            while all(count.values()):
                count[s[i]] -= 1
                i += 1
            res += i
        return res

'''
1365. How Many Numbers Are Smaller Than the Current Number
Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it.
That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].
Return the answer in an array.
Example:
Input: nums = [8,1,2,2,3]
Output: [4,0,1,1,3]
Explanation: 
For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3). 
For nums[1]=1 does not exist any smaller number than it.
For nums[2]=2 there exist one smaller number than it (1). 
For nums[3]=2 there exist one smaller number than it (1). 
For nums[4]=3 there exist three smaller numbers than it (1, 2 and 2).
'''
class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sortlst = sorted(nums)
        dic = {}
        for i, num in enumerate(sortlst):
            if num not in dic:
                dic[num] = i
        for i, num in enumerate(nums):
            sortlst[i] = dic[num]
        return sortlst

'''
1366. Rank Teams by Votes
In a special ranking system, each voter gives a rank from highest to lowest to all teams participating in the competition.
The ordering of teams is decided by who received the most position-one votes. If two or more teams tie in the first position, we consider the second position to resolve the conflict, if they tie again, we continue this process until the ties are resolved.
If two or more teams are still tied after considering all positions, we rank them alphabetically based on their team letter.
You are given an array of strings votes which is the votes of all voters in the ranking systems. Sort all teams according to the ranking system described above.
Return a string of all teams sorted by the ranking system.
Example:
Input: votes = ["ABC","ACB","ABC","ACB","ACB"]
Output: "ACB"
'''
class Solution(object):
    def rankTeams(self, votes):
        """
        :type votes: List[str]
        :rtype: str
        """
        dic = collections.defaultdict(lambda : [0]*26)
        for vote in votes:
            for i, char in enumerate(vote):
                dic[char][i] += 1
        voted_names = sorted(dic.keys())
        return "".join(sorted(voted_names, key=lambda x: dic[x], reverse=True))

'''
1382. Balance a Binary Search Tree
Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them.
A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.
Example:
Input: root = [1,null,2,null,3,null,4,null,null]
Output: [2,1,3,null,null,null,4]
Explanation: This is not the only correct answer, [3,1,4,null,2] is also correct.
'''
class Solution(object):
    def balanceBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        result = []
        def ArrayToBST(cur):
                if not cur: return
                ArrayToBST(cur.left)
                result.append(cur.val)
                ArrayToBST(cur.right)
        ArrayToBST(root)
        return self.traversal(result,0,len(result)-1)
    
    def traversal(self,lst,left,right):
        if left>right:return
        mid = (left+right)//2
        midnode = TreeNode(lst[mid])
        midnode.left = self.traversal(lst,left,mid-1)
        midnode.right = self.traversal(lst,mid+1,right)
        return midnode

'''
1372. Longest ZigZag Path in a Binary Tree
You are given the root of a binary tree.
A ZigZag path for a binary tree is defined as follow:
Choose any node in the binary tree and a direction (right or left).
If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
Change the direction from right to left or from left to right.
Repeat the second and third steps until you can't move in the tree.
Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).
Return the longest ZigZag path contained in that tree.
Example:
Input: root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1,null,1]
Output: 3
'''
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        return max(self.trversal(root.left, 0, false), self.trversal(root.right, 0, true))

    def trversal(self, root, length, goleft):
        if not root: return length
        length += 1
        # goleft means should we should go left
        if goleft:
            leftlength = self.trversal(root.left, length, not goleft)
            rightlength = max(self.trversal(root.right, 0, goleft), length)
        else:
            leftlength = max(self.trversal(root.left, 0, goleft), length)
            rightlength = self.trversal(root.right, length, not goleft)
        return max(leftlength, rightlength)

'''
1376. Time Needed to Inform All Employees
A company has n employees with a unique ID for each employee from 0 to n - 1. The head of the company is the one with headID.
Each employee has one direct manager given in the manager array where manager[i] is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.
The head of the company wants to inform all the company employees of an urgent piece of news. He will inform his direct subordinates, and they will inform their subordinates, and so on until all employees know about the urgent news.
The i-th employee needs informTime[i] minutes to inform all of his direct subordinates (i.e., After informTime[i] minutes, all his direct subordinates can start spreading the news).
Return the number of minutes needed to inform all the employees about the urgent news.
Example:
Input: n = 1, headID = 0, manager = [-1], informTime = [0]
Output: 0
'''
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        arr = [[] for _ in range(n)]
        ans = 0
        for i in range(n):
            if manager[i] != -1:
                arr[manager[i]].append(i)

        q = deque([(headID, informTime[headID])])
        while q:
            size = len(q)

            for _ in range(size):
                t = q.popleft()

                for x in arr[t[0]]:
                    if informTime[x] == 0:
                        ans = max(ans, t[1])
                    else:
                        q.append((x, t[1] + informTime[x]))
        return ans

'''
1394. Find Lucky Integer in an Array
Given an array of integers arr, a lucky integer is an integer that has a frequency in the array equal to its value.
Return the largest lucky integer in the array. If there is no lucky integer return -1.
Example:
Input: arr = [2,2,3,4]
Output: 2
Explanation: The only lucky number in the array is 2 because frequency[2] == 2.
'''
class Solution:
    def findLucky(self, arr: List[int]) -> int:
        dic = collections.Counter(arr)
        result = -1
        for i, num in dic.items():
            if i == num: result = max(result, num)
        return result

'''
1395. Count Number of Teams
There are n soldiers standing in a line. Each soldier is assigned a unique rating value.
You have to form a team of 3 soldiers amongst them under the following rules:
Choose 3 soldiers with index (i, j, k) with rating (rating[i], rating[j], rating[k]).
A team is valid if: (rating[i] < rating[j] < rating[k]) or (rating[i] > rating[j] > rating[k]) where (0 <= i < j < k < n).
Return the number of teams you can form given the conditions. (soldiers can be part of multiple teams).
Example:
Input: rating = [2,5,3,4,1]
Output: 3
'''
class Solution(object):
    def numTeams(self, rating):
        """
        :type rating: List[int]
        :rtype: int
        """
        greater = collections.defaultdict(int)
        less = collections.defaultdict(int)
        n = len(rating)
        res = 0
        for i in range(n):
            for j in range(i+1, n):
                if rating[i] > rating[j]:
                    less[i] += 1
                else:
                    greater[i] += 1
        for i in range(n-2):
            for j in range(i+1, n):
                if rating[i] > rating[j]:
                    res += less[j]
                else:
                    res += greater[j]
        return res

'''
1396. Design Underground System
An underground railway system is keeping track of customer travel times between different stations. They are using this data to calculate the average time it takes to travel from one station to another.
Implement the UndergroundSystem class:
void checkIn(int id, string stationName, int t)
A customer with a card ID equal to id, checks in at the station stationName at time t.
A customer can only be checked into one place at a time.
void checkOut(int id, string stationName, int t)
A customer with a card ID equal to id, checks out from the station stationName at time t.
double getAverageTime(string startStation, string endStation)
Returns the average time it takes to travel from startStation to endStation.
The average time is computed from all the previous traveling times from startStation to endStation that happened directly, meaning a check in at startStation followed by a check out from endStation.
The time it takes to travel from startStation to endStation may be different from the time it takes to travel from endStation to startStation.
There will be at least one customer that has traveled from startStation to endStation before getAverageTime is called.
You may assume all calls to the checkIn and checkOut methods are consistent. If a customer checks in at time t1 then checks out at time t2, then t1 < t2. All events happen in chronological order.
Example:
Input
["UndergroundSystem","checkIn","checkIn","checkIn","checkOut","checkOut","checkOut","getAverageTime","getAverageTime","checkIn","getAverageTime","checkOut","getAverageTime"]
[[],[45,"Leyton",3],[32,"Paradise",8],[27,"Leyton",10],[45,"Waterloo",15],[27,"Waterloo",20],[32,"Cambridge",22],["Paradise","Cambridge"],["Leyton","Waterloo"],[10,"Leyton",24],["Leyton","Waterloo"],[10,"Waterloo",38],["Leyton","Waterloo"]]
Output
[null,null,null,null,null,null,null,14.00000,11.00000,null,11.00000,null,12.00000]
'''
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

'''
1399. Count Largest Group
You are given an integer n.
Each number from 1 to n is grouped according to the sum of its digits.
Return the number of groups that have the largest size.
Example 1:
Input: n = 13
Output: 4
Explanation: There are 9 groups in total, they are grouped according sum of its digits of numbers from 1 to 13:
[1,10], [2,11], [3,12], [4,13], [5], [6], [7], [8], [9].
There are 4 groups with largest size.
'''
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

##1400##

'''
1402. Reducing Dishes
A chef has collected data on the satisfaction level of his n dishes. Chef can cook any dish in 1 unit of time.
Like-time coefficient of a dish is defined as the time taken to cook that dish including previous dishes multiplied by its satisfaction level i.e. time[i] * satisfaction[i].
Return the maximum sum of like-time coefficient that the chef can obtain after dishes preparation.
Dishes can be prepared in any order and the chef can discard some dishes to get this maximum value.
Example:
Input: satisfaction = [-1,-8,0,5,-9]
Output: 14
'''
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        satisfaction.sort(reverse = True)
        res = total = 0
        for num in satisfaction:
            if num + total  > 0:
                total += num
                res += total
            else:
                break
        return res

'''
1406. Stone Game III
Alice and Bob continue their games with piles of stones. There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.
Alice and Bob take turns, with Alice starting first. On each player's turn, that player can take 1, 2, or 3 stones from the first remaining stones in the row.
The score of each player is the sum of the values of the stones taken. The score of each player is 0 initially.
The objective of the game is to end with the highest score, and the winner is the player with the highest score and there could be a tie. The game continues until all the stones have been taken.
Assume Alice and Bob play optimally.
Return "Alice" if Alice will win, "Bob" if Bob will win, or "Tie" if they will end the game with the same score.
Example:
Input: values = [1,2,3,7]
Output: "Bob"
'''
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)
        dp = [0,0,0]

        for i in range(n - 1, -1, -1):
            takeOne = stoneValue[i] - dp[(i + 1) % 3]
            takeTwo = float('-inf')
            if i + 1 < n:
                takeTwo = stoneValue[i] + stoneValue[i + 1] - dp[(i + 2) % 3];
            takeThree = float('-inf')
            if i + 2 < n:
                takeThree = stoneValue[i] + stoneValue[i + 1] + stoneValue[i + 2] - dp[(i + 3) % 3];
            dp[i % 3] = max(takeOne, takeTwo, takeThree)

        scoreDiff = dp[0]
        if scoreDiff > 0:
            return "Alice"
        elif scoreDiff < 0:
            return "Bob"
        else:
            return "Tie"

'''
1408. String Matching in an Array
Given an array of string words. Return all strings in words which is substring of another word in any order. 
String words[i] is substring of words[j], if can be obtained removing some characters to left and/or right side of words[j].
Example:
Input: words = ["mass","as","hero","superhero"]
Output: ["as","hero"]
Explanation: "as" is substring of "mass" and "hero" is substring of "superhero".
["hero","as"] is also a valid answer.
'''
class Solution(object):
    def stringMatching(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        total = '.'.join(words)
        res = []
        print(total)
        left,right = 0,0
        for word in words:
            right=right+len(word)+1
            if word in total[:left] or word in total[right:]:
                res.append(word)
            left=left+len(word)+1    
        return res

'''
1416. Restore The Array
A program was supposed to print an array of integers.
The program forgot to print whitespaces and the array is printed as a string of digits s and all we know is that all integers in the array were in the range [1, k] and there are no leading zeros in the array.
Given the string s and the integer k, return the number of the possible arrays that can be printed as s using the mentioned program. 
Since the answer may be very large, return it modulo 109 + 7.
Example:
Input: s = "1000", k = 10000
Output: 1
'''
class Solution:
    def numberOfArrays(self, s: str, k: int) -> int:
        n = len(s)
        klength = len(str(k))
        MOD = int(1e9+7)
        dp = [0 for _ in range(n+1)]
        dp[-1] = 1
        for i in range(n-1, -1, -1):
            if s[i] == '0':
                continue
            num, j = 0, i
            while j < n:
                if j-i+1 > klength: break
                if int(s[i:j+1]) > k: break
                num += dp[j+1]
                j += 1
            dp[i] = num % MOD
        return dp[0]

'''
1419. Minimum Number of Frogs Croaking
You are given the string croakOfFrogs, which represents a combination of the string "croak" from different frogs, that is, multiple frogs can croak at the same time, so multiple "croak" are mixed.
Return the minimum number of different frogs to finish all the croaks in the given string.
A valid "croak" means a frog is printing five letters 'c', 'r', 'o', 'a', and 'k' sequentially. The frogs have to print all five letters to finish a croak. If the given string is not a combination of a valid "croak" return -1.
Example:
Input: croakOfFrogs = "croakcroak"
Output: 1 
Explanation: One frog yelling "croak" twice.
'''
class Solution(object):
    def minNumberOfFrogs(self, croakOfFrogs):
        """
        :type croakOfFrogs: str
        :rtype: int
        """
        dic = collections.defaultdict(int)
        inuse, result = 0, 0
        for char in croakOfFrogs:
            if char == 'c':
                inuse += 1
            elif char == 'k':
                inuse -= 1
            dic[char] += 1
            result = max(inuse, result)
            if not dic['c'] >= dic['r'] >= dic['o'] >= dic['a'] >= dic['k']:
                return -1
        if inuse == 0 and len(set(dic.values()))==1:
            return result
        else: return -1

'''
1431. Kids With the Greatest Number of Candies
There are n kids with candies. You are given an integer array candies, where each candies[i] represents the number of candies the ith kid has, and an integer extraCandies, denoting the number of extra candies that you have.
Return a boolean array result of length n, where result[i] is true if, after giving the ith kid all the extraCandies, they will have the greatest number of candies among all the kids, or false otherwise.
Note that multiple kids can have the greatest number of candies.
Example:
Input: candies = [2,3,5,1,3], extraCandies = 3
Output: [true,true,true,false,true] 
'''
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_candies = max(candies)
        res = []
        for num in candies:
            res.append(num+extraCandies >= max_candies)
        return res

'''
1456. Maximum Number of Vowels in a Substring of Given Length
Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.
Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.
Example:
Input: s = "abciiidef", k = 3
Output: 3
'''
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        window = 0
        for i in range(k):
            if s[i] in vowels:
                window += 1
        maxwindow = window
        for i in range(k, len(s)):
            if s[i-k] in vowels:
                window -= 1
            if s[i] in vowels:
                window += 1
            if window == k:
                return window
            maxwindow = max(maxwindow, window)
        return maxwindow

'''
1472. Design Browser History
You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.
Implement the BrowserHistory class:
BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
void visit(string url) Visits url from the current page. It clears up all the forward history.
string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. 
Return the current url after moving back in history at most steps.
string forward(int steps) Move steps forward in history. 
If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.
Example:
Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]
'''
class PageNode:
    def __init__(self,val,prev=None,next=None):
        self.val = val
        self.prev = prev
        self.next = next

class BrowserHistory:

    def __init__(self, homepage: str):
        self.curPage = PageNode(homepage)

    def visit(self, url: str) -> None:
        node = self.curPage
        node.next = PageNode(url)
        node.next.prev = node
        self.curPage = self.curPage.next

    def back(self, steps: int) -> str:
        while self.curPage.prev and steps:
            self.curPage = self.curPage.prev
            steps -= 1
        return self.curPage.val

    def forward(self, steps: int) -> str:
        while self.curPage.next and steps:
            self.curPage = self.curPage.next
            steps -= 1
        return self.curPage.val

'''
1480. Running Sum of 1d Array
Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).
Return the running sum of nums.
Example:
Input: nums = [1,2,3,4]
Output: [1,3,6,10]
'''
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        res = []
        prefixSum  = 0
        for num in nums:
            prefixSum += num
            res.append(prefixSum)
        return res

'''
1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
Given a weighted undirected connected graph with n vertices numbered from 0 to n - 1, and an array edges where edges[i] = [ai, bi, weighti] represents a bidirectional and weighted edge between nodes ai and bi.
A minimum spanning tree (MST) is a subset of the graph's edges that connects all vertices without cycles and with the minimum possible total edge weight.
Find all the critical and pseudo-critical edges in the given graph's minimum spanning tree (MST).
An MST edge whose deletion from the graph would cause the MST weight to increase is called a critical edge. On the other hand, a pseudo-critical edge is that which can appear in some MSTs but not all.
Note that you can return the indices of the edges in any order.
Example:
Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
Output: [[0,1],[2,3,4,5]]
Success
Details 
Runtime: 2483 ms, faster than 100.00% of Python online submissions for Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree.
Memory Usage: 13.7 MB, less than 100.00% of Python online submissions for Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree.
'''
class Solution(object):
    def __init__(self):
        self.parents = []
        self.count = []
        
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
    
    def kruskal(self, edges, n, edge_not_use=None, edge_must_use=None):
        totalweight, c = 0, 1
        self.parents = [i for i in range(n)]
        self.count = [1 for _ in range(n)]
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
        
    def findCriticalAndPseudoCriticalEdges(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[List[int]]
        """
        edges = [(u, v, w, i) for i, (u, v, w) in enumerate(edges)]
        edges = sorted(edges, key=lambda x: x[2])
        result, min_weight = [[], []], self.kruskal(edges, n)
        for i, (nodea,nodeb,weight, index) in enumerate(edges):
            if self.kruskal(edges, n, i) > min_weight:
                result[0].append(index)
            elif self.kruskal(edges, n, None, (nodea,nodeb)) == min_weight:
                result[1].append(index)
        return result

'''
1491. Average Salary Excluding the Minimum and Maximum Salary
You are given an array of unique integers salary where salary[i] is the salary of the ith employee.
Return the average salary of employees excluding the minimum and maximum salary. Answers within 10-5 of the actual answer will be accepted.
Example:
Input: salary = [4000,3000,1000,2000]
Output: 2500.00000
'''
class Solution:
    def average(self, salary: List[int]) -> float:
        salary.sort()
        return (sum(salary[1:-1]))/(len(salary)-2)

'''
1493. Longest Subarray of 1's After Deleting One Element
Given a binary array nums, you should delete one element from it.
Return the size of the longest non-empty subarray containing only 1's in the resulting array. Return 0 if there is no such subarray.
Example:
Input: nums = [1,1,0,1]
Output: 3
'''
class Solution:
    def longestSubarray(self, nums):
        numZeros = 1
        left = 0
        for right in range(len(nums)):
            numZeros -= nums[right] == 0
            if numZeros < 0:
                numZeros += nums[left] == 0
                left += 1
        return right - left

'''
1497. Check If Array Pairs Are Divisible by k
Given an array of integers arr of even length n and an integer k.
We want to divide the array into exactly n / 2 pairs such that the sum of each pair is divisible by k.
Return true If you can find a way to do that or false otherwise.
Example:
Input: arr = [1,2,3,4,5,10,6,7,8,9], k = 5
Output: true
'''
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

##1500##

'''
1539. Kth Missing Positive Number
Given an array arr of positive integers sorted in a strictly increasing order, and an integer k.
Return the kth positive integer that is missing from this array.
Example:
Input: arr = [2,3,4,7,11], k = 5
Output: 9
'''
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        index = 0
        integer = 1
        while index < len(arr):
            if arr[index] != integer:
                k -= 1
                if k == 0:
                    return integer
            else:
                index += 1
            integer += 1
        return arr[-1] + k


'''
1545. Find Kth Bit in Nth Binary String
Given two positive integers n and k, the binary string Sn is formed as follows:
S1 = "0"
Si = Si - 1 + "1" + reverse(invert(Si - 1)) for i > 1
Where + denotes the concatenation operation, reverse(x) returns the reversed string x, and invert(x) inverts all the bits in x (0 changes to 1 and 1 changes to 0).
For example, the first four strings in the above sequence are:
S1 = "0"
S2 = "011"
S3 = "0111001"
S4 = "011100110110001"
Return the kth bit in Sn. It is guaranteed that k is valid for the given n.
Example:
Input: n = 3, k = 1
Output: "0"
'''
class Solution(object):
    def findKthBit(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        num = '0'
        for _ in range(n-1):
            num = num + '1' + ''.join('1' if x == '0' else '0' for x in num)[::-1]
        return num[k-1]

'''
1546. Maximum Number of Non-Overlapping Subarrays With Sum Equals Target
Given an array nums and an integer target, return the maximum number of non-empty non-overlapping subarrays such that the sum of values in each subarray is equal to target.
Example:
Input: nums = [1,1,1,1,1], target = 2
Output: 2
Explanation: There are 2 non-overlapping subarrays [1,1,1,1,1] with sum equals to target(2).
'''
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

'''
1547. Minimum Cost to Cut a Stick
Given a wooden stick of length n units. The stick is labelled from 0 to n. For example, a stick of length 6 is labelled as follows:
Given an integer array cuts where cuts[i] denotes a position you should perform a cut at.
You should perform the cuts in order, you can change the order of the cuts as you wish.
The cost of one cut is the length of the stick to be cut, the total cost is the sum of costs of all cuts. When you cut a stick, it will be split into two smaller sticks (i.e. the sum of their lengths is the length of the stick before the cut). Please refer to the first example for a better explanation.
Return the minimum total cost of the cuts.
Example:
Input: n = 7, cuts = [1,3,4,5]
Output: 16
'''
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        cuts.append(0)
        cuts.append(n)
        cuts.sort()
        m = len(cuts)
        dp = [[0 for _ in range(m)] for _ in range(m)]
        for l in range(2, m):
            for i in range(m - l):
                j = i + l
                dp[i][j] = float('inf')
                for k in range(i+1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j]+cuts[j] - cuts[i])
        return dp[0][m - 1]

'''
1557. Minimum Number of Vertices to Reach All Nodes
Given a directed acyclic graph, with n vertices numbered from 0 to n-1, and an array edges where edges[i] = [fromi, toi] represents a directed edge from node fromi to node toi.
Find the smallest set of vertices from which all nodes in the graph are reachable. It's guaranteed that a unique solution exists.
Notice that you can return the vertices in any order.
Example:
Input: n = 6, edges = [[0,1],[0,2],[2,5],[3,4],[4,2]]
Output: [0,3]
'''
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        des = set()
        for edge in edges:
            des.add(edge[1])
        res = []
        for i in range(n):
            if i not in des:
                res.append(i)
        return res



'''
1577. Number of Ways Where Square of Number Is Equal to Product of Two Numbers
Given two arrays of integers nums1 and nums2, return the number of triplets formed (type 1 and type 2) under the following rules:
Type 1: Triplet (i, j, k) if nums1[i]2 == nums2[j] * nums2[k] where 0 <= i < nums1.length and 0 <= j < k < nums2.length.
Type 2: Triplet (i, j, k) if nums2[i]2 == nums1[j] * nums1[k] where 0 <= i < nums2.length and 0 <= j < k < nums1.length.
Example:
Input: nums1 = [7,4], nums2 = [5,2,8,9]
Output: 1
Explanation: Type 1: (1, 1, 2), nums1[1]2 = nums2[1] * nums2[2]. (42 = 2 * 8). 
'''
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

'''
1579. Remove Max Number of Edges to Keep Graph Fully Traversable
Alice and Bob have an undirected graph of n nodes and three types of edges:
Type 1: Can be traversed by Alice only.
Type 2: Can be traversed by Bob only.
Type 3: Can be traversed by both Alice and Bob.
Given an array edges where edges[i] = [typei, ui, vi] represents a bidirectional edge of type typei between nodes ui and vi, find the maximum number of edges you can remove so that after removing the edges, the graph can still be fully traversed by both Alice and Bob.
The graph is fully traversed by Alice and Bob if starting from any node, they can reach all other nodes.
Return the maximum number of edges you can remove, or return -1 if Alice and Bob cannot fully traverse the graph.
Example:
Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
Output: 2
Explanation: If we remove the 2 edges [1,1,2] and [1,1,3]. The graph will still be fully traversable by Alice and Bob. Removing any additional edge will not make it so. So the maximum number of edges we can remove is 2.
'''
class Solution:
    def __init__(self):
        self.parents = []
        self.count = []
        self.result = 0

    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        edges_3 = ([t, u, v] for t, u, v in edges if t==3)
        edges_2 = ([t, u, v] for t, u, v in edges if t==2)
        edges_1 = ([t, u, v] for t, u, v in edges if t==1)
        self.parents = [i for i in range(n+1)]
        self.count = [1 for _ in range(n+1)]
        def help(eds):
            for _, u, v in eds:
                if self.find(u) == self.find(v):
                    self.result += 1
                else: self.union(u, v)
        help(edges_3)
        tempa,tempb = copy.deepcopy(self.parents), copy.deepcopy(self.count)
        help(edges_2)
        if not self.allconnected(n): return -1
        self.parents, self.count = tempa,tempb
        help(edges_1)
        if not self.allconnected(n): return -1
        return self.result

    def find(self, node):
        if (node != self.parents[node]):
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]
    
    def union(self, a, b):
        a_parent, b_parent = self.find(a), self.find(b)
        a_size, b_size = self.count[a_parent], self.count[b_parent]
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.count[b_parent] += a_size
            else:
                self.parents[b_parent] = a_parent
                self.count[a_parent] += b_size
    
    def allconnected(self, n):
        return len({self.find(i) for i in range(1, n+1)}) == 1
                
'''
1588. Sum of All Odd Length Subarrays
Given an array of positive integers arr, return the sum of all possible odd-length subarrays of arr.
A subarray is a contiguous subsequence of the array.
Example:
Input: arr = [1,4,2,5,3]
Output: 58
'''
class Solution(object):
    def sumOddLengthSubarrays(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        window = 1
        res = 0
        while window <= len(arr):
            for i in range(len(arr)-window+1):
                res += sum(arr[i:i+window])
            window += 2
        return res


'''
1590. Make Sum Divisible by P
Given an array of positive integers nums, remove the smallest subarray (possibly empty) such that the sum of the remaining elements is divisible by p. It is not allowed to remove the whole array.
Return the length of the smallest subarray that you need to remove, or -1 if it's impossible.
A subarray is defined as a contiguous block of elements in the array.
Example:
Input: nums = [3,1,4,2], p = 6
Output: 1
Explanation: The sum of the elements in nums is 10, which is not divisible by 6. We can remove the subarray [4], and the sum of the remaining elements is 6, which is divisible by 6.
'''
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

##1600##
'''
1603. Design Parking System
Design a parking system for a parking lot. The parking lot has three kinds of parking spaces: big, medium, and small, with a fixed number of slots for each size.
Implement the ParkingSystem class:
ParkingSystem(int big, int medium, int small) Initializes object of the ParkingSystem class. The number of slots for each parking space are given as part of the constructor.
bool addCar(int carType) Checks whether there is a parking space of carType for the car that wants to get into the parking lot. carType can be of three kinds: big, medium, or small, which are represented by 1, 2, and 3 respectively. A car can only park in a parking space of its carType. If there is no space available, return false, else park the car in that size space and return true.
Example:
Input
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]
Output
[null, true, true, false, false]
'''
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.spots = [big, medium, small]

    def addCar(self, carType: int) -> bool:
        if self.spots[carType - 1] > 0:
            self.spots[carType - 1] -= 1
            return True
        else:
            return False

'''
1636. Sort Array by Increasing Frequency
Given an array of integers nums, sort the array in increasing order based on the frequency of the values.
If multiple values have the same frequency, sort them in decreasing order.
Return the sorted array.
Example:
Input: nums = [1,1,2,2,2,3]
Output: [3,1,1,2,2,2]
'''
class Solution(object):
    def frequencySort(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        dic = collections.Counter(nums)
        for num, freq in sorted(sorted(dic.items(), key=lambda x:x[0], reverse=True), key=lambda x:x[1]):
            for i in range(freq):
                result.append(num)
        return result

'''
1638. Count Substrings That Differ by One Character
Given two strings s and t, find the number of ways you can choose a non-empty substring of s and replace a single character by a different character such that the resulting substring is a substring of t. In other words, find the number of substrings in s that differ from some substring in t by exactly one character.
For example, the underlined substrings in "computer" and "computation" only differ by the 'e'/'a', so this is a valid way.
Return the number of substrings that satisfy the condition above.
A substring is a contiguous sequence of characters within a string.
Example:
Input: s = "aba", t = "baba"
Output: 6
'''
class Solution(object):
    def countSubstrings(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        def check(s1, s2):
            count = 0
            for a, b in zip(s1, s2):
                if a != b:
                    count += 1
                if count == 2:
                    return False
            return count == 1
        min_len = min(len(s), len(t))
        result = 0
        for d in range(1, min_len+1):
            for i in range(len(s)-d+1):
                s1 = s[i:i+d]
                for j in range(len(t)-d+1):
                    s2 = t[j:j+d]
                    result += check(s1, s2)
        return result

'''
1658. Minimum Operations to Reduce X to Zero
You are given an integer array nums and an integer x. In one operation, you can either remove the leftmost or the rightmost element from the array nums and subtract its value from x. Note that this modifies the array for future operations.
Return the minimum number of operations to reduce x to exactly 0 if it is possible, otherwise, return -1.
Example:
Input: nums = [1,1,4,2,3], x = 5
Output: 2
Explanation: The optimal solution is to remove the last two elements to reduce x to zero.
'''
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        target = -x
        for i in nums:
            target += i
        if target == 0: return len(nums)
        dic = {0:-1}
        totalsum, res = 0, float('-inf')
        for i,num in enumerate(nums):
            totalsum += num
            if totalsum-target in dic:
                res = max(res, i-dic[totalsum-target])
            dic[totalsum] = i
        return -1 if res==float('-inf') else len(nums)-res

'''
1675. Minimize Deviation in Array
You are given an array nums of n positive integers.
You can perform two types of operations on any element of the array any number of times:
If the element is even, divide it by 2.
For example, if the array is [1,2,3,4], then you can do this operation on the last element, and the array will be [1,2,3,2].
If the element is odd, multiply it by 2.
For example, if the array is [1,2,3,4], then you can do this operation on the first element, and the array will be [2,2,3,4].
The deviation of the array is the maximum difference between any two elements in the array.
Return the minimum deviation the array can have after performing some number of operations.
Example:
Input: nums = [1,2,3,4]
Output: 1
'''
class Solution:
    def minimumDeviation(self, nums: List[int]) -> int:
        pq = [-num*2 if num%2==1 else -num for num in nums]
        heapq.heapify(pq)
        min_val = float('inf')
        min_deviation = float('inf')
        for num in nums:
            min_val = min(min_val, num if num%2 == 0 else num*2)
        while True:
            max_val = -heapq.heappop(pq)
            min_deviation = min(min_deviation, max_val - min_val)
            if max_val % 2 == 1:
                break
            max_val //= 2
            min_val = min(min_val, max_val)
            heapq.heappush(pq, -max_val)
        return min_deviation

'''
1697. Checking Existence of Edge Length Limited Paths
An undirected graph of n nodes is defined by edgeList, where edgeList[i] = [ui, vi, disi] denotes an edge between nodes ui and vi with distance disi. Note that there may be multiple edges between two nodes.
Given an array queries, where queries[j] = [pj, qj, limitj], your task is to determine for each queries[j] whether there is a path between pj and qj such that each edge on the path has a distance strictly less than limitj .
Return a boolean array answer, where answer.length == queries.length and the jth value of answer is true if there is a path for queries[j] is true, and false otherwise.
Example:
Input: n = 3, edgeList = [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], queries = [[0,1,2],[0,2,5]]
Output: [false,true]
'''
class Solution:
    def __init__(self):
        self.parents = []

    def distanceLimitedPathsExist(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[bool]:
        self.parents = [i for i in range(n)]
        edges = sorted(edges, key=lambda x: x[2])
        result = [False for _ in range(len(queries))]
        i = 0
        for index, (u, v, limit) in sorted(enumerate(queries), key=lambda x: x[1][2]):
            while i < len(edges) and edges[i][2] < limit:
                self.union(edges[i][0], edges[i][1])
                i += 1
            result[index] = self.find(u)==self.find(v)
        return result

    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        x_parent, y_parent = self.find(x), self.find(y)
        if x_parent != y_parent:
            self.parents[x_parent] = y_parent

##1700##

'''
1706. Where Will the Ball Fall
You have a 2-D grid of size m x n representing a box, and you have n balls. The box is open on the top and bottom sides.
Each cell in the box has a diagonal board spanning two corners of the cell that can redirect a ball to the right or to the left.
A board that redirects the ball to the right spans the top-left corner to the bottom-right corner and is represented in the grid as 1.
A board that redirects the ball to the left spans the top-right corner to the bottom-left corner and is represented in the grid as -1.
We drop one ball at the top of each column of the box. Each ball can get stuck in the box or fall out of the bottom. A ball gets stuck if it hits a "V" shaped pattern between two boards or if a board redirects the ball into either wall of the box.
Return an array answer of size n where answer[i] is the column that the ball falls out of at the bottom after dropping the ball from the ith column at the top, or -1 if the ball gets stuck in the box.
Example:
Input: grid = [[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]]
Output: [1,-1,-1,-1,-1]
'''
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        balls = [i for i in range(len(grid[0]))]
        for row in grid:
            balls = [self.single(i, row) if i>-1 else -1 for i in balls]
        return balls

    def single(self, pos, row):
        if (pos==0 and row[0]==-1) or (pos==len(row)-1 and row[-1]==1):
            return -1
        if row[pos] == 1:
            if row[pos+1] == -1:
                return -1
            else: return pos + 1
        else:
            if row[pos-1] == 1:
                return -1
            else: return pos - 1

'''
1711. Count Good Meals
A good meal is a meal that contains exactly two different food items with a sum of deliciousness equal to a power of two.
You can pick any two different foods to make a good meal.
Given an array of integers deliciousness where deliciousness[i] is the deliciousness of the i​​​​​​th​​​​​​​​ item of food, return the number of different good meals you can make from this list modulo 109 + 7.
Note that items with different indices are considered different even if they have the same deliciousness value.
Example:
Input: deliciousness = [1,3,5,7,9]
Output: 4
Explanation: The good meals are (1,3), (1,7), (3,5) and, (7,9).
Their respective sums are 4, 8, 8, and 16, all of which are powers of 2.
'''
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

'''
1721. Swapping Nodes in a Linked List
You are given the head of a linked list, and an integer k.
Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).
Example :
Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]
'''
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        prev = right = ListNode(0, head)
        for i in range(k):
            prev = prev.next
            right = right.next
        left = ListNode(0, head)
        while right != None:
            left = left.next
            right = right.next
        prev.val, left.val = left.val, prev.val
        return head

'''
1732. Find the Highest Altitude
There is a biker going on a road trip. The road trip consists of n + 1 points at different altitudes. The biker starts his trip on point 0 with altitude equal 0.
You are given an integer array gain of length n where gain[i] is the net gain in altitude between points i​​​​​​ and i + 1 for all (0 <= i < n). Return the highest altitude of a point.
Example:
Input: gain = [-5,1,5,0,-7]
Output: 1
'''
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        maxAltitude = 0
        currentAltitude = 0
        for g in gain:
            currentAltitude += g
            maxAltitude = max(maxAltitude, currentAltitude)
        return maxAltitude

'''
1768. Merge Strings Alternately
You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.
Return the merged string.
Example:
Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
'''
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        res = ""
        left, right = 0, 0
        while left < len(word1) and right < len(word2):
            res += word1[left]
            res += word2[right]
            left += 1
            right += 1
        if left < len(word1):
            res += word1[left:]
        else:
            res += word2[right:]
        return res


'''
1798. Maximum Number of Consecutive Values You Can Make
You are given an integer array coins of length n which represents the n coins that you own.
The value of the ith coin is coins[i]. 
You can make some value x if you can choose some of your n coins such that their values sum up to x.
Return the maximum number of consecutive integer values that you can make with your coins starting from and including 0.
Note that you may have multiple coins of the same value.
Example:
Input: coins = [1,3]
Output: 2
Explanation: You can make the following values:
- 0: take []
- 1: take [1]
You can make 2 consecutive integer values starting from 0.
'''
class Solution(object):
    def getMaximumConsecutive(self, coins):
        """
        :type coins: List[int]
        :rtype: int
        """
        result = 0
        for num in sorted(coins):
            if num > result+1:
                return result+1
            result += num
        return result+1

##1800##

'''
1802. Maximum Value at a Given Index in a Bounded Array
You are given three positive integers: n, index, and maxSum. You want to construct an array nums (0-indexed) that satisfies the following conditions:
nums.length == n
nums[i] is a positive integer where 0 <= i < n.
abs(nums[i] - nums[i+1]) <= 1 where 0 <= i < n-1.
The sum of all the elements of nums does not exceed maxSum.
nums[index] is maximized.
Return nums[index] of the constructed array.
Note that abs(x) equals x if x >= 0, and -x otherwise.
Example:
Input: n = 4, index = 2,  maxSum = 6
Output: 2
'''
class Solution:
    def check(self, mid):
        left_offset = max(mid - self.index, 0)
        result = (mid + left_offset) * (mid - left_offset + 1) // 2
        right_offset = max(mid - ((self.n - 1) - self.index), 0)
        result += (mid + right_offset) * (mid - right_offset + 1) // 2
        return result - mid

    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        self.n, self.index = n, index
        maxSum -= n
        left, right = 0, maxSum
        while left < right:
            mid = (left + right + 1) // 2
            if self.check(mid) <= maxSum:
                left = mid
            else:
                right = mid - 1
        result = left + 1
        return result

'''
1822. Sign of the Product of an Array
There is a function signFunc(x) that returns:
1 if x is positive.
-1 if x is negative.
0 if x is equal to 0.
You are given an integer array nums. Let product be the product of all values in the array nums.
Return signFunc(product).
Example:
Input: nums = [-1,-2,-3,-4,3,2,1]
Output: 1
'''
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        sign = 1
        for num in nums:
            if num == 0: return 0
            elif num < 0: sign *= -1
        return sign

'''
1857. Largest Color Value in a Directed Graph
There is a directed graph of n colored nodes and m edges. The nodes are numbered from 0 to n - 1.
You are given a string colors where colors[i] is a lowercase English letter representing the color of the ith node in this graph (0-indexed).
You are also given a 2D array edges where edges[j] = [aj, bj] indicates that there is a directed edge from node aj to node bj.
A valid path in the graph is a sequence of nodes x1 -> x2 -> x3 -> ... -> xk such that there is a directed edge from xi to xi+1 for every 1 <= i < k.
The color value of the path is the number of nodes that are colored the most frequently occurring color along that path.
Return the largest color value of any valid path in the given graph, or -1 if the graph contains a cycle.
Example:
Input: colors = "abaca", edges = [[0,1],[0,2],[2,3],[3,4]]
Output: 3
'''
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        l = len(colors)
        d = collections.defaultdict(set) # child node set
        indegree = [0 for i in range(l)] # indegree
        for s, e in edges:
            d[s].add(e)
            indegree[e] += 1
        deq = collections.deque() # queue
        opt = [[0 for i in range(26)] for j in range(l)] # opt of max color
        for i in range(l):
            if indegree[i] == 0: # if head
                deq.append(i) # add into queue
        while deq:
            i = deq.popleft()
            opt[i][ord(colors[i]) - ord("a")] += 1 # add on the color vlaue
            for j in d[i]: # the later child node
                for k in range(26): # update maxvlaue of child node
                    opt[j][k] = max(opt[i][k], opt[j][k])
                indegree[j] -= 1 # update indegree
                if indegree[j] == 0: # if degree is 0 then no later path
                    deq.append(j) # add into queue
        if sum(indegree) > 0: # there is circle if not finish
            return -1
        return max([max(i) for i in opt]) # no circle, this is finsih result

'''
1909. Remove One Element to Make the Array Strictly Increasing
Given a 0-indexed integer array nums, return true if it can be made strictly increasing after removing exactly one element, or false otherwise.
If the array is already strictly increasing, return true.
The array nums is strictly increasing if nums[i - 1] < nums[i] for each index (1 <= i < nums.length).
Example:
Input: nums = [1,2,10,5,7]
Output: true
'''
class Solution(object):
    def canBeIncreasing(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        count,index = 0,-1
        for i in range(len(nums)-1):
            if nums[i] >= nums[i+1]:
                count += 1
                index = i
        if count==0: return True
        if count==1:
            if index == 0 or index == len(nums)-2:
                return True
            if nums[index-1] < nums[index+1] or (index + 2 and  nums[index] < nums[index+2]):
                return True
        return False

##2000##

'''
2090. K Radius Subarray Averages
You are given a 0-indexed array nums of n integers, and an integer k.
The k-radius average for a subarray of nums centered at some index i with the radius k is the average of all elements in nums between the indices i - k and i + k (inclusive). If there are less than k elements before or after the index i, then the k-radius average is -1.
Build and return an array avgs of length n where avgs[i] is the k-radius average for the subarray centered at index i.
The average of x elements is the sum of the x elements divided by x, using integer division. The integer division truncates toward zero, which means losing its fractional part.
For example, the average of four elements 2, 3, 1, and 5 is (2 + 3 + 1 + 5) / 4 = 11 / 4 = 2.75, which truncates to 2.
Example 1:
Input: nums = [7,4,3,9,1,8,5,2,6], k = 3
Output: [-1,-1,-1,5,4,4,-1,-1,-1]
'''
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        windowSize = 2 * k + 1
        windowSum = 0
        result = [-1] * n
        if windowSize > n: 
            return result
        for i in range(n):
            windowSum += nums[i]
            if i - windowSize >= 0:
                windowSum -= nums[i - windowSize]
            if i >= windowSize - 1:
                result[i - k] = windowSum // windowSize
        return result

'''
2048. Next Greater Numerically Balanced Number
An integer x is numerically balanced if for every digit d in the number x, there are exactly d occurrences of that digit in x.
Given an integer n, return the smallest numerically balanced number strictly greater than n.
Example:
Input: n = 1
Output: 22
'''
class Solution(object):
    def nextBeautifulNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        from itertools import permutations
        mylist = [1, 22, 122, 333, 1333, 4444, 14444, 22333, 55555, 122333, 155555, 224444, 666666,1224444]
        res=[]
        def digit_combination(a,length):
            a = list(str(a))
            comb = permutations(a, length)
            for each in comb:
                s = [str(i) for i in each]
                result = int("".join(s))
                res.append(result)
        for every in mylist:
            digit_combination(every,len(str(every)))
        res.sort()
        for idx in res:
            if(idx>n):
                return idx

##2100##

'''
2101. Detonate the Maximum Bombs
You are given a list of bombs. The range of a bomb is defined as the area where its effect can be felt. This area is in the shape of a circle with the center as the location of the bomb.
The bombs are represented by a 0-indexed 2D integer array bombs where bombs[i] = [xi, yi, ri]. xi and yi denote the X-coordinate and Y-coordinate of the location of the ith bomb, whereas ri denotes the radius of its range.
You may choose to detonate a single bomb. When a bomb is detonated, it will detonate all bombs that lie in its range. These bombs will further detonate the bombs that lie in their ranges.
Given the list of bombs, return the maximum number of bombs that can be detonated if you are allowed to detonate only one bomb.
Example:
Input: bombs = [[2,1,3],[6,1,4]]
Output: 2
'''
class Solution:
    def __init__(self):
        self.count = 0
        self.linkvisit = []
    
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        n = len(bombs)
        result = 0
        self.linkvisit = [False for _ in range(n)]
        for i in range(n):
            if self.linkvisit[i]: continue
            self.count = 0
            self.dfs(i, [False for _ in range(n)], bombs)
            result = max(result, self.count)
        return result

    def dfs(self, idx, visit, bombs):
        self.count += 1
        visit[idx] = True
        for i in range(len(bombs)):
            if not visit[i] and self.inRange(bombs[idx], bombs[i]):
                self.linkvisit[i] = True
                visit[i] = True
                self.dfs(i, visit, bombs)

    def inRange(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) <= (a[2]**2)

'''
2114. Maximum Number of Words Found in Sentences
A sentence is a list of words that are separated by a single space with no leading or trailing spaces.
You are given an array of strings sentences, where each sentences[i] represents a single sentence.
Return the maximum number of words that appear in a single sentence.
Example:
Input: sentences = ["alice and bob love leetcode", "i think so too", "this is great thanks very much"]
Output: 6
'''
class Solution(object):
    def mostWordsFound(self, sentences):
        """
        :type sentences: List[str]
        :rtype: int
        """
        return max([len(x.split()) for x in sentences])

'''
2122. Recover the Original Array
Alice had a 0-indexed array arr consisting of n positive integers. She chose an arbitrary positive integer k and created two new 0-indexed integer arrays lower and higher in the following manner:
lower[i] = arr[i] - k, for every index i where 0 <= i < n
higher[i] = arr[i] + k, for every index i where 0 <= i < n
Unfortunately, Alice lost all three arrays. However, she remembers the integers that were present in the arrays lower and higher, but not the array each integer belonged to. Help Alice and recover the original array.
Given an array nums consisting of 2n integers, where exactly n of the integers were present in lower and the remaining in higher, return the original array arr. In case the answer is not unique, return any valid array.
Note: The test cases are generated such that there exists at least one valid array arr.
Example:
Input: nums = [2,10,6,4,8,12]
Output: [3,7,11]
Explanation:
If arr = [3,7,11] and k = 1, we get lower = [2,6,10] and higher = [4,8,12].
Combining lower and higher gives us [2,6,10,4,8,12], which is a permutation of nums.
Another valid possibility is that arr = [5,7,9] and k = 3. In that case, lower = [2,4,6] and higher = [8,10,12]. 
'''
class Solution(object):
    def recoverArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def check(nums, k, cnt):
            ans = []
            for num in nums:
                if cnt[num] == 0: 
                    continue
                if cnt[num + k] == 0: 
                    return False, []
                cnt[num] -= 1
                cnt[num + k] -= 1
                ans += [num + k//2]
            return True, ans
        
        nums = sorted(nums)
        cnt = Counter(nums)
        for i in range(1, len(nums)):
            k = nums[i] - nums[0]
            if k != 0 and k % 2 == 0:
                a, b = check(nums, k, cnt.copy())
                if a: return b

'''
2130. Maximum Twin Sum of a Linked List
In a linked list of size n, where n is even, the ith node (0-indexed) of the linked list is known as the twin of the (n-1-i)th node, if 0 <= i <= (n / 2) - 1.
For example, if n = 4, then node 0 is the twin of node 3, and node 1 is the twin of node 2. These are the only nodes with twins for n = 4.
The twin sum is defined as the sum of a node and its twin.
Given the head of a linked list with even length, return the maximum twin sum of the linked list.
Example:
Input: head = [5,4,2,1]
Output: 6
'''
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        prev, slow, fast = None, head, head
        while fast and fast.next:
            fast = fast.next.next
            nextNode = slow.next
            slow.next = prev
            prev = slow
            slow = nextNode
        maxSum = -float('inf')
        while slow:
            maxSum = max(maxSum, slow.val + prev.val)
            slow = slow.next
            prev = prev.next
        return maxSum

'''
2161. Partition Array According to Given Pivot
You are given a 0-indexed integer array nums and an integer pivot. Rearrange nums such that the following conditions are satisfied:
Every element less than pivot appears before every element greater than pivot.
Every element equal to pivot appears in between the elements less than and greater than pivot.
The relative order of the elements less than pivot and the elements greater than pivot is maintained.
More formally, consider every pi, pj where pi is the new position of the ith element and pj is the new position of the jth element. For elements less than pivot, if i < j and nums[i] < pivot and nums[j] < pivot, then pi < pj. Similarly for elements greater than pivot, if i < j and nums[i] > pivot and nums[j] > pivot, then pi < pj.
Return nums after the rearrangement.
Example:
Input: nums = [9,12,5,10,14,3,10], pivot = 10
Output: [9,5,3,10,10,12,14]
'''
class Solution(object):
    def pivotArray(self, nums, pivot):
        """
        :type nums: List[int]
        :type pivot: int
        :rtype: List[int]
        """
        left,right = [],[]
        same = 0
        for num in nums:
            if num < pivot:
                left.append(num)
            elif num == pivot:
                same += 1
            else:
                right.append(num)
        return left + [pivot]*same + right

'''
2187. Minimum Time to Complete Trips
You are given an array time where time[i] denotes the time taken by the ith bus to complete one trip.
Each bus can make multiple trips successively; that is, the next trip can start immediately after completing the current trip. Also, each bus operates independently; that is, the trips of one bus do not influence the trips of any other bus.
You are also given an integer totalTrips, which denotes the number of trips all buses should make in total. Return the minimum time required for all buses to complete at least totalTrips trips.
Example:
Input: time = [1,2,3], totalTrips = 5
Output: 3
'''
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        left = 1
        right = min(time) * totalTrips
        while left<right:
            mid = (left+right)//2
            if sum(mid//t for t in time) >= totalTrips:
                right = mid
            else:
                left = mid+1
        return left

'''
2200. Find All K-Distant Indices in an Array
You are given a 0-indexed integer array nums and two integers key and k. A k-distant index is an index i of nums for which there exists at least one index j such that |i - j| <= k and nums[j] == key.
Return a list of all k-distant indices sorted in increasing order.
Example:
Input: nums = [3,4,9,1,3,9,5], key = 9, k = 1
Output: [1,2,3,4,5,6]
'''
class Solution(object):
    def findKDistantIndices(self, nums, key, k):
        """
        :type nums: List[int]
        :type key: int
        :type k: int
        :rtype: List[int]
        """
        result = []
        prev = 0
        for i, num in enumerate(nums):
            if num == key:
                result += range(max(prev, i-k), min(i+k, len(nums)-1)+1)
                prev = min(i+k, len(nums)-1)+1
        return result

'''
2218. Maximum Value of K Coins From Piles
There are n piles of coins on a table. Each pile consists of a positive number of coins of assorted denominations.
In one move, you can choose any coin on top of any pile, remove it, and add it to your wallet.
Given a list piles, where piles[i] is a list of integers denoting the composition of the ith pile from top to bottom, and a positive integer k, return the maximum total value of coins you can have in your wallet if you choose exactly k coins optimally.
Example:
Input: piles = [[1,100,3],[7,8,9]], k = 2
Output: 101
'''
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        opt = [[0 for _ in range(k + 1)] for _ in range(len(piles) + 1)]
        return self.func(piles, 0, k, opt)
    
    def func(self, piles, i, k, opt):
        if opt[i][k] > 0: return opt[i][k]
        if i == len(piles) or k == 0: return 0
        res = self.func(piles, i + 1, k, opt)
        cur = 0
        j = 0
        while j<len(piles[i]) and j<k:
            cur += piles[i][j]
            res = max(res, self.func(piles, i + 1, k - j - 1, opt) + cur)
            j += 1
        opt[i][k] = res
        return res

'''
2248. Intersection of Multiple Arrays
Given a 2D integer array nums where nums[i] is a non-empty array of distinct positive integers, return the list of integers that are present in each array of nums sorted in ascending order.
Example:
Input: nums = [[3,1,2,4,5],[1,2,3,4],[3,4,5,6]]
Output: [3,4]
'''
class Solution(object):
    def intersection(self, nums):
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        result = set(nums[0])
        for i in range(1, len(nums)):
            result = result & set(nums[i])
        return sorted(list(result))

##2300##

'''
2300. Successful Pairs of Spells and Potions
You are given two positive integer arrays spells and potions, of length n and m respectively, where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.
You are also given an integer success. A spell and potion pair is considered successful if the product of their strengths is at least success.
Return an integer array pairs of length n where pairs[i] is the number of potions that will form a successful pair with the ith spell.
Example:
Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
Output: [4,0,3]
'''
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        n, m = len(spells), len(potions)
        potions.sort()
        res = []
        for spell in spells:
            left = 0
            right = m
            while left < right:
                mid = left + (right-left)//2
                if spell * potions[mid] >= success:
                    right = mid
                else: left = mid + 1
            res.append(m - left)
        return res
'''
2305. Fair Distribution of Cookies
You are given an integer array cookies, where cookies[i] denotes the number of cookies in the ith bag. You are also given an integer k that denotes the number of children to distribute all the bags of cookies to. All the cookies in the same bag must go to the same child and cannot be split up.
The unfairness of a distribution is defined as the maximum total cookies obtained by a single child in the distribution.
Return the minimum unfairness of all distributions.
Example:
Input: cookies = [8,15,10,20,8], k = 2
Output: 31
Explanation: One optimal distribution is [8,15,8] and [10,20]
- The 1st child receives [8,15,8] which has a total of 8 + 15 + 8 = 31 cookies.
- The 2nd child receives [10,20] which has a total of 10 + 20 = 30 cookies.
The unfairness of the distribution is max(31,30) = 31.
It can be shown that there is no distribution with an unfairness less than 31.
'''
class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        result = float("inf")
        distriChild = [0 for _ in range(k)]
        def traversal(start):
            nonlocal result
            temp = max(distriChild)
            if start == len(cookies):
                result = min(temp, result)
                return
            if result <= temp:
                return
            for i in range(k):
                distriChild[i] += cookies[start]
                traversal(start+1)
                distriChild[i] -= cookies[start]
        traversal(0)
        return result

'''
2316. Count Unreachable Pairs of Nodes in an Undirected Graph
You are given an integer n. There is an undirected graph with n nodes, numbered from 0 to n - 1. You are given a 2D integer array edges where edges[i] = [ai, bi] denotes that there exists an undirected edge connecting nodes ai and bi.
Return the number of pairs of different nodes that are unreachable from each other.
Example:
Input: n = 3, edges = [[0,1],[0,2],[1,2]]
Output: 0
'''
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

'''
2326. Spiral Matrix IV
You are given two integers m and n, which represent the dimensions of a matrix.
You are also given the head of a linked list of integers.
Generate an m x n matrix that contains the integers in the linked list presented in spiral order (clockwise), starting from the top-left of the matrix. If there are remaining empty spaces, fill them with -1.
Return the generated matrix.
Example:
Input: m = 3, n = 5, head = [3,0,2,6,8,1,7,9,4,2,5,5,0]
Output: [[3,0,2,6,8],[5,0,-1,-1,1],[5,2,4,9,7]]
'''
class Solution(object):
    def spiralMatrix(self, m, n, head):
        """
        :type m: int
        :type n: int
        :type head: Optional[ListNode]
        :rtype: List[List[int]]
        """
        result = [[-1 for _ in range(n)] for _ in range(m)]
        x, y, di, dj = 0, 0, 0, 1
        while head:
            result[x][y] = head.val
            nextx, nexty = x+di, y+dj
            if not (0<= nextx <m and 0<= nexty <n and (result[nextx][nexty] == -1)):
                di, dj = dj, -di
            x, y = x+di, y+dj
            head = head.next
        return result

'''
2336. Smallest Number in Infinite Set
You have a set which contains all positive integers [1, 2, 3, 4, 5, ...].
Implement the SmallestInfiniteSet class:
SmallestInfiniteSet() Initializes the SmallestInfiniteSet object to contain all positive integers.
int popSmallest() Removes and returns the smallest integer contained in the infinite set.
void addBack(int num) Adds a positive integer num back into the infinite set, if it is not already in the infinite set.
Example:
Input
["SmallestInfiniteSet", "addBack", "popSmallest", "popSmallest", "popSmallest", "addBack", "popSmallest", "popSmallest", "popSmallest"]
[[], [2], [], [], [], [1], [], [], []]
Output
[null, null, 1, 2, 3, null, 1, 4, 5]
'''
class SmallestInfiniteSet:

    def __init__(self):
        self.heap = []
        self.curr = 1
        
    def popSmallest(self) -> int:
        result = self.curr
        if self.heap and self.heap[0] < result:
            result = heappop(self.heap)
        else:
            self.curr += 1

        while self.heap and result == self.heap[0]:
            heappop(self.heap)

        return result

    def addBack(self, num: int) -> None:
        heappush(self.heap, num)

'''
2348. Number of Zero-Filled Subarrays
Given an integer array nums, return the number of subarrays filled with 0.
A subarray is a contiguous non-empty sequence of elements within an array.
Example:
Input: nums = [1,3,0,0,2,0,0,4]
Output: 6
'''
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        appear = 0
        res = 0
        for num in nums:
            if num == 0:
                appear += 1
                res += appear
            else:
                appear = 0
        return res

'''
2352. Equal Row and Column Pairs
Given a 0-indexed n x n integer matrix grid, return the number of pairs (ri, cj) such that row ri and column cj are equal.
A row and column pair is considered equal if they contain the same elements in the same order (i.e., an equal array).
Example:
Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
Output: 1
'''
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        dic = defaultdict(int)
        n = len(grid)
        cnt = 0
        for row in grid:
            dic[str(row)] += 1
        for i in range(n):
            col = []
            for j in range(n):
                col.append(grid[j][i])
            cnt += dic[str(col)]
        return cnt

'''
2360. Longest Cycle in a Graph
You are given a directed graph of n nodes numbered from 0 to n - 1, where each node has at most one outgoing edge.
The graph is represented with a given 0-indexed array edges of size n, indicating that there is a directed edge from node i to node edges[i]. If there is no outgoing edge from node i, then edges[i] == -1.
Return the length of the longest cycle in the graph. If no cycle exists, return -1.
A cycle is a path that starts and ends at the same node.
Example:
Input: edges = [3,3,4,2,3]
Output: 3
'''
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        longest_cycle_len = -1
        time_step = 1
        node_visited_at_time = [0] * len(edges)

        for current_node in range(len(edges)):
            if node_visited_at_time[current_node] > 0:
                continue
            start_time = time_step
            u = current_node
            while u != -1 and node_visited_at_time[u] == 0:
                node_visited_at_time[u] = time_step
                time_step += 1
                u = edges[u]
            if u != -1 and node_visited_at_time[u] >= start_time:
                longest_cycle_len = max(longest_cycle_len, time_step - node_visited_at_time[u])
        return longest_cycle_len

'''
2405. Optimal Partition of String
Given a string s, partition the string into one or more substrings such that the characters in each substring are unique. That is, no letter appears in a single substring more than once.
Return the minimum number of substrings in such a partition.
Note that each character should belong to exactly one substring in a partition.
Example:
Input: s = "abacaba"
Output: 4
'''
class Solution:
    def partitionString(self, s: str) -> int:
        res = 0
        hashset = set()
        for char in s:
            if char in hashset:
                hashset.clear()
                res += 1
            hashset.add(char)
        return res+1

'''
2439. Minimize Maximum of Array
You are given a 0-indexed array nums comprising of n non-negative integers.
In one operation, you must:
Choose an integer i such that 1 <= i < n and nums[i] > 0.
Decrease nums[i] by 1.
Increase nums[i - 1] by 1.
Return the minimum possible value of the maximum integer of nums after performing any number of operations.
Example:
Input: nums = [3,7,1,6]
Output: 5
'''
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        totalsum, result = 0, 0
        for i, num in enumerate(nums):
            totalsum += num
            result = max(result, (totalsum + i) // (i + 1))
        return result

'''
2444. Count Subarrays With Fixed Bounds
You are given an integer array nums and two integers minK and maxK.
A fixed-bound subarray of nums is a subarray that satisfies the following conditions:
The minimum value in the subarray is equal to minK.
The maximum value in the subarray is equal to maxK.
Return the number of fixed-bound subarrays.
A subarray is a contiguous part of an array.
Example:
Input: nums = [1,3,5,2,7,5], minK = 1, maxK = 5
Output: 2
'''
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        res = 0
        minFound, maxFound = False, False
        start = 0
        minStart, maxStart = 0, 0
        for i, num in enumerate(nums):
            if num < minK or num > maxK:
                minFound, maxFound = False, False
                start = i+1
            if num == minK:
                minFound = True
                minStart = i
            if num == maxK:
                maxFound = True
                maxStart = i
            if minFound and maxFound:
                res += (min(minStart, maxStart) - start + 1)
        return res

'''
2454. Next Greater Element IV
You are given a 0-indexed array of non-negative integers nums. For each integer in nums, you must find its respective second greater integer.
The second greater integer of nums[i] is nums[j] such that:
j > i
nums[j] > nums[i]
There exists exactly one index k such that nums[k] > nums[i] and i < k < j.
If there is no such nums[j], the second greater integer is considered to be -1.
For example, in the array [1, 2, 4, 3], the second greater integer of 1 is 4, 2 is 3, and that of 3 and 4 is -1.
Return an integer array answer, where answer[i] is the second greater integer of nums[i].
Example:
Input: nums = [2,4,0,9,6]
Output: [9,6,6,-1,-1]
'''
class Solution(object):
    def secondGreaterElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = [-1 for i in range(len(nums))]
        stack1, stack2 = [], []
        for i, num in enumerate(nums):
            while stack2 and num > nums[stack2[-1]]:
                result[stack2.pop()] = num
            temp = []
            while stack1 and num > nums[stack1[-1]]:
                temp.append(stack1.pop())
            stack2 += temp[::-1]
            stack1.append(i)
        return result

'''
2485. Find the Pivot Integer
Given a positive integer n, find the pivot integer x such that:
The sum of all elements between 1 and x inclusively equals the sum of all elements between x and n inclusively.
Return the pivot integer x. If no such integer exists, return -1. It is guaranteed that there will be at most one pivot index for the given input.
Example:
Input: n = 8
Output: 6
'''
class Solution(object):
    def pivotInteger(self, n):
        """
        :type n: int
        :rtype: int
        """
        leftsum, rightsum = 0, (1+n)*n/2
        for i in range(1, n+1):
            leftsum += i
            if leftsum == rightsum:
                return i
            rightsum -= i
        return -1

'''
2492. Minimum Score of a Path Between Two Cities
You are given a positive integer n representing n cities numbered from 1 to n.
You are also given a 2D array roads where roads[i] = [ai, bi, distancei] indicates that there is a bidirectional road between cities ai and bi with a distance equal to distancei.
The cities graph is not necessarily connected.
The score of a path between two cities is defined as the minimum distance of a road in this path.
Return the minimum possible score of a path between cities 1 and n.
Note:
A path is a sequence of roads between two cities.
It is allowed for a path to contain the same road multiple times, and you can visit cities 1 and n multiple times along the path.
The test cases are generated such that there is at least one path between 1 and n.
Example:
Input: n = 4, roads = [[1,2,9],[2,3,6],[2,4,5],[1,4,7]]
Output: 5
'''
class Solution:
    def __init__(self):
            self.parents = {}
            self.count = {}

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        for i in range(1,n+1): self.count[i] = 1
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
        a_size, b_size = self.count.get(a_parent, 0), self.count.get(b_parent, 0)
        
        if a_parent != b_parent:
            if a_size < b_size:
                self.parents[a_parent] = b_parent
                self.count[b_parent] += a_size
            else:
                self.parents[b_parent] = a_parent
                self.count[a_parent] += b_size

'''
2560. House Robber IV
There are several consecutive houses along a street, each of which has some money inside. There is also a robber, who wants to steal money from the homes, but he refuses to steal from adjacent homes.
The capability of the robber is the maximum amount of money he steals from one house of all the houses he robbed.
You are given an integer array nums representing how much money is stashed in each house. More formally, the ith house from the left has nums[i] dollars.
You are also given an integer k, representing the minimum number of houses the robber will steal from. It is always possible to steal at least k houses.
Return the minimum capability of the robber out of all the possible ways to steal at least k houses.
Example:
Input: nums = [2,3,5,9], k = 2
Output: 5
'''
class Solution:
    def minCapability(self, nums: List[int], k: int) -> int:
        l, r = min(nums), max(nums)
        while l < r:
            mid = (l + r) // 2
            last = take = 0
            for num in nums:
                if last:
                    last = 0
                    continue
                if num <= mid:
                    take += 1
                    last = 1
            if take >= k:
                r = mid
            else:
                l = mid + 1
        return l
    
'''
2595. Number of Even and Odd Bits
You are given a positive integer n.
Let even denote the number of even indices in the binary representation of n (0-indexed) with value 1.
Let odd denote the number of odd indices in the binary representation of n (0-indexed) with value 1.
Return an integer array answer where answer = [even, odd].
Example:
Input: n = 17
Output: [2,0]
'''
class Solution:
    def evenOddBit(self, n: int) -> List[int]:
        even, odd = 0, 0
        digit = 0
        while n:
            if n&1:
                print(digit)
                if digit%2:
                    odd += 1
                else:
                    even += 1
            digit += 1
            n >>= 1
        return [even, odd]