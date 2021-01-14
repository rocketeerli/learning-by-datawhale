# task01

## [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int flag = 0;
        ListNode* head = new ListNode(), *p = head;
        while(l1 != nullptr || l2 != nullptr) {
            int val = flag * 1;
            if (l1 == nullptr) {
                val += l2->val;
            } else if (l2 == nullptr) {
                val += l1->val;
            } else{
                val += l1->val + l2->val;
            }
            if (val >= 10) {
                flag = 1;
                val = val - 10;
            } else flag = 0;
            p->next = new ListNode(val);
            p = p->next;
            if (l1 != nullptr) l1 = l1->next;
            if (l2 != nullptr) l2 = l2->next;
        }
        if (flag == 1) {
            p->next = new ListNode(1);
        }
        return head->next;
    }
};
```

第一想法是直接进行模拟，计算出两个数，相加，然后再转成链表。但很显然，链表长度很长的话，数据会溢出。

然后就是设置一个进位符，按位进行相加，结果对10取模，放回结果链表的对应位置。

## [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

```c++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size(), total = n + m;
        if (total % 2) {
            return findkmin(nums1, nums2, (total + 1)/2);
        } else {
            return double(findkmin(nums1, nums2, total/2)+findkmin(nums1, nums2, total/2+1))/2;
        }
    }
    int findkmin(vector<int>& nums1, vector<int>& nums2, int k) {
        int n = nums1.size(), m = nums2.size();
        int index1 = 0, index2 = 0;
        while (true) {
            if (index1 == n) return nums2[index2 + k - 1];
            if (index2 == m) return nums1[index1 + k - 1];
            if (k == 1) return min(nums1[index1], nums2[index2]);
            int km = k / 2 - 1;
            int i_1 = min(index1 + km, n-1), i_2 = min(index2 + km, m-1);
            int mid1 = nums1[i_1], mid2 = nums2[i_2];
            if (mid1 <= mid2) {
                k -= i_1 - index1 + 1;
                index1 = i_1 + 1;
            } else {
                k -= i_2 - index2 + 1;
                index2 = i_2 + 1;
            }
        }
    }
};
```

第一眼看到 log 的时间复杂度，就猜到是使用二分查找的方式。但直接二分，比较复杂，下标梳理清楚非常不容易。

需要转换问题，转换成寻找第 k 小的数。这里也有一个问题，就是为什么需要向前的步长是 k/2-1，而不是 k/2。保证前面排除的过程中，每次排除k/2个值。

## [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size(), len = 0, start = 0, max_len = 0;
        if (n < 2) return s;
        vector<vector<int>> dp(n, vector<int>(n));
        while (len < n) {
            for (int i = 0; i < n-len; i++) {
                int j = i + len;
                if (s[i] == s[j]) {
                    if (len < 3) dp[i][j] = 1;
                    else dp[i][j] = dp[i+1][j-1];
                    if (dp[i][j] && len > max_len) {
                        start = i;
                        max_len = len;
                    }
                } else {
                    dp[i][j] = 0;
                }
            }
            len++;
        }
        return s.substr(start, max_len+1);
    }
};
```

**动态规划**

最主要的是状态转移方程，字符串问题的动态规划问题的状态转移矩阵基本都是二维的。

这里，需要注意的是，由于转移矩阵中，当前位置(i, j)的值是由左下角位置(i+1, j-1)的值决定的，因此，需要行从下到上或是列从左到右进行转移矩阵的赋值。我直接使用左下到右上的方式。

另一个需要注意的点就是，生成转移矩阵的同时，需要记录最大长度和起始下标。
