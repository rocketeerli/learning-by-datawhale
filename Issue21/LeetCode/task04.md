# Task04

## [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

和昨天的15题（三数之和）一样，可以使用排序+双指针的方法。

```c++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int sum = INT_MAX / 2, n = nums.size();
        if (n < 3) return sum;
        sort(nums.begin(), nums.end());
        for (int first = 0; first < n-2; first++) {
            // if (first > 0 && nums[first] == nums[first-1]) continue;
            int left = first+1, right = n-1;
            int tar = target - nums[first];
            while (left < right) {
                int tmp = nums[left] + nums[right];
                int dis = abs(tar - tmp);
                if (dis < abs(target - sum)) sum = tmp + nums[first];
                if (tmp == tar) return target;
                else if (tmp > tar) right--;
                else if (tmp < tar) left++;
            }
        }
        return sum;
    }
};
```

第一眼看，以为是昨天题的进阶版，增加了个`target`，但其实确实比昨天的题简单很多，因为这里不需要去除重复的元素。为了降低一下复杂度，使用双指针的方法，将时间复杂度控制在$O(n^2)$（看题解说是$O(n^3)$的暴力遍历方法也可以，emmm）。

## [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```c++
class Solution {
public:
    bool isValid(string s) {
        int n = s.size();
        stack<char> st;
        for (int i = 0; i < n; i++) {
            if(st.empty()) st.push(s[i]);
            else if (st.top() == '(' && s[i] == ')' ||
                    st.top() == '{' && s[i] == '}' ||
                    st.top() == '[' && s[i] == ']') {
                st.pop();
            } else st.push(s[i]);
        }
        return st.empty();
    }
};
```

直接使用栈，利用后进先出的原则，进行匹配。

## [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr) return l2;
        if (l2 == nullptr) return l1;
        ListNode* head = new ListNode(0), *index = head;
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val <= l2->val) {
                index->next = l1;
                l1 = l1->next;
            } else {
                index->next = l2;
                l2 = l2->next;
            }
            index = index->next;
        }
        if(l1 == nullptr) index->next = l2;
        if(l2 == nullptr) index->next = l1;
        return head->next;
    }
};
```

链表题，直接归并，没有太多技巧。

看了下题解，有递归和迭代两种方式，我这种属于迭代。其实思路都是相同的。