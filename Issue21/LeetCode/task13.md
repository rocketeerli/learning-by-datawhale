# Task13

## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == NULL || headB == NULL) return NULL;
        ListNode* indexA = headA, * indexB = headB;
        while (indexA != indexB) {
            if (indexA == NULL) indexA = headB;
            if (indexB == NULL) indexB = headA;
            if (indexA == indexB) break;
            indexA = indexA->next;
            indexB = indexB->next;
        }
        return indexA;
    }
};
```

嗯，使用双指针的方式，这是解决链表相交问题的典型方法。本题双指针的方法大致有两种：

1. 两个指针分别遍历两个链表，到尾部，计算两个链表的长度。然后计算差值为k，再让长链表先走k步，然后二者再同时出发，如果中间第一次遇到相同的链表节点，则为相交的入口节点。
2. 两个指针分别从两个链表的头部开始出发，如果其中一个到达尾部，则从另外一个链表的头部继续出发，直到遇到相同的节点为止。如果该节点是非空的，则为要找的入口节点；否则，二者不相交。两个指针走过的长度一定是相同的。（可以好好想一想）

我这里使用的是第二种方法，这是二哥之前教过我的，但今天还是提交了好几次才通过。。。

## [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int num = 0, sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == num) sum++;
            else if (sum == 0) {
                num = nums[i];
                sum = 1;
            } else {
                sum--;
            }
        }
        return num;
    }
};
```

突然发现，这是之前面试别人的面试题\~

每次记录当前出现次数最多的数字，和其出现次数，如果有出现与其数字值不同的，则次数减一；否则次数加一。当次数减为0时，遇到下一个不相同的数字，更新出现次数最多的数字为该数字，然后记录其出现次数 。继续以上步骤。

时间复杂度为O(n)，空间复杂度为O(1)。

看了题解，发现这个算法叫**Boyer-Moore 投票算法**。题解中有很多其他的方法，但感觉都是为了这个算法做铺垫。

## [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

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
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) return head;
        ListNode* head_pre = new ListNode(0), * index = head->next;
        head_pre->next = head;
        while (index != nullptr) {
            head->next = index->next;
            index->next = head_pre->next;
            head_pre->next = index;
            index = head->next;
        }
        return head_pre->next;
    }
};
```

使用虚拟头结点的方式，不断地把遇到的节点放到该节点的后面，直到遇到链表尾部。

题解中给出了两种方法，迭代和递归，我这里使用的是迭代的方式。