# Task11

## [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int  res = 0;
        for (int i = 0; i < nums.size(); i++) {
            res ^= nums[i];
        }
        return res;
    }
};
```

位运算，首先初始化结果变量为0，然后对所有元素进行异或操作，最后得到的结果就是只出现一次的数字。

由于所有元素都遍历了一次，因此时间复杂度为O(n)；没有开辟额外的辅助空间，因此空间复杂度为O(1)。

这道题二哥给我讲过，第一次听到这个思路感觉很神奇，因此印象很深刻。

## [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

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
    bool hasCycle(ListNode *head) {
        if (head == NULL) return false;
        ListNode* fast=head->next, *slow=head;
        while (fast != NULL && fast->next != NULL && slow != NULL) {
            if (fast == slow) return true;
            slow = slow->next;
            fast = fast->next->next;
        }
        return false;
    }
};
```

判断链表是否有环的典型方法就是快慢指针，快指针每次走两步，慢指针每次走一步。如果有环，二者一定会在环中某个位置相遇。

时间复杂度为O(N)，空间复杂度为O(1)。

## [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

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
    ListNode *detectCycle(ListNode *head) {
        if (head == NULL) return head;
        ListNode* slow = head, *fast = slow->next;
        int n = 0, flag = 0;
        while(slow != NULL && fast != NULL && fast->next != NULL) {
            if (fast == slow) {
                flag = 1;
                break;
            }
            slow = slow->next;
            fast = fast->next->next;
            n++;
        }
        if (flag == 0) return NULL;
        slow = fast = head;
        while (n-- >= 0) {
            fast = fast->next;
        }
        while(slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```

首先利用上一题快慢指针的思想，判断链表是否有环。同时需要记录向前走的步数n，这样的话，慢指针走了n步，快指针走了2n+1步。如果有环，二者之差为环长度的整数倍。

那么，可以重置快慢指针，设置快指针先走n+1步，然后二者再一起走，移动的速度相同。这样相遇的时候，快指针比慢指针多走的步数是环长度的倍数，因此，最后一定会在环的入口节点相遇。

看了题解，发现，第二个快慢指针的快指针初始值，可以直接使用第一个快慢指针的慢指针的下一个节点。这样就相当于使用了三个指针。

时间复杂度为O(n)，空间复杂度O(1)。