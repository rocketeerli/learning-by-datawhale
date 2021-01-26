# Task16

## [237. 删除链表中的节点](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

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
    void deleteNode(ListNode* node) {
         node->val = node->next->val;
         node->next = node->next->next;
    }
};
```

最开始以为会有一个头结点，后来发现只有一个被删除的节点。。

然后直接利用其非尾结点的特点，将下一个节点的值覆盖掉当前节点，然后再删除下一个节点即可。

## [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

看到题的第一想法就是计算所有数的乘积，然后挨个做除法。但这里明确要求不能使用 除法，。，。

```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size(), l = 1, r = 1;
        vector<int> res(n), left(n), right(n);
        for (int i = 0; i < n; i++) {
            left[i] = i > 0 ? nums[i-1] * left[i-1] : l;
            right[n-i-1] = i > 0 ? nums[n-i] * right[n-i] : r;
        }
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                res[i] = right[i];
            } else if (i == n-1) {
                res[i] = left[i];
            } else {
                res[i] = left[i]*right[i];
            }
        }
        return res;
    }
};
```

使用了数组，存储左右两边的乘积值，最后，第i处的值为第i处左边的乘积乘上右边的乘积。

如果空间复杂度为O(1)，则借助结果数组，首先存储左侧的乘积，再挨个计算，乘上右侧的乘积。本质上还是存储两个方向的乘积。

## [292. Nim 游戏](https://leetcode-cn.com/problems/nim-game/)

```c++
class Solution {
public:
    bool canWinNim(int n) {
        return n % 4 != 0;
    }
};
```

只要是四的倍数，都不可能赢。因为无论是取 1 个，还是 2 个，或是 3 个，对手都可以拿走相应的数目使其再次成为4的倍数，最后4为不可能赢的数。