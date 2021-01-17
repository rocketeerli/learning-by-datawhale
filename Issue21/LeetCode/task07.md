# Task07

## [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if (matrix.size() == 0) return res;
        int rows = matrix.size(), cols = matrix[0].size(), size = rows * cols;
        res = vector<int>(size);
        int index = 0, dis = 0;
        while (index < size) {
            for (int i = dis; i < cols-dis;i++) {
                res[index++] = matrix[dis][i];
            }
            if (index == size) break;
            for (int i = dis+1; i < rows - dis - 1; i++) {
                res[index++] = matrix[i][cols-dis-1];
            }
            if (index == size) break;
            for (int i = cols-dis-1; i >= dis;i--) {
                res[index++] = matrix[rows-dis-1][i];
            }
            if (index == size) break;
            for (int i = rows-dis-2; i > dis;i--) {
                res[index++] = matrix[i][dis];
            }
            dis++;
        }
        return res;
    }
};
```

直接模拟，螺旋操作，写一个循环，从最外圈开始遍历。

我这里写的是按层进行遍历，对每一层，进行一圈遍历。由于每个数值遍历了一次，时间复杂度为O(mn)，由于没有开辟新的辅助空间，因此，空间复杂度为O(1)。

还有一种模拟方向的方法，利用事先存储好的四个放向信息进行模拟。每次移动，根据当前的方向，继续向上下左右四个方向前进，直到所有的数值全都都遍历一遍。时间复杂度同样是 O(mn)，由于这里需要记录每一个数值是否被访问过，因此空间复杂度为 O(mn)。

## [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res = vector<vector<int>>(n, vector<int>(n));
        int left = 0, top = 0, right = n - 1, bottom = n-1;
        int num = 1, size = n*n;
        while (num <= size) {
            for (int i = left; i <= right; i++) res[top][i] = num++;
            top++;
            for (int i = top; i <= bottom; i++) res[i][right] = num++;
            right--;
            for (int i = right; i >= left;i--) res[bottom][i] = num++;
            bottom--;
            for (int i = bottom; i >= top; i--) res[i][left] = num++;
            left++;
        }
        return res;
    }
};
```

参考了一下上一题的 `c++` 题解，发现果然很简单。

这题重点在，由于矩阵肯定是方阵，正方形，所以少了很多边界条件的判断，直接模拟一遍即可。

时间复杂度为O(mn)，空间复杂度为O(1)。

看了下这道题的题解，发现跟我写的一毛一样。。。emmm，我真的没抄，。，。

## [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

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
    ListNode* rotateRight(ListNode* head, int k) {
        if (head == nullptr) return head;
        ListNode* slow = head, *fast = head;
        int n = k;
        while (fast != nullptr && n > 0) {
            fast = fast->next;
            n--;
        }
        if (n > 0) {
            n = n % (k-n);
            fast = head;
            while(n-- > 0) fast = fast->next;
        }
        if (fast == nullptr) return head;
        while (fast->next != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }
        fast->next = head;
        head = slow->next;
        slow->next = nullptr;
        return head;
    }
};
```

链表题常见的解题思路——快慢指针。

这里唯一的一个坑就是，k值有可能超过链表的长度，因此，这里首先需要对超出链表长度的情况进行分析，再分析好后，利用快慢指针，即可得到新的头结点和旧的尾结点，将旧的尾结点和旧的头结点连接起来，再重新赋值新的头结点，并将新的尾结点的指向设成空，即可。

跟官方题解不同的是，我这里首先进行判断，如果链表很长，长度超过k，这时，寻找快指针的位置就不需要先遍历一遍链表了。

如果无论什么情况下都先确定链表长度的话，那么，链表一定会被遍历两次。一次是获取长度，另一次是快指针遍历。