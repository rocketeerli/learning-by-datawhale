# Task15

## [231. 2的幂](https://leetcode-cn.com/problems/power-of-two/)

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        int num = 1;
        for (int i = 0; i < 30; i++) {
            if (num == n) return true;
            else if (num > n) return false;
            num *= 2;
        }
        return num == n;
    }
};
```

由于int的数值范围，只有32位，第一位还是符号位，所以2的幂的数有 31 个，分别计算出这31个数，进行判断即可。

看了下题解，发现位运算是真的好用啊。。。虽然我也考虑到二进制位了，但没有发现规律。。

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        if (n <= 0) return false;
        return (n & (n - 1)) == 0;
    }
};
```

这里需要注意的是，最后 return 的括号不能省略。

## [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    vector<TreeNode*> qup, quq;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        vector<TreeNode*> qu;
        deepSearch(qu, root, p, q);
        int res = 0, n = min(qup.size(), quq.size());
        for (int i = 1; i < n; i ++) {
            if (qup[i] != quq[i]) break;
            res++;
        }
        return quq[res];
    }
    void deepSearch(vector<TreeNode*>& qu, TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == NULL) return;
        qu.push_back(root);
        if (root == p) qup = qu;
        if (root == q) quq = qu;
        deepSearch(qu, root->left, p, q);
        deepSearch(qu, root->right, p, q);
        qu.pop_back();
    }
};
```

深度优先搜索，分别计算两个节点深度遍历时经过的节点，然后从头开始进行判断，找到第一个不相等的父节点，它的父节点即为最近的公共祖先节点。每个节点遍历了1-2次，因此时间复杂度为O(N)；开辟了三个长度最大为N的可变数组，因此空间复杂度O(N)。

emmm，看了下题解，发现我没考虑到这是二叉搜索树，因此没有用到节点值的信息。。。这个相当于下一题的答案了。。

如果使用节点值的信息，直接遍历一次，找到数值在两个节点中间的祖先节点。由于仅遍历一次，且不需要额外的空间存储路径信息，因此时间复杂度为O(N)，空间复杂度为O(1)。

代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (true) {
            if (root->val > p->val && root->val > q->val) {
                root = root->left;
            } else if (root->val < p->val && root->val < q->val) {
                root = root->right;
            } else break;
        }
        return root;
    }
};
```

## [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> qup, quq;
    TreeNode* node1, *node2;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        vector<TreeNode*> qu;
        node1 = p, node2 = q;
        deepSearch(qu, root);
        int res = 0, n = min(qup.size(), quq.size());
        for (int i = 1; i < n; i++) {
            if (qup[i] != quq[i]) break;
            res++;
        }
        return qup[res];
    }
    void deepSearch(vector<TreeNode*>& qu, TreeNode* root) {
        if (root == NULL) return;
        qu.push_back(root);
        if (root == node1) qup = qu;
        if (root == node2) quq = qu;
        deepSearch(qu, root->left);
        deepSearch(qu, root->right);
        qu.pop_back();
    }
};
```

直接使用上一道题的第一种解法\~使用数组存储两个节点路径上的所有父节点。

题解中递归的方法不是很懂，看起来有点复杂。。。