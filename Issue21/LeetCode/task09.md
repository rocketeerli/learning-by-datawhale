# Task09

## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)





## [89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)





## [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```

二叉树求深度，最直接的方法就是递归了\~

分别求左右子树的高度，返回其中的最大值加一。递归终止的条件是根节点为空。

题解中还有广度优先搜索的方法，感觉没有递归这种好理解。