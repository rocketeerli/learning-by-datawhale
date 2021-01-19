# Task09

## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        vector<int> nums = nums1;
        int i = 0, j = 0;
        while(i < m && j < n) {
            if (nums[i] < nums2[j]) {
                nums1[i+j] = nums[i];
                i++;
            } else {
                nums1[i+j] = nums2[j];
                j++;
            }
        }
        while (j < n) nums1[i+j++] = nums2[j];
        while (i < m) nums1[i+++j] = nums[i];
    }
};
```

两个排序数组，采用归并的方法。每个元素遍历了一次，所以时间复杂度为O(m+n)；额外开了一个数组，空间复杂度为O(m+n)。（也可以是O(m)，我这里为了方便，直接赋值，写成了O(m+n)的。）

想了挺长时间，感觉还是得额外开空间，没想到空间复杂度为O(1)的方法\~看了题解，才发现可以从后往前移动双指针。。。我咋就没想到呢，。，。

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