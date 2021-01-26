# Task14

## [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        return quickSelect(nums, k, 0, nums.size()-1);
    }
    int quickSelect(vector<int>& nums, int k, int left, int right) {
        int p = partition(nums, left, right);
        if (p == k-1) {
            return nums[p];
        } else if (p > k-1) {
            return quickSelect(nums, k, left, p-1);
        } else {
            return quickSelect(nums, k, p+1, right);
        }
    }
    int partition(vector<int>& nums, int l, int r) {
        int p = nums[l];
        int left = l, right = r;
        while (left < right) {
            while(left < right && nums[right] <= p) right--;
            while(left < right && nums[left] >= p) left++;
            swap(nums[left], nums[right]);
        }
        swap(nums[left], nums[l]);
        return left;
    }
};
```

使用快速排序的精简版，这也是看到题目第一眼就想到的方法，但还是写了好久。。。

题解中的快排方法，在划分之前，增加了一个随机选取比较值的方式，因为这个用来划分的基数对于这个算法来讲，十分重要，如果选择不合适，很可能让时间复杂度变得很大，即$O(n^2)$。但快排的时间复杂度期望值是线性的。

堆排序，，emmm，先待定吧，，不想看了。。。

## [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

```c++
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> s;
        for (int num : nums) {
            if (s.count(num)) return true;
            s.insert(num);
        }
        return false;
    }
};
```

可以排序，也可以哈希。我这里是使用哈希的方法，用空间换时间。

哈希的话，时间复杂度为O(N)，空间复杂度也为O(N)。排序的方法，时间复杂度会高一些。

## [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

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
    int kthSmallest(TreeNode* root, int k) {
        int num = 0;
        vector<int> res;
        midorder(root, res);
        return res[k-1];
    }
    void midorder(TreeNode* root, vector<int>& res) {
        if (root == nullptr) return;
        midorder(root->left, res);
        res.push_back(root->val);
        midorder(root->right, res);
    }
};
```

中序遍历一遍，相当于把所有数据都排序好了，然后直接取第k个即可。时间复杂度为O(N)，空间复杂度也为O(N)。

上面的方法是递归，比较好理解，也容易写，也可以使用迭代的方式。

使用迭代的方式，可以不用遍历所有节点，记录遍历到的数据，当找到第 k 小的数时，跳出迭代。