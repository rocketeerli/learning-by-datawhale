# Task10

## [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        int buy = -prices[0], sell = 0, n = prices.size();
        for (int i = 1; i < n; i++) {
            sell = max(buy+prices[i], sell);
            buy = max(buy, -prices[i]);
        }
        return sell;
    }
};
```

股票问题，上来就使用动态规划。首先，分别记录两个状态——买入和卖出状态；然后不断更新这两个状态；最后返回最后一天的卖出收益。所有元素遍历了一遍，所以时间复杂度为O(n)；使用了两个额外的状态变量，空间复杂度为O(1)。

然而这道题使用动态规划设置两个状态有点大材小用的感觉，直接寻找历史最低点，然后不断更新当前天数卖出的收益即可，这样相当于少维护了一个变量，在速度上会有所提升。（这也是官方题解里面最好的方法。）

## [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        int buy = -prices[0], sell = 0;
        for (int i = 1; i < prices.size(); i++) {
            int buy_pre = buy;
            buy = max(buy, sell-prices[i]);
            sell = max(sell, buy_pre+prices[i]);
        }
        return sell;
    }
};
```

第二种股票问题，直接动态规划，思路大致与上一题一致，只不过没有限制交易次数。所有元素遍历了一遍，所以时间复杂度为O(n)；使用了两个额外的状态变量和一个辅助变量，空间复杂度为O(1)。

看了题解，发现动态规划又有点大材小用了，。，。直接贪心，遍历一遍，计算差值即可\~

## [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

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
    int res = INT_MIN;
    int maxPathSum(TreeNode* root) {
        maxRoot(root);
        return res;
    }
    int maxRoot(TreeNode* root) {
        if (root == nullptr) return 0;
        int left = max(maxRoot(root->left), 0);
        int right = max(maxRoot(root->right), 0);
        int num = root->val + left + right;
        res = max(num, res);
        return root->val + max(left, right);
    }
};
```

这题，，没太看懂，，直接瞅的题解，然后发现，这也能是困难题？？？！！！好吧，我不会。

思路就是，计算每个节点作为根节点时，以它为起点的最大路径和。递归地进行计算。

如果计算完成，那么再把它另一个子树的路径和加上，即不以它为起点（将两段路径连接起来）更新最大路径和。

最后返回最终的最大路径和。