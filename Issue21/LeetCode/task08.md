# Task08

## [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/) 

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        if (m == 0) return 1;
        vector<vector<int>> dp = vector<vector<int>>(m, vector<int>(n, 1));
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```

提交了才发现，这道题一个月前做过，。只是换了系统，浏览器没加载缓存。。

还是一样，**动态规划**，很明显的递归方程：$dp[i] [j] = dp[i-1] [j] + dp[i] [j-1]$ where $ i > 0, j > 0$. 

首先初始化第一行和第一列全为1，然后按照递归方程，填写状态矩阵，返回右下角（即dp[m-1] [n-1]）的值。

第一次做的时候，还以为可以直接找规律，让两数相乘，实际上并不行\~。

## [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)





## [78. 子集](https://leetcode-cn.com/problems/subsets/)



