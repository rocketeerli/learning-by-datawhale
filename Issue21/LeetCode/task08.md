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

```c++
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 1) return 1;
        vector<int> res(n+1);
        res[0] = 1, res[1] = 1;
        for (int i = 2; i <= n; i++) {
            res[i] = res[i-1] + res[i-2];
        }
        return res[n];
    }
};
```

一上来就想到了动态规划，，与斐波那契数列的思想一毛一样。

看了题解才发现，自己好特么青涩。。。除了动态规划，还有两种更简洁的方法。

其一，是矩阵快速幂。利用矩阵连乘的方法，直接计算出转移矩阵，即可得出结果。这属于**线性代数**中的矩阵表示方法。

其二，是求递归方程的通项。直接求递归方程的表示，首先计算特征方程的特征根，然后将初始的两个值，第 0 个和第 1 个代入方程，求出系数，即可得到递归方程的通项。这种方法是属于**组合数学**中的递归方程，

## [78. 子集](https://leetcode-cn.com/problems/subsets/)

```c++
class Solution {
public:
    vector<int> origin;
    vector<int> flag;
    vector<vector<int>> subsets(vector<int>& nums) {
        origin = nums;
        int n = nums.size();
        flag = vector<int>(n);
        vector<vector<int>> res;
        vector<int> num;
        for (int i = 0; i <= n; i++) {
            backtrack(i, res, num, 0);
        }
        return res;
    }
    void backtrack(int n, vector<vector<int>>& res, vector<int> num, int len) {
        if (n == len) {
            res.push_back(num);
            return;
        }
        for (int i = 0; i < origin.size(); i++) {
            if (flag[i] == 0) {
                if (num.size() > 0 && num[num.size()-1] >= origin[i]) continue;
                num.push_back(origin[i]);
                flag[i] = 1;
                backtrack(n, res, num, len+1);
                num.pop_back();
                flag[i] = 0;
            }
        }
    }
};
```

与前天的第46题全排列思路一样，都是使用回溯的方法。

这里有两点不同：

1. 首先，需要生成不同长度的子集，因此回溯的终止条件 n 需要是可变的，从 `0` 到 `nums.size()`；
2. 其次，与全排列不同的是，子集是没有顺序的，即这里需要过滤掉顺序不同但元素相同的数组。

在全排列的基础上，增加这两点后，就完全可以了。

题解中的方法一，我觉得很好，是属于组合数学的枚举思想，直接使用二进制的方法，从0开始，每次加1，遍历所有的子集，思路很清晰。

题解中的方法二，同样也是枚举思想，只不过是使用递归的方式进行实现，感觉单论思路的话，还是方法一比较清晰。

看来官方题解并不推荐回溯\~只不过我刚学会，这里高低得用一下。