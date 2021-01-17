# Task06

## [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

```c++
class Solution {
public:
    string multiply(string num1, string num2) {
        int n1 = num1.length(), n2 = num2.length();
        string result = "0";
        for (int i = n2-1; i >= 0; i--) {
            int num_2 = num2[i] - '0';
            int flag = 0;
            vector<int> mul(n2-1-i);
            for (int j = n1-1; j >= 0; j--) {
                int num_1 = num1[j] - '0';
                int total = num_1 * num_2 + flag;
                flag = total / 10;
                mul.push_back(total % 10);
            }
            if (flag > 0) mul.push_back(flag);
            int n = mul.size();
            string s(n, '0');
            for (int i = 0; i < n; i++) {
                s[i] += mul[n-i-1];
            }
            result = add(result, s);
        }
        int index = 0;
        while (result.size() - index > 1 && result[index] == '0') index++;
        return result.substr(index);
    }
    string add(string num1, string num2) {
        int n1 = num1.length(), n2 = num2.length(), n = max(n1, n2), flag = 0;
        vector<int> sum;
        for (int i = 0; i < n; i++) {
            int num_1 = i < n1 ? num1[n1-i-1]-'0' : 0;
            int num_2 = i < n2 ? num2[n2-i-1]-'0' : 0;
            int total = num_1 + num_2 + flag;
            flag = total/10;
            sum.push_back(total % 10);
        }
        if (flag == 1) sum.push_back(1);
        n = sum.size();
        string result(n, '0');
        for (int i = 0; i < n; i++) {
            result[i] += sum[n-i-1];
        }
        return result;
    }
};
```

思路与之前字符串相加很相似，直接切分，按位转成整数再进行相乘，然后将每一位的结果相加即可。按位相乘的时间复杂度为$O(nm)$，按位相加的次数有$m$次，每次相加的字符串最长为 $m+n$，所以时间复杂度为$O(m^2 + mn)$。

这里，重点结构主要有两个，首先要写一个字符串加法，然后再写一个整数与字符串相乘的结构。

使用竖式乘法进行模拟，需要很多的累加操作。虽然一遍通过了，但时间和空间的效果都不是很好。

看了下题解，优化的方法是，直接开一个已知长度的数组，每次按位相乘，更新数组对应位置的数据即可。优化的重点在最后的累加上，之前的按位相乘的步骤都是一样的，按位相乘的时间复杂度为$O(nm)$，按位相加的时间复杂度为 $O(m+n)$， 所以总的时间复杂度为$O(mn)$。

## [46. 全排列](https://leetcode-cn.com/problems/permutations/)

```c++
class Solution {
public:
    vector<int> origin;
    vector<int> flag;
    vector<vector<int>> permute(vector<int>& nums) {
        origin = nums;
        flag = vector<int>(nums.size());
        vector<vector<int>> res;
        vector<int> num;
        backtrack(res, num, 0);
        return res;
    }
    void backtrack(vector<vector<int>>& res, vector<int>& num, int index) {
        if (index == origin.size()) {
            res.push_back(num);
            return;
        }
        for (int i = 0; i < origin.size(); i++) {
            if (flag[i] == 0) {
                num.push_back(origin[i]);
                flag[i] = 1;
                backtrack(res, num, index+1);
                num.pop_back();
                flag[i] = 0;
            }
        }
    }
};
```

全排列是回溯算法的典型问题，之前组里的分享会上听师妹讲过这类问题，但这次做，还是忘记了。。。很好的一道题，很适合用来学习回溯算法。

回溯问题的思想主要就是深度优先搜索，基本的想法也很简单，每次向前试探，然后回退，继续下一次的试探。其基本的解题结构如下：

首先，主函数的结构为：

```c++
main() {
	initial value;
	backtrack();
}
```

回溯函数使用递归的方式实现：

```c++
backtrack(important parameters) {
	1. Judge boundary conditions. （边界条件，一般用于剪枝）
	2. Judge end conditions.（递归的结束条件）
	3. Traverse.（继续向下遍历搜索，相当于深度优先搜索）
	4. Recovery operation.（回退到之前的状态）
}
```

这里直接套用这个模板即可。由于没有剪枝操作，这里的边界条件没有。递归结束的条件就是数组长度等于所有的数据数目。继续向下遍历相当于深度优先搜索，递归调用回溯函数。恢复状态，相当于把 `push` 进去的数据再 `pop` 出来。

## [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        int sum = nums[0];
        for (int i = 1; i < n; i++) {
            if (nums[i-1] > 0) {
                nums[i] = nums[i] + nums[i-1];
            } else {
                nums[i] = nums[i];
            }
            sum = max(sum, nums[i]);
        }
        return sum;
    }
};
```

直接原地动归即可，中间存储一个最大值的结果变量。