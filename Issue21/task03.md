# task03

## [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int result = 0, n = height.size();
        int left = 0, right = n-1;
        while (left < right) {
            int h;
            if (height[left] <= height[right])  {
                h = height[left];
                left++;
            } else {
                h = height[right];
                right--;
            }
            result = max(result, h * (right - left + 1));
        }
        return result;
    }
};
```

盛水的题，第一想法就是使用单调栈，寻找左右两边的最大高度，但这里的水是可以透过中间的柱子的，所以并不能直接按照高度直接放进单调栈中。

第二种想法就是暴力遍历求解，时间复杂度为 $O(n^2)$，尝试了一下，但很显然超出时间限制了。

最后也没想到使用双指针，看了题解，发现这也可以？？？

双指针的思想，一般是用在链表上，快慢指针，这里是前后的双指针。只要思路有了，代码编写还是很简单的。

## [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string result = "";
        if (strs.size() == 0) return result;
        int length = 0;
        while(true) {
            if (length == strs[0].size()) return result;
            char ch = strs[0][length];
            for (int i = 1; i < strs.size(); i++) {
                string s = strs[i];
                if (s.size() <= length || s[length] != ch) return result;
            }
            length++;
            result += ch;
        }
        return result;
    }
};
```

简单题确实好做，一顿操作猛如虎，一看超越百分五。。。

我是直接纵向遍历的方式写的，第一想法也是最后使用的方法就是依次遍历每个字符串的每个字符，方法是对的，但时间很慢。

题解中，前两种方法是横向和纵向遍历，时间复杂度都是$O(n)$，第三种方法是分治，个人感觉，可以用，但没必要，时间上并没有优化。二分查找的思想也很好，但时间复杂度还是挺高。

题解评论里的高赞还是很靠谱的，时间很快，仅排序了一次。思路很好，改成 `c++` 版本如下：

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.size() == 0) return "";
        int length = 0;
        sort(strs.begin(), strs.end());
        string left = strs[0], right = strs[strs.size()-1];
        int m = min(left.size(), right.size());
        for (; length < m && left[length] == right[length]; length++);
        return left.substr(0, length);
    }
};
```

## [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

没啥思路，待做。。。