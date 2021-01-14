# Task03

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

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        int n = nums.size();
        if (n < 3) return result;
        sort(nums.begin(), nums.end());
        for (int left = 0; left < n-2; left++) {
            if (left > 0 && nums[left] == nums[left-1]) continue;
            int mid = left+1, right = n-1, target = -nums[left];
            while(mid < right) {
                if (nums[mid] + nums[right] == target) {
                    result.push_back(vector<int>{nums[left], nums[mid], nums[right]});
                    mid++;right--;
                    while(mid < right && nums[mid] == nums[mid-1]) mid++;
                    while(right > mid && nums[right] == nums[right+1]) right--;
                } else if (nums[mid] + nums[right] > target) {
                    right--;
                } else {
                    mid++;
                }
            }
        }
        return result;
    }
};
```

第一眼看这题，就知道是使用双指针的方法，但还是不会，还是看了题解，还是做了一下午，尝试了好多种方法，最后使用了题解评论里面的高赞解。在提交了7次错误后，第八次终于通过了，，泪目啊！

最开始的思路是想，先固定住两端，移动中间的，相当于是使用三个指针的方法。在解决了一系列问题之后，最后一个没能解决的问题就是如何移动两端的指针，一旦处理不好，就会导致遗漏情况。

然后，看了题解，是固定住一个值，依然固执地认为，双指针一定是要在两端进行移动。so, 我继续选择固定中间的指针，最后出现的问题就是，没办法去除重复的值，重复值太多了（现在，想了一下，应该也是可以的）。使用 `c++` 的 `unique` 方法，先对结果进行排序，再进行去除的方法，超出了时间限制。

最后，使用了高赞的题解。确实比官方题解写得清除，和我之前其中一个思路很相似，所以很快就写好了。然而还是出现了问题，主要还是左右指针移动过程中，如何排除重复值的问题。这里需要注意的一点是，左指针需要和它左边的值进行比较，右指针需要和它右边的值进行比较，这样才可以。（我是刚好反过来了，，找了很久的bug，最后对比代码，才发现，惭愧惭愧\~）。

## 总结

来个总结吧，今天的问题，确实让我对双指针有了更深的理解。花了很长的时间，还是很值得的。