# Task17

## [344. 反转字符串](https://leetcode-cn.com/problems/reverse-string/)

```c++
class Solution {
public:
    void reverseString(vector<char>& s) {
        if (s.size() <= 1) return;
        int left = 0, right = s.size()-1;
        while (left < right) swap(s[left++], s[right--]);
    }
};
```

使用双指针，从两端同时移动，边移动边交换。

由于 while 循环中，一共执行了 N/2 次交换数据，因此时间复杂度为O(N)；由于没有额外开辟新的空间，所以空间复杂度为O(1)。

## [557. 反转字符串中的单词 III](https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/)

```c++
class Solution {
public:
    string reverseWords(string s) {
        for (int left = 0; left < s.size();) {
            int right = left;
            while (right < s.size() && s[right] != ' ') right++;
            int tmp = right--;
            while (left < right) {
                swap(s[left++], s[right--]);
            }
            left = tmp+1;
        }
        return s;
    }
};
```

首先根据空格对字符串进行拆分，对每个小字符串，进行上一道题的交换操作，最后返回原始的字符串即可。

看了下题解，我这里的方法属于原地解方法。另一种方法是额外开辟新的空间，每找到一个单词，则从后向前遍历，加入到数组中。