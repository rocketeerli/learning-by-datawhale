# task02

## [7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

```c++
class Solution {
public:
    int reverse(int x) {
        long y = 0;
        while (x != 0) {
            y = y * 10;
            y += x % 10;
            x = x / 10;
        }
        if (y < INT_MIN || y > INT_MAX) return 0;
        else return y;
    }
};
```

借用了隐式类型转换的方法。需要注意的是，求余数运算已经有符号了，不需要再显示地存储结果的正负符号。

## [8. 字符串转换整数(atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

```c++
class Solution {
public:
    int myAtoi(string s) {
        int flag = 0, positive = 1;
        long result = 0;
        for (int i = 0; i < s.size(); i++) {
            if (flag == 0 && s[i] == ' ') continue;
            if (flag == 0) {
                if (s[i] == '-') positive = -1;
                else if (s[i] == '+') positive = 1;
                else if (s[i] >= '0' && s[i] <= '9') result+=s[i]-'0';
                else return 0;
                flag = 1;
            } else {
                if (s[i] >= '0' && s[i] <= '9') {
                    result *= 10;
                    result += s[i] - '0';
                    if (result*positive < INT_MIN) return INT_MIN;
                    if (result*positive > INT_MAX) return INT_MAX;
                } else {
                    return result*positive;
                }
            }
        }
        return result*positive;
    }
};
```

这个思路倒是没啥，直接顺序遍历一遍即可。主要是条件的判断，不要遗漏情况。还有数据类型，注意不要溢出。

看题解的意思是，需要使用有穷状态自动机（DFA）？！emmm

## [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        long y = x, m = 0;
        while(x != 0) {
            m = m * 10;
            m += x % 10;
            x = x / 10;
        }
        return y == m;
    }
};
```

又是回文数，。

如果不转成字符串，需要注意翻转之后的数值是否溢出的问题。我这里还是像之前一样，使用了 long 类型。（有点玩赖的赶脚）

看了题解，发现可以仅翻转一半的数字，这样完全不会有数值溢出的问题了。确实啊，很有道理。