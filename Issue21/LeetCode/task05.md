# Task05

## [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode* head = new ListNode(0), *index = head;
        int n = lists.size(), k = 0;
        if (n == 0) return nullptr;
        while (k < n || lists[k] != nullptr) {
            for (; k < n && lists[k] == nullptr; k ++);
            if (k >= n) break;
            ListNode* listnode = lists[k];
            int j = k;
            for (int i = k+1; i < n; i++) {
                if (lists[i] != nullptr && listnode->val > lists[i]->val) {
                    listnode = lists[i];
                    j = i;
                }
            }
            lists[j] = lists[j]->next;
            index->next = listnode;
            index = index->next;
        }
        return head->next;
    }
};
```

这道题，感觉并不能算是困难题，最多算是中等。。。

看到链表的问题，第一想法就是归并，事实上我也是这样写的。但考虑了一下，如果每次都需要进行比较，那么时间复杂度会很高，复杂度大概是$O(nm)$，其中 $n$ 是所有链表中所有数的个数，即链表长度和，$m$ 是所有链表的个数，即$k$。虽然时间复杂度很高，但空间复杂度为$O(1)$。

还有一种可以大幅度提高速度的方法，用空间换时间，直接读取所有的数据，然后对其进行排序，再遍历，生成链表。读取数据的时间为$O(n)$，排序的时间为$O(lgn)$，遍历生成链表的时间为$O(n)$，因此总的时间复杂度为$o(nlgn)$。但空间复杂度很大，为$O(n)$。

题解中，给出了一种折中的方法，即使用优先队列，每次选出 $k$ 个数据，进行排序（其实，除了第一次，其余每次，都有 $k-1$ 个已排好序的元素）。这样，时间复杂度为$O(kn \times lgk)$，空间复杂度为$O(k)$。

个人认为，链表题通常会限制空间复杂度，所以对辅助空间应该有限制。否则，体现不出链表结构的优势。

## [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 0) return 0;
        int left = 0, pre = nums[left];
        for (int right = 1; right < nums.size(); right++) {
            if (pre != nums[right]) {
                pre = nums[right];
                nums[++left] = pre;
            }
        }
        return left+1;
    }
};
```

题目的双指针思想很明显，题解也很简略，直接将后面不重复的元素覆盖掉前面指针的数值即可，没有太多可想的。

## [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if (nums.size() == 0) return -1;
        int k = searchk(nums), left = 0, right = nums.size()-1;
        if (nums[k] > target || k > 0 && nums[k-1] < target) return -1;
        if (nums[0] <= target) {
            right = k==0?right:k-1;
        } else {
            left = k;
        }
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
    int searchk(vector<int>& nums) {
        int n = nums.size(), left = 0, right = n - 1;
        if (nums[left] <= nums[right]) return 0;
        while (left < right) {
            if (right - left <= 1) return nums[left] < nums[right] ? left : right;
            int mid = (left + right) / 2;
            if (nums[left] < nums[mid]) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return right;
    }
};
```

这道题我见过，还问过别人，。，。刚看第一眼就知道需要进行二分搜索，直接遍历虽然也可以，但显然题目并不想这样搞。

首先，二分查找，找到旋转点。

然后，确定两段升序序列，选择包含目标值的升序序列再次进行二分查找。

思路大概是这样，但有很多细节方面的问题。

1. 其一就是，如果旋转点就是下标0的位置，那么跟没旋转的效果是一样的。（这时的搜索旋转点和查找值的逻辑可能会稍微不同）
2. 其二，二分搜索时，需要注意，如果左右两指针仅相差一的情况。（注意，左边不一定一直大于或小于右边）
3. 其三，注意边界条件。

然后看了下题解，直接一步二分就可以了，，，。。。果然厉害！

又看了一下，还是感觉我这个思路比较容易理解，也更容易想到。