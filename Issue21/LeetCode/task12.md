# Task12

## [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

```c++
class LRUCache {
public:
    unordered_map<int, vector<int>> mp;
    int total = 0, cap;
    queue<int> q;
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if (mp.count(key) == 0) return -1;
        mp[key][1]++;
        q.push(key);
        return mp[key][0];
    }
    
    void put(int key, int value) {
        if (mp.count(key)) {
            mp[key][0] = value;
            q.push(key);
            mp[key][1]++;
        } else {
            if (total == cap) {
                while(mp[q.front()][1] > 1) {
                    mp[q.front()][1]--;
                    q.pop();
                }
                mp.erase(q.front());
                q.pop();
                total--;
            }
            total++;
            mp[key] = vector<int>{value, 1};
            q.push(key);
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

首先设置一个map数据结构，存储数据的键值对。然后利用一个队列，存储每次用到的键值。同时记录队列里每个键值的数量。

每次put或者get一个数据时，都将key加入到队列中，然后更新它们在队列中的数量。

在缓存满了的时候，从队列里pop元素，直到pop出的元素在队列中只出现一次，停止。这个元素就是需要被移除的元素，删除他，然后加入新的元素，继续加入到队列中，更新其value值和队列里出现的次数。

题解中使用的是双向链表，，链表的逻辑太容易被绕晕了\~如果面试只是讲讲思想的话，还可以\~

## [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

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
    ListNode* sortList(ListNode* head) {
        ListNode* head_pre = new ListNode(INT_MIN), * index = head_pre;
        head_pre->next = head;
        while (head != nullptr) {
            if (head->val >= index->val) {
                index = index->next;
                head = head->next;
                continue;
            }

            ListNode* index_pre = head_pre;
            while(index_pre->next->val < head->val) index_pre = index_pre->next;
            index->next = head->next;
            head->next = index_pre->next;
            index_pre->next = head;
            head = index->next;
        }
        return head_pre->next;
    }
};
```

每次head向前走一步，比较与其前一节点值的大小，如果大于或等于，则继续向前；否则，从头结点开始寻找，直到找到一个下一节点的值比其大的节点，将head插入到这个节点后面，更新head的值。继续下一步。

时间复杂度较高，为$O(n^2)$。

看了下题解，对链表的话，可以采用归并的方法进行排序。

## [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    unordered_map<int, int> mp;
    stack<int> s;
    priority_queue<int, vector<int>, greater<int>> q;
    MinStack() {
    }
    
    void push(int x) {
        s.push(x);
        mp[x]++;
        q.push(x);
    }
    
    void pop() {
        mp[s.top()]--;
        s.pop();
    }
    
    int top() {
        return s.top();
    }
    
    int getMin() {
        while(mp[q.top()] <= 0) q.pop();
        return q.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

直接使用优先队列的方式，，感觉有点作弊。。。

看了题解，发现是用辅助栈，emm，，高级啊。。