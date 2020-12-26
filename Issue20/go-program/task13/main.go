package main

import (
	"fmt"
	"sync"
	"time"
)

// SafeCounter 的并发使用是安全的。
type SafeCounter struct {
	v     map[string]int
	rwmux sync.RWMutex
}

// Inc 增加给定 key 的计数器的值。
func (c *SafeCounter) Inc(key string) {
	// 写操作使用写锁
	c.rwmux.Lock()
	defer c.rwmux.Unlock()
	// Lock 之后同一时刻只有一个 goroutine 能访问 c.v
	c.v[key]++
}

// Value 返回给定 key 的计数器的当前值。
func (c *SafeCounter) Value(key string) int {
	// 读的时候加读锁
	c.rwmux.RLock()
	// Lock 之后同一时刻只有一个 goroutine 能访问 c.v
	defer c.rwmux.RUnlock()
	return c.v[key]
}

func main() {
	c := SafeCounter{v: make(map[string]int)}
	for i := 0; i < 1000; i++ {
		go c.Inc("somekey")
	}

	time.Sleep(time.Second)
	fmt.Println(c.Value("somekey"))
}
