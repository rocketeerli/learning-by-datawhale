## 并发编程

### 定义

- 并发的好处

1. 不阻塞等待其他任务的执行，从而浪费时间，影响系统性能。
2. 并行可以使系统变得简单些，将复杂的大任务切换成许多小任务执行，单独测试。

- 进程等待的原因： 通常受限来源于**进程I/O**或**CPU**。
  - 进程I/O限制， 如：等待网络或磁盘访问
  - CPU限制， 如：大量计算

### Go 并发

每个go程序至少都有一个 `Goroutine`：主 `Goroutine`（在运行进程时自动创建）。

在 `sync` 包里包含了： `WaitGroup`、`Mutex`、`Cond`、`Once`、`Pool`.

#### 1. `WaitGroup`

Add(n)把计数器设置为n,Done()会将计数器每次减1，Wait()函数会阻塞代码运行，直到计数器减0。

```go
// 这是我们将在每个goroutine中运行的函数。
// 注意，等待组必须通过指针传递给函数。
func worker(id int, wg *sync.WaitGroup) {
	defer wg.Done()
	fmt.Printf("Worker %d starting\n", id)
	time.Sleep(time.Second)
	fmt.Printf("Worker %d done\n", id)
}

func main() {
	var wg sync.WaitGroup

	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go worker(i, &wg)
	}
	wg.Wait()
}
```

注意：

- 计数器不能为负值
- `WaitGroup` 对象不是引用类型

#### 2. `Once`

`sync.Once` 可以控制函数只能被调用一次，不能多次重复调用。

```go
var doOnce sync.Once

func main() {
	DoSomething()
	DoSomething()
}

func DoSomething() {
	doOnce.Do(func() {
		fmt.Println("Run once - first time, loading...")
	})
	fmt.Println("Run this every time")
}
```

#### 3. 互斥锁 `Mutex`

互斥锁是并发程序对共享资源进行访问控制的主要手段,在 `go` 中的 `sync` 中提供了 `Mutex` 的支持。

```go
// SafeCounter 的并发使用是安全的。
type SafeCounter struct {
	v   map[string]int
	mux sync.Mutex
}

// Inc 增加给定 key 的计数器的值。
func (c *SafeCounter) Inc(key string) {
  c.mux.Lock()
  defer c.mux.Unlock()
  // Lock 之后同一时刻只有一个 goroutine 能访问 c.v
  c.v[key]++
}

// Value 返回给定 key 的计数器的当前值。
func (c *SafeCounter) Value(key string) int {
	c.mux.Lock()
	// Lock 之后同一时刻只有一个 goroutine 能访问 c.v
	defer c.mux.Unlock()
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
```

使用 `sync.Mutex`，读操作与写操作都会被阻塞。其实读操作的时候我们是不需要进行阻塞的，因此sync中还有另一个锁：读写锁 `RWMutex`,这是一个单写多读模型。

`sync.RWMutex`分为：读、写锁。在读锁占用下，会阻止写，但不会阻止读，多个 `goroutine` 可以同时获取读锁，调用 `RLock()` 函数即可，`RUnlock()` 函数释放。写锁会阻止任何 `goroutine` 进来，整个锁被当前`goroutine`，此时等价于`Mutex`,写锁调用 `Lock` 启用，通过 `UnLock()` 释放。

使用 `sync.RWMutex` 改写上述代码：

```go
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
```

#### 4. 条件变量 `Cond`（待学。。。）

`sync.Cond` 是条件变量，它可以让一系列的 `Goroutine `都在满足特定条件时被唤醒。

条件变量通常与互斥锁一起使用，条件变量可以在共享资源的状态变化时通知相关协程。 

- `NewCond`: 创建一个 `Cond` 的条件变量。
- `Broadcast`: 广播通知，调用时可以加锁，也可以不加。
- `Signal`: 单播通知，只唤醒一个等待 c 的 `goroutine`。
- `Wait`: 等待通知, `Wait()`会自动释放 `c.L`，并挂起调用者的 `goroutine`。之后恢复执行，`Wait()` 会在返回时对 `c.L` 加锁。

注意：除非被 `Signal` 或者 `Broadcast` 唤醒，否则 `Wait()` 不会返回。

#### 5. 原子操作

原子操作即是进行过程中不能被中断的操作。针对某个值的原子操作在被进行的过程中，CPU绝不会再去进行其他的针对该值的操作。 为了实现这样的严谨性，原子操作仅会由一个独立的CPU指令代表和完成。

在 `sync/atomic` 中，提供了一些原子操作，包括加法（`Add`）、比较并交换（`Compare And Swap`，简称 `CAS`）、加载（`Load`）、存储（`Store`）和交换（`Swap`）。

#### 6. 临时对象池 `Pool`

`sync.Pool` 可以作为临时对象的保存和复用的集合。

#### 7. 通道Channel

这里引入一下`CSP模型`，`CSP` 是 `Communicating Sequential Process` 的简称，中文可以叫做通信顺序进程，是一种并发编程模型，由 [Tony Hoare](https://en.wikipedia.org/wiki/Tony_Hoare) 于 1977 年提出。

简单来说是实体之间通过发送消息进行通信，这里发送消息时使用的就是通道，或者叫 `Channel`。`Goroutine`对应并发实体。

- 使用方法：

```go
ch := make(chan int, 1)

// 读操作
x <- ch

// 写操作
ch <- x
```

当channel关闭后会引发下面相关问题：

- 重复关闭 `Channel` 会 `panic`
- 向关闭的 `Channel` 发数据 会 `Panic`，读关闭的 `Channel` 不会 `Panic`，但读取的是默认值

对于最后一点读操作默认值怎么区分呢？例如：Channel本身的值是默认值又或者是读到的是关闭后的默认值，可以通过下面进行区分：

```go
val, ok := <-ch
if ok == false {
    // channel closed
}
```

- 分类
  - 无缓冲的 `Channel`: 发送与接受同时进行。(如果没有`Goroutine`读取`Channel(<-Channel)`，发送者(`Channel<-x`)会一直阻塞。)
  - 有缓冲的`Channel`: 发送与接受并非同时进行。当队列为空，接受者阻塞;队列满，发送者阻塞。

#### 8. Select

- 每个 `case` 都必须是一个通信
- 所有 `channel` 表达式都会被求值
- 如果没有 `default` 语句，`select` 将阻塞，直到某个通信可以运行
- 如果多个`case`都可以运行，`select`会随机选择一个执行

特点：

1. 随机选择
2. 检查 `chan`

注意：**当多个channel需要读取数据的时候，就必须使用 `for+select`**。