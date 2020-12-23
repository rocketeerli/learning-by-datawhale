## 异常处理

### error

Go语言内置了一个简单的错误接口作为一种错误处理机制，接口定义如下：

```go
type error interface {
	Error() string
}
```

#### error 的两种构造方法

* `errors.New()`: 

```go
err := errors.New("This is an error")
if err != nil {
  fmt.Print(err)
}
```

* `fmt.Errorf()`:

```go
err := fmt.Errorf("This is an error")
if err != nil {
  fmt.Print(err)
}
```

### panic

- panic后面的程序不会被执行

- 为了让程序能正常运行下去，需要使用 recover。

#### 原理

panic()

>当函数F调用panic时，F的正常执行立即停止。任何被F延迟执行的函数都将以正常的方式运行，然后F返回其调用者。对调用方G来说，对F的调用就像调用panic一样。

recover()

> 在defer函数（但不是它调用的任何函数）内执行恢复调用，通过恢复正常执行来停止panicking序列，并检索传递给panic调用的错误值。如果在defer函数之外调用recover，则不会停止panicking的序列。

### 源码分析

```go
func New(text string) error {
	return &errorString{text}
}

// errorString is a trivial implementation of error.
type errorString struct {
	s string
}

func (e *errorString) Error() string {
	return e.s
}
```

1. New 函数返回格式为给定文本的错误

2. 即使文本是相同的，每次对 New 的调用都会返回一个不同的错误值。