## 函数

### 定义

```go
func functionName([parameter list]) [returnTypes]{
   //body
}
```

### 值传递和引用传递

- 值传递： `func xxx(int a)`
- 引用传递：`func xxx(int *a)` （传地址）

### 变长参数

在go语言中也支持变长参数，但需要注意的是**变长参数必须放在函数参数的最后一个**，否则会报错。

```go
func main() {
	slice := []int{7, 9, 3, 5, 1}
	x := min(slice...)
	fmt.Printf("The minimum is: %d", x)
}
func min(s ...int) int {
	if len(s) == 0 {
		return 0
	}
	min := s[0]
	for _, v := range s {
		if v < min {
			min = v
		}
	}
	return min
}
```

### 多返回值

go语言中函数还支持一个特性那就是：多返回值。

通过返回结果与一个错误值，这样可以使函数的调用者很方便的知道函数是否执行成功，这样的模式也被称为`command, ok`模式。

### 命名返回值

当需要返回的时候，我们只需要一条简单的不带参数的 `return` 语句。

### 匿名函数

```go
func main() {
	f := func() string {
		return "hello world"
	}
	fmt.Println(f())
}
```

### 闭包

闭包可以解释为**一个函数与这个函数外部变量的一个封装**。粗略的可以理解为一个类，类里面有变量和方法，**其中闭包所包含的外部变量对应着类中的静态变量。** 

- 最开始我们先声明一个函数`add`，在函数体内返回一个匿名函数
- 其中的`n`,`str`与下面的匿名函数构成了整个的闭包，`n`与`str`就像类中的静态变量只会初始化一次，所以说尽管后面多次调用这个整体函数，里面都不会再重新初始化了
- 而且对于外部变量的操作是累加的，这与类中的静态变量也是一致的

**在汇编代码中，闭包返回的不仅仅是匿名函数，还包括所引用的环境变量指针，这与我们之前的解释也是类似的，闭包通过操作指针来调用对应的变量。**

##### 问题

尝试一下如何通过闭包来实现斐波那契数列。

```go
package main

import "fmt"

func fb() func() int{
	a, b := 0, 1
	return func() int {
		tmp := b
		b = b + a
		a = tmp
		return b
	}
}

func main() {
	f := fb()
	n := 10
	for i := 0; i <n; i++ {
		fmt.Println(f())
	}
}
```

