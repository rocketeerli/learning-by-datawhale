# 变量、常量、枚举

## 变量

### 声明方式：

    var identifier type
    var identifier1, identifier2 type

## 常量

在运行时不会被修改的量

### 声明：

可以省略类型说明符[type]

    const identifier [type] = value
    const b = "abc"

**注意：**

`iota`，特殊常量，可认为是可以被编译器修改的常量。在 `const`关键字出现时将被重置为 0(const 内部的第一行之前)，`const` 中每新增一行常量声明将使 `iota` 计数一次(`iota` 可理解为 `const` 语句块中的行索引)。第一个 `iota` 等于 0，每当 `iota` 在新的一行被使用时，它的值都会自动加 1；

## 枚举

枚举，将变量的值一一列举出来

Go语言中没有枚举这种数据类型的，但是可以使用`const`配合`iota`模式来实现

### 普通模式

    const (
        a = 0
    	b = 1
    	c = 2
    	d = 3
    )

### 自增模式

1. `iota` 只能用于常量表达式
2. 它默认开始值是0，`const` 中每增加一行加1,同行值相同

```golang
const (
	a = iota //0
	c        //1
	d        //2
)
const (
	e, f = iota, iota //e=0, f=0
	g    = iota       //g=1
)
```

3. 若中间中断iota，必须显式恢复。

```
const (
  a = iota    //0
  b           //1
  c = 100     //100
  d           //100
  e = iota    //4
)
```

