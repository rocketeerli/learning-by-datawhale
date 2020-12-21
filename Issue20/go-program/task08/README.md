# 结构体、方法和接口

## 结构体

- Go 语言中没有“类”的概念，也不支持像继承这种面向对象的概念。

- Go 语言中结构体的组合方式比面向对象具有更高的扩展性和灵活性。

### 语法

```go
type identifier struct {
  field1 type1
  field2 type2
  ...
}
```

#### 例子

* 普通模式

```
type Student struct {
	Name string
	Age int
}
```

* 字段类型可以是任何类型

```go
type ListNode struct {
  Val int
  Next *ListNode
}
```

* 匿名字段

可以不给字段指定名字

```go
type Person struct {
	ID string
	int
}
```

注意：**定义结构体的字段时首字母为小写在其他包是不能直接访问该字段的。**

#### 创建结构体

```go
s1 := new(Student) //第一种方式
s2 := Student{"james", 35} //第二种方式
s3 := &Student { //第三种方式
	Name: "LeBron",
	Age:  36,
}
```

#### 赋值

```go
s1.Name = "james"
s1.Age = 35
```

将定义的结构体首字母也变为小写那么在其他包内就不能直接创建该结构体

可以使用匿名字段：

```go
p := new(Person)
p.ID = "123"
p.int = 10
```

**对于一个结构体来说，每一种数据类型只能有一个匿名字段。**

### 标签

在go语言中结构体除了字段的名称和类型外还有一个可选的标签tag，标记的tag只有reflect包可以访问到，一般用于`orm`或者`json`的数据传递。 （不太懂`orm`是啥）

```go
type Student struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}
```

使用go自带的json包将声明的结构体变量转变为json字符串。

```go
func ToJson(s *Student) (string, error) {
	bytes, err := json.Marshal(s)
	if err != nil {
		return "", nil
	}
	return string(bytes), nil
}
```

### 内嵌结构体

结构体作为一种数据类型也可以将其生命为匿名字段，此时我们称其为内嵌结构体

* 例子

```go
type A struct {
	X, Y int
}

type B struct {
	A
	Name string
}
```

## 方法

### 定义

方法与函数类似，只不过在方法定义时会在func和方法名之间增加一个参数，如下所示：

```go
func (r Receiver)func_name(){
  // body
}
```

### 方法接收者

对于一个方法来说接收者分为两种类型：值接收者和指针接收者。

- 使用指针接收者的话，在方法体内的修改就会影响原来的变量。
- 使用值接收者定义的方法使用指针来调用也是可以的，反过来也是如此

## 接口

### 定义

```go
type Namer interface {
    Method1(param_list) return_type
    Method2(param_list) return_type
    ...
}
```

###  实现接口

在go语言中不需要显示的去实现接口，只要一个类型实现了该接口中定义的所有方法就是默认实现了该接口。

### 类型断言

判断接口的类型

```go
func IsDog(a Animal) bool {
	if v, ok := a.(Dog); ok {
		fmt.Println(v)
		return true
	}
	return false
}
```

对传递进来的参数进行判断，判断其是否为Dog类型，如果是Dog类型的话就会将其进行转换为v，ok用来表示是否断言成功。

### 空接口

空接口是一个比较特殊的类型，因为其内部没有定义任何方法所以空接口可以表示任何一个类型，比如可以进行下面的操作：

```go
var any interface{}

any = 1
fmt.Println(any)

any = "hello"
fmt.Println(any)

any = false
fmt.Println(any)
```