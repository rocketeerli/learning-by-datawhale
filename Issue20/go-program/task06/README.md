## 数组、切片

### 数组

#### 定义

```go
//方式一
var arr1 = [5]int{}
//方式二
var arr2 = [5]int{1,2,3,4,5}
//方式三
var arr3 = [5]int{3:10}
```

#### 遍历

* ```go
  for i := 0; i < len(arr1); i++ {}
  ```

* ```go
  for index, value := range arr1
  ```

#### 多维数组

```go
var arr4 = [5][5]int{
		{1, 2, 3, 4, 5},
		{6, 7, 8, 9, 10},
	}
```

#### 作为函数参数

go语言中数组默认是值传递的。在传递数组时会对其进行拷贝，所以如果传递的是大数组的话会非常占内存，所以一般情况下很少直接传递一个数组，避免这种情况我们可以使用以下两种方式：

- 传递数组的指针
- 传递切片

#### 指针数组与数组指针

- 指针数组来说：一个数组里面装的都是指针
- 数组指针： 指向数组的指针

### 切片

切片长度是不固定的，可以追加元素，如果以达到当前切片容量的上限会再自动扩容。

#### 定义

```go
//方法一
var s1 = []int{}
//方法二
var s2 = []int{1, 2, 3}
//方法三
var s3 = make([]int, 5)
//方法四
var s4 = make([]int, 5, 10)  // 长度为5，容量为10
```

#### 切片操作

* 切片扩充(`append`)
* 切片拼接(`append(a, b...)`)

* 切片复制 (`copy`)

  - 声明b切片时，其长度比a切片长，复制结果是怎么样的？（**结尾为原来的值**）

  - 声明b切片时，其长度比a切片短，复制结果是怎么样的？（**仅拷贝a的前len(b)个值**）

  - 声明b切片时，其长度被定义为0，那么调用copy函数会报错吗？（**不会，b依然为[]**）

##### 问题

- 编写程序看看切片的容量与数组的大小有什么关系呢？（**创建切片时，默认容量是长度乘上2；追加时，首先判断容量是否足够，足够直接追加；否则判断是否超过1024，超过，容量乘上1.25；没超过，容量乘上2**）

- 如果我们在切片上再做切片那么他们会指向相同的底层数组吗？（**会**）修改其中一个切片会影响其他切片的值么？（**会**）其中一个切片扩容到容量大小之后会更换底层数组，那么之前的其他切片也会指向新的底层数组吗？（**不会，只有扩容后的切片指向新的底层数组**）

**当我们的切片容量大于底层数组容量时，会自动创建一个新的底层数组，取消对原数组的引用**

既然切片是引用底层数组的，需要注意的就是小切片引用大数组的问题，如果底层的大数组一直有切片进行引用，那么垃圾回收机制就不会将其收回，造成内存的浪费，最有效的做法是copy需要的数据后再进行操作。