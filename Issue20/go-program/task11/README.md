## 反射

### 定义

> 反射的概念是由Smith在1982年首次提出的，主要是指程序可以访问、检测和修改它本身状态或行为的一种能力。
>
> Go 语言提供了一种机制在运行时更新变量和检查它们的值、调用它们的方法，但是在编译时并不知道这些变量的具体类型，这称为反射机制。

### 作用（？？？）

反射是为了解决在运行期，对某个实例一无所知的情况下，如何调用其方法。

* **在编写不定传参类型函数的时候，或传入类型过多时**（例如，对象关系映射？？？）
* **不确定调用哪个函数，需要根据某些条件来动态执行**

```go
func bridge(funcPtr interface{}, args ...interface{})
```

第一个参数funcPtr以接口的形式传入函数指针，函数参数args以可变参数的形式传入，bridge函数中可以用反射来动态执行funcPtr函数。

### 实现

Go的反射基础是接口和类型系统，Go的反射机制是通过接口来进行的。

Go 语言在 reflect 包里定义了各种类型，实现了反射的各种函数，通过它们可以在运行时检测类型的信息、改变类型的值。

#### 反射三定律

1. **反射可以将“接口类型变量”转换为“反射类型对象”。**

reflect包里的两个方法**reflect.Value**和**reflect.Type**，将接口类型变量分别转换为反射类型对象。

```go
package main

import (
	"fmt"
	"reflect"
)

func main() {
	var Num float64 = 3.14

	v := reflect.ValueOf(Num)
	t := reflect.TypeOf(Num)

	fmt.Println("Reflect : Num.Value = ", v)
	fmt.Println("Reflect : Num.Type  = ", t)
}
```

函数签名：

```go
func TypeOf(i interface{}) Type
```

```go
func (v Value) Interface() (i interface{})
```

2. **反射可以将“反射类型对象”转换为“接口类型变量”。**

可以使用 Interface 方法恢复其接口类型的值。

```go
package main
import (
    "fmt"
    "reflect"
)
func main() {
    var Num = 3.14
    v := reflect.ValueOf(Num)
    t := reflect.TypeOf(Num)
    fmt.Println(v)
    fmt.Println(t)

    origin := v.Interface().(float64)
    fmt.Println(origin)
}
```

3. **如果要修改“反射类型对象”，其值必须是“可写的”。**

- 直接使用 `set` 方法修改值会出现错误。

```go
v.SetFloat(6.18)
```

原因： **反射对象v包含的是副本值，所以无法修改。**(可以先拿到指针，再去通过指针去访问元素，进行修改，具体实现过程见下方实践小节)

- 可以通过 `CanSet` 函数来判断反射对象是否可以修改

```go
fmt.Println("v的可写性:", v.CanSet())
```

#### 小结

1.反射对象包含了接口变量中存储的值以及类型。

2.如果反射对象中包含的值是原始值，那么可以通过反射对象修改原始值；

3.如果反射对象中包含的值不是原始值（反射对象包含的是副本值或指向原始值的地址），则该反射对象不可以修改。

### 实践

1. 通过反射修改内容

**通过反射修改内容**

```
var f float64 = 3.41
fmt.Println(f)
p := reflect.ValueOf(&f)
v := p.Elem()
v.SetFloat(6.18)s
fmt.Println(f)
```

`reflect.Elem()` 方法获取这个指针指向的元素类型。这个获取过程被称为取元素，等效于对指针类型变量做了一个*操作

2. 通过反射调用方法（感觉有点高大上）

```
package main

import (
        "fmt"
        "reflect"
)

func hello() {
  fmt.Println("Hello world!")
}

func main() {
  hl := hello
  fv := reflect.ValueOf(hl)
  fv.Call(nil)
}
```

### 反射的缺点

反射会使得代码执行效率较慢，原因有

1. 涉及到内存分配以及后续的垃圾回收

2. `reflect` 实现里面有大量的枚举，也就是for循环，比如类型之类的