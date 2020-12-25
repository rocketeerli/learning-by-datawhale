## 单元测试

在go标准库中有一个叫做`testing`的测试框架，可以进行单元测试，命令是`go test xxx`。

测试文件通常是以`xx_test.go`命名，放在同一包下面。

### 单元测试基础

- 函数代码：

```go
package rocketeerli
import "fmt"
func Add(a, b int) {
	fmt.Printf("a + b = %d", a+b)
}
```

- 测试代码：

```go
package rocketeerli

import (
	"fmt"
	"testing"
)

func TestAdd(t *testing.T) {
	type args struct {
		a int
		b int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
		{
			name: "",
			args: args{
				a: 12,
				b: 25,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Add(tt.args.a, tt.args.b)
		})
	}
}
```

1. 运行测试文件

`go test .\rocketeerli`

结果：

> ok      task12/rocketeerli      0.200s

1. 使用 `-v` 参数，打印 log 信息

```shell
go test .\rocketeerli -v
```

> `===` RUN   TestAdd
> `===` RUN   TestAdd/#00
> a + b = 37--- PASS: TestAdd (0.00s)
>     --- PASS: TestAdd/#00 (0.00s)
> PASS
> ok      task12/rocketeerli      0.217s

2. 使用 `-cover` 指定单测覆盖率

覆盖率可以简单理解为进行单元测试mock的时候，能够覆盖的代码行数占总代码行数的比率

```shell
go test .\rocketeerli -cover
```

结果：

> ok      task12/rocketeerli      0.176s  coverage: 100.0% of statements

3. 使用(table-driven tests)表格驱动型测试，可以填写很多测试样例

具体代码见上面，在 `TODO` 里面我们可以填写很多单元测试样例。

### 基准测试（不会用\~）

基准测试函数名字必须以 `Benchmark` 开头，代码在 `xxx_test.go` 中。

- 运行

```shell
go test -benchmem -run=. -bench=.
```

### mock/stub测试（？？？不懂呀！！！）

`gomock` 是官方提供的 `mock` 框架，同时有 `mockgen` 工具来辅助生成测试代码。

- 安装

```shell
go get -u github.com/golang/mock/gomock
go get -u github.com/golang/mock/mockgen
```

- 打桩

最大的好处：

**不直接依赖的实例，而是使用依赖注入降低耦合性。**

### 浏览器实时测试

`goconvey` 可以在浏览器进行实时查看单元测试结果。

可以使用convey进行单测。

