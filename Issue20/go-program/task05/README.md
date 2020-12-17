## 字典、字符串

### 字典

#### 定义字典

三种定义字典的方式：

```golang
var m1 map[string]int
m2 := make(map[int]interface{}, 100)
m3 := map[string]string{
	"name": "james",
	"age":  "35",
}
```

注意：

* 定义字典时不需要为其指定容量，因为map是可以动态增长的
* 不能使用不能比较的元素作为字典的key，例如数组，切片等。
* value可以是任意类型的，如果使用 `interface{}` 作为value类型，那么就可以接受各种类型的值

#### 判断字典中的元素是否存在

```
if value, ok := m3["name"]; ok {
		fmt.Println(value)
	}
```

#### 遍历字典

```
for key, value := range m3 {
		fmt.Println("key: ", key, " value: ", value)
}
```

* 默认字典是无序的，因此每次运行时输出都不同。
* 可以将函数作为值类型存入到字典中

### 字符串

#### 定义

创建字符串之后其值是不可变的。

以下操作不被允许：

```
s := "hello"
s[0] = 'T'
```

想要修改一个字符串的内容，我们可以将其转换为字节切片，再将其转换为字符串:

```
func main() {
	s := "hello"
	b := []byte(s)
	b[0] = 'g'
	s = string(b)
	fmt.Println(s) //gello
}
```

* 如果字符串中包含中文就不能直接使用byte切片对其进行操作（将字符串转为 `rune` 切片）

### 包

* `strings` 包: `strings.HasPrefix`, `strings.HasSuffix`, `strings.Contains`
* `strconv` 包: 实现了基本数据类型与字符串之间的转换。`strconv.Atoi("-42")`, `strconv.Itoa(-42)`

#### 拼接

* `SPrintf`: `s = fmt.Sprintf("%v%v", s, i)`
* `+`: `s += strconv.Itoa(i)`
* `bytes.Buffer` : `buf.WriteString(strconv.Itoa(i))`
* `strings.Builder`: `builder.WriteString(strconv.Itoa(i))`

实验结论： 通过`strings.Builder`拼接字符串是最高效的。