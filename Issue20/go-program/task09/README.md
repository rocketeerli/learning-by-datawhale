## 包管理

### 概念

Go语言通过包管理来封装模块和复用代码，这里只介绍 `Go Modules` 管理方法。

`Go Modules`于Go语言1.11版本时引入，在1.12版本正式支持，是由Go语言官方提供的包管理解决方案

### 使用方法

* 环境变量：`go env` 查看配置

* 更改：`go env -w GO111MODULE=on`

* 环境变量说明

  GO111MODULE

  - auto：只要项目包含了 go.mod 文件的话启用 Go modules，目前在 Go1.11 至 Go1.14 中仍然是默认值。
  - on：启用 Go modules，推荐设置，将会是未来版本中的默认值。
  - off：禁用 Go modules，不推荐设置。

  GOPROXY：用于设计 `Go Module` 的代理。

  GOSUMDB：用于在拉取模块的时候保证模块版本数据的一致性。

* 初始化模块

  `Go Modules` 的使用方法比较灵活，在目录下包含 `go.mod` 文件即可。

  可以直接使用命令 `go mod init [module name]` 会自动创建 `go.mod` 文件。

* `go get` 命令：用于拉取新的依赖

  参数：

  ```shell
  -d 只下载不安装
  -f 只有在你包含了 -u 参数的时候才有效，不让 -u 去验证 import 中的每一个都已经获取了，这对于本地 fork 的包特别有用
  -fix 在获取源码之后先运行 fix，然后再去做其他的事情
  -t 同时也下载需要为运行测试所需要的包
  -u 强制使用网络去更新包和它的依赖包
  -v 显示执行的命令
  ```

* #### 常用命令

  ```go
  go mod init  // 初始化go.mod
  go mod tidy  // 更新依赖文件
  go mod download  // 下载依赖文件
  go mod vendor  // 将依赖转移至本地的vendor文件
  go mod edit  // 手动修改依赖文件
  go mod graph  // 查看现有的依赖结构
  go mod verify  // 校验依赖
  ```

