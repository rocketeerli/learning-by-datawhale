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

