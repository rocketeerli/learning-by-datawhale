package main

import(
	"fmt"
	"strconv"
)

func main() {
	slice := []int{7, 9, 3, 5, 1}
	x := min(slice...)
	fmt.Printf("The minimum is: %d\n", x)
	test()
	fmt.Println("----------------")
	test1()
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

func test() {
	f := func() string {
		return "hello world"
	}
	fmt.Println(f())
}

func add() func(int) int {
	n := 10
	str := "string"
	return func(x int) int {
		n = n + x
		str += strconv.Itoa(x)
		fmt.Print(str, " ")
		return n
	}
}

func test1() {
	f := add()
	fmt.Println(f(1))
	fmt.Println(f(2))
	fmt.Println(f(3))

	f = add()
	fmt.Println(f(1))
	fmt.Println(f(2))
	fmt.Println(f(3))
}

