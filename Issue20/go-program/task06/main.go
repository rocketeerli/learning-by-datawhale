package main

import(
    "fmt"
)

func main() {
    var arr = [5]int{3:10}
    fmt.Println(arr)
    fmt.Println("-------------")
    test1()
    fmt.Println("-------------")
    test2()
    fmt.Println("-------------")
    test3()
}

func test1() {
	a := []int{1, 2, 3}
	b := a[1:3]
	//a[1] = 10
	b = append(b, 4)
	b = append(b, 5)
	b = append(b, 6)
	b = append(b, 7)
	a[1] = 10
	fmt.Println(a)
	fmt.Println(b)
}

func test2() {
	a := []int{1, 2, 3}
	b := a[1:3]
	fmt.Println(b)
	a = append(a, b...)
	fmt.Println(a)
}

func test3() {
    a := []int{1, 2, 3}
    b := make([]int, 4)
    copy(b, a)
    fmt.Println(a)
    fmt.Println(b)
}

