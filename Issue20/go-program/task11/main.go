package main

import (
	"fmt"
	"reflect"
)

func main() {
	var Num float64 = 3.14
	// 将“接口类型变量”转换为“反射类型对象”。
	v := reflect.ValueOf(Num)
	t := reflect.TypeOf(Num)
	fmt.Println("Reflect : Num.Value = ", v)
	fmt.Println("Reflect : Num.Type  = ", t)
	// 将“反射类型对象”转换为“接口类型变量”。
	origin := v.Interface().(float64)
	fmt.Println(origin)
	// 如果要修改“反射类型对象”，其值必须是“可写的”。
	// 首先通过 CanSet 函数判断是否可以修改
	fmt.Println("v的可写性:", v.CanSet())
	if ok := v.CanSet(); ok {
		fmt.Println("可以修改其值")
		v.SetFloat(5.59)
		fmt.Println("Reflect : Num.Value = ", v)
	}
	// 通过反射修改内容
	var f float64 = 3.15
	p := reflect.ValueOf(&f)
	fmt.Println("p的可写性:", p.CanSet())
	vl := p.Elem()
	fmt.Println("vl的可写性:", vl.CanSet())
	vl.SetFloat(5.69)
	fmt.Println(f)
}
