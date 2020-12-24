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
