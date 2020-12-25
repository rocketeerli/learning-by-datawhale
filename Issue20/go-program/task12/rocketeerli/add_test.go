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

func BenchmarkAdd(t *testing.B) {
	for i := 0; i < t.N; i++ {
		fmt.Sprintf("hello")
	}
}
