#include <stdio.h>


void foo(int* arr) {
  int* a = arr;  
  a[0] = 10; 
  a[1] = 20;
  a[2] = 30;
  a[3] = 40;
} 

int main() { 
  
  int a[4] = {1,2,3,4};
  int* b = a;
  for (int i=0; i<4; ++i) {
    printf("b[%d] = %d\n", i, b[i]);
  }
  
  foo(b);

  for (int i=0; i<4; ++i) {
    printf("updated b[%d] = %d\n", i, b[i]);
  }
  return 0; 
}
