#include <stdio.h>
#include <type_traits>
#include <iostream>

using namespace std;

template <typename T>
class primitives {
 public:
  T a;
  T b;
  
  primitives(T _a, T _b) {
    this->a = _a;
    this->b = _b;
  }
  
  T sum() {
    T result = a + b; 
    return result;
  } 
};


template<typename T>
void ncclAllReduce(T a, T b) {
  primitives<T> prims(a, b);                          //when T != float
  //primitives<int32_t> prims(1, 2);                  //when T == float
  //calling one of the prims functions

  cout << prims.sum();
  printf("\n");
}

int main()
{
  ncclAllReduce<float>(2.5, 2.7);
  ncclAllReduce<int>(1,5);
  return 0;
}
