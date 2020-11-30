#include <stdio.h>
#include <type_traits>
#include <iostream>

using namespace std;


template <typename T>
class primitives {
 public:
  int a;
  int b;
  
  primitives(int _a, int _b) {
    this->a = _a;
    this->b = _b;
  }
  
  T sum() {
    T result = (T)a + (T)b; 
    return result;
  }
};


template<typename T>
void foo() {
  if (std::is_same<T, float>::value) {
    primitives<int32_t> prims(5, 6);                    //when T == float
    cout << prims.sum() << endl;
  }
  else {
    primitives<T> prims(1, 2);                          //when T != float
    cout << prims.sum() << endl;
  }
}

int main()
{
  foo<float>();
  foo<int>();
  return 0;
}
