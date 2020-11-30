#include<stdio.h> 
#include <stdint.h>

//float floatCast(int8_t num) {
//  return static_cast<float> (num); 
//}


int main() 
{

    //thrust::device_ptr<int8_t> dev_ptr1(thisOutput1);
    //thrust::device_ptr<float> dev_ptr2(thisOutput);
    //thrust::transform(dev_ptr1, dev_ptr1+size, dev_ptr2, floatCast);
    int arr[6] = {10, 20, 30, 40, 50, 60};
    printf(arr.begin());
    return 0;
}
