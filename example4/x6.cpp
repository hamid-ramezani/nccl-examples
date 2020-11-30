#include <stdio.h>
#include <math.h>


template <int BITS>
void find_meta_seq(const float* input, unsigned char* meta, int num_elem, int bucket_size) {
  const unsigned int divisor = (1 << BITS) - 1;
  float* meta_buf = (float*)meta;
  for (int i = 0; i < (num_elem + bucket_size - 1) / bucket_size; i += 1) {
    float mmin = input[i * bucket_size];
    float mmax = input[i * bucket_size];
    for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, num_elem); j++) {
      mmin = fminf(mmin, input[j]);
      mmax = fmaxf(mmax, input[j]);
    }
    meta_buf[2 * i] = (mmax - mmin) / divisor;
    meta_buf[2 * i + 1] = mmin;
  }
  //printf("size of meta_buff = %d\n", sizeof(meta_buf));
  for (int i=0; i<8; ++i) { 
    printf("meta_buff[%d] = %f\n", i, meta_buf[i]);
  }
}


void func1() {
   float arr[8];
   for (int i=0; i<8; i++) {
      arr[i] = static_cast<float>(i+1); 
   }
   const float* input = arr;
   unsigned char b = 0;
   unsigned char* meta = &b;
   find_meta_seq<3>(input, meta, 8, 2);
   //printf("size of meta = %d\n", sizeof(meta));
   for (int i=0; i<8; ++i) { 
     printf("meta[%d] = %f\n", i, meta[i]);
   }
}


int main() {
   func1();
   //printf("Hello, World!\n");
   return 0;
}
