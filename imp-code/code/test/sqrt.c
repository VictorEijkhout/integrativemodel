#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main() {
  int factor;
  for (factor=1; factor<42000; factor++) {
    int product = factor*factor;
    double root = sqrt(product);
    int found = (int)root;
    if (found<factor) printf("root %e < %d\n",root,factor);
  }
  return 0;
}
