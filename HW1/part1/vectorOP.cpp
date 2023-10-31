#include "PPintrin.h"
#include <stdio.h>

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  // All ones
  maskAll = _pp_init_ones();

  // All zeros
  maskIsNegative = _pp_init_ones(0);

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  __pp_vec_float x, result;
  __pp_vec_int y, count;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_mask maskAll, maskIsZero, maskIsNotZero;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if (i + VECTOR_WIDTH > N)
    {
      maskAll = _pp_init_ones(N - i);
    }
    else
    {
      maskAll = _pp_init_ones();
    }

    _pp_vload_float(x, values + i, maskAll);  // x = values[i];
    _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];

    _pp_veq_int(maskIsZero, y, zero, maskAll);  // if (y == 0) {
    _pp_vset_float(result, 1.f, maskIsZero);    //   output[i] = 1.f;
    maskIsNotZero = _pp_mask_not(maskIsZero);   // } else {
    _pp_vmove_float(result, x, maskIsNotZero);  //   result = x;
    _pp_vsub_int(count, y, one, maskIsNotZero); //   count = y - 1;

    __pp_mask maskLoop = maskIsNotZero;
    __pp_mask maskIsNotDone = _pp_init_ones(0);

    // check if count > 0 before the loop
    _pp_vgt_int(maskIsNotDone, count, zero, maskLoop);
    maskLoop = _pp_mask_and(maskLoop, maskIsNotDone);

    while (_pp_cntbits(maskLoop) > 0)
    {
      _pp_vmult_float(result, result, x, maskLoop); //   result *= x;
      _pp_vsub_int(count, count, one, maskLoop);    //   count--;
      _pp_vgt_int(maskIsNotDone, count, zero, maskLoop);
      maskLoop = _pp_mask_and(maskLoop, maskIsNotDone);
    }

    __pp_vec_float temp;
    __pp_vec_float upperLimit = _pp_vset_float(9.999999f);
    _pp_vgt_float(maskIsNotZero, result, upperLimit, maskIsNotZero); //   if (result > 9.999999f) {
    _pp_vmove_float(temp, upperLimit, maskIsNotZero);                //     result = 9.999999f;
    _pp_vmove_float(result, temp, maskIsNotZero);                    //   }
    _pp_vstore_float(output + i, result, maskAll);                   //   output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  __pp_vec_float sum = _pp_vset_float(0.0f);
  __pp_vec_float x;
  __pp_mask maskAll;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(x, values + i, maskAll);
    _pp_vadd_float(sum, sum, x, maskAll);
  }

  __pp_vec_float temp;
  int numActiveLanes = VECTOR_WIDTH;
  while (numActiveLanes > 1)
  {
    int halfNumActiveLanes = numActiveLanes / 2;

    maskAll = _pp_init_ones(halfNumActiveLanes);
    _pp_hadd_float(temp, sum);
    _pp_interleave_float(sum, temp);

    numActiveLanes = halfNumActiveLanes;
  }

  float result;
  _pp_vstore_float(&result, sum, maskAll);

  return result;
}