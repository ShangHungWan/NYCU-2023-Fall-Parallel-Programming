# HW1

## Q1-1

We could observe that vector utilization decreased as `VECTOR_WIDTH` changes from statistics below.
Because when `N % VECTOR_WIDTH != 0`, we can't not utilize 100% of the last vector.
We only use parts of it because we don't want to be overwritten.

### VECTOR_WIDTH = 2

```shell
$ make && ./myexp -s 10000
g++ -I./common -O3 -std=c++17 -Wall -c PPintrin.cpp
g++ -I./common -O3 -std=c++17 -Wall -c logger.cpp
g++  -I./common logger.o PPintrin.o main.cpp serialOP.cpp vectorOP.cpp -o myexp
CLAMPED EXPONENT (required) 
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              2
Total Vector Instructions: 203407
Vector Utilization:        80.0%
Utilized Vector Lanes:     325579
Total Vector Lanes:        406814
************************ Result Verification *************************
ClampedExp Passed!!!

ARRAY SUM (bonus)
****************** Printing Vector Unit Statistics *******************
Vector Width:              2
Total Vector Instructions: 10002
Vector Utilization:        100.0%
Utilized Vector Lanes:     20003
Total Vector Lanes:        20004
************************ Result Verification *************************
ArraySum Passed!!!
```

### VECTOR_WIDTH = 4

```shell
$ make && ./myexp -s 10000
g++ -I./common -O3 -std=c++17 -Wall -c PPintrin.cpp
g++ -I./common -O3 -std=c++17 -Wall -c logger.cpp
g++  -I./common logger.o PPintrin.o main.cpp serialOP.cpp vectorOP.cpp -o myexp
CLAMPED EXPONENT (required) 
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              4
Total Vector Instructions: 118217
Vector Utilization:        74.4%
Utilized Vector Lanes:     352003
Total Vector Lanes:        472868
************************ Result Verification *************************
ClampedExp Passed!!!

ARRAY SUM (bonus)
****************** Printing Vector Unit Statistics *******************
Vector Width:              4
Total Vector Instructions: 5002
Vector Utilization:        100.0%
Utilized Vector Lanes:     20005
Total Vector Lanes:        20008
************************ Result Verification *************************
ArraySum Passed!!!
```

### VECTOR_WIDTH = 8

```shell
$ make && ./myexp -s 10000
g++ -I./common -O3 -std=c++17 -Wall -c PPintrin.cpp
g++ -I./common -O3 -std=c++17 -Wall -c logger.cpp
g++  -I./common logger.o PPintrin.o main.cpp serialOP.cpp vectorOP.cpp -o myexp
CLAMPED EXPONENT (required) 
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              8
Total Vector Instructions: 64532
Vector Utilization:        71.5%
Utilized Vector Lanes:     369363
Total Vector Lanes:        516256
************************ Result Verification *************************
ClampedExp Passed!!!

ARRAY SUM (bonus)
****************** Printing Vector Unit Statistics *******************
Vector Width:              8
Total Vector Instructions: 2502
Vector Utilization:        100.0%
Utilized Vector Lanes:     20009
Total Vector Lanes:        20016
************************ Result Verification *************************
ArraySum Passed!!!
```

### VECTOR_WIDTH = 16

```shell
$ make && ./myexp -s 10000
g++ -I./common -O3 -std=c++17 -Wall -c PPintrin.cpp
g++ -I./common -O3 -std=c++17 -Wall -c logger.cpp
g++  -I./common logger.o PPintrin.o main.cpp serialOP.cpp vectorOP.cpp -o myexp
CLAMPED EXPONENT (required) 
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              16
Total Vector Instructions: 33707
Vector Utilization:        70.2%
Utilized Vector Lanes:     378595
Total Vector Lanes:        539312
************************ Result Verification *************************
ClampedExp Passed!!!

ARRAY SUM (bonus)
****************** Printing Vector Unit Statistics *******************
Vector Width:              16
Total Vector Instructions: 1252
Vector Utilization:        99.9%
Utilized Vector Lanes:     20017
Total Vector Lanes:        20032
************************ Result Verification *************************
ArraySum Passed!!!
```

## Q2-1

Observe the below diff result of `test1.vec.restr.align.s` and `test1.vec.restr.align.avx2.s`, we could find out that use 16 bytes to align.
However, `avx2` version use unaligned version instruction: `MOVAPS`.
In the beginning, we could look up the [document](https://www.felixcloutier.com/x86/movaps).
We could find out there are three types of MOVAPS alignment, they are 16-bytes, 32-bytes, and 64-bytes.
Firstly, 16-bytes couldn't align correctly. Thus, we could try out other two types.
Secondly, the diff result showed that `avx2` version that the different size of its instruction is 32.

```plain
...
        movaps  16(%rbx,%rcx,4), %xmm1                        |         vmovups 32(%rbx,%rcx,4), %ymm1
        addps   (%r15,%rcx,4), %xmm0                          |         vmovups 64(%rbx,%rcx,4), %ymm2
        addps   16(%r15,%rcx,4), %xmm1                        |         vmovups 96(%rbx,%rcx,4), %ymm3
...
```

Therefore, we guess that we need to align the variable to 32 bytes:

```c
void test(float *__restrict a, float *__restrict b, float *__restrict c, int N)
{
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  fasttime_t time1 = gettime();
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < N; j++)
    {
      c[j] = a[j] + b[j];
    }
  }
  fasttime_t time2 = gettime();

  double elapsedf = tdiff(time1, time2);
  std::cout << "Elapsed execution time of the loop in test1():\n"
            << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}
```

We could find out that the `avx2` version use `vmovaps` instead of `vmovups` now:

```plain
...
        movaps  16(%rbx,%rcx,4), %xmm1                        |         vmovaps 32(%rbx,%rcx,4), %ymm1
        addps   (%r15,%rcx,4), %xmm0                          |         vmovaps 64(%rbx,%rcx,4), %ymm2
        addps   16(%r15,%rcx,4), %xmm1                        |         vmovaps 96(%rbx,%rcx,4), %ymm3
...
```

## Q2-2

test1, unvectorized, not fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
8.5956sec (N: 1024, I: 20000000)
```

test1, vectorized, not fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
2.10834sec (N: 1024, I: 20000000)
```

test1, AVX2, not fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
1.06596sec (N: 1024, I: 20000000)
```

test1, unvectorized, fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
8.38047sec (N: 1024, I: 20000000)
```

test1, vectorized, fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
2.09753sec (N: 1024, I: 20000000)
```

test1, AVX2, fixed:

```plain
Running test1()...
Elapsed execution time of the loop in test1():
1.07915sec (N: 1024, I: 20000000)
```

We could get conclusions:

1. for unfixed version, vectorized version is 4.0770 times faster than unvectorized one.
2. for unfixed version, AVX2 version is 8.0637 times faster than unvectorized one and 1.9779 times faster than vectorized one.
3. for fixed version, vectorized version is 3.9954 times faster than unvectorized one.
4. for fixed version, AVX2 version is 7.7658 times faster than unvectorized one and 1.9437 times faster than vectorized one.

We could observe that the width of instruction in the asm is 16 bytes.
Thus, we could obtain the bit width by 16 * 8 = 128.

```plain
...
	movups	(%rbx,%rcx,4), %xmm0
	movups	16(%rbx,%rcx,4), %xmm1
	movups	(%r15,%rcx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%r15,%rcx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%r14,%rcx,4)
	movups	%xmm0, 16(%r14,%rcx,4)
	movups	32(%rbx,%rcx,4), %xmm0
	movups	48(%rbx,%rcx,4), %xmm1
	movups	32(%r15,%rcx,4), %xmm2
...
```

From Q2-1, we already know the width of instruction in the asm is 32 bytes.
Thus, we could obtain the bit width by 32 * 8 = 256.

Also, we fix the code of `test2.cpp` and `test3.cpp` and get their execution time:

test2, not fixed:

```plain
Running test2()...
Elapsed execution time of the loop in test2():
14.0207sec (N: 1024, I: 20000000)
```

test2, fixed:

```plain
Running test2()...
Elapsed execution time of the loop in test2():
2.04426sec (N: 1024, I: 20000000)
```

test3, not fixed:

```plain
Running test3()...
Elapsed execution time of the loop in test3():
15.8326sec (N: 1024, I: 20000000)
```

test3, fixed:

```plain
Running test3()...
Elapsed execution time of the loop in test3():
4.07231sec (N: 1024, I: 20000000)
```

## Q2-3

For `test2.cpp`, we could find that the latter version use instructions like `movaps`, `maxps` instead of a lot of branch operations.
Thus, we could infer that those vector instructions can speed up by compiler. In the contract, branch operations like `jmp` can't.

Furthermore, we could find that the latter version use `maxps`, which can obtain maximum value of float without branch.
So, if we put both line (`c[j] = b[j]` and `c[j] = a[j]`) in the `if` branch. We can leverage the `maxps`.
In the contract, it will use `mov` firstly and use `cmp`, `jmp` to handle branch operation, which has higher overhead.

```plain
...
.LBB0_3:                                #   Parent Loop BB0_2 |         je      .LBB0_13
                                        # =>  This Inner Loop | .LBB0_2:                                # =>This Loop Header:
        movaps  (%r15,%rcx,4), %xmm0                          |                                         #     Child Loop BB0_
        movaps  16(%r15,%rcx,4), %xmm1                        |         xorl    %ecx, %ecx
        maxps   (%rbx,%rcx,4), %xmm0                          |         jmp     .LBB0_3
        maxps   16(%rbx,%rcx,4), %xmm1                        |         .p2align      4, 0x90
        movups  %xmm0, (%r14,%rcx,4)                          | .LBB0_11:                               #   in Loop: Header=B
        movups  %xmm1, 16(%r14,%rcx,4)                        |         addq    $4, %rcx
        movaps  32(%r15,%rcx,4), %xmm0                        |         cmpq    $1024, %rcx                     # imm = 0x400
        movaps  48(%r15,%rcx,4), %xmm1                        |         je      .LBB0_12
        maxps   32(%rbx,%rcx,4), %xmm0                        | .LBB0_3:                                #   Parent Loop BB0_2
        maxps   48(%rbx,%rcx,4), %xmm1                        |                                         # =>  This Inner Loop
        movups  %xmm0, 32(%r14,%rcx,4)                        |         movaps  (%r15,%rcx,4), %xmm1
        movups  %xmm1, 48(%r14,%rcx,4)                        |         movups  %xmm1, (%rbx,%rcx,4)
        addq    $16, %rcx                                     |         movaps  (%r14,%rcx,4), %xmm0
        cmpq    $1024, %rcx                     # imm = 0x400 |         ucomiss %xmm1, %xmm0
        jne     .LBB0_3                                       |         ja      .LBB0_4
...
```
