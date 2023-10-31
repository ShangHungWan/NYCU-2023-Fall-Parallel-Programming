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

## Q2-2

## Q2-3

