CC := g++
exe := pi.out
CFLAGS := -fopenmp -Wall

all:$(exe)

%.out: %.cpp
	$(CC) ${CFLAGS} $^ -o $@

.PHONY: clean time
clean:
	rm *.out
time:
	time ./$(exe)