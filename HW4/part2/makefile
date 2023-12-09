CC := mpicxx
exe := matmul

all:$(exe)

$(exe): main.cc matrix.cc
	$(CC) main.cc matrix.cc -o $(exe)
	~/copy.sh

.PHONY: clean
clean:
	rm $(exe) > /dev/null 2>&1