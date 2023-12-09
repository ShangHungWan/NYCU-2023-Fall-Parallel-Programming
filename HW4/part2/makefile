CC := mpicxx
exe := matmul

all:$(exe)

$(exe): main.cc matrix.cc
	$(CC) main.cc matrix.cc -o $(exe)
	~/copy.sh

.PHONY: clean test
clean:
	rm $(exe) > /dev/null 2>&1
test:
	mpirun -np 4 --hostfile /home/.grade/HW4/mat2_hosts ./matmul < /home/.grade/HW4/data-set/data2_10 > out && diff out /home/.grade/HW4/data-set/ans2_10