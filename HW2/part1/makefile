CC := gcc
exe := pi.out
CFLAGS := -Wall -pthread -Iinclude mtwister.c

all:$(exe)

%.out: %.c
	$(CC) ${CFLAGS} $^ -o $@

.PHONY: clean
clean:
	rm *.out
