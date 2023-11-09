CC := gcc
exe := pi.out
CFLAGS := -Wall

all:$(exe)

%.out: %.c
	$(CC) ${CFLAGS} $^ -o $@ -Iinclude mtwister.c

.PHONY: clean
clean:
	rm *.out