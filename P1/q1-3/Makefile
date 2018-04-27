CC=g++
TARGET=main0
SRC=zeta0.cpp mach0.cpp main0.cpp

all:	$(SRC)
	$(CC) -c $(SRC)
	$(CC) -o $(TARGET) zeta0.o mach0.o main0.o
clean:
	rm -f *.o $(TARGET)

run: main0
	./main0

utest: all
	./main0 utest

vtest: all
	./main0 vtest
