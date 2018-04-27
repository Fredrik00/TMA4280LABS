CC=g++
TARGET=main2
SRC=zeta2.cpp mach2.cpp main2.cpp

all:	$(SRC)
	$(CC) -c $(SRC) -fopenmp
	$(CC) -o $(TARGET) zeta2.o mach2.o main2.o -fopenmp
clean:
	rm -f *.o $(TARGET)

run: main2
	OMP_NUM_THREADS=4 ./main2

utest: all
	OMP_NUM_THREADS=4 ./main2 utest

vtest: all
	OMP_NUM_THREADS=4 ./main2 vtest
