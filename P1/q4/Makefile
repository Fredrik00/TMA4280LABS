CC=g++
TARGET=main1
SRC=zeta1.cpp mach1.cpp main1.cpp

all:	$(SRC)
	$(CC) -c $(SRC)
	$(CC) -o $(TARGET) zeta1.o mach1.o main1.o
clean:
	rm -f *.o $(TARGET)

run: main1
	./main1

utest: all
	./main1 utest

vtest: all
	./main1 vtest
