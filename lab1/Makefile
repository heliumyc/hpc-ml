CFLAGS:=-O3  # -g = debug, -O2 for optimized code CPPFLAGS for g++
CC:=gcc
N1:=1000000
N2:=300000000
REP1:=1000
REP2:=20

.PHONY: all

all: dp1 dp2;

dp1:
	$(CC) $(CFLAGS) -o dp1 dp1.c;

dp2:
	$(CC) $(CFLAGS) -o dp2 dp2.c;


test:
	./dp1 $(N1) $(REP1)
	./dp1 $(N2) $(REP2)
	./dp2 $(N1) $(REP1)
	./dp2 $(N2) $(REP2)

clean:
	rm -f *.o;
	rm -f dp1 dp2;

