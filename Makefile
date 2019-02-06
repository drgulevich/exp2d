.PHONY : clean

CFLAGS  = -O3
#CFLAGS  = -g
#CFLAGS  = -g -Wall
#CFLAGS  = -ggdb -Wall
#  -g    adds debugging information to the executable file
#  -Wall turns on compiler warnings

### office version
INCLUDES = -I/usr/include/lapacke
LIBS = -lblas

### home version
#INCLUDES = -I/usr/local/include/lapacke
#LIBS = -llapacke -lblas -lgfortran

# -Wl,xx,yy: pass option as an options xx,yy to the linker
liblattice.so : lattice.o
	gcc $(CFLAGS) -shared -Wl,-soname,liblattice.so -o liblattice.so lattice.o -fopenmp -glomp -lfftw3 -lm $(LIBS)

# -fPIC: position-independent code
lattice.o : lattice.c
	gcc $(CFLAGS) -c -fPIC lattice.c -o lattice.o $(INCLUDES)

clean :
	-rm -vf liblattice.so lattice.o lattice.pyc

