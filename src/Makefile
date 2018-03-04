default: steps_glad

clean:
	rm -f *.o ../bin/steps_glad

# Set these variables to the point to the correct locations
GSL_LIB=$(HOME)/local/lib
GSL_INCLUDE=$(HOME)/local/include

steps_glad: main.cpp data.cpp data.h prob_functions.cpp prob_functions.h
	mkdir -p ../bin
	g++ -Wall -m64 -O3 -I${GSL_INCLUDE} -o ../bin/steps_glad main.cpp data.cpp prob_functions.cpp -L${GSL_LIB} -lgsl -lgslcblas -lm
