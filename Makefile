CFLAGS=-Iinclude/ -Wall -Werror

BIN=bin/
INC=include/
SRC=src/
OBJ=object/
TST=tests/

objects=$(OBJ)neural_network.o $(OBJ)print_mv.o $(OBJ)funcs.o
r_objects=neural_network.o print_mv.o funcs.o

test_srcs=$(TST)complete_training_tests.c $(TST)gradient_average_tests.c $(TST)brief_training_tests.c $(TST)backpropegation_tests.c $(TST)feed_forward_tests.c $(TST)alloc_free_tests.c $(TST)rand_tests.c $(TST)hadamard_tests.c
ifeq ($(DEBUG),true)
	CFLAGS+=-g
endif

ifeq ($(MEM), true)
	CFLAGS+=-fsanitize=address
endif

tests: $(r_objects) $(test_srcs)
	make complete_training_tests
	make gradient_average_tests
	make brief_training_tests
	make backpropegation_tests
	make feed_forward_tests
	make alloc_free_tests
	make rand_tests
	make hadamard_tests
	make funcs_tests

mynum_ai : $(r_objects) $(TST)mynum_ai.c
	gcc $(CFLAGS) -o $(BIN)mynum_ai $(objects) $(TST)mynum_ai.c -lm -lgsl

mnist_ai : $(r_objects) mnist_reader.o $(TST)mnist_ai.c
	gcc $(CFLAGS) -o $(BIN)mnist_ai $(objects) $(OBJ)mnist_reader.o $(TST)mnist_ai.c -lm -lgsl

mnist_reader_tests : mnist_reader.o print_mv.o $(TST)mnist_reader_tests.c
	gcc $(CFLAGS) -o $(BIN)mnist_reader_tests $(OBJ)mnist_reader.o $(OBJ)print_mv.o $(TST)mnist_reader_tests.c -lm -lgsl

complete_training_tests : $(r_objects) $(TST)complete_training_tests.c
	gcc $(CFLAGS) -o $(BIN)complete_training_tests $(objects) $(TST)complete_training_tests.c -lm -lgsl

gradient_average_tests : $(r_objects) $(TST)gradient_average_tests.c
	gcc $(CFLAGS) -o $(BIN)gradient_average_tests $(objects) $(TST)gradient_average_tests.c -lm -lgsl

brief_training_tests : $(r_objects) $(TST)brief_training_tests.c
	gcc $(CFLAGS) -o $(BIN)brief_training_tests $(objects) $(TST)brief_training_tests.c -lm -lgsl

backpropegation_tests : $(r_objects) $(TST)backpropegation_tests.c
	gcc $(CFLAGS) -o $(BIN)backpropegation_tests $(objects) $(TST)backpropegation_tests.c -lm -lgsl

feed_forward_tests : $(r_objects) $(TST)feed_forward_tests.c
	gcc $(CFLAGS) -o $(BIN)feed_forward_tests $(objects) $(TST)feed_forward_tests.c -lm -lgsl

alloc_free_tests : $(r_objects) $(TST)alloc_free_tests.c
	gcc $(CFLAGS) -o $(BIN)alloc_free_tests $(objects) $(TST)alloc_free_tests.c -lm -lgsl

rand_tests : $(r_objects) $(TST)rand_tests.c
	gcc $(CFLAGS) -o $(BIN)rand_tests $(objects) $(TST)rand_tests.c -lm -lgsl

hadamard_tests : $(r_objects) $(TST)hadamard_tests.c 
	gcc $(CFLAGS) -o $(BIN)hadamard_tests $(objects) $(TST)hadamard_tests.c -lm -lgsl
	
mnist_reader.o : $(SRC)mnist_reader.c $(INC)mnist_reader.h
	gcc $(CFLAGS) -c -o $(OBJ)mnist_reader.o $(SRC)mnist_reader.c

neural_network.o : $(SRC)neural_network.c $(INC)neural_network.h
	gcc $(CFLAGS) -c -o $(OBJ)neural_network.o $(SRC)neural_network.c 

funcs_tests : funcs.o print_mv.o $(TST)funcs_tests.c 
	gcc $(CFLAGS) -o $(BIN)funcs_tests $(OBJ)print_mv.o $(OBJ)funcs.o $(TST)funcs_tests.c -lm -lgsl

funcs.o : $(SRC)funcs.c $(INC)funcs.h
	gcc $(CFLAGS) -c -o $(OBJ)funcs.o $(SRC)funcs.c

print_mv.o : $(SRC)print_mv.c $(INC)print_mv.h
	gcc $(CFLAGS) -c -o $(OBJ)print_mv.o $(SRC)print_mv.c

clean : 
	rm -f $(OBJ)*.o 
	rm -f $(BIN)*
