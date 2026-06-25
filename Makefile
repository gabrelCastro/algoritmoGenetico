# Makefile - AG Paralelo com OpenMP
CXX      ?= g++
CXXFLAGS ?= -O2 -fopenmp -Wall -std=c++17
TARGET    = ga_rastrigin
SRC       = src/ga_rastrigin.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Teste rapido de sanidade (deve convergir e imprimir tempo)
run: $(TARGET)
	./$(TARGET) --pop 500 --dim 100 --gen 100 --threads 4 --load 50 --verbose

clean:
	rm -f $(TARGET)

.PHONY: all run clean
