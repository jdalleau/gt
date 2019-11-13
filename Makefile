all: ./src/main.cpp
	g++ -Wall -fopenmp -std=c++17 ./src/main.cpp -o main -I./include -O3
