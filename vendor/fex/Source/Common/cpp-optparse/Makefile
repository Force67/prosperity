ifeq ($(WARN),1)
ifeq ($(CXX),g++)
WARN_FLAGS = -O3 -g -Wall -Wextra -Werror # -Weffc++
else ifeq ($(CXX),clang++)
WARN_FLAGS = -O3 -g -Wall -Wextra -Werror
else ifeq ($(CXX),icpc)
WARN_FLAGS = -O3 -ipo -g -Wall -wd981 -wd383 -wd2259 -Werror # -Weffc++
endif
endif

ifeq ($(CPP11),1)
STD_FLAGS = -std=c++0x
endif

BIN = testprog
OBJECTS = OptionParser.o testprog.o

$(BIN): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(WARN_FLAGS) $(STD_FLAGS) $(LINKFLAGS)

%.o: %.cpp OptionParser.h
	$(CXX) $(WARN_FLAGS) $(STD_FLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean test

test: testprog
	./test.sh

clean:
	rm -f *.o $(BIN)
