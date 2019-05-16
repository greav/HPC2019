CXX       = g++

DEFINES   = -DDISABLE_RANDOM_SEED

# флаги компиляции
CXXFLAGS  =
CXXFLAGS += -std=c++11 -O2
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(DEFINES)
CXXFLAGS += $(shell pkg-config --cflags blitz)

# флаги сборки (библиотеки)
LDFLAGS   = 
LDFLAGS  += -llapack -lblas -lgfortran
LDFLAGS  += $(shell pkg-config --libs blitz)

SOURCES   = main.cc
BINARY    = autoreg

VISUAL    = visual
VISUAL_SOURCES = visual.cc
VISUAL_LDFLAGS = $(shell pkg-config --libs freeglut) -lGL

$(BINARY): $(SOURCES) *.hh Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

$(VISUAL): Makefile *.hh $(VISUAL_SOURCES)
	$(CXX) $(CXXFLAGS) $(VISUAL_SOURCES) $(VISUAL_LDFLAGS) -o $(VISUAL)

run: ../tests autoreg.model
run: $(BINARY)
	(cd ../tests; $(PWD)/$(BINARY))

debug: ../tests autoreg.model
debug: $(BINARY)
	(cd ../tests; gdb $(PWD)/$(BINARY))

../tests:
	mkdir -p ../tests

../tests/autoreg.model: autoreg.model
	cp ../input/autoreg.model ../tests

clean:
	rm -f $(BINARY) $(VISUAL)
