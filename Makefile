#Target options
MAJOR_VER := 1
MINOR_VER := 0
TARGET:= damm
TARGET_SYMLINK := lib$(TARGET).so
TARGET_SONAME := lib$(TARGET).so.$(MAJOR_VER)
TARGET := lib$(TARGET).so.$(MAJOR_VER).$(MINOR_VER)
SRC = transpose.cc #multiply.cc #union.cc reduce.cc fused_union.cc broadcast.cc fused_reduce.cc

#Directories
SRCDIR = src
INCDIR = inc
TESTDIR = test
PREFIX ?= /usr
INSTALL_LIB ?= $(PREFIX)/lib/damm
INSTALL_INC ?= $(PREFIX)/include/damm
HEADERS = $(INCDIR)/*.h

#Toolchains
CXX = g++-13
CXX_STD = -std=c++23
CXX_SUFFIX = cc
LD = $(CXX)

#Compile Options
OPT = -O3
DEBUG_SYM = -g
CARRAY_DIR = /usr/include/carray
ORACLE_DIR = /usr/include/oracle
LDFLAGS = $(OMPFLAGS)
SIMDFLAGS = -msse4.1 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mfma
OMPFLAGS = -fopenmp -lgomp
CXXFLAGS = $(CXX_STD) $(DEBUG_SYM) -fPIC $(OPT) -march=native -funroll-loops \
	-I$(INCDIR) -I $(CARRAY_DIR) -I$(ORACLE_DIR) \
	$(SIMDFLAGS) $(OMPFLAGS) 
LDLIBS =
TEST_LDFLAGS = -L ./ -Wl,-rpath=. -fopenmp
TEST_LDLIBS = -ldamm
TEST_TARGETS = simd_test \
				broadcast_test \
				multiply_test \
				transpose_test \
				reduce_test \
				union_test \
				fused_reduce_test \
				fused_union_test \
				householder_test inverse_test decompose_test

PERF_TARGETS =  broadcast_perf \
				multiply_perf \
				transpose_perf \
				reduce_perf \
				union_perf \
				fused_reduce_perf \
				fused_union_perf \

TEST_SOURCES = $(TEST_TARGETS:%=%.$(CXX_SUFFIX))

#Shell type
SHELL := /bin/bash

#build rules
OBJ = $(SRC:%.$(CXX_SUFFIX)=$(SRCDIR)/%.o)
TEST_OBJ = $(TEST_SRC:%.$(CXX_SUFFIX)=$(TESTDIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJ) 
	$(LD) -o $@ -shared $(OBJ) $(LDFLAGS) $(LDLIBS) -Wl,-soname,$(TARGET_SONAME)
	$(shell ln -sf $(TARGET) $(TARGET_SONAME))
	$(shell ln -sf $(TARGET) $(TARGET_SYMLINK))

$(TEST_TARGETS): %: $(TESTDIR)/%.o $(TARGET)
	$(LD) -o $@ $(TESTDIR)/$@.o $(TEST_LDFLAGS) $(TEST_LDLIBS)

$(PERF_TARGETS): %: $(TESTDIR)/%.o $(TARGET)
	$(LD) -o $@ $(TESTDIR)/$@.o $(TEST_LDFLAGS) $(TEST_LDLIBS)

unit_test:$(TEST_TARGETS)
	for test in $(TEST_TARGETS); do \
		./$$test;  \
	done;

perf_test:$(PERF_TARGETS)
	for test in $(PERF_TARGETS); do \
		taskset -c 0 $$test;  \
	done;

clean:
	$(RM) $(SRCDIR)/*.o $(TESTDIR)/*.o gmon.out *_report.txt

cleanall: clean
	$(RM) $(TARGET) $(TARGET_SYMLINK) $(TARGET_SONAME) $(TEST_TARGETS) 

install: $(TARGET) $(HEADERS)
	install -d $(INSTALL_LIB)
	install -d $(INSTALL_INC)
	install -m 644 $(HEADERS) $(INSTALL_INC)
	install -m 755 $(TARGET_SYMLINK) $(INSTALL_LIB)
	install -m 755 $(TARGET_SONAME) $(INSTALL_LIB)
	install -m 755 $(TARGET) $(INSTALL_LIB)

uninstall:
	$(RM) -r $(INSTALL_LIB) $(INSTALL_INC)

.PHONY: all clean help $(TEST_TARGETS) $(TARGET) 

.DEFAULT_GOAL := $(TARGET)
