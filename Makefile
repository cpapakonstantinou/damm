#Target options
MAJOR_VER := 1
MINOR_VER := 0
TARGET:= damm
TARGET_SYMLINK := lib$(TARGET).so
TARGET_SONAME := lib$(TARGET).so.$(MAJOR_VER)
TARGET := lib$(TARGET).so.$(MAJOR_VER).$(MINOR_VER)
SRC = transpose.cc multiply.cc union.cc reduce.cc fused_union.cc broadcast.cc fused_reduce.cc

PREFIX ?= /usr/lib
INSTALLDIR ?= $(PREFIX)/damm
HEADERS = inc/common.h inc/damm.h inc/multiply.h inc/transpose.h inc/union.h inc/reduce.h inc/fused_reduce.h inc/fused_union.h inc/broadcast.h inc/decompose.h inc/solve.h inc/inverse.h inc/macros.h inc/householder.h

#Unit test options
TEST_TARGET ?= multiply_test
TEST_SRC = $(TEST_TARGET).cc

#Directories
SRCDIR = src
INCDIR = inc
TESTDIR = test

#Toolchains
CXX = g++-13
CXX_STD = -std=c++23
CXX_SUFFIX = cc
LD = $(CXX)

#Compile Options
OPT = -O3
CARRAY_DIR = /usr/include/carray
ASYNC_DIR = /usr/include/async
ORACLE_DIR = /usr/include/oracle
LDFLAGS = 
SIMDFLAGS = -msse4.1 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mfma
CXXFLAGS = -g $(CXX_STD) -fPIC $(OPT) -march=native -I$(INCDIR) -I $(CARRAY_DIR) -I $(ASYNC_DIR) -I$(ORACLE_DIR) $(SIMDFLAGS) -funroll-loops
LDLIBS =
TEST_LDFLAGS = -pg -L ./ -Wl,-rpath=.
TEST_LDLIBS = -ldamm
TEST_TARGETS = multiply_test transpose_test reduce_test union_test fused_reduce_test fused_union_test householder_test inverse_test decompose_test broadcast_test
TEST_SOURCES = $(TEST_TARGETS:%=%.cc)

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

$(TEST_TARGET): $(TEST_OBJ)
	$(LD) -o $@ $(TEST_OBJ) $(TEST_LDFLAGS) $(TEST_LDLIBS)

$(TEST_TARGETS): %: $(TESTDIR)/%.o $(TARGET)
	$(LD) -o $@ $(TESTDIR)/$@.o $(TEST_LDFLAGS) $(TEST_LDLIBS)

unit_test:$(TEST_TARGETS)
	for test in $(TEST_TARGETS); do \
		./$$test; \
	done;

clean:
	$(RM) $(SRCDIR)/*.o $(TESTDIR)/*.o gmon.out *_report.txt

cleanall: clean
	$(RM) $(TARGET) $(TARGET_SYMLINK) $(TARGET_SONAME) $(TEST_TARGETS) 

install: $(TARGET) $(HEADERS)
	install -d $(INSTALLDIR)
	install -m 644 $(HEADERS) $(INSTALLDIR)
	install -m 755 $(TARGET_SYMLINK) $(INSTALLDIR)
	install -m 755 $(TARGET_SONAME) $(INSTALLDIR)
	install -m 755 $(TARGET) $(INSTALLDIR)

uninstall:
	$(RM) -r $(INSTALLDIR)

.PHONY: all clean help $(TEST_TARGET) $(TARGET) 

.DEFAULT_GOAL := $(TARGET)
