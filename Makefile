#Target options
MAJOR_VER := 1
MINOR_VER := 0
TARGET:= damm
TARGET_SYMLINK := lib$(TARGET).so
TARGET_SONAME := lib$(TARGET).so.$(MAJOR_VER)
TARGET := lib$(TARGET).so.$(MAJOR_VER).$(MINOR_VER)
SRC = transpose.cc multiply.cc union.cc reduce.cc

PREFIX ?= /usr/lib/
INSTALLDIR ?= $(PREFIX)/
HEADERS = inc/common.h inc/damm.h inc/multiply.h inc/transpose.h inc/union.h

#Unit test options
TEST_TARGET ?= householder_test
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
OPT = -O0
CARRAY_DIR = /usr/include/carray
ASYNC_DIR = /usr/include/async
LDFLAGS = 
CXXFLAGS = -g $(CXX_STD) -fPIC $(OPT) -march=native -I$(INCDIR) -I $(CARRAY_DIR) -I $(ASYNC_DIR) -msse4.1 -mavx512f -funroll-loops
LDLIBS =
TEST_LDFLAGS = -pg -L ./ -Wl,-rpath=.
TEST_LDLIBS = -ldamm

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

profile_test:
	./$(TEST_TARGET)
	gprof ./$(TEST_TARGET) gmon.out -b > $(TEST_TARGET)_report.txt 

clean:
	$(RM) $(SRCDIR)/*.o $(TESTDIR)/*.o gmon.out *_report.txt

cleanall: clean
	$(RM) $(TARGET) $(TARGET_SYMLINK) $(TARGET_SONAME) $(TEST_TARGET) 

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
