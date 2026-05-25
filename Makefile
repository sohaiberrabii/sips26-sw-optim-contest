SHELL := /bin/sh

DEBUG ?= 0
OPENCV ?= 1
OPENMP ?= 1
OPENCL ?= 0
JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE ?= cmake

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH_FLAGS ?= -mcpu=native
  else
    ARCH_FLAGS ?= -march=native
  endif
else
  ARCH_FLAGS ?= -march=native -mtune=native
endif

ifeq ($(DEBUG),1)
  BUILD_DIR ?= build-debug
  BUILD_TYPE := Debug
  BASE_FLAGS := -Wall -g3 -O0 -fno-omit-frame-pointer
  OPT_FLAGS :=
  LTO ?= 0
  DEBUG_DEFS := -DMOTION_ENABLE_DEBUG
else
  BUILD_DIR ?= build
  BUILD_TYPE := Release
  BASE_FLAGS := -Wall -funroll-loops -fstrict-aliasing $(ARCH_FLAGS)
  OPT_FLAGS := -O3 -DNDEBUG
  LTO ?= 1
  DEBUG_DEFS :=
endif

ifeq ($(OPENCV),1)
  OPENCV_ARG := -DMOTION_OPENCV_LINK=ON
else
  OPENCV_ARG := -DMOTION_OPENCV_LINK=OFF
endif

ifeq ($(OPENMP),1)
  OPENMP_ARG := -DMOTION_OPENMP_LINK=ON
else
  OPENMP_ARG := -DMOTION_OPENMP_LINK=OFF
endif

ifeq ($(OPENCL),1)
  OPENCL_ARG := -DMOTION_OPENCL_LINK=ON
else
  OPENCL_ARG := -DMOTION_OPENCL_LINK=OFF
endif

BREW_OPENCV := $(shell brew --prefix opencv 2>/dev/null)
ifeq ($(OPENCV),1)
  ifneq ($(BREW_OPENCV),)
    OPENCV_HINT := -DOpenCV_DIR=$(BREW_OPENCV)/lib/cmake/opencv4
  endif
endif

BREW_LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
ifeq ($(OPENMP),1)
  ifeq ($(UNAME_S),Darwin)
    ifneq ($(BREW_LIBOMP),)
      OPENMP_HINTS := -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include" \
                      -DOpenMP_C_LIB_NAMES=omp \
                      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include" \
                      -DOpenMP_CXX_LIB_NAMES=omp \
                      -DOpenMP_omp_LIBRARY=$(BREW_LIBOMP)/lib/libomp.dylib
    endif
  endif
endif

CMAKE_ARGS := -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
              -DCMAKE_POLICY_DEFAULT_CMP0069=NEW \
              -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
              $(OPENMP_ARG) \
              $(OPENCV_ARG) \
              $(OPENCL_ARG) \
              $(OPENCV_HINT) \
              $(OPENMP_HINTS) \
              -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=$(if $(filter 1,$(LTO)),ON,OFF)

.PHONY: all configure clean distclean help

all: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_ARGS) \
		-DCMAKE_C_FLAGS="$(BASE_FLAGS) $(DEBUG_DEFS)" \
		-DCMAKE_CXX_FLAGS="$(BASE_FLAGS) $(DEBUG_DEFS)" \
		-DCMAKE_C_FLAGS_RELEASE="$(OPT_FLAGS)" \
		-DCMAKE_CXX_FLAGS_RELEASE="$(OPT_FLAGS)"

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean

distclean:
	rm -rf build build-debug build-release

help:
	@echo "Usage:"
	@echo "  make              Optimized benchmark build -> $(BUILD_DIR)/bin/motion"
	@echo "  DEBUG=1 make      Debug build -> build-debug/bin/motion"
	@echo "  OPENCV=0 make     Disable OpenCV text/ID drawing support"
	@echo "  OPENMP=0 make     Disable OpenMP linking"
	@echo "  LTO=0 make        Disable link-time optimization"
	@echo "  make clean        Clean current BUILD_DIR ($(BUILD_DIR))"
	@echo "  make distclean    Remove build directories"
