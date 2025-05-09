.PHONY: clean all check python

DIR := ${CURDIR}

SDIR = $(DIR)/src
LDIR = $(DIR)/lib
ODIR = $(DIR)/obj
IDIR = $(DIR)/include
TDIR = $(DIR)/tests
CDIR = $(DIR)/cython
PDIR = $(DIR)/ksw

HS_SDIR = $(DIR)/hyperspherical/src
HS_IDIR = $(DIR)/hyperspherical/include

LP_SDIR = $(DIR)/libpshtlight
LP_IDIR = $(DIR)/libpshtlight

NEWDIRS = $(LDIR) $(ODIR) $(TDIR)/obj $(TDIR)/bin
$(info $(shell mkdir -p -v $(NEWDIRS)))

CFLAGS = -Wall -fpic -std=c99
OMPFLAG = -fopenmp
OPTFLAG = -march=native -O2 -ffast-math

RF_OBJECTS = $(ODIR)/radial_functional.o \
             $(ODIR)/common.o \
             $(ODIR)/hyperspherical.o

LP_OBJECTS = $(ODIR)/ylmgen_c.o \
             $(ODIR)/c_utils.o \
	     $(ODIR)/walltime_c.o 

EST_OBJECTS = $(ODIR)/estimator.o

FIS_OBJECTS = $(ODIR)/fisher.o

TEST_OBJECTS = $(TDIR)/obj/seatest.o \
               $(TDIR)/obj/test_radial_functional.o \
               $(TDIR)/obj/test_estimator.o \
               $(TDIR)/obj/test_fisher.o \
               $(TDIR)/obj/run_tests.o

LINK_COMMON = -lm
LINK_FFTW = -L$(FFTWROOT) -lfftw3 -lfftw3f
LINK_MKL = -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
CFLAGS_MKL = -m64  -I"${MKLROOT}/include"

all: $(LDIR)/libradial_functional.so ${LDIR}/libksw_estimator.so ${LDIR}/libksw_fisher.so

python: $(LDIR)/libradial_functional.so setup.py $(CDIR)/radial_functional.pyx $(CDIR)/radial_functional.pxd
	python setup.py build_ext --inplace

$(LDIR)/libradial_functional.so: $(RF_OBJECTS)
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -shared -o $(LDIR)/libradial_functional.so $(RF_OBJECTS) $(LINK_COMMON) -lgomp

$(ODIR)/radial_functional.o: $(SDIR)/radial_functional.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(HS_IDIR)

$(ODIR)/common.o: $(HS_SDIR)/common.c $(HS_IDIR)/common.h $(HS_IDIR)/svnversion.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(HS_IDIR)

$(ODIR)/hyperspherical.o: $(HS_SDIR)/hyperspherical.c $(HS_IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(HS_IDIR)

$(LDIR)/libksw_estimator.so: $(EST_OBJECTS) $(LP_OBJECTS) $(IDIR)/ksw_estimator.h
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(OMPFLAG) $(OPTFLAG) -shared -o $@ $< $(LP_OBJECTS) $(LINK_COMMON) $(LINK_FFTW) $(LINK_MKL) -lgomp

$(LDIR)/libksw_fisher.so: $(FIS_OBJECTS) $(LP_OBJECTS) $(IDIR)/ksw_fisher.h
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(OMPFLAG) $(OPTFLAG) -shared -o $@ $< $(LP_OBJECTS) $(LINK_COMMON) $(LINK_FFTW) $(LINK_MKL) -lgomp

$(ODIR)/estimator.o: $(SDIR)/estimator.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) $(LINK_COMMON) $(LINK_FFTW) $(LINK_MKL) -lgomp

$(ODIR)/fisher.o: $(SDIR)/fisher.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(CFLAGS_MKL) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) $(LINK_COMMON) $(LINK_MKL)

$(ODIR)/ylmgen_c.o: $(LP_SDIR)/ylmgen_c.c $(LP_IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) $(LINK_COMMON) 

$(ODIR)/c_utils.o: $(LP_SDIR)/c_utils.c $(LP_IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) $(LINK_COMMON) 

$(ODIR)/walltime_c.o: $(LP_SDIR)/walltime_c.c $(LP_IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) $(LINK_COMMON) 

check: check_c check_python

check_c: $(TDIR)/bin/run_tests
	$(TDIR)/bin/run_tests

check_python:
	cd $(TDIR); python -m pytest python/

$(TDIR)/bin/run_tests: $(TEST_OBJECTS) $(LDIR)/libradial_functional.so $(LDIR)/libksw_estimator.so $(LDIR)/libksw_fisher.so
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -o $@ $(TEST_OBJECTS) -L$(LDIR) -lradial_functional -lksw_estimator -lksw_fisher $(LINK_COMMON) $(LINK_FFTW) $(LINK_MKL) -lgomp -Wl,-rpath,$(LDIR)

$(TDIR)/obj/run_tests.o: $(TDIR)/src/run_tests.c $(TDIR)/include/seatest.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(TDIR)/include -I$(IDIR) -I$(LP_IDIR)

$(TDIR)/obj/seatest.o: $(TDIR)/src/seatest.c $(TDIR)/include/seatest.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(TDIR)/include

$(TDIR)/obj/test_radial_functional.o: $(TDIR)/src/test_radial_functional.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(HS_IDIR) -I$(TDIR)/include

$(TDIR)/obj/test_estimator.o: $(TDIR)/src/test_estimator.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) -I$(TDIR)/include

$(TDIR)/obj/test_fisher.o: $(TDIR)/src/test_fisher.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(LP_IDIR) -I$(TDIR)/include

clean:
	rm -rf $(ODIR)
	rm -rf $(LDIR)
	rm -rf $(TDIR)/obj
	rm -rf $(TDIR)/bin
	rm -f $(CDIR)/*.c
	rm -f $(PDIR)/*.so
	rm -rf $(DIR)/build
	rm -rf $(PDIR)/__pycache__
	rm -rf $(TDIR)/python/__pycache__
	rm -rf $(DIR)/*.egg-info
	rm -f $(CDIR)/*.html
	rm -rf $(DIR)/cython_debug
