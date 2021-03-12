HDR   = ../interface/
OBJ   = ../obj/
LIB   = ../lib/
BIN   = ../bin/
PRG   = ../test/

HdrSuf  =  .h
SrcSuf  =  .cc
ObjSuf  =  .o
PrgSuf  =  .cpp
BinSuf  =  .exe
LibSuf  =  .so

HDRS  =  $(wildcard $(HDR)*$(HdrSuf))
SRCS  =  $(wildcard *$(SrcSuf))
_OBJS =  $(patsubst %$(SrcSuf), %$(ObjSuf), $(SRCS))
OBJS  =  $(addprefix $(OBJ),$(_OBJS))
PRGS  =  $(wildcard $(PRG)*$(PrgSuf))

_BINS    =  $(wildcard $(PRG)*$(PrgSuf))
__BINS   =  $(_BINS:$(PrgSuf)=$(BinSuf))
___BINS  =  $(notdir $(__BINS))
BINS	 =  $(addprefix $(BIN),${___BINS})

LINKDEF   =  $(wildcard ${HDR}*LinkDef.h)
DICTHDRS  =  $(patsubst $(LINKDEF),,$(HDRS)) $(LINKDEF)

ROOTCFLAGS    = $(shell root-config --cflags)
ROOTGLIBS     = $(shell root-config --glibs)

BASELIBS = -I$(shell echo $$CMSSW_RELEASE_BASE | sed -e 's/$$/\/src/')
CMSSWLIBS = -I$(shell echo $$CMSSW_BASE | sed -e 's/$$/\/src/')

ARCHL = -m64
CXX  =  g++
CXXFLAGS  =  -Wall -O -fPIC -I$(HDR) $(ROOTCFLAGS) $(BASELIBS) $(CMSSWLIBS)
CPP  =  g++
CPPFLAGS  = -Wall  $(ARCHL) -I$(HDR) $(ROOTCFLAGS) $(BASELIBS) $(CMSSWLIBS)


#################################################
#if mac 64
ARCH  =  $(shell root-config --arch)
ifeq ($(ARCH),macosx64)
LibSuf  =  .dylib

ARCHL = -m64

CPPFLAGS  =  -Wall -W -Woverloaded-virtual -O2 $(ARCHL) -pipe -I$(HDR) $(ROOTCFLAGS)

CXXFLAGS  = -fPIC  -Wall -W -Woverloaded-virtual -O2 $(ARCHL) -pipe -I$(HDR) $(ROOTCFLAGS) 

LDFLAGS  =  -dynamiclib -shared -03 -single_module -undefined dynamic_lookup $(ARCHL)
SONAME	 =  libPlotter.dylib
SOFLAGS  =
endif
#################################################

.PHONY: all exe test clean exef

all: $(LIB)$(SONAME)

exe: $(BINS) 


test:
	@echo "HDRS = $(HDRS)"
	@echo "DICTHDRS = $(DICTHDRS)"
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "PRGS = $(PRGS)"
	@echo "BINS = $(BIN)%$(BinSuf): $(PRG)%$(PrgSuf)"
	@echo "CMSSWPATH = $(CMSSWPATH)"


$(BIN)%$(BinSuf): $(PRG)%$(PrgSuf) $(HDRS) $(LIB)$(SONAME)
	$(CPP) $< $(CPPFLAGS) -L$(LIB) $(ROOTGLIBS) -o $@


clean:
	rm -f $(OBJ)*$(ObjSuf)  $(OBJ)mydict* $(BIN)*$(BinSuf) $(LIB)*$(LibSuf) 
