srcdir = .


include ../../Mk/macports.autoconf.mk

ifneq ($(HAVE_GETDELIM),yes)
OBJS+= getdelim.o
endif
ifneq ($(HAVE_GETLINE),yes)
OBJS+= getline.o
endif
ifneq ($(HAVE_STRLCAT),yes)
OBJS+= strlcat.o
endif
ifneq ($(HAVE_STRLCPY),yes)
OBJS+= strlcpy.o
endif

# disable pkgIndex.tcl and install
PKGINDEX=
INSTALLTARGET=

all:: $(OBJS)

include $(srcdir)/../../Mk/macports.tea.mk

test::

distclean::
	rm -f Makefile
