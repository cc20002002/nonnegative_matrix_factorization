srcdir = .


include ../../Mk/macports.autoconf.mk

SRCS = mpcommon.tcl signalcatch.tcl

INSTALLDIR= ${DESTDIR}${TCL_PACKAGE_PATH}/mpcommon1.0

all:: pkgIndex.tcl

pkgIndex.tcl: $(SRCS)
	$(SILENT) ../pkg_mkindex.sh .

clean::
	rm -f pkgIndex.tcl

distclean:: clean
	rm -f Makefile

install:: all
	$(INSTALL) -d -o "${DSTUSR}" -g "${DSTGRP}" -m "${DSTMODE}" "${INSTALLDIR}"
	$(SILENT) set -x; for file in ${SRCS}; do \
		$(INSTALL) -o "${DSTUSR}" -g "${DSTGRP}" -m 444 "$(srcdir)/$$file" "${INSTALLDIR}/$$file"; \
	done
	$(INSTALL) -o "${DSTUSR}" -g "${DSTGRP}" -m 444 pkgIndex.tcl "${INSTALLDIR}"

test:: ;
