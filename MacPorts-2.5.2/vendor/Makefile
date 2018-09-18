TCL_SUBDIR=			tcl/unix
TCL_TARGETS=		install-binaries install-libraries

TCLTHREAD_SUBDIR=	thread
TCLTREHAD_TARGETS=	install-binaries install-libraries

TCLX_SUBDIR=		tclx
TCLX_TARGETS=		install-binaries install-libraries

TCLLIB_SUBDIR=		tcllib
TCLLIB_TARGETS= 	install-libraries

PREFIX=			/opt/local
DESTROOT=   		/Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/vendor-destroot

.PHONY: all clean distclean install destroot test
.PHONY: destroot-tcl destroot-tcllthread destroot-tclx destroot-tcllib
.PHONY: install-tcl install-tcllthread install-tclx install-tcllib
.PHONY: %-tcl %-tcllthread %-tclx %-tcllib

# for make all, run destroot (where destroot will depend on all of each
# subpackage)
all: destroot

%-tcl:
	@echo ===\> making $(@:%-tcl=%) in ${DIRPRFX}tcl/unix
	@umask 0022; $(MAKE) -C tcl/unix $(@:%-tcl=%)

%-tclthread:
	@echo ===\> making $(@:%-tclthread=%) in ${DIRPRFX}thread
	@umask 0022; $(MAKE) -C thread $(@:%-tclthread=%)

# thread links against libtclstub
all-tclthread: all-tcl

%-tclx:
	@echo ===\> making $(@:%-tclx=%) in ${DIRPRFX}tclx
	@umask 0022; $(MAKE) -C tclx $(@:%-tclx=%)

# tclx also links against libtclstub
all-tclx: all-tcl

%-tcllib:
	@echo ===\> making $(@:%-tcllib=%) in ${DIRPRFX}tcllib
	@umask 0022; $(MAKE) -C tcllib TCLSH_PROG=/Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/tclsh $(@:%-tcllib=%)

# tcllib requires a working tclsh
all-tcllib: all-tcl

destroot: destroot-tcl destroot-tclthread destroot-tclx destroot-tcllib
destroot-tcl: all-tcl
	@echo ===\> staging to destroot in ${DIRPRFX}tcl/unix
	@umask 0022; $(MAKE) -C tcl/unix DESTDIR=${DESTROOT} install-binaries install-libraries

destroot-tclthread: all-tclthread
	@echo ===\> staging to destroot in ${DIRPRFX}thread
	@umask 0022; $(MAKE) -C thread DESTDIR=${DESTROOT} install-binaries install-libraries

destroot-tclx: all-tclx
	@echo ===\> staging to destroot in ${DIRPRFX}tclx
	@umask 0022; $(MAKE) -C tclx DESTDIR=${DESTROOT} install-binaries install-libraries

destroot-tcllib: all-tcllib
	@echo ===\> staging to destroot in ${DIRPRFX}tcllib
	@umask 0022; $(MAKE) -C tcllib DESTDIR=${DESTROOT} TCLSH_PROG=/Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/tclsh install-libraries
	@chmod -R ugo+rX ${DESTROOT}${PREFIX}/libexec/macports/lib/tcllib*

install: install-tcl install-tclthread install-tclx install-tcllib

TCL_PACKAGE_PATH=/opt/local/libexec/macports/lib
OLD_TCL_FILES=  $(DESTDIR)$(TCL_PACKAGE_PATH)/tcl8/8.4/http-2.7.12.tm \
                $(DESTDIR)$(TCL_PACKAGE_PATH)/tcl8/8.4/platform-1.0.12.tm \
                $(DESTDIR)$(TCL_PACKAGE_PATH)/tcl8/8.4/platform-1.0.13.tm \
                $(DESTDIR)$(TCL_PACKAGE_PATH)/tcl8/8.5/tcltest-2.3.5.tm
install-tcl:
	rm -rf $(OLD_TCL_FILES)
	@echo ===\> making $(@:%-tcl=%) in ${DIRPRFX}tcl/unix
	@umask 0022; $(MAKE) -C tcl/unix install-binaries install-libraries

install-tclthread:
	rm -rf $(DESTDIR)$(TCL_PACKAGE_PATH)/thread2.7.{0,2}
	@echo ===\> making $(@:%-tclthrad=%) in ${DIRPRFX}thread
	@umask 0022; $(MAKE) -C thread install-binaries install-libraries

install-tclx:
	@echo ===\> making $(@:%-tclx=%) in ${DIRPRFX}tclx
	@umask 0022; $(MAKE) -C tclx install-binaries install-libraries

install-tcllib:
	rm -rf $(DESTDIR)$(TCL_PACKAGE_PATH)/tcllib1.1{5,7}
	@echo ===\> making $(@:%-tcllib=%) in ${DIRPRFX}tcllib
	@umask 0022; $(MAKE) -C tcllib install-libraries
	@chmod -R ugo+rX $(DESTDIR)${PREFIX}/libexec/macports/lib/tcllib*

test:

clean: clean-tcl clean-tcllib clean-tclx clean-tclthread
	rm -rf ${DESTROOT}

distclean: distclean-tcl distclean-tcllib distclean-tclx distclean-tclthread
	rm -f Makefile
	rm -rf vendor-destroot
	rm -f tclsh
