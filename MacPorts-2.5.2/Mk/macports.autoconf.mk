# vim:ts=8:noet:sts=8:sw=8

SHELL			= /bin/sh

CC			= clang
CFLAGS			= -g -O2 -std=c99 $(CFLAGS_QUICHEEATERS) $(CFLAGS_PEDANTIC) $(CFLAGS_WERROR)
CPPFLAGS		=  -DHAVE_CONFIG_H -I/Users/xiaodangan/code/nmf/MacPorts-2.5.2/src -I/Users/xiaodangan/code/nmf/MacPorts-2.5.2/src -I. -I/Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/vendor-destroot//opt/local/libexec/macports/include
CFLAGS_QUICHEEATERS	= -Wextra -Wall
CFLAGS_PEDANTIC		= -pedantic
CFLAGS_WERROR		= 

UNIVERSAL_ARCHFLAGS	=  -arch x86_64 -arch i386

READLINE_CFLAGS		=
MD5_CFLAGS		=
SQLITE3_CFLAGS		= 
CURL_CFLAGS		= -I/anaconda3/include

LDFLAGS			= 
SHLIB_LD		= ${CC} -dynamiclib ${CFLAGS} ${LDFLAGS} -Wl,-single_module
STLIB_LD		= ${AR} cr
SHLIB_CFLAGS		= -fno-common
SHLIB_LDFLAGS		= 
SHLIB_SUFFIX		= .dylib
TCL_STUB_LIB_SPEC	= -L/Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/tcl/unix -ltclstub8.5

LIBS			= 
READLINE_LIBS		= 
MD5_LIBS		= 
SQLITE3_LIBS		= -lsqlite3
CURL_LIBS		= -lcurl
INSTALL			= /usr/bin/install -c
MTREE			= /usr/sbin/mtree
LN_S			= ln -s
XCODEBUILD		= /usr/bin/xcodebuild

INTREE_TCLSH		= /Users/xiaodangan/code/nmf/MacPorts-2.5.2/vendor/tclsh
TCLSH			= /opt/local/libexec/macports/bin/tclsh8.5
TCL_PACKAGE_PATH	= /opt/local/libexec/macports/lib

DSCL			= /usr/bin/dscl
DSEDITGROUP		= /usr/sbin/dseditgroup
DSTUSR			= root
DSTGRP			= admin
DSTMODE			= 0755
RUNUSR			= macports

prefix			= /opt/local
sysconfdir		= ${prefix}/etc
exec_prefix		= ${prefix}
bindir			= ${exec_prefix}/bin
datarootdir		= ${prefix}/share
datadir			= ${datarootdir}
libdir			= ${exec_prefix}/lib
localstatedir		= ${prefix}/var
infodir			= ${datarootdir}/info

mpconfigdir		= ${sysconfdir}/macports

MACPORTS_VERSION	= 2.5.2

SILENT			= @

HAVE_GETDELIM		= yes
HAVE_GETLINE		= yes
HAVE_STRLCAT		= yes
HAVE_STRLCPY		= yes
