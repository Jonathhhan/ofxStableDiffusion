# All variables and this file are optional, if they are not present the PG and the
# makefiles will try to parse the correct values from the file system.
#
# Variables that specify exclusions can use % as a wildcard to specify that anything in
# that position will match. A partial path can also be specified to, for example, exclude
# a whole folder from the parsed paths from the file system
#
# Variables can be specified using = or +=
# = will clear the contents of that variable both specified from the file or the ones parsed
# from the file system
# += will add the values to the previous ones in the file or the ones parsed from the file 
# system
# 
# The PG can be used to detect errors in this file, just create a new project with this addon 
# and the PG will write to the console the kind of error and in which line it is

meta:
	ADDON_NAME = ofxStableDiffusion
	ADDON_DESCRIPTION = Stable Diffusion: https://github.com/leejet/stable-diffusion.cpp
	ADDON_AUTHOR = Jonathan Frank
	ADDON_TAGS = "Stable Diffusion" "Artificial Intelligence" "Image Generation"
	ADDON_URL = https://github.com/Jonathhhan/ofxStableDiffusion

common:
	ADDON_INCLUDES += libs/stable-diffusion/include
	# stable-diffusion.cpp is bundled as a separately built native library.
	ADDON_SOURCES_EXCLUDE += libs/stable-diffusion/source/%
	ADDON_SOURCES_EXCLUDE += libs/stable-diffusion/build/%
	ADDON_INCLUDES_EXCLUDE += libs/stable-diffusion/source/%
	ADDON_INCLUDES_EXCLUDE += libs/stable-diffusion/build/%

linux64:
	ADDON_LIBS += libs/stable-diffusion/lib/Linux64/libstable-diffusion.so
	ADDON_LDFLAGS += -Wl,-rpath=../../../../addons/ofxStableDiffusion/libs/stable-diffusion/lib/Linux64

linux:

linuxarmv6l:

linuxarmv7l:

msys2:

vs:
	ADDON_LIBS += libs/stable-diffusion/lib/vs/stable-diffusion.lib

android/armeabi:

android/armeabi-v7a:

osx:
	ADDON_LIBS += libs/stable-diffusion/lib/osx/libstable-diffusion.dylib
	ADDON_LDFLAGS += -Wl,-rpath,@loader_path/../../../../addons/ofxStableDiffusion/libs/stable-diffusion/lib/osx

ios:
	# iOS requires static linking
	ADDON_LIBS += libs/stable-diffusion/lib/ios/libstable-diffusion.a
	ADDON_CFLAGS += -DIOS_PLATFORM

emscripten:
