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

linux64:
	ADDON_LDFLAGS += -Wl,-rpath=../../../../addons/ofxStableDiffusion/libs/stable-diffusion.cpp/lib/linux64

linux:

linuxarmv6l:

linuxarmv7l:

msys2:

vs:

android/armeabi:

android/armeabi-v7a:

osx:

ios:

emscripten:
