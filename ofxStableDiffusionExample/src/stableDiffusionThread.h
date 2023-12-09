#pragma once

#include "ofMain.h"

class stableDiffusionThread : public ofThread {
public:
	void* userData;
private:
	void threadedFunction();
};
