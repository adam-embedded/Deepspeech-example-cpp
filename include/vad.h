//
// Created by adam slaymark on 31/07/2023.
//

#ifndef AWSLEX_VAD_H
#define AWSLEX_VAD_H
#include <iostream>
//#include <fvad.h>
#include <sndfile.h>

void initVAD(double sampleRate, int vad_aggressiveness);
int is_speech(const int16_t *Buffer, size_t length);

#endif //AWSLEX_VAD_H
