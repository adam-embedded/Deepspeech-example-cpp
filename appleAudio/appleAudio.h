//
// Created by adam slaymark on 31/07/2023.
//

#ifdef __APPLE__

#ifndef AWSLEX_APPLEAUDIO_H
#define AWSLEX_APPLEAUDIO_H

#include <iostream>

// Callback Function to be defined in user code, allows user abstraction between apple and linux audio drivers
// Passes const char buffer or int16 to callback
void callbackAbstraction(const char* inputBuffer, uint32_t bufferSize);
void callbackAbstraction(const unsigned char* inputBuffer, uint32_t bufferSize);
void callbackAbstraction(const int16_t * inputBuffer, uint32_t bufferSize);
void callbackAbstraction(const int32_t * inputBuffer, uint32_t bufferSize);

//output in mono only
void outputCallbackAbstraction(char * inputBuffer, uint32_t bufferSize);

void initAudio(double sampleRate, uint8_t bitDepth, int inBufferSize, int outBufferSize);
void startAudioStream();
void stopAudioStream();

#endif //AWSLEX_APPLEAUDIO_H

#endif
