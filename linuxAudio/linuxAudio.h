//
// Created by adam slaymark on 31/07/2023.
//

#ifndef AWSLEX_LINUXAUDIO_H
#define AWSLEX_LINUXAUDIO_H

#include <iostream>

// Callback Function to be defined in user code, allows user abstraction between apple and linux audio drivers
// Passes const char buffer to callback
void callbackAbstraction(const int16_t * inputBuffer, uint32_t bufferSize, void *user);

void initAudio(uint32_t sampleRate,uint8_t bitDepth,int inBufSize, int outBufSize, const std::string& name="default");
void startAudioStream(); // only for apple
void stopAudioStream(); //only for apple

void PlayAudio(char * buffer, uint32_t bufferSize);
void RecordAudio();

#endif //AWSLEX_LINUXAUDIO_H
