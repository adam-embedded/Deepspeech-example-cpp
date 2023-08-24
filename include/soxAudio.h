//
// Created by adam slaymark on 05/08/2023.
//

#ifndef AWSLEX_SOXAUDIO_H
#define AWSLEX_SOXAUDIO_H

void initSox(int sampleRate, uint8_t bitDepth);
void writeAudioSox(char * buffer, uint32_t bufferSize);
void closeSox();

#endif //AWSLEX_SOXAUDIO_H
