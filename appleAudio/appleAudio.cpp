#ifdef __APPLE__

#include <iostream>
#include <AudioToolbox/AudioToolbox.h>

#include "appleAudio.h"

#include <vector>
//#define DEBUG_STREAM
//#define WRITE_AUDIO

#ifdef WRITE_AUDIO
    #include <sndfile.h>
#endif

#define NUM_OUT_BUFFERS 3
constexpr UInt32 kNumChannels = 1;

// Struct to hold audio data and other necessary variables
struct AudioCaptureData {
    AudioQueueRef queue;
    AudioQueueBufferRef buffers[3]; // Use multiple buffers to continuously capture audio
    AudioStreamBasicDescription dataFormat;
    int8_t BitDepth;
#ifdef WRITE_AUDIO
    SF_INFO sfInfo;
    SNDFILE * outfile;
#endif
};

struct AudioOutputData {
    AudioQueueRef queue;
    AudioQueueBufferRef buffers[NUM_OUT_BUFFERS]; // Use multiple buffers to continuously capture audio
    AudioStreamBasicDescription dataFormat;
};

// Callback function called when a buffer is filled with audio data
static void AudioInputCallback(void* userData, AudioQueueRef queue, AudioQueueBufferRef buffer, const AudioTimeStamp* startTime, UInt32 numPackets, const AudioStreamPacketDescription* packetDesc) {
    AudioCaptureData* captureData = static_cast<AudioCaptureData*>(userData);

    if (numPackets > 0) {
#ifdef DEBUG_STREAM
        printf("Packets: %ul ",numPackets);
        printf("Buffer Size: %ul ", buffer->mAudioDataByteSize);
        puts("");
#endif
        //create stdlib compatible audio buffer
        //if (captureData->BitDepth == 16) {
            const auto *inputBuffer = reinterpret_cast<const int16_t *>(buffer->mAudioData);
            callbackAbstraction(inputBuffer, buffer->mAudioDataByteSize);
        //}

#ifdef WRITE_AUDIO
        // Write audio data to the WAV file
        sf_write_raw(captureData->outfile, inputBuffer, buffer->mAudioDataByteSize);
#endif
    }


    // Re-enqueue the buffer for continuous recording
    AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
}

//#define AMPLITUDE 32767
//// Function to generate a 16-bit PCM square wave
//void generateSquareWave(double frequency, uint32_t numSamples, double sampleRate, int16_t * outBuf) {
//    const double amplitude = 0.5;
//    double halfPeriod = 0.5 / frequency;
//
//    for (int i=0; i < numSamples;i++){
//        double time = (double)i / sampleRate;
//        double value = (fmod(time, halfPeriod * 2.0) < halfPeriod) ? amplitude * INT16_MAX : -amplitude * INT16_MAX;
//        outBuf[i] = (int16_t)value;
//    }
//
//}
//
//static void AudioOutputCallback(void *inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer){
//
//    auto *outputBuffer = reinterpret_cast<int16_t *>(inBuffer->mAudioData);
//    size_t numFrames = inBuffer->mAudioDataBytesCapacity / (sizeof(int16_t));
//
//    //auto * tmpBuffer = new int16_t [numFrames/kNumChannels];
//    int16_t *tmpBuffer = (int16_t*) std::malloc(numFrames);
//
//    generateSquareWave(1000.0, numFrames/kNumChannels, 16000, tmpBuffer);
//
//    //outputCallbackAbstraction(tmpBuffer, numFrames/kNumChannels); // send buffer to abstraction
//
//    for (uint32_t frame = 0; frame < numFrames/kNumChannels; ++frame){
//        for (uint8_t channel=0; channel < kNumChannels; ++channel){
//            // Copy audio samples, move mono audio to stereo
//            outputBuffer[frame * kNumChannels + channel] = tmpBuffer[frame];
//        }
//    }
//
//
//
//    //reload buffer back to queue
//    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
//
//    free(tmpBuffer);
//}

static AudioCaptureData captureData;
static AudioOutputData outputData;

#ifndef SAMPLE_APPLICATION
void initAudio(const double sampleRate, const uint8_t bitDepth, int bufferByteSize, int outBufferSize) {

    captureData.BitDepth = bitDepth;
    bufferByteSize = bufferByteSize * (bitDepth / 8);
    if (bitDepth == 16 || bitDepth == 32){
        // Set up the audio format for recording (16-bit signed int, mono)
        //AudioCaptureData captureData;
        captureData.dataFormat.mSampleRate = sampleRate;
        captureData.dataFormat.mFormatID = kAudioFormatLinearPCM;
        captureData.dataFormat.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
        captureData.dataFormat.mFramesPerPacket = 1;
        captureData.dataFormat.mChannelsPerFrame = 1;
        captureData.dataFormat.mBitsPerChannel = bitDepth;
        captureData.dataFormat.mBytesPerPacket = (bitDepth / 8);
        captureData.dataFormat.mBytesPerFrame = (bitDepth / 8);

    } else {
        throw std::runtime_error("Application only compatible with 16 bit or 32 bit mono currently");
    }

#ifdef WRITE_AUDIO
    const char* outputFilename = "recorded_audio.wav";
    captureData.sfInfo.channels = captureData.dataFormat.mChannelsPerFrame;
    captureData.sfInfo.samplerate = captureData.dataFormat.mSampleRate;
    captureData.sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
#endif
    // Create an audio queue for capturing audio
    AudioQueueNewInput(&captureData.dataFormat, AudioInputCallback, &captureData, nullptr, nullptr, 0, &captureData.queue);

    // Create audio buffers
    //const int bufferByteSize = 1024; // Adjust buffer size as needed
    for (int i = 0; i < 3; i++) {
        AudioQueueAllocateBuffer(captureData.queue, bufferByteSize, &captureData.buffers[i]);
        AudioQueueEnqueueBuffer(captureData.queue, captureData.buffers[i], 0, nullptr);
    }

#ifdef WRITE_AUDIO
    // Open the WAV file for writing
    captureData.outfile = sf_open(outputFilename, SFM_WRITE, &captureData.sfInfo);
    if (!captureData.outfile) {
        std::cerr << "Error opening the output WAV file.\n";
        throw std::runtime_error("");
    }
#endif

//    //Setup Audio Output
//    outputData.dataFormat.mSampleRate = sampleRate;
//    outputData.dataFormat.mFormatID = kAudioFormatLinearPCM;
//    outputData.dataFormat.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
//    outputData.dataFormat.mFramesPerPacket = 1;
//    outputData.dataFormat.mChannelsPerFrame = kNumChannels;
//    outputData.dataFormat.mBitsPerChannel = bitDepth;
//    outputData.dataFormat.mBytesPerPacket = (bitDepth / 8) * kNumChannels;
//    outputData.dataFormat.mBytesPerFrame = (bitDepth / 8) * kNumChannels;
//
//    //create audio queue for outputting audio
//    AudioQueueNewOutput(&outputData.dataFormat, AudioOutputCallback, &outputData, nullptr, kCFRunLoopCommonModes, 0, &outputData.queue);
//
//    //Allocate buffers and enqueue audio buffers
//    for (auto & buffer : outputData.buffers){
//        AudioQueueAllocateBuffer(outputData.queue, outBufferSize, &buffer);
//        //AudioQueueEnqueueBuffer(outputData.queue, buffer, 0, nullptr);
//        //We have to call the callback initially for it to get stuck in a loop, likely if there is no buffer on the callback, hold the thread until a buffer is available
//        AudioOutputCallback(nullptr, outputData.queue, buffer); // Fill buffer up initially with values
//
//    }

}
#endif

void startAudioStream(){
    // Start recording
    AudioQueueStart(captureData.queue, nullptr);

    //Start playback
    //AudioQueueStart(outputData.queue, nullptr);
}

void stopAudioStream(){
    // Stop recording
    AudioQueueStop(captureData.queue, true);

    // Clean up
    AudioQueueDispose(captureData.queue, true);
    #ifdef WRITE_AUDIO
        sf_close(captureData.outfile);
    #endif

    //Stop Playback
    //AudioQueueStop(outputData.queue, true);

    //cleanup dispose and release resources
//    for (auto & buffer : outputData.buffers){
//        AudioQueueFreeBuffer(outputData.queue, buffer);
//    }
    //AudioQueueDispose(outputData.queue, true);
}

#ifdef SAMPLE_APPLICATION
int main() {

    const char* outputFilename = "recorded_audio.wav";
    // Set up the audio format for recording (44100 Hz, 16-bit signed int, mono)
    //AudioCaptureData captureData;
    captureData.dataFormat.mSampleRate = 16000.0;
    captureData.dataFormat.mFormatID = kAudioFormatLinearPCM;
    captureData.dataFormat.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    captureData.dataFormat.mFramesPerPacket = 1;
    captureData.dataFormat.mChannelsPerFrame = 1;
    captureData.dataFormat.mBitsPerChannel = 16;
    captureData.dataFormat.mBytesPerPacket = 2;
    captureData.dataFormat.mBytesPerFrame = 2;

#ifdef WRITE_AUDIO
    captureData.sfInfo.channels = captureData.dataFormat.mChannelsPerFrame;
    captureData.sfInfo.samplerate = captureData.dataFormat.mSampleRate;
    captureData.sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
#endif
    // Create an audio queue for capturing audio
    AudioQueueNewInput(&captureData.dataFormat, AudioInputCallback, &captureData, nullptr, nullptr, 0, &captureData.queue);

    // Create audio buffers
    const int bufferByteSize = 1024; // Adjust buffer size as needed
    for (int i = 0; i < 3; i++) {
        AudioQueueAllocateBuffer(captureData.queue, bufferByteSize, &captureData.buffers[i]);
        AudioQueueEnqueueBuffer(captureData.queue, captureData.buffers[i], 0, nullptr);
    }

#ifdef WRITE_AUDIO
    // Open the WAV file for writing
    captureData.outfile = sf_open(outputFilename, SFM_WRITE, &captureData.sfInfo);
    if (!captureData.outfile) {
        std::cerr << "Error opening the output WAV file.\n";
        return 1;
    }
#endif

    // Start recording
    AudioQueueStart(captureData.queue, nullptr);

    // Keep the program running for a while to capture audio
    sleep(10); // Recording for 10 seconds

    // Stop recording
    AudioQueueStop(captureData.queue, true);

    // Clean up
    AudioQueueDispose(captureData.queue, true);

#ifdef WRITE_AUDIO
    sf_close(captureData.outfile);
#endif

    return 0;
}
#endif

#endif