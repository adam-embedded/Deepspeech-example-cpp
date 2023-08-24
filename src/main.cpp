//
// Created by adam slaymark on 31/07/2023.
//
#include <iostream>
#include <deque>
#include <vector>
#include <mutex>
#include "cmath"
#include <thread>
#include <chrono>

#include "vad.h"
#include "deep.h"

#ifdef __APPLE__
#include "appleAudio.h"
#endif

#ifdef LINUX
#include "linuxAudio.h"
#endif


#include <sndfile.h>
static SF_INFO sfInfo;
static SNDFILE * outfile;
static SNDFILE * preFile;

#define RATIO 0.75f

static bool stop = false;

double sampleRate = 16000.0;
const int8_t bitDepth = 16;
const int blocksPerSecond = 50;
const int BlockSize = int(sampleRate / blocksPerSecond);
const int padding_ms = 300;
const int frame_duration_ms = 1000 * BlockSize/sampleRate;
const int num_padding_frames = floor(padding_ms/frame_duration_ms);



bool triggered = false;

static struct {
    std::deque<std::pair<std::vector<int16_t>,bool>> circ_buff;
    std::mutex lock;
}audio_adc;

static struct {
    std::vector<std::vector<int16_t>> frames;
    std::mutex lock;
    bool execute = false;
}deepstreamBuf;

//this is a threaded callback function for audio, this can be from linux and apple
// Callback is triggered when buffer is full
void callbackAbstraction(const int16_t * inputBuffer, uint32_t bufferSize){

//    if (bufferSize < (BlockSize*2)){
//        //fprintf(stderr,"Buffer Under run\n");
//        return;
//    }
    //input buffer is detached from thread within this abstraction. it will be re-attached within the driver.

    //convert char array to 16bit array
    int division = bitDepth/8;

    sf_write_short(preFile,inputBuffer, bufferSize/division);

    //if speech detected append vector
    int speech = is_speech(inputBuffer, bufferSize/division);


    //printf("||Is Speech: %d \n", speech);

    if (speech < 0) {
        //fprintf(stderr, "VAD processing failed\n");
        return;
    }

    //write array to vector
    std::vector<int16_t> tmpVec;
    tmpVec.insert(tmpVec.end(),inputBuffer, inputBuffer + (bufferSize/division));

    if (!triggered){
        //lock variable to thread to stop race conditions
        audio_adc.lock.lock();
        //Check if buffer has exceeded size, if it has, remove a sample before adding another.
        if (audio_adc.circ_buff.size() == num_padding_frames){
            audio_adc.circ_buff.pop_front();
        }
        //write to circular buffer
        //add to buffer and mark with speech value
        audio_adc.circ_buff.emplace_back(tmpVec, bool(speech));

        int num_voiced = 0;
        for (auto value : audio_adc.circ_buff){
            if (value.second) num_voiced++;
        }
        //if (num_voiced > 1) std::cout << "Num Voiced: " << num_voiced << std::endl;

        if (num_voiced > (RATIO * float(num_padding_frames))){
            triggered = true;
            //lock deepstream buffer
            deepstreamBuf.lock.lock();
            for (auto & i : audio_adc.circ_buff){
                deepstreamBuf.frames.push_back(i.first);
            }
            deepstreamBuf.lock.unlock();

            //clear the circular buffer
            audio_adc.circ_buff.clear();
        }
        audio_adc.lock.unlock();
    }
    else {
        //Place frames within deepstream buffer
        deepstreamBuf.lock.lock();
        deepstreamBuf.frames.emplace_back(tmpVec);


        audio_adc.lock.lock();
        audio_adc.circ_buff.emplace_back(tmpVec,speech);

        int num_not_voiced = 0;
        for (const auto& value : audio_adc.circ_buff){
            if (!value.second) num_not_voiced++;
        }
        if (num_not_voiced > (RATIO * float(num_padding_frames))) {
            triggered = false;
            audio_adc.circ_buff.clear();
            puts("Complete");
            deepstreamBuf.execute = true;
        }
        audio_adc.lock.unlock();
        deepstreamBuf.lock.unlock();
    }
}

static void handle_sig(int sig)
{
    std::cout << "Waiting for process to finish... got signal : " << sig << std::endl;
    stop = true;
}

static void handle_pipe(int sig)
{
    std::cout << "Received signal: " << sig << std::endl;
}

int main(){

    // Listen to ctrl+c and assert
    signal(SIGINT, handle_sig);
    signal(SIGABRT, handle_sig);
    signal(SIGKILL, handle_sig);
    signal(SIGTERM, handle_sig);
    signal(SIGPIPE, handle_pipe);

    const char* outputFilename = "VAD_recorded_audio.wav";
    const char* preFilename = "VAD_pre.wav";
    sfInfo.channels = 1;
    sfInfo.samplerate = sampleRate;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    outfile = sf_open(outputFilename, SFM_WRITE, &sfInfo);
    preFile = sf_open(preFilename,SFM_WRITE, &sfInfo);
    if (!outfile) {
        std::cerr << "Error opening the output WAV file.\n";
        throw std::runtime_error("");
    }

    Deepspeech deepspeech;
    //Initialise Audio
    initAudio(sampleRate, bitDepth, BlockSize, 0);
    //initialise VAD, pointer to be created to reference
    initVAD(sampleRate, 3);

    startAudioStream();

    bool test = false;

    //Main thread loop
    while(!stop){
        deepstreamBuf.lock.lock();
        if (deepstreamBuf.execute) {

            //compile all frames in to 1 vector
            std::vector<int16_t> tmpVec;
            for (auto frame: deepstreamBuf.frames){
                tmpVec.insert(tmpVec.end(), frame.begin(),frame.end());
            }

            deepstreamBuf.frames.clear();
            deepstreamBuf.execute = false;
            deepstreamBuf.lock.unlock();

            // convert vector in to array of fixed length
            int16_t deepBuf[tmpVec.size()];
            for (int i=0;i< (sizeof(deepBuf) / sizeof(deepBuf[0]));i++){
                deepBuf[i] = tmpVec[i];
            }

            uint32_t bufferSize = (sizeof(deepBuf) / sizeof(deepBuf[0]));
            std::string str;

            if (!test) sf_write_short(outfile,deepBuf,bufferSize);

            //TODO - we may need to set this up as streaming to allow higher accuracy...

            bool status = deepspeech.processBuffer(deepBuf, str, bufferSize);
            // if process successful, print result
            if (status) {
                std::cout << "Result" << str << std::endl;
            }
        }
        else {
            deepstreamBuf.lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(2500));
        }
    }
    stopAudioStream();
    deepspeech.close();
    sf_close(outfile);
    sf_close(preFile);

    return 0;
}