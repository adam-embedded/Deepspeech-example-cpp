/***
  This file is part of PulseAudio.

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <vector>
#include <cstdio>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#include <pulse/pulseaudio.h>
#include <pulse/simple.h>
#include <pulse/error.h>

#include "linuxAudio.h"

#include <sndfile.h>
static SF_INFO sfInfo;
static SNDFILE * outfile;

//static int BUFSIZE = 0;

#define BUFSIZE 320

static pa_simple *s = nullptr;

static int error;

// PulseAudio context and stream variables
pa_mainloop* mainloop;
pa_context* context;
pa_stream* recordStream;
/* The Sample format to use */
static pa_sample_spec ss = {};


void closeStream(){
    if (s)
        pa_simple_free(s);
    std::exit(1);
}

void initAudio(uint32_t sampleRate,uint8_t bitDepth,int inBufSize, int outBufSize, const std::string& name){

    std::cout << "sample Rate: " << sampleRate << std::endl;

    const char* outputFilename = "linuxPlay.wav";
    sfInfo.channels = 1;
    sfInfo.samplerate = 16000;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    outfile = sf_open(outputFilename, SFM_WRITE, &sfInfo);
    if (!outfile) {
        std::cerr << "Error opening the output WAV file.\n";
        std::exit(1);
    }

    //BUFSIZE = outBufSize;

    ss = {
            .format = PA_SAMPLE_S16LE,
            .rate = sampleRate,
            .channels = 1
    };

    /* Create a new playback stream */
    if (!(s = pa_simple_new(nullptr, "./lexSpeech", PA_STREAM_PLAYBACK, nullptr, "playback", &ss, nullptr, nullptr, &error))) { //argv[0]
        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
        closeStream();
    }


}

void RecordAudio(){
    bool rete = false;
    static pa_simple *r = nullptr;
    // Create the recording stream
    if (!(r = pa_simple_new(NULL, "./lexSpeech1", PA_STREAM_RECORD, NULL, "record", &ss, NULL, NULL, &error))) {
        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
        closeStream();
    }

    while (!rete){
        int16_t buf[BUFSIZE];



        /* Record some data ... */
        if (pa_simple_read(r, buf, sizeof(buf), &error) < 0) {
            fprintf(stderr, __FILE__": pa_simple_read() failed: %s\n", pa_strerror(error));
            closeStream();
        }
        //sf_write_raw(outfile, buf, sizeof(buf));

        //remove 100 samples from beginning
//        for (int i=0; i< 100; i++){
//
//        }

        callbackAbstraction(buf, sizeof(buf), &rete);
    }
    pa_simple_free(r);

}


void PlayAudio(char * buffer, uint32_t bufferSize){



    if (pa_simple_write(s, buffer, bufferSize, &error) < 0) {
        fprintf(stderr, __FILE__": pa_simple_write() failed: %s\n", pa_strerror(error));
        closeStream();
        throw std::runtime_error("");
    }


    /* Make sure that every single sample was played */
    if (pa_simple_drain(s, &error) < 0) {
        fprintf(stderr, __FILE__": pa_simple_drain() failed: %s\n", pa_strerror(error));
        closeStream();
        throw std::runtime_error("");
    }
}



//
//int main(int argc, char*argv[]) {
//
//    /* The Sample format to use */
//    static const pa_sample_spec ss = {
//            .format = PA_SAMPLE_S16LE,
//            .rate = 16000,
//            .channels = 1
//    };
//
//    pa_simple *s = NULL;
//    int ret = 1;
//    int error;
//
//    /* replace STDIN with the specified file if needed */
//    if (argc > 1) {
//        int fd;
//
//        if ((fd = open(argv[1], O_RDONLY)) < 0) {
//            fprintf(stderr, __FILE__": open() failed: %s\n", strerror(errno));
//            goto finish;
//        }
//
//        if (dup2(fd, STDIN_FILENO) < 0) {
//            fprintf(stderr, __FILE__": dup2() failed: %s\n", strerror(errno));
//            goto finish;
//        }
//
//        close(fd);
//    }
//
//    /* Create a new playback stream */
//    if (!(s = pa_simple_new(NULL, argv[0], PA_STREAM_PLAYBACK, NULL, "playback", &ss, NULL, NULL, &error))) {
//        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
//        goto finish;
//    }
//
//    for (;;) {
//        uint8_t buf[BUFSIZE];
//        ssize_t r;
//
//#if 0
//        pa_usec_t latency;
//
//        if ((latency = pa_simple_get_latency(s, &error)) == (pa_usec_t) -1) {
//            fprintf(stderr, __FILE__": pa_simple_get_latency() failed: %s\n", pa_strerror(error));
//            goto finish;
//        }
//
//        fprintf(stderr, "%0.0f usec    \r", (float)latency);
//#endif
//
//        /* Read some data ... */
//        if ((r = read(STDIN_FILENO, buf, sizeof(buf))) <= 0) {
//            if (r == 0) /* EOF */
//                break;
//
//            fprintf(stderr, __FILE__": read() failed: %s\n", strerror(errno));
//            goto finish;
//        }
//
//        /* ... and play it */
//        if (pa_simple_write(s, buf, (size_t) r, &error) < 0) {
//            fprintf(stderr, __FILE__": pa_simple_write() failed: %s\n", pa_strerror(error));
//            goto finish;
//        }
//
//    }
//
//    /* Make sure that every single sample was played */
//    if (pa_simple_drain(s, &error) < 0) {
//        fprintf(stderr, __FILE__": pa_simple_drain() failed: %s\n", pa_strerror(error));
//        goto finish;
//    }
//
//    ret = 0;
//
//    finish:
//
//    if (s)
//        pa_simple_free(s);
//
//    return ret;
//}