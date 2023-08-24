//
// Created by adam slaymark on 30/07/2023.
//

#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#ifndef NO_DIR
#include <dirent.h>
#include <unistd.h>
#endif // NO_DIR
#include <vector>

#include <sox.h>
#include "deepspeech.h"
#include "deep.h"

//#define NO_SOX

//Temporary globals
//int stream_size = 0;
//int extended_stream_size = 0;

typedef struct
{
    const char *string;
    double cpu_time_overall;
} ds_result;

struct meta_word
{
    std::string word;
    float start_time;
    float duration;
};

char * CandidateTranscriptToString(const CandidateTranscript *transcript)
{
    std::string retval = "";
    for (int i = 0; i < transcript->num_tokens; i++)
    {
        const TokenMetadata &token = transcript->tokens[i];
        retval += token.text;
    }
    return strdup(retval.c_str());
}

std::vector<meta_word> CandidateTranscriptToWords(const CandidateTranscript *transcript)
{
    std::vector<meta_word> word_list;

    std::string word = "";
    float word_start_time = 0;

    // Loop through each token
    for (int i = 0; i < transcript->num_tokens; i++)
    {
        const TokenMetadata &token = transcript->tokens[i];

        // Append token to word if it's not a space
        if (strcmp(token.text, u8" ") != 0)
        {
            // Log the start time of the new word
            if (word.length() == 0)
            {
                word_start_time = token.start_time;
            }
            word.append(token.text);
        }

        // Word boundary is either a space or the last token in the array
        if (strcmp(token.text, u8" ") == 0 || i == transcript->num_tokens - 1)
        {
            float word_duration = token.start_time - word_start_time;

            if (word_duration < 0)
            {
                word_duration = 0;
            }

            meta_word w;
            w.word = word;
            w.start_time = word_start_time;
            w.duration = word_duration;

            word_list.push_back(w);

            // Reset
            word = "";
            word_start_time = 0;
        }
    }

    return word_list;
}

ds_result LocalDsSTT(ModelState *aCtx, const short *aBuffer, size_t aBufferSize)
{
    ds_result res = {0};

    clock_t ds_start_time = clock();


    res.string = DS_SpeechToText(aCtx, aBuffer, aBufferSize);

    clock_t ds_end_infer = clock();

    res.cpu_time_overall =
            ((double)(ds_end_infer - ds_start_time)) / CLOCKS_PER_SEC;

    return res;
}

typedef struct
{
    char *buffer;
    size_t buffer_size;
} ds_audio_buffer;

ds_audio_buffer GetAudioBuffer(const char *path, int desired_sample_rate)
{
    ds_audio_buffer res = {0};

#ifndef NO_SOX
    sox_format_t *input = sox_open_read(path, NULL, NULL, NULL);
    assert(input);

    // Resample/reformat the audio so we can pass it through the MFCC functions
    sox_signalinfo_t target_signal = {
            static_cast<sox_rate_t>(desired_sample_rate), // Rate
            1,                                            // Channels
            16,                                           // Precision
            SOX_UNSPEC,                                   // Length
            NULL                                          // Effects headroom multiplier
    };

    sox_signalinfo_t interm_signal;

    sox_encodinginfo_t target_encoding = {
            SOX_ENCODING_SIGN2, // Sample format
            16,                 // Bits per sample
            0.0,                // Compression factor
            sox_option_default, // Should bytes be reversed
            sox_option_default, // Should nibbles be reversed
            sox_option_default, // Should bits be reversed (pairs of bits?)
            sox_false           // Reverse endianness
    };

#if TARGET_OS_OSX
    // It would be preferable to use sox_open_memstream_write here, but OS-X
    // doesn't support POSIX 2008, which it requires. See Issue #461.
    // Instead, we write to a temporary file.
    char *output_name = tmpnam(NULL);
    assert(output_name);
    sox_format_t *output = sox_open_write(output_name, &target_signal,
                                          &target_encoding, "raw", NULL, NULL);
#else
    sox_format_t *output = sox_open_memstream_write(&res.buffer,
                                                  &res.buffer_size,
                                                  &target_signal,
                                                  &target_encoding,
                                                  "raw", NULL);
#endif

    assert(output);

    if ((int)input->signal.rate < desired_sample_rate)
    {
        fprintf(stderr, "Warning: original sample rate (%d) is lower than %dkHz. "
                        "Up-sampling might produce erratic speech recognition.\n",
                desired_sample_rate, (int)input->signal.rate);
    }

    // Setup the effects chain to decode/resample
    char *sox_args[10];
    sox_effects_chain_t *chain =
            sox_create_effects_chain(&input->encoding, &output->encoding);

    interm_signal = input->signal;

    sox_effect_t *e = sox_create_effect(sox_find_effect("input"));
    sox_args[0] = (char *)input;
    assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &input->signal) ==
           SOX_SUCCESS);
    free(e);

    e = sox_create_effect(sox_find_effect("rate"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
           SOX_SUCCESS);
    free(e);

    e = sox_create_effect(sox_find_effect("channels"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
           SOX_SUCCESS);
    free(e);

    e = sox_create_effect(sox_find_effect("output"));
    sox_args[0] = (char *)output;
    assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
           SOX_SUCCESS);
    free(e);

    // Finally run the effects chain
    sox_flow_effects(chain, NULL, NULL);
    sox_delete_effects_chain(chain);

    // Close sox handles
    sox_close(output);
    sox_close(input);
#endif // NO_SOX
#if TARGET_OS_OSX
    res.buffer_size = (size_t)(output->olength * 2);
    res.buffer = (char *)malloc(sizeof(char) * res.buffer_size);
    FILE *output_file = fopen(output_name, "rb");
    assert(fread(res.buffer, sizeof(char), res.buffer_size, output_file) == res.buffer_size);
    fclose(output_file);
    unlink(output_name);
#endif

    return res;
}

void ProcessFile(ModelState *context, const char *path, bool show_times)
{
    ds_audio_buffer audio = GetAudioBuffer(path, DS_GetModelSampleRate(context));

    // Pass audio to DeepSpeech
    // We take half of buffer_size because buffer is a char* while
    // LocalDsSTT() expected a short*
    ds_result result = LocalDsSTT(context,
                                  (const short *)audio.buffer,
                                  audio.buffer_size / 2);
    free(audio.buffer);

    if (result.string)
    {
        printf("%s\n", result.string);
        DS_FreeString((char *)result.string);
    }

    if (show_times)
    {
        printf("cpu_time_overall=%.05f\n",
               result.cpu_time_overall);
    }
}

std::vector<std::string> SplitStringOnDelim(std::string in_string, std::string delim)
{
    std::vector<std::string> out_vector;
    char *tmp_str = new char[in_string.size() + 1];
    std::copy(in_string.begin(), in_string.end(), tmp_str);
    tmp_str[in_string.size()] = '\0';
    const char *token = strtok(tmp_str, delim.c_str());
    while (token != NULL)
    {
        out_vector.push_back(token);
        token = strtok(NULL, delim.c_str());
    }
    delete[] tmp_str;
    return out_vector;
}

static ModelState *ctx;

Deepspeech::Deepspeech() {
    int status = DS_CreateModel(model.c_str(), &ctx);
    if (status != 0)
    {
        char *error = DS_ErrorCodeToErrorMessage(status);
        fprintf(stderr, "Could not create model: %s\n", error);
        free(error);
        //return 1;
        throw std::runtime_error("");
    }
    status = DS_EnableExternalScorer(ctx, scorer.c_str());
    if (status != 0)
    {
        fprintf(stderr, "Could not enable external scorer.\n");
        //return 1;
        throw std::runtime_error("");
    }
    //here set up hot words
    status = DS_AddHotWord(ctx, std::string("alexa").c_str(), 10.0f);
    if (status != 0)
    {
        fprintf(stderr, "Could not enable hot-word.\n");
        //return 1;
        std::exit(1);
    }

    //Get sample rate of model

    int rate = DS_GetModelSampleRate(ctx);
    std::cout << "Model Sample Rate: " << rate << std::endl;

}

bool Deepspeech::processBuffer(int16_t *buffer, std::string& str, uint32_t bufferSize) {

//    for (int i=0; i<20000;i++){
//        std::cout << buffer[i] << std::endl;
//    }
    const char* chstr;

    chstr = DS_SpeechToText(ctx, buffer, bufferSize);

    if (chstr) {
        printf("Internal Result: %s\n",chstr);
        str.assign(chstr);
        DS_FreeString((char *) chstr);
        return true;
    } else {
        return false;
    }
}

void Deepspeech::close(){
    DS_FreeModel(ctx);
}
