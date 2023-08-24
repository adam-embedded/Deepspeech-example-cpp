//
// Created by adam slaymark on 01/08/2023.
//

#ifndef AWSLEX_DEEPSPEECH_H
#define AWSLEX_DEEPSPEECH_H

#include "deepspeech.h"

class Deepspeech {
public:
    explicit Deepspeech();
    static bool processBuffer(int16_t* buffer, std::string& str, uint32_t BufferSize);
    static void close();

private:

    std::string model = "/Users/adam/CLionProjects/awsLex/assets/deepspeech-0.9.3-models.tflite";
    std::string scorer = "/Users/adam/CLionProjects/awsLex/assets/deepspeech-0.9.3-models.scorer";

    typedef struct
    {
        const char *string;
        double cpu_time_overall;
    } ds_result;


};

#endif //AWSLEX_DEEPSPEECH_H
