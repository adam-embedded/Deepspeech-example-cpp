//
// Created by adam slaymark on 31/07/2023.
//

/*
 * Copyright (c) 2016 Daniel Pirch
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#define _POSIX_C_SOURCE 200809L

#include <fvad.h>
#include <cstdlib>
#include <unistd.h>
#include <sndfile.h>

#include "vad.h"

static Fvad *internal_vad;
int is_speech(const int16_t *Buffer, size_t length){
    return  fvad_process(internal_vad, Buffer, length);
}


void initVAD(double sampleRate, int vad_aggressiveness){
    internal_vad = fvad_new();

    //internal_vad = vad;
    if (!internal_vad) {
        throw std::runtime_error("VAD New Failed to allocate Memory");
    }
    //set Sample Rate
    if (fvad_set_sample_rate(internal_vad, int(sampleRate)) < 0) {
        throw std::runtime_error("invalid sample rate");
    }
    fvad_set_mode(internal_vad, vad_aggressiveness);

}



int runVAD(int argc, char *argv[])
{
    int retval;
    const char *in_fname, *out_fname[2] = {NULL, NULL}, *list_fname = NULL;
    SNDFILE *in_sf = NULL, *out_sf[2] = {NULL, NULL};
    SF_INFO in_info = {0}, out_info[2];
    FILE *list_file = NULL;
    int mode, frame_ms = 10;
    Fvad *vad = NULL;

    /*
     * create fvad instance
     */
    vad = fvad_new();
    if (!vad) {
        fprintf(stderr, "Failed to allocate Memory\n");
        goto fail;
    }

    //set Sample Rate
    if (fvad_set_sample_rate(vad, in_info.samplerate) < 0) {
        fprintf(stderr, "invalid sample rate: %d Hz\n", in_info.samplerate);
        goto fail;
    }
    fvad_set_mode(vad, 3);


    /*
     * run main loop
     */
//    if (!process_sf(in_sf, vad,
//                    (size_t)in_info.samplerate / 1000 * frame_ms, out_sf, list_file))
//        goto fail;

    /*
     * cleanup
     */
    success:
    retval = EXIT_SUCCESS;
    goto end;

    argfail:
    fprintf(stderr, "Try '%s -h' for more information.\n", argv[0]);
    fail:
    retval = EXIT_FAILURE;
    goto end;

    end:
    if (in_sf) sf_close(in_sf);
    for (int i = 0; i < 2; i++)
        if (out_sf[i]) sf_close(out_sf[i]);
    if (list_file) fclose(list_file);
    if (vad) fvad_free(vad);

    return retval;
}