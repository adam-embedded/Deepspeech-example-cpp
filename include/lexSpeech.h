//
// Created by adam slaymark on 08/08/2023.
//

#ifndef AWSLEX_LEXSPEECH_H
#define AWSLEX_LEXSPEECH_H

#include <iostream>
#include <sstream>
#include <string>
#include <zlib.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

// Function to decode base64 data
std::string Base64Decode(const std::string &input) {
    BIO *bio, *b64;
    BUF_MEM *bptr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new_mem_buf(input.c_str(), input.length());
    bio = BIO_push(b64, bio);
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_get_mem_ptr(bio, &bptr);

    std::string result(bptr->data, bptr->length);

    BIO_free_all(bio);
    return result;
}

// Function to decompress Gzip data
std::string GzipDecompress(const std::string &input) {
    z_stream zs;
    memset(&zs, 0, sizeof(zs));

    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) {
        return "";
    }

    zs.avail_in = input.size();
    zs.next_in = (Bytef *)(input.c_str());

    int ret;
    char buffer[32768];
    std::string output;

    do {
        zs.avail_out = sizeof(buffer);
        zs.next_out = reinterpret_cast<Bytef *>(buffer);
        ret = inflate(&zs, Z_NO_FLUSH);

        if (output.size() < zs.total_out) {
            output.append(buffer, zs.total_out - output.size());
        }

    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END) {
        return "";
    }

    return output;
}

#endif //AWSLEX_LEXSPEECH_H
