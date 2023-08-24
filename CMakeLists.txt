cmake_minimum_required(VERSION 3.16)
project(Deepspeech LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
#set(BUILD_SHARED_LIBS ON CACHE STRING "Link to shared libraries by default.")

#Load required services/packages: This basic example uses S3.
find_package(AWSSDK REQUIRED COMPONENTS lexv2-runtime)

include_directories(include deepspeech/inc)

if (UNIX AND NOT APPLE)
    SET(LINUX TRUE)
endif ()
if (LINUX)
    message("Linux OS")
    include_directories(linuxAudio)
    find_library(AUDIO pulse)
    if (NOT AUDIO)
        message(FATAL_ERROR "Pulse audio Not found")
    else ()
        message("-- PULSE Audio Found")
    endif ()

    find_library(AUDIO_UNIT pulse-simple)
    if (NOT AUDIO_UNIT)
        message(FATAL_ERROR "pulse-simple not found")
    else ()
        message("-- pulse-simple Found")
    endif ()

endif ()
if(APPLE)
    message("Mac OSX detected")
    include_directories(appleAudio)
    find_library(AUDIO CoreAudio)
    if (NOT AUDIO)
        message(FATAL_ERROR "CoreAudio Not found")
    else ()
        message("-- CoreAudio Found")
    endif ()

    find_library(AUDIO_UNIT AudioToolbox)
    if (NOT AUDIO_UNIT)
        message(FATAL_ERROR "audio unit not found")
    else ()
        message("-- Audio Unit Found")
    endif ()


endif ()

find_library(DEEPSPEECH NAMES libdeepspeech.so HINTS deepspeech/lib)

find_library(SOX_LIB sox)

find_library(SND_FILE sndfile)
if (NOT SND_FILE)
    message(FATAL_ERROR "libsndfile not found")
else ()
    message("-- libsndfile Found")
endif ()

find_library(FVAD fvad)
if (NOT FVAD)
    message(FATAL_ERROR "libfvad not found")
else ()
    message("-- libfvad Found")
endif ()

find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)


if(APPLE)
    add_executable(appleAudio appleAudio/appleAudio.cpp)
    target_link_libraries(appleAudio ${AWSSDK_LINK_LIBRARIES} ${SOX_LIB} ${DEEPSPEECH} ${AUDIO} ${AUDIO_UNIT} ${SND_FILE})
endif ()

if (LINUX)
    add_executable(linuxAudio linuxAudio/linuxAudio.cpp)
    target_link_libraries(linuxAudio ${AUDIO} ${AUDIO_UNIT} ${SND_FILE})
endif ()

add_executable(testVAD src/main.cpp
        src/deepspeech.cpp
        src/vad.cpp
        appleAudio/appleAudio.cpp

        )
target_link_libraries(testVAD ${AWSSDK_LINK_LIBRARIES} ${SOX_LIB} ${DEEPSPEECH} ${AUDIO} ${AUDIO_UNIT} ${SND_FILE} ${FVAD})

