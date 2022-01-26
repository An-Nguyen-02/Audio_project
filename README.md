# Audio_project
## Introduction
The human auditory system is very effective in identifying a musical piece performed
by different people (this could also be different instrumental renditions) at a different
pace. The aim of this project – music synchronization- is to replicate this human
ability using audio processing tools.
Dynamic Time Warping is an algorithm used in measuring the similarities between
two signals (which might be of different speeds). DTW is employed in this project to
find the warping path between the extracted features of the given audios of different
lengths (ie different speeds). The Librosa library has a function that can carry out
DTW efficiently.
The aim of this project is to find the similarities between the reference audio and the
comparing audios then set the pace of the other audio to that of the reference audio.
## How to use
1. Read the report and reference to understand in details what we're doing
2. Download the python file of the project
3. Change the input files Audio_A and Audio_B in main()
4. Run the file and see the result

## Future improvement
1. Clearer reasoning why male and female alignments are hard.
2. Triming the silences in sample to have better alignments.
