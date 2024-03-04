import os 
import soundfile as sf
import numpy as np

def read_sources(row, librispeech_dir, wham_dir):
    """ Get sources and info to mix the sources """
    # Get info about the mixture
    sources_path_list = []
    sources_path_list.append([row['source1_path']])
    sources_path_list.append([row['source2_path']])
    sources_path_list.append([row['source3_path']])
    sources_path_list.append([row['source4_path']])
    mixture_id = row['mixture_ID']
    
    sources_list = []
    max_length = 0
    # Read the files to make the mixture
    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir,
                                    sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        # Get max_length
        if max_length < len(source):
            max_length = len(source)
        sources_list.append(source)
    # Read the noise
    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)
    # if noises have 2 channels take the first
    if len(noise.shape) > 1:
        noise = noise[:, 0]
    # if noise is too short extend it
    if len(noise) < max_length:
        noise = extend_noise(noise, max_length)
    sources_list.append(noise)
    gain_list.append(row['noise_gain'])

    return mixture_id, gain_list, sources_list


def extend_noise(noise, max_length):
    """ Concatenate noise using hanning window"""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    # Increasing window
    i_w = window[:len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2::-1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   np.multiply(
                                       noise_ex[len(noise_ex) - len(d_w):],
                                       d_w) + np.multiply(
                                       noise[:len(i_w)], i_w),
                                   noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex
