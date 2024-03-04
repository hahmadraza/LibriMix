import soundfile as sf
import librosa
import tqdm
import os

target_sample_rate = 48000  # Desired sample rate

input_dir = "/mnt2/Libri4Mix_Dataset_v1/wham_noise/tr"
output_dir = "/mnt3/svoice_48k_data/wham_noise_48k/tr"
for id in tqdm.tqdm(os.listdir(input_dir)):
    try:

        input_file = os.path.join(input_dir, id)
        output_file = os.path.join(output_dir, id)
        audio_data, current_sample_rate = librosa.load(input_file, sr=None)
        resampled_audio = librosa.resample(audio_data, current_sample_rate, target_sample_rate)

        librosa.output.write_wav(output_file, resampled_audio, target_sample_rate)
    except:
        continue

input_dir = "/mnt2/Libri4Mix_Dataset_v1/wham_noise/tt"
output_dir = "/mnt3/svoice_48k_data/wham_noise_48k/tt"
for id in tqdm.tqdm(os.listdir(input_dir)):
    try:
    
        input_file = os.path.join(input_dir, id)
        output_file = os.path.join(output_dir, id)
        audio_data, current_sample_rate = librosa.load(input_file, sr=None)
        resampled_audio = librosa.resample(audio_data, current_sample_rate, target_sample_rate)

        librosa.output.write_wav(output_file, resampled_audio, target_sample_rate)
    except:
        continue

input_dir = "/mnt2/Libri4Mix_Dataset_v1/wham_noise/cv"
output_dir = "/mnt3/svoice_48k_data/wham_noise_48k/cv"
for id in tqdm.tqdm(os.listdir(input_dir)):
    try:

        input_file = os.path.join(input_dir, id)
        output_file = os.path.join(output_dir, id)
        audio_data, current_sample_rate = librosa.load(input_file, sr=None)
        resampled_audio = librosa.resample(audio_data, current_sample_rate, target_sample_rate)

        librosa.output.write_wav(output_file, resampled_audio, target_sample_rate)
    except:
        continue
