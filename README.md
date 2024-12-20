### About the dataset
LibriMix is an open source dataset for source separation in noisy 
environments. It is derived from LibriSpeech signals (clean subset) 
and WHAM noise. It offers a free alternative to the WHAM dataset 
and complements it. It will also enable cross-dataset experiments.

### Generating LibriMix
To generate LibriMix 48k data, clone the repo and run follwing scripts : 

```
git clone https://github.com/hahmadraza/LibriMix.git
cd LibriMix 

python scripts/create_librispeech_metadata_48k.py --librispeech_dir /mnt3/svoice_48k_data/data

python scripts/create_wham_metadata_48k.py --wham_dir /mnt3/svoice_48k_data/wham_noise_48k

python scripts/create_librimix_metadata_48k.py --librispeech_dir /mnt3/svoice_48k_data/data --wham_dir /mnt3/svoice_48k_data/wham_noise_48k --librispeech_md_dir /mnt3/svoice_48k_data/data/metadata \
--wham_md_dir /mnt3/svoice_48k_data/wham_noise_48k/metadata --metadata_outdir /mnt3/svoice_48k_data/data/Libri3Mix --nsrc 3

python create_librimix_from_metadata_48k.py --librispeech_dir /mnt3/svoice_48k_data/data --metadata_dir /mnt3/svoice_48k_data/data/Libri3Mix --wham_dir /mnt3/svoice_48k_data/wham_noise_48k \
--librimix_outdir /mnt3/Libri3Mix_48k --n_src 3 --freqs 48k -modes min --types mix_both
```

For Original LibriMix data download and creation, run
```
./generate_librimix.sh storage_dir
```

Make sure that SoX is installed on your machine.

For windows :
```
conda install -c groakat sox
```

For Linux :
```
conda install -c conda-forge sox
```

You can either change `storage_dir` and `n_src` by hand in 
the script or use the command line.  
By default, LibriMix will be generated for 2 and 3 speakers,
at both 16Khz and 8kHz, 
for min max modes, and all mixture types will be saved (mix_clean, 
mix_both and mix_single). This represents around **430GB** 
of data for Libri2Mix and **332GB** for Libri3Mix. 
You will also need to store LibriSpeech and wham_noise_augmented during
generation for an additional **30GB** and **50GB**.
Please refer to 
[this section](#Features) if you want to generate less data.
You will also find a detailed storage usage description in each metadata folder.


### Features
In LibriMix you can choose :
* The number of sources in the mixtures.
* The sample rate  of the dataset from 16 KHz to any frequency below. 
* The mode of mixtures : min (the mixture ends when the shortest source
 ends) or max (the mixtures ends with the longest source)
 * The type of mixture : mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)

You can customize the generation by editing ``` generate_librimix.sh ```.
 
### Note on scripts
For the sake of transparency, we have released the metadata generation 
scripts. However, we wish to avoid any changes to the dataset, 
especially to the test subset that shouldn't be changed under any 
circumstance.

### Why use LibriMix
More than just an open source dataset, LibriMix aims towards generalizable speech separation.
You can checkout section 3.3 of our paper [here](https://arxiv.org/pdf/2005.11262.pdf) for more details.

### Related work
If you wish to implement models based on LibriMix you can checkout 
[Asteroid](https://github.com/mpariente/asteroid) and the 
[recipe](https://github.com/mpariente/asteroid/tree/master/egs/librimix/ConvTasNet)
associated to LibriMix for reproducibility.

Along with LibriMix, SparseLibriMix a dataset aiming towards more realistic, conversation-like scenarios
has been released [here](https://github.com/popcornell/SparseLibriMix).

(contributors: [@JorisCos](https://github.com/JorisCos), [@mpariente](https://github.com/mpariente) and [@popcornell](https://github.com/popcornell) )

### Citing Librimix 

```BibTex
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
