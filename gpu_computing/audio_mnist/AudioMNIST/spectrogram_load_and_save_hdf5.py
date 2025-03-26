import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import h5py
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# MelSpectrogram transformation
ToSpectrogram = torchaudio.transforms.MelSpectrogram()
ToDB = torchaudio.transforms.AmplitudeToDB()

class AudioDataset(Dataset):
    def __init__(self, annotations_file, hdf5_file="spectrograms.h5", max_workers=4):
        self.annotations = pd.read_csv(annotations_file, header=None, names=['Path', 'Label'], delimiter=',')
        self.annotations = pd.concat([self.annotations] * 10, ignore_index=True)
        self.hdf5_file = hdf5_file
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Create HDF5 file if it doesn’t exist
        if not os.path.exists(self.hdf5_file):
            with h5py.File(self.hdf5_file, 'w') as f:
                f.create_group("data")  # Group for storing spectrograms

        # Data augmentation (optional)
        self.augmentations = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __len__(self):
        return len(self.annotations)
    
    def load_audio_and_compute_spectrogram(self, path, augmentation):
        """Loads audio and computes MelSpectrogram."""
        with h5py.File(self.hdf5_file, 'a') as f:
            if path in f["data"]:  # Load from HDF5 if it exists
                return torch.tensor(f["data"][path])

        audio = torch.zeros((1, 16000))
        data = torchaudio.load(path)
        audio[:, :data[0].size()[1]] = data[0][0]
        
        if augmentation is not None:
            audio = self.augmentations(audio)  # Apply augmentation
        
        spectrogram = ToDB(ToSpectrogram(audio))

        # Save to HDF5
        with h5py.File(self.hdf5_file, 'a') as f:
            f["data"].create_dataset(path, data=spectrogram.numpy(), compression="gzip")

        return spectrogram

    def __getitem__(self, index):
        path = self.annotations['Path'][index]
        label = self.annotations['Label'][index]
        future = self.executor.submit(self.load_audio_and_compute_spectrogram, path, self.augmentations)
        return future.result(), label

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MelSpectrogram HDF5 Loader')
    parser.add_argument('--numworkers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--hdf5file', type=str, default='spectrograms.h5', help='HDF5 file to store spectrograms')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Instantiate dataset with HDF5 storage
    audioMNIST_100 = AudioDataset('./100_files.csv', hdf5_file=args.hdf5file, max_workers=args.numworkers)

    start_time = time.time()

    # DataLoader
    data_loader = DataLoader(audioMNIST_100, batch_size=40, shuffle=True)

    for i, sample in enumerate(data_loader):
        continue

    execution_time = time.time() - start_time

    print(f"Data loading took {execution_time:.2f} seconds.")

    # Stop the thread pool executor
    audioMNIST_100.executor.shutdown()
