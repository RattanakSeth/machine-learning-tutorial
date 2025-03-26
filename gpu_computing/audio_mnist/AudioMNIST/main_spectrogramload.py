import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from concurrent.futures import ThreadPoolExecutor
import time
import os
import h5py

ToSpectrogram = torchaudio.transforms.MelSpectrogram()
ToDB = torchaudio.transforms.AmplitudeToDB()

class AudioDataset(Dataset):
    def __init__(self, annotations_file, max_workers=4, save_dir="./melspectrograms"):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        self.annotations = pd.concat([self.annotations] * 10, ignore_index=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # we can add data augmentation
        self.augmentations = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        # in the other case
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists
        # self.augmentations = None

    def __len__(self):
        return len(self.annotations)
    
    def load_audio_and_do_melspectrogram(self, path, augmentation):
        save_path = os.path.join(self.save_dir, os.path.basename(path) + ".pt")

        # Load from .pt file if it exists
        if os.path.exists(save_path):
            return torch.load(save_path)
    
        audio = torch.zeros((1, 16000))
        data = torchaudio.load(path)
        audio[:, :data[0].size()[1]] = data[0][0]
        if augmentation is not None:
            audio = self.augmentations(audio) # for data augmentation
        spectrogram = ToDB(ToSpectrogram(audio))
        # Save spectrogram for future use
        torch.save(spectrogram, save_path)
        return spectrogram

    def __getitem__(self, index):
        path = self.annotations['Path'][index]
        label = self.annotations['Label'][index]
        future = self.executor.submit(self.load_audio_and_do_melspectrogram, path, self.augmentations)
        return future.result(), label

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MelSpectrogram loader example')
    parser.add_argument('--numworkers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--savedir', type=str, default='./melspectrograms',
                        help='directory to save mel spectrograms')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    audioMNIST_100 = AudioDataset('./100_files.csv', max_workers=args.numworkers, save_dir=args.savedir)

    start_time = time.time()

    data_loader = DataLoader(audioMNIST_100, batch_size=40, shuffle=True)
    
    for i, sample in enumerate(data_loader):
        continue

    execution_time = time.time() - start_time

    print(f"The data loading took {execution_time:.2f} seconds.")

    # be sure to stop threadexecutor (the system should do that nicely at the end of the script)
    audioMNIST_100.executor.shutdown()
