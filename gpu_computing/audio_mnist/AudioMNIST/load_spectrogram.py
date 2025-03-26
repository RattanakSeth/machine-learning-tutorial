import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from concurrent.futures import ThreadPoolExecutor
import time
import os
import csv
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

class PTDataset(Dataset):
    """Loads spectrograms from .pt files."""
    def __init__(self, annotations_file, pt_dir="melspectrograms"):
        self.annotations = pd.read_csv(annotations_file, header=None, names=['Path', 'Label'], delimiter=',')
        self.pt_dir = pt_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = self.annotations['Path'][index]
        # path = self.annotations['Path'][index]
        label = self.annotations['Label'][index]
        pt_path = os.path.join(self.pt_dir, os.path.basename(path) + ".pt")
        # print("path: ", pt_path)
        # spectrogram, label = torch.load(pt_path, weights_only=False)
        # return spectrogram, label
        spectrogram = torch.load(pt_path, weights_only=False)
        # print("spectrogram: ", spectrogram)
        return spectrogram, label
    
class HDF5Dataset(Dataset):
    """Loads spectrograms from HDF5."""
    def __init__(self, annotations_file, hdf5_file="spectrograms.h5"):
        self.annotations = pd.read_csv(annotations_file, header=None, names=['Path', 'Label'], delimiter=',')
        self.hdf5_file = hdf5_file

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = self.annotations['Path'][index]
        with h5py.File(self.hdf5_file, 'r') as f:
            spectrogram = torch.tensor(f["data"][path])
        label = self.annotations['Label'][index]
        return spectrogram, label

def makeCSVList():
    path = 'melspectrograms/'
    annot_list = []

    #########################################
    # Changement d'organisation des répertoires par rapport à l'auteur
    # Nous avons un seul répertoire avec tous les fichiers sons à l'intérieur
    # nous avons également que 3000 fichiers audio
    ########################################
    for file in os.listdir(path):
        if file.lower().endswith('.pt'):   # check pour ne pas intégrer d'autres types de fichier
            file_path = path + file
            label = file[0]
            annot_list.append((file_path, label))

    # print(annot_list)
    # return annot_list
    with open('spectrogram_pt.csv', mode='w') as csv_file:  
        csv_writer = csv.writer(csv_file, lineterminator='\n')   # lineterminator='\n' => rend le code Windows compliant
        for item in annot_list:
            csv_writer.writerow([item[0], item[1]])  

# Function to measure execution time
def measure_time(dataset, batch_size=40):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for batch_idx, (spectrograms, labels) in enumerate(data_loader):
        batch_time = time.time() - start_time
        print(f"Batch {batch_idx + 1}: Loaded {len(spectrograms)} samples in {batch_time:.2f} seconds")
        start_time = time.time()

    total_time = time.time() - start_time
    return total_time

if __name__ == '__main__':
    # makeCSVList()
    # import argparse
    # parser = argparse.ArgumentParser(description='MelSpectrogram loader example')
    # parser.add_argument('--numworkers', type=int, default=4, metavar='N',
    #                     help='number of workers (default: 4)')
    # parser.add_argument('--load_csv', type=int, default=1, metavar='N',
    #                     help='number of workers (default: 1)')
    # # parser.add_argument('--savedir', type=str, default='./melspectrograms',
    # #                     help='directory to save mel spectrograms')

    # args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    pt_dataset = PTDataset('./100_files.csv')
    hdf5_dataset = HDF5Dataset('./100_files.csv', hdf5_file="spectrograms.h5")


    # Compare execution times
    print("Testing .pt file loading speed...")
    pt_time = measure_time(pt_dataset)

    print("\nTesting HDF5 loading speed...")
    hdf5_time = measure_time(hdf5_dataset)
    # start_time = time.time()

    # data_loader = DataLoader(audioMNIST_100, batch_size=40, shuffle=True)
    
    # for i, sample in enumerate(data_loader):
    #     continue

    # execution_time = time.time() - start_time

    # print(f"The data loading took {execution_time:.2f} seconds.")

    # be sure to stop threadexecutor (the system should do that nicely at the end of the script)
    # audioMNIST_100.executor.shutdown()
