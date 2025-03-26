import torch
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchaudio
import multiprocessing as mp

class ImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        # self.annotations = pd.concat([self.annotations] * 10, ignore_index=True)

    def __len__(self):
        return(len(self.annotations))
    
    # TODO: change this
    def __getitem__(self, index):
        audio = torch.zeros((1, 16000))
        data = torchaudio.load(self.annotations['Path'][index])
        audio[:, :data[0].size()[1]] = data[0][0]
        label = self.annotations['Label'][index]
        return(audio, label)

if __name__ == '__main__':
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as prof:
        import time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device)) #Use GPU if available

        # Create dataset loader and associated dataloader
        image500 = ImageDataset('./train_500Images.csv')

        # Get start time
        start_time = time.time()

        data_loader = DataLoader(image500, batch_size=30, shuffle=True, num_workers=8)
        
        # Effectively load data and print it
        # for i, sample in enumerate(data_loader):
        #     # print(i, sample[0].shape, sample[1])
        #     prof.step()
        #     continue

                # Chargement de toutes les données sans les parcourir
        # with torch.no_grad():  # Désactive le calcul du gradient pour accélérer le processus
        #     all_data = list(data_loader)

        prof.step()

        execution_time = time.time() - start_time

        print(f"The data loading tooks {execution_time} seconds.")

    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
