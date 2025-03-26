import torch
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchaudio
import multiprocessing as mp
from torch.profiler import profile, ProfilerActivity
import time

class AudioDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        # self.annotations = pd.concat([self.annotations] * 10, ignore_index=True)

    def __len__(self):
        return(len(self.annotations))
    
    def __getitem__(self, index):
        audio = torch.zeros((1, 16000))
        data = torchaudio.load(self.annotations['Path'][index])
        audio[:, :data[0].size()[1]] = data[0][0]
        label = self.annotations['Label'][index]
        return(audio, label)

def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device)) #Use GPU if available

    # Create dataset loader and associated dataloader
    audioMNIST_100 = AudioDataset('./100_files.csv')
    record_cpu_times = []
    for num_workers in [1, 2, 4, 8]:  # Testing different values
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
             # Get start time
            start_time = time.time()

            # Test different num_workers values
            print(f"\nTesting with num_workers={num_workers} ...")

            data_loader = DataLoader(
                audioMNIST_100, 
                batch_size=30, 
                shuffle=True, 
                num_workers=num_workers, 
                multiprocessing_context=mp.get_context('spawn')
                )
            
            # Effectively load data and print it
            for i, sample in enumerate(data_loader):
                # print(i, sample[0].shape, sample[1])
                prof.step()
                continue

            # with torch.no_grad():  # Désactive le calcul du gradient pour accélérer le processus
            #     all_data = list(data_loader)

            prof.step()

            execution_time = time.time() - start_time
            print(f"The data loading took {execution_time} seconds.")
            print(f"Execution Time with num_workers={num_workers}: {execution_time:.4f} seconds")

        self_cpu_total = record_cpu_times.append(_format_time(prof.key_averages(group_by_stack_n=5).self_cpu_time_total))
        # new_cpu_time = self_cpu_total 
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        print("original value: ", prof.key_averages(group_by_stack_n=5).self_cpu_time_total)
        print("self cpu: ", _format_time(prof.key_averages(group_by_stack_n=5).self_cpu_time_total))
        # del data_loader
        # torch.cuda.empty_cache()  # If using CUDA
    

    # obtained results
    print(record_cpu_times)
    # for idx, num_workers in [1, 2, 4, 8]:
    #     print("user {} worker got {} seconds".format(num_workers, record_cpu_times[i] if i > 0 else record_cpu_times[i] - record_cpu_times[i -1]))