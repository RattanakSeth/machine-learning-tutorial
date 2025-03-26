import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, dataDir):
        self.data_dir = dataDir

    def dataPaths(self):
        filepaths = []
        # labels = []
        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldPath = os.path.join(self.data_dir, fold)
            filelist = os.listdir(foldPath)
            for file in filelist:
                fpath = os.path.join(foldPath, file)
                filepaths.append(fpath)
                # labels.append(fold)
        print("file path: ", filepaths)
        return filepaths

    def dataFrame(self, files, labels):

        Fseries = pd.Series(files, name='filepaths')
        return Fseries
        # Lseries = pd.Series(labels, name='labels')
        # return pd.concat([Fseries, Lseries], axis=1)

    # def split_(self):
    #     files, labels = self.dataPaths()
    #     df = self.dataFrame(files, labels)
    #     strat = df['labels']
    #     trainData, dummyData = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=strat)
    #     strat = dummyData['labels']
    #     validData, testData = train_test_split(dummyData, train_size=0.5, shuffle=True, random_state=42, stratify=strat)
    #     return trainData, validData, testData



dataDir = '500images'

data = Dataset(dataDir)
data.dataPaths()