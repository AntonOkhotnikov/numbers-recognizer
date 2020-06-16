import csv
import torch
import torchaudio
from torchaudio.compliance.kaldi import mfcc
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class DRDataset(Dataset):
    
    def __init__(self, root_path, max_len, enrich_target=True):
        """Digits Recognizer Dataset

        Args:
            root_path (str): Path to root of unpacked numbers.zip
            max_len (float): Maximum audio length in seconds (pad with zeroes if less)
            enrich_target (bool): Insert symbol to target at 4 position from the end
        """
        csv_path = '/'.join([root_path, 'train.csv'])
        paths, self.genders, self.labels = self.__read_csv(csv_path)
        self.paths = ['/'.join([root_path, path]) for path in paths]
        self.max_len = max_len
        if enrich_target:
            self.labels = [self.__enrich_target(label) for label in self.labels]
        
    def __getitem__(self, index):
        """Generate one sample"""
        
        filepath, gender, y = self.paths[index], self.genders[index], self.labels[index]

        # 0. Load file
        waveform, sample_rate = torchaudio.load(filepath)

        # 1. Pad signal if necessary
        if waveform.shape[1] / sample_rate < self.max_len:
            to_pad = int(self.max_len * sample_rate - waveform.shape[1])
            waveform = F.pad(input=waveform, pad=(1, to_pad), mode='constant', value=0)
        else:
            raise NotImplementedError("Signal max length is above expected")

        # 2. Make MFCC - use default Kaldi params
        # X = mfcc(waveform, sample_frequency=sample_rate, dither=1).to(device)
        X = mfcc(waveform, sample_frequency=sample_rate, dither=1)
        
        y = self.labels[index]

        return X, y
        
    def __len__(self):
        return len(self.paths)

    def __read_csv(self, csv_path, delimiter=',', skip_header=True):

        with open(csv_path) as csv_stream:
            reader = csv.reader(csv_stream, delimiter=delimiter)
            if skip_header:
                next(reader)

            paths = []
            genders = []
            labels = []

            for row in reader:
                path, gender, label = row
                paths.append(path)
                genders.append(gender)
                labels.append(label)

            return paths, genders, labels
    
    @staticmethod
    def __enrich_target(target_str, label='*'):
        """Insert a character that means word 'тысяч(-а/-и)'

        Args:
            target_str ([type]): [description]
            label (str, optional): [description]. Defaults to '*'.

        Returns:
            [type]: [description]
        """
        if len(target_str) > 3:
            target_list = list(target_str)
            target_list.insert(-3, label)
            return ''.join(target_list)
        return target_str


def get_train_dev_loaders(root_path, params, max_len, train_ratio=0.8, enrich_target=True):
    """Returns Train and Val generators

    Args:
        root_path (str): [description]
        params (dict): [description]
        max_len (float): [description]
        train_ratio (float, optional): [description]. Defaults to 0.8.
        enrich_target (bool, optional): [description]. Defaults to True.

    Returns:
        train_loader (gen): Train data generator
        val_loader (gen): Validation data generator
    """
    
    # 0. Build dataset
    dataset = DRDataset(root_path, max_len, enrich_target=enrich_target)

    # 1. Split train and eval
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=(1 - train_ratio))

    # 2. Make subsets
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    # 3. Pack subsets to Loaders
    loaders = []
    for ds in [train_ds, val_ds]:
        loaders.append(DataLoader(ds, **params))

    return loaders


if __name__ == "__main__":

    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

    root_path='/home/anton/work/projects/myna/data/numbers'

    train_loader, val_loader = get_train_dev_loaders(root_path, params, train_ratio=0.9, max_len=3.9, enrich_target=True)

    print(device)

    for x, y in train_loader:
        print(x, y)
