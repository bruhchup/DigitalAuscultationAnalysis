import torch
import torchaudio
import librosa
from torchaudio import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import math
import os

DESIRED_DURATION = 8
DESIRED_SR = 16000


class ICBHI(Dataset):
    def __init__(self, data_path, split, metadatafile='metadata.csv', duration=DESIRED_DURATION,
                 samplerate=DESIRED_SR, device="cpu", fade_samples_ratio=16, pad_type="circular", meta_label=""):

        self.data_path = data_path
        self.csv_path = os.path.join(self.data_path, metadatafile)
        self.split = split
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["split"] == self.split]
        self.meta_label = meta_label
        self.duration = duration
        self.samplerate = samplerate
        self.targetsample = self.duration * self.samplerate
        self.pad_type = pad_type
        self.device = device
        self.fade_samples_ratio = fade_samples_ratio
        self.fade_samples = int(self.samplerate / self.fade_samples_ratio)
        self.fade = T.Fade(fade_in_len=self.fade_samples, fade_out_len=self.fade_samples, fade_shape='linear')
        self.fade_out = T.Fade(fade_in_len=0, fade_out_len=self.fade_samples, fade_shape='linear')

        if self.meta_label != "":
            self.pth_path = os.path.join(self.data_path, f"icbhi-4{self.split}_duration{self.duration}_metalabel-{meta_label}.pth")
        else:
            self.pth_path = os.path.join(self.data_path, f"icbhi-4{self.split}_duration{self.duration}.pth")

        if os.path.exists(self.pth_path):
            print(f"Loading dataset {self.split}...")
            pth_dataset = torch.load(self.pth_path, weights_only=False)
            self.data, self.labels, self.metadata_labels = pth_dataset['data'], pth_dataset['label'], pth_dataset['meta_label']
            print(f"Dataset {self.split} loaded!")
        else:
            print(f"File {self.pth_path} does not exist. Creating dataset...")
            self.data, self.labels, self.metadata_labels = self.get_dataset()
            data_dict = {"data": self.data, "label": self.labels, "meta_label": self.metadata_labels}
            print(f"Dataset {self.split} created!")
            torch.save(data_dict, self.pth_path)
            print(f"File {self.pth_path} saved!")

    def get_sample(self, i):
        ith_row = self.df.iloc[i]
        filepath = ith_row['filepath']
        filepath = os.path.join(self.data_path, filepath)
        onset = ith_row['onset']
        offset = ith_row['offset']
        bool_wheezes = ith_row['wheezes']
        bool_crackles = ith_row['crackles']

        metalabel_colname = str(self.meta_label) + '_class_num'
        if metalabel_colname in ith_row.index:
            metadata_label = ith_row[metalabel_colname]
        else:
            metadata_label = 0

        if not bool_wheezes:
            if not bool_crackles:
                label = 0  # Normal
            else:
                label = 1  # Crackle
        else:
            if not bool_crackles:
                label = 2  # Wheeze
            else:
                label = 3  # Both

        sr = librosa.get_samplerate(filepath)
        audio, _ = torchaudio.load(filepath, int(onset * sr), (int(offset * sr) - int(onset * sr)))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.samplerate:
            resample = T.Resample(sr, self.samplerate)
            audio = resample(audio)

        return self.fade(audio), label, metadata_label

    def get_dataset(self):
        dataset = []
        labels = []
        metadata_labels = []

        for i in range(len(self.df)):
            audio, label, metadata_label = self.get_sample(i)
            if audio.shape[-1] > self.targetsample:
                audio = audio[..., :self.targetsample]
            else:
                if self.pad_type == 'circular':
                    ratio = math.ceil(self.targetsample / audio.shape[-1])
                    audio = audio.repeat(1, ratio)
                    audio = audio[..., :self.targetsample]
                    audio = self.fade_out(audio)
                elif self.pad_type == 'zero':
                    tmp = torch.zeros(1, self.targetsample, dtype=torch.float32)
                    diff = self.targetsample - audio.shape[-1]
                    tmp[..., diff // 2:audio.shape[-1] + diff // 2] = audio
                    audio = tmp
            dataset.append(audio)
            labels.append(label)
            metadata_labels.append(metadata_label)

        return torch.unsqueeze(torch.vstack(dataset), 1), torch.tensor(labels), torch.tensor(metadata_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.metadata_labels[idx]
