import os
import sys

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from src.entity.artifact_entity import DataPreprocessingArtifacts
from src.entity.config_entity import CustomDatasetConfig
from src.exceptions import CustomException
from src.logger import logging


# Custom Dataset class for loading the data using annotations
class IndianLanguageDataset(Dataset):
    try:
        def __init__(self, dataset_config: CustomDatasetConfig, transformations: MelSpectrogram,
                     preprocessing_artifacts: DataPreprocessingArtifacts, validation: False):

            self.dataset_config = dataset_config
            self.transformations = transformations
            self.preprocessing_artifacts = preprocessing_artifacts
            if validation:
                self.annotations = pd.read_csv(
                    self.preprocessing_artifacts.test_metadata_path)
            else:
                self.annotations = pd.read_csv(
                    self.preprocessing_artifacts.train_metadata_path)
            self.audio_dir = self.dataset_config.audio_dir
            self.num_samples = self.dataset_config.num_samples
            self.target_sample_rate = self.dataset_config.sample_rate

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            audio_sample_path = self._get_audio_sample_path(idx)
            label = self._get_audio_sample_label(idx)
            signal, sr = torchaudio.load(audio_sample_path)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = self.transformations(signal)
            return signal, label

        def _get_audio_sample_path(self, idx):
            """
            Args:
              idx: index of the audio sample

            Returns:
              The path to the audio file.
            """
            class_name = f"{self.annotations.iloc[idx, 1]}"
            path = os.path.join(self.audio_dir, class_name,
                                self.annotations.iloc[idx, 0])
            return path

        def _get_audio_sample_label(self, idx):
            """            
            Args:
              idx: the index of the audio sample in the dataframe

            Returns:
              The audio sample label
            """
            return self.annotations.iloc[idx, 2]

        def _cut_if_necessary(self, signal):
            """
            > If the number of samples in the signal is greater than the number of samples given in config, then
            cut the signal to the number of samples in the model

            Args:
              signal: The signal to be processed.

            Returns:
              The signal is being returned.
            """
            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
            return signal

        def _right_pad_if_necessary(self, signal):
            """
            If the length of the signal is less than the number of samples, pad the signal with zeros

            Args:
              signal: the input signal, which is a tensor of shape (batch_size, num_samples, num_channels)

            Returns:
              The signal is being returned.
            """
            length_signal = signal.shape[1]
            if length_signal < self.num_samples:
                num_missing = self.num_samples - length_signal
                last_dim_padding = (0, num_missing)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
            return signal

        def _resample_if_necessary(self, signal, sr):
            """
            > If the sample rate of the input signal is not the same as the target sample rate, then resample
            the input signal to the target sample rate

            Args:
              signal: the audio signal
              sr: The sample rate of the audio signal.

            Returns:
              The signal is being returned.
            """
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sr, self.target_sample_rate)
                signal = resampler(signal)
            return signal

        def _mix_down_if_necessary(self, signal):
            """
            If the input signal has more than one channel, average the channels together

            Args:
              signal: the input signal, which is a tensor of shape (batch_size, num_channels, num_samples)

            Returns:
              The signal is being returned.
            """
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            return signal

    except Exception as e:
        raise CustomException(e, sys)
