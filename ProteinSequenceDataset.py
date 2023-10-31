import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from config import config

# Ścieżki do embeddingów:
embeds_map = {
    "T5" : "embeddings/t5",
    "ProtBERT" : "embeddings/protbert",
    "EMS2" : "embeddings/ems2"
}

# Długości wektorów w poszczególnych embeddingach:
embeds_dim = {
    "T5" : 1024,
    "ProtBERT" : 1024,
    "EMS2" : 1280
}
class ProteinSequenceDataset(Dataset):
    """
    Klasa służąca do ładowania i obsługi danych dotyczących sekwencji białek w kontekście
    ich embeddingów oraz etykiet.
    """

    def __init__(self, datatype, embeddings_source):
        """
         Inicjalizuje obiekt ProteinSequenceDataset.

         Parameters:
         ----------
         datatype : str
             Rodzaj danych; oczekuje się jednej z wartości, np. "train" lub "test".

         embeddings_source : str
             Źródło embeddingów białkowych; akceptowane wartości to "ProtBERT", "EMS2" i "T5".
         """

        super(ProteinSequenceDataset).__init__()
        self.datatype = datatype

        if embeddings_source in ["ProtBERT", "EMS2"]:
            embeds = np.load(embeds_map[embeddings_source] + "/" + datatype + "_embeddings.npy")
            ids = np.load(embeds_map[embeddings_source] + "/" + datatype + "_ids.npy")

        if embeddings_source == "T5":
            embeds = np.load(embeds_map[embeddings_source] + "/" + datatype + "_embeds.npy")
            ids = np.load(embeds_map[embeddings_source] + "/" + datatype + "_ids.npy")

        embeds_list = []
        for l in range(embeds.shape[0]):
            embeds_list.append(embeds[l, :])
        self.df = pd.DataFrame(data={"EntryID": ids, "embed": embeds_list})

        if datatype == "train":
            df_labels = pd.read_pickle(
                "other/train_targets_top" + str(config.num_labels) + ".pkl")
            self.df = self.df.merge(df_labels, on="EntryID")

    def __len__(self):
        """
        Zwraca liczbę rekordów w ramce danych.

        Returns:
        -------
        int
            Liczba rekordów w ramce danych.
        """

        return len(self.df)

    def __getitem__(self, index):
        """
        Zwraca embedding białka i ewentualne etykiety lub identyfikator dla podanego indeksu.

        Parameters:
        ----------
        index : int
            Indeks rekordu w ramce danych.

        Returns:
        -------
        tuple
            Dla datatype="train": Krotka zawierająca tensor embeddingu (torch.Tensor) oraz tensor etykiet (torch.Tensor).
            Dla datatype="test": Krotka zawierająca tensor embeddingu (torch.Tensor) oraz identyfikator białka (str).
        """

        embed = torch.tensor(self.df.iloc[index]["embed"], dtype=torch.float32)
        if self.datatype == "train":
            targets = torch.tensor(self.df.iloc[index]["labels_vect"], dtype=torch.float32)
            return embed, targets
        if self.datatype == "test":
            id = self.df.iloc[index]["EntryID"]
            return embed, id