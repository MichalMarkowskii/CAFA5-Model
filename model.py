# Biblioteki
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Moduły torcha:
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy


# %%
# Klasa pomocnicza do łatwiejszego zarządzania parametrami ogólnymi

class config:
    train_sequences_path = "cafa/train_sequences.fasta"
    train_labels_path = "cafa/train_terms.tsv"
    test_sequences_path = "cafa/testsuperset.fasta"

    # biorąc pod uwagę X najważniejszych terminów GO we wszystkich zestawach białek,
    # (tych które najczęściej się powtarzają) generujemy dla każdego białka wektor o długości X,
    # aby wskazać prawdziwe prawdopodobieństwo, że każdy z X terminów GO znajduje się w białku (0 lub 1).
    # (ograniczenie związane z mocą obliczeniową)
    num_labels = 500 # ^^^
    n_epochs = 5
    batch_size = 128 # liczba sekwencji na 1 batch
    lr = 0.001 # learning rate

# %%

# Skrypt ten przetwarza etykiety dla pewnych identyfikatorów białek w kontekście terminów Gene Ontology (GO).
# Najpierw skrypt wczytuje identyfikatory oraz etykiety z pliku. Następnie wybiera daną liczbę
# najczęściej występujących terminów GO i filtruje etykiety, aby zachować tylko te odpowiadające wybranym terminom
# oraz wcześniej wczytanym identyfikatorom. Na podstawie tego tworzy słownik z etykietami dla każdego
# identyfikatora oraz wypełnia macierz, reprezentującą obecność konkretnych terminów GO dla każdego identyfikatora.

print("GENERATE TARGETS FOR ENTRY IDS ("+str(config.num_labels)+" MOST COMMON GO TERMS)")
ids = np.load("embeddings/t5/train_ids.npy")
labels = pd.read_csv(config.train_labels_path, sep = "\t")
top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False)
labels_names = top_terms[:config.num_labels].index.values
train_labels_sub = labels[(labels.term.isin(labels_names)) & (labels.EntryID.isin(ids))]
id_labels = train_labels_sub.groupby('EntryID')['term'].apply(list).to_dict()

go_terms_map = {label: i for i, label in enumerate(labels_names)}
labels_matrix = np.empty((len(ids), len(labels_names)))

for index, id in tqdm(enumerate(ids)):
    id_gos_list = id_labels[id]
    temp = [go_terms_map[go] for go in labels_names if go in id_gos_list]
    labels_matrix[index, temp] = 1

labels_list = []
for l in range(labels_matrix.shape[0]):
    labels_list.append(labels_matrix[l, :])

labels_df = pd.DataFrame(data={"EntryID":ids, "labels_vect":labels_list})
labels_df.to_pickle("other/train_targets_top"+str(config.num_labels)+".pkl")
print("GENERATION FINISHED!")

# %%

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

# %%
class ProteinSequenceDataset(Dataset):
    """
    Klasa służąca do ładowania i obsługi danych dotyczących sekwencji białek w kontekście
    ich embeddingów oraz etykiet.

    Attributes:
    ----------
    df : pd.DataFrame
        Ramka danych zawierająca identyfikatory białek oraz ich embeddingi.
        Dla danych treningowych dodatkowo zawiera etykiety.
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


# %%
# Test czy działa
# dataset = ProteinSequenceDataset(datatype="train",embeddings_source="T5")
# embeddings, labels = dataset.__getitem__(0)
# print("COMPONENTS FOR FIRST PROTEIN : ")
# print("EMBEDDINGS VECTOR : \n ", embeddings, "\n")
# print("TARGETS LABELS VECTOR : \n ", labels, "\n")

# %%
class MultiLayerPerceptron(torch.nn.Module):
    """
    Model klasyfikacyjny oparty na warstwach w pełni połączonych.

    Attributes:
    ----------
    linear1 : nn.Linear
        Pierwsza warstwa w pełni połączona.
    activation1 : nn.ReLU
        Pierwsza funkcja aktywacji.
    linear2 : nn.Linear
        Druga warstwa w pełni połączona.
    activation2 : nn.ReLU
        Druga funkcja aktywacji.
    linear3 : nn.Linear
        Trzecia warstwa w pełni połączona.
    """

    def __init__(self, input_dim, num_classes):
        """
        Inicjalizuje model MultiLayerPerceptron.

        Parameters:
        ----------
        input_dim : int
            Wymiar danych wejściowych.
        num_classes : int
            Liczba klas wyjściowych.
        """

        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 1012)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1012, 712)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(712, num_classes)

    def forward(self, x):
        """
        Przepuszcza dane wejściowe przez model.

        Parameters:
        ----------
        x : torch.Tensor
            Tensor danych wejściowych.

        Returns:
        -------
        torch.Tensor
            Tensor przewidywań modelu.
        """

        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x

class CNN1D(nn.Module):
    """
    Model klasyfikacyjny oparty na jednowymiarowej konwolucji.

    Attributes:
    ----------
    conv1 : nn.Conv1d
        Pierwsza warstwa konwolucyjna.
    pool1 : nn.MaxPool1d
        Pierwsza warstwa łączenia.
    conv2 : nn.Conv1d
        Druga warstwa konwolucyjna.
    pool2 : nn.MaxPool1d
        Druga warstwa łączenia.
    fc1 : nn.Linear
        Pierwsza warstwa w pełni połączona.
    fc2 : nn.Linear
        Druga warstwa w pełni połączona.
    """

    def __init__(self, input_dim, num_classes):
        """
        Inicjalizuje model CNN1D.

        Parameters:
        ----------
        input_dim : int
            Wymiar danych wejściowych.
        num_classes : int
            Liczba klas wyjściowych.
        """

        super(CNN1D, self).__init__()
        # (batch_size, channels, embed_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=1)
        # (batch_size, 3, embed_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (batch_size, 3, embed_size/2 = 512)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, dilation=1, padding=1, stride=1)
        # (batch_size, 8, embed_size/2 = 512)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (batch_size, 8, embed_size/4 = 256)
        self.fc1 = nn.Linear(in_features=int(8 * input_dim/4), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Przepuszcza dane wejściowe przez model.

        Parameters:
        ----------
        x : torch.Tensor
            Tensor danych wejściowych.

        Returns:
        -------
        torch.Tensor
            Tensor przewidywań modelu.
        """

        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerClassifier(nn.Module):
    """
    Model klasyfikacyjny oparty na enkoderze z architektury transformerowej.

    Attributes:
    ----------
    embedding : nn.Linear
        Warstwa embeddingów.
    transformer_encoder : nn.TransformerEncoder
        Enkoder transformerowy.
    classifier : nn.Linear
        Warstwa klasyfikacyjna.
    """

    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_layers=6):
        """
        Inicjalizuje model TransformerClassifier.

        Parameters:
        ----------
        input_dim : int
            Wymiar danych wejściowych.
        num_classes : int
            Liczba klas wyjściowych.
        d_model : int, optional
            Liczba cech w modelu transformerowym. Domyślnie 512.
        nhead : int, optional
            Liczba głów w mechanizmie uwagi. Domyślnie 8.
        num_layers : int, optional
            Liczba warstw w enkoderze. Domyślnie 6.
        """

        super(TransformerClassifier, self).__init__()

        # Warstwa embeddingów
        self.embedding = nn.Linear(input_dim, d_model)

        # Enkoder transformerowy
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Warstwa klasyfikacyjna
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Przepuszcza dane wejściowe przez model.

        Parameters:
        ----------
        x : torch.Tensor
            Tensor danych wejściowych.

        Returns:
        -------
        torch.Tensor
            Tensor przewidywań modelu.
        """

        x = self.embedding(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x[0, :, :]
        x = self.classifier(x)
        return x

# %%

def train_model(embeddings_source, model_type="linear", train_size=0.9):
    """
    Trenuje model na podstawie danych treningowych z zadanych embeddingów i określonego typu modelu.

    Parameters:
    ----------
    embeddings_source : str
        Określa źródło embeddingów białkowych; akceptowane wartości to "ProtBERT", "EMS2" i "T5".

    model_type : str, optional (default = "linear")
        Rodzaj modelu do trenowania; możliwe wartości to "linear", "convolutional" i "transformer".

    train_size : float, optional (default = 0.9)
        Ustala stosunek danych treningowych do walidacyjnych.

    Returns:
    -------
    model : torch.nn.Module
        Wytrenowany model.

    losses_history : dict
        Słownik z historią strat dla zbiorów treningowego i walidacyjnego.

    scores_history : dict
        Słownik z historią wyników F1 dla zbiorów treningowego i walidacyjnego.
    """

    # Wczytanie danych treningowych i podział na zbiór treningowy i walidacyjny
    train_dataset = ProteinSequenceDataset(datatype="train", embeddings_source=embeddings_source)
    train_set, val_set = random_split(train_dataset,
                                      lengths=[int(len(train_dataset) * train_size),
                                               len(train_dataset) - int(len(train_dataset) * train_size)])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    # Wybór modelu do trenowania w zależności od parametru model_type
    if model_type == "linear":
        model = MultiLayerPerceptron(input_dim=embeds_dim[embeddings_source], num_classes=config.num_labels).to(
            'cpu')
    elif model_type == "convolutional":
        model = CNN1D(input_dim=embeds_dim[embeddings_source], num_classes=config.num_labels).to('cpu')
    elif model_type == "transformer":
        model = TransformerClassifier(input_dim=embeds_dim[embeddings_source], num_classes=config.num_labels).to(
            'cpu')

    # Inicjalizacja optymalizatora, planera zmiany współczynnika uczenia oraz funkcji straty
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
    CrossEntropy = torch.nn.CrossEntropyLoss()
    f1_score = MultilabelF1Score(num_labels=config.num_labels).to('cpu')

    print("BEGIN TRAINING...")

    train_loss_history = []
    val_loss_history = []
    train_f1score_history = []
    val_f1score_history = []

    # Pętla trenowania
    for epoch in range(config.n_epochs):
        print("EPOCH ", epoch + 1)

        # Faza treningowa
        losses = []
        scores = []
        for embed, targets in tqdm(train_dataloader):
            embed, targets = embed.to('cpu'), targets.to('cpu')
            optimizer.zero_grad()
            preds = model(embed)
            loss = CrossEntropy(preds, targets)
            score = f1_score(preds, targets)
            losses.append(loss.item())
            scores.append(score.item())
            loss.backward()
            optimizer.step()

        # Zapisywanie wyników dla zbioru treningowego
        avg_loss = np.mean(losses)
        avg_score = np.mean(scores)
        train_loss_history.append(avg_loss)
        train_f1score_history.append(avg_score)

        # Faza walidacji
        losses = []
        scores = []
        for embed, targets in val_dataloader:
            embed, targets = embed.to('cpu'), targets.to('cpu')
            preds = model(embed)
            loss = CrossEntropy(preds, targets)
            score = f1_score(preds, targets)
            losses.append(loss.item())
            scores.append(score.item())

        # Zapisywanie wyników dla zbioru walidacyjnego
        avg_loss = np.mean(losses)
        avg_score = np.mean(scores)
        val_loss_history.append(avg_loss)
        val_f1score_history.append(avg_score)

        scheduler.step(avg_loss)
        print("\n")

    print("TRAINING FINISHED")
    losses_history = {"train": train_loss_history, "val": val_loss_history}
    scores_history = {"train": train_f1score_history, "val": val_f1score_history}

    return model, losses_history, scores_history


# %%

ems2_model, ems2_losses, ems2_scores = train_model(embeddings_source="EMS2",model_type="transformer")

# %%

t5_model, t5_losses, t5_scores = train_model(embeddings_source="T5",model_type="transformer")

# %%

protbert_model, protbert_losses, protbert_scores = train_model(embeddings_source="ProtBERT",model_type="transformer")

# %%

# Wizualizacja wyników dla każdego rodzaju embeddingów w kontekście straty walidacyjnej
plt.figure(figsize=(10, 4))
plt.plot(ems2_losses["val"], label="EMS2")
plt.plot(t5_losses["val"], label="T5")
plt.plot(protbert_losses["val"], label="ProtBERT")
plt.title("Validation Losses for # Vector Embeddings")
plt.xlabel("Epochs")  # Oś X przedstawia epoki
plt.ylabel("Average Loss")  # Oś Y przedstawia średnią stratę
plt.legend()
plt.show()

# Wizualizacja wyników dla każdego rodzaju embeddingów w kontekście średniego wyniku F1 walidacyjnego
plt.figure(figsize=(10, 4))
plt.plot(ems2_scores["val"], label="EMS2")
plt.plot(t5_scores["val"], label="T5")
plt.plot(protbert_scores["val"], label="ProtBERT")
plt.title("Validation F1-Scores for # Vector Embeddings")
plt.xlabel("Epochs")  # Oś X przedstawia epoki
plt.ylabel("Average F1-Score")  # Oś Y przedstawia średni wynik F1
plt.legend()
plt.show()

# %%

def predict(embeddings_source):
    """
    Generuje predykcje dla zestawu testowego na podstawie podanego źródła embeddingów.

    Parameters:
    ----------
    embeddings_source : str
        Nazwa źródła embeddingów, które mają zostać użyte do predykcji.
        Akceptowane wartości to: "T5", "ProtBERT" i "EMS2".

    Returns:
    -------
    submission_df : pd.DataFrame
        Ramka danych zawierająca identyfikatory białek ("Id"), odpowiadające im terminy GO ("GO term")
        oraz przewidywane prawdopodobieństwa ("Confidence").
    """

    # Wczytanie danych testowych oraz przygotowanie loadera do batchowania danych.
    test_dataset = ProteinSequenceDataset(datatype="test", embeddings_source=embeddings_source)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Wybór odpowiedniego modelu na podstawie embeddingów.
    if embeddings_source == "T5":
        model = t5_model
    elif embeddings_source == "ProtBERT":
        model = protbert_model
    elif embeddings_source == "EMS2":
        model = ems2_model

    # Przełączenie modelu w tryb ewaluacji (ważne - może mieć wpływ na wynik ewaluacji).
    model.eval()

    # Wczytanie etykiet treningowych i wybór najczęściej występujących terminów GO.
    labels = pd.read_csv(config.train_labels_path, sep="\t")
    top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False)
    labels_names = top_terms[:config.num_labels].index.values

    print("GENERATE PREDICTION FOR TEST SET...")

    # Przygotowanie tablic na wyniki.
    ids_ = np.empty(shape=(len(test_dataloader) * config.num_labels,), dtype=object)
    go_terms_ = np.empty(shape=(len(test_dataloader) * config.num_labels,), dtype=object)
    confs_ = np.empty(shape=(len(test_dataloader) * config.num_labels,), dtype=np.float32)

    # Iteracja po danych testowych i generowanie predykcji.
    for i, (embed, id) in tqdm(enumerate(test_dataloader)):
        embed = embed.to('cpu')
        confs_[i * config.num_labels:(i + 1) * config.num_labels] = torch.nn.functional.sigmoid(
            model(embed)).squeeze().detach().cpu().numpy()
        ids_[i * config.num_labels:(i + 1) * config.num_labels] = id[0]
        go_terms_[i * config.num_labels:(i + 1) * config.num_labels] = labels_names

    # Tworzenie końcowej ramki danych z predykcjami.
    submission_df = pd.DataFrame(data={"Id": ids_, "GO term": go_terms_, "Confidence": confs_})

    print("PREDICTIONS DONE")
    return submission_df


# %%
# T5 EMBEDDINGS
submission_t5 = predict("T5")

# %%

submission_t5.head(50)

# %%

submission_t5.to_csv('submission_t5.tsv', sep='\t', index=False)

# %%
# ProtBERT EMBEDDINGS

submission_protbert = predict("ProtBERT")

# %%

submission_protbert.head(50)

# %%

submission_protbert.to_csv('submission_protbert.tsv', sep='\t', index=False)

# %%
# EMS2 EMBEDDINGS

submission_ems2 = predict("EMS2")

# %%

submission_ems2.head(50)

# %%

submission_ems2.to_csv('submission_ems2.tsv', sep='\t', index=False)

# %%

print(submission_t5["Confidence"].mean())
print(submission_protbert["Confidence"].mean())
print(submission_ems2["Confidence"].mean())
