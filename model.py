# Biblioteki
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from CNN1D import CNN1D
from MultiLayerPerceptron import MultiLayerPerceptron
from ProteinSequenceDataset import ProteinSequenceDataset
from TransformerClassifier import TransformerClassifier

# Moduły torcha:
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
from torchmetrics.classification import MultilabelAccuracy

from config import config

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

def train_model(embeddings_source, model_type="linear", train_size=0.9):
    """
    Trenuje model na podstawie dostarczonych embeddingów i typu modelu.

    Parameters:
    ----------
    embeddings_source : str
        Źródło embeddingów białek; "ProtBERT", "EMS2" lub "T5".
    model_type : str, opcjonalnie
        Typ modelu do wytrenowania; "linear" (liniowy), "convolutional" (konwolucyjny) lub "transformer" (transformer).
    train_size : float, opcjonalnie
        Stosunek danych do treningu do walidacji.

    Returns:
    -------
    model : torch.nn.Module
        Wytrenowany model.
    loss_history : tuple
        Krotka zawierająca historię strat w trakcie treningu i walidacji.
    f1_history : tuple
        Krotka zawierająca historię wyników F1 w trakcie treningu i walidacji.
    precision_history : tuple
        Krotka zawierająca historię precyzji w trakcie treningu i walidacji.
    recall_history : tuple
        Krotka zawierająca historię czułości w trakcie treningu i walidacji.
    accuracy_history : tuple
        Krotka zawierająca historię dokładności w trakcie treningu i walidacji.
    """

    # Wczytanie danych treningowych i podział na zbiory treningowe i walidacyjne
    train_dataset = ProteinSequenceDataset(datatype="train", embeddings_source=embeddings_source)
    train_set, val_set = random_split(train_dataset, [int(len(train_dataset) * train_size),
                                                      len(train_dataset) - int(len(train_dataset) * train_size)])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    # Definicja modelu na podstawie wybranego typu
    if model_type == "transformer":
        model = TransformerClassifier(num_tokens=1000, dim_model=embeds_dim[embeddings_source], num_heads=8, num_encoder_layers=6, num_decoder_layers=6)
    elif model_type == "linear":
        model = MultiLayerPerceptron(input_dim=embeds_dim[embeddings_source], num_classes=config.num_labels).to('cpu')
    elif model_type == "convolutional":
        model = CNN1D(input_dim=embeds_dim[embeddings_source], num_classes=config.num_labels).to('cpu')

    # Inicjalizacja optymalizatora, schedulera, funkcji straty i metryk
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    f1_score = MultilabelF1Score(num_labels=config.num_labels).to('cpu')
    precision_score = MultilabelPrecision(num_labels=config.num_labels).to('cpu')
    recall_score = MultilabelRecall(num_labels=config.num_labels).to('cpu')
    accuracy_score = MultilabelAccuracy(num_labels=config.num_labels).to('cpu')

    # Historie metryk
    train_loss_history, val_loss_history = [], []
    train_f1_history, val_f1_history = [], []
    train_precision_history, val_precision_history = [], []
    train_recall_history, val_recall_history = [], []
    train_accuracy_history, val_accuracy_history = [], []

    print("BEGIN TRAINING...")
    for epoch in range(config.n_epochs):
        print("EPOCH ", epoch + 1)

        # Faza treningu
        model.train()
        for embed, targets in tqdm(train_dataloader):
            embed, targets = embed.to('cpu'), targets.to('cpu')
            optimizer.zero_grad()
            preds = model(embed)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

        # Aktualizacja metryk treningowych
        train_loss_history.append(loss.item())
        train_f1_history.append(f1_score(preds, targets).item())
        train_precision_history.append(precision_score(preds, targets).item())
        train_recall_history.append(recall_score(preds, targets).item())
        train_accuracy_history.append(accuracy_score(preds, targets).item())

        # Faza walidacji
        model.eval()
        with torch.no_grad():
            for embed, targets in val_dataloader:
                embed, targets = embed.to('cpu'), targets.to('cpu')
                preds = model(embed)
                loss = loss_fn(preds, targets)

            # Aktualizacja metryk walidacyjnych
            val_loss_history.append(loss.item())
            val_f1_history.append(f1_score(preds, targets).item())
            val_precision_history.append(precision_score(preds, targets).item())
            val_recall_history.append(recall_score(preds, targets).item())
            val_accuracy_history.append(accuracy_score(preds, targets).item())

        scheduler.step(np.mean(val_loss_history[-len(val_dataloader):]))
        print("\n")

    print("TRAINING FINISHED")

    # Zwracanie historii metryk
    loss_history = {"train": train_loss_history, "val": val_loss_history}
    f1_history = {"train": train_f1_history, "val": val_f1_history}
    precision_history = {"train": train_precision_history, "val": val_precision_history}
    recall_history = {"train": train_recall_history, "val": val_recall_history}
    accuracy_history = {"train": train_accuracy_history, "val": val_accuracy_history}

    return model, loss_history, f1_history, precision_history, recall_history, accuracy_history


# %%

# Training the model with EMS2 embeddings
ems2_model, ems2_loss_history, ems2_f1_history, ems2_precision_history, ems2_recall_history, ems2_accuracy_history = train_model(
    embeddings_source="EMS2", model_type="linear")

# %%

# Training the model with T5 embeddings
t5_model, t5_loss_history, t5_f1_history, t5_precision_history, t5_recall_history, t5_accuracy_history = train_model(
    embeddings_source="T5", model_type="linear")

# %%

# Training the model with ProtBERT embeddings
protbert_model, protbert_loss_history, protbert_f1_history, protbert_precision_history, protbert_recall_history, protbert_accuracy_history = train_model(
    embeddings_source="ProtBERT", model_type="linear")


# %%

# Wizualizacja wyników dla każdego rodzaju embeddingów w kontekście straty walidacyjnej
plt.figure(figsize=(10, 4))
plt.plot(ems2_loss_history["val"], label="EMS2")
plt.plot(t5_loss_history["val"], label="T5")
plt.plot(protbert_loss_history["val"], label="ProtBERT")
plt.title("Validation Losses for # Vector Embeddings")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.show()

# Wizualizacja wyników dla każdego rodzaju embeddingów w kontekście średniego wyniku F1 walidacyjnego
plt.figure(figsize=(10, 4))
plt.plot(ems2_f1_history["val"], label="EMS2")
plt.plot(t5_f1_history["val"], label="T5")
plt.plot(protbert_f1_history["val"], label="ProtBERT")
plt.title("Validation F1-Scores for # Vector Embeddings")
plt.xlabel("Epochs")
plt.ylabel("Average F1-Score")
plt.legend()
plt.show()

# Precyzja
plt.figure(figsize=(10, 4))
plt.plot(ems2_precision_history["val"], label="EMS2")
plt.plot(t5_precision_history["val"], label="T5")
plt.plot(protbert_precision_history["val"], label="ProtBERT")
plt.title("Validation Precision for Different Vector Embeddings")
plt.xlabel("Epochs")
plt.ylabel("Average Precision")
plt.legend()
plt.show()

# Czułość
plt.figure(figsize=(10, 4))
plt.plot(ems2_recall_history["val"], label="EMS2")
plt.plot(t5_recall_history["val"], label="T5")
plt.plot(protbert_recall_history["val"], label="ProtBERT")
plt.title("Validation Recall for Different Vector Embeddings")
plt.xlabel("Epochs")
plt.ylabel("Average Recall")
plt.legend()
plt.show()

# Dokładność
plt.figure(figsize=(10, 4))
plt.plot(ems2_accuracy_history["val"], label="EMS2")
plt.plot(t5_accuracy_history["val"], label="T5")
plt.plot(protbert_accuracy_history["val"], label="ProtBERT")
plt.title("Validation Accuracy for Different Vector Embeddings")
plt.xlabel("Epochs")
plt.ylabel("Average Accuracy")
plt.legend()
plt.show()

# %%
def calculate_and_print_average_metrics(metric_history):
    if 'val' in metric_history:
        avg_value = sum(metric_history['val']) / len(metric_history['val']) if metric_history['val'] else 0
        print(f"(Validation): {avg_value:.4f}")

# Example usage with each metric history dictionary
print("EMS-2\nAverage F1:")
calculate_and_print_average_metrics(ems2_f1_history)
print("Average Loss:")
calculate_and_print_average_metrics(ems2_loss_history)
print("Average Precision:")
calculate_and_print_average_metrics(ems2_precision_history)
print("Average Accuracy:")
calculate_and_print_average_metrics(ems2_accuracy_history)
print("Average Recall:")
calculate_and_print_average_metrics(ems2_recall_history)

print("\nT5\nAverage F1:")
calculate_and_print_average_metrics(t5_f1_history)
print("Average Loss:")
calculate_and_print_average_metrics(t5_loss_history)
print("Average Precision:")
calculate_and_print_average_metrics(t5_precision_history)
print("Average Accuracy:")
calculate_and_print_average_metrics(t5_accuracy_history)
print("Average Recall:")
calculate_and_print_average_metrics(t5_recall_history)

print("\nProtBERT\nAverage F1:")
calculate_and_print_average_metrics(protbert_f1_history)
print("Average Loss:")
calculate_and_print_average_metrics(protbert_loss_history)
print("Average Precision:")
calculate_and_print_average_metrics(protbert_precision_history)
print("Average Accuracy:")
calculate_and_print_average_metrics(protbert_accuracy_history)
print("Average Recall:")
calculate_and_print_average_metrics(protbert_recall_history)

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
#print(submission_ems2)
