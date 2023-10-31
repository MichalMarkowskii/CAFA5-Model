import pandas as pd
import numpy as np
from tqdm import tqdm

from model import config

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