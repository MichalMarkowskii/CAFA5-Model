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
    n_epochs = 6
    batch_size = 128 # liczba sekwencji na 1 batch
    lr = 0.001 # learning rate