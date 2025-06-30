"""
Example usage for SGFCMedParallel
Author: Atcharin Klomsae
Date: 23/4/2020

This script demonstrates how to use SGFCMedParallel to cluster strings,
print prototypes, membership matrix, and predict cluster of new strings.
"""

import random
from sgfcmed_parallel import SGFCMedParallel  # Import the clustering class

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42) # จะทำให้ผลลัพธ์การสุ่ม เหมือนเดิมทุกครั้งที่รัน โปรแกรม

    # Define a list of strings to cluster
    data = ["book", "back", "boon", "cook", "look", "cool", "kick", "lack", "rack", "tack"]

    # Create the model with 2 clusters and fuzzifier m=2.0
    model = SGFCMedParallel(C=2, m=2.0) # สร้าง object

    # Fit the model on the data
    model.fit(data)

    # Print the final prototype strings representing each cluster
    print("Prototypes:", model.prototypes())

    # Print the fuzzy membership matrix for each input string
    print("\nMembership Matrix:")
    for s, u in zip(data, model.membership()):
        # Format each string with its membership values
        print(f"{s:>6} → {[round(val, 3) for val in u]}")

    # Define new strings to classify using the trained model
    new_data = ["hack", "rook", "cook"]

    # Predict the cluster index (0 or 1) for each new string
    pred = model.predict(new_data)

    # Display predictions in a readable format
    print("\nPredictions:")
    for s, c in zip(new_data, pred):
        print(f"{s} → Cluster {c+1}")
