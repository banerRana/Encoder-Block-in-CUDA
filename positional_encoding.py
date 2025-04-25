#positional_encoding.py
import numpy as np 

#One thing to note about dimensionality of word embeddings: the dimensionality represents the total number of features that are encoded in the vector representation. 

def positional_encoding (seq_len, d_model):
    """
    Create sinusoidal positional embeddings to be added to token embeddings

    Args:
        seq_len (int): length of the sequence
        d_model (int): embedding dimension

    Returns:
        np.ndarray of shape (seq_len, d_model)
    """
    #creating an empty grid (a matrix) with all zeros. Each row will be for 1 word and each column is 1 part of its position pattern
    PE = np.zeros ((seq_len, d_model))

    #looping through every word position (0, 1, 2...)
    for pos in range (seq_len):

        #looping through every even-numbered column in the vector (sine for even columns)
        for i in range (0, d_model, 2):

            #As the position of the embedding gets bigger, the frequency of the waves changes more slowly (longer waves, slower oscillations). The "10000^(1/d_model)" scales the frequency logarithmically. +
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i] = np.sin(angle)

            #for the odd column - the cosine wave is used (done if it's within the bounds of the vector size)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(angle)

        #return the full positioning encoding matrix
    return PE.astype(np.float32)

##Helper function to apply the positional encoding
def apply_positional_encoding(X):
    """
    Add positional encoding directly to the input matrix
    Assumes X is of shape (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    return X + positional_encoding (seq_len, d_model)

if __name__ == "__main__":
    test = np.zeros((4, 8))  # dummy embeddings (4 words, 8-sized vector)
    print("Original Input:")
    print(test)

    pe = positional_encoding(4, 8)
    print("\nPositional Encoding:")
    print(pe)

    combined = apply_positional_encoding(test)
    print("\nCombined:")
    print(combined)

    
    