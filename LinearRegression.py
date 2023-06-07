#!/usr/bin/env python3

# Standard Packages
import csv

# Machine Learning Packages
from neolinearalgebra import Matrix


def load_data(filename, delimiter=',', header=False) -> tuple[Matrix, list]:
    '''Load a dataset stored in .csv format.

    Args:
        #filename (str): Filename of stored dataset
        #delimiter (str): Demiliting character (Default=',')
        #header (bool): ... (Default=False)

    Returns:
        #object: Matrix containing loaded dataset.
        #list: List containing feature headers.
    '''
    output, features = [], []

    with open(filename) as file:
        file_reader = csv.reader(file, delimiter=delimiter, quotechar='|')

        for observation in file_reader:
            if header == True:
                features.extend(observation)
                header = False
            else:
                output.append([observation])

    return Matrix(output), features


def save_weights(filename, weights) -> None:
    '''

    Args:
        #filename (str):
        #weights (object):
    
    Returns:
        None
    '''
    pass


class LinearRegression:
    def __init__(self, weights: dict[str, Matrix]) -> None:
        '''#
        
        Args:
            #
        
        Returns:
            None
        '''
        self.weights = weights

    def forward_pass(X_batch: Matrix, y_batch: Matrix) -> tuple[float, dict[str, Matrix]]:
        '''
        '''
        assert X_batch.shape[0] == y_batch.shape[0]
        
        assert X_batch.shape[1] == self.weights['W'].shape[0]
        
        assert self.weights['B'].shape[0] == self.weights['B'].shape[1] == 1
        
        N = X_batch @ self.weights['W']
        
        P = N + self.weights['B']
        
        loss = ((y_batch - P) * (y_batch - P)).mean()
        
        forward_info: Dict[str, Matrix] = {}
        forward_info['X'] = X_batch
        forward_info['N'] = N
        forward_info['P'] = P
        forward_info['y'] = y_batch
        
        return loss, forward_info
        
    def loss_gradients(forward_info: dict[str, Matrix]) -> dict[str, Matrix]:
        batch_size = forward_info['X'].shape[0]
        
        dLdP = -2 * (forward_info['y'] - forward_info['P'])
        
        dPdN = Matrix((forward_info['N'].shape[0], forward_info['N'].shape[1], 1))
        
        dPdB = Matrix((self.weights['B'].shape[0], self.weights['B'].shape[1], 1))
        
        dLdN = dLdP * dPdN
        
        dNdW = forward_info['X'].transpose()
        
        dLdW = dNdW @ dLdN
        
        dLdB = (dLdP * dPdB).sum(axis=0)
        
        loss_gradients: Dict[str, Matrix] = {}
        loss_gradients['W'] = dLdW
        loss_gradients['B'] = dLdB
        
        return loss_gradients
        
    #def train() -> :
        '''
        '''

X, features = load_data('data/dataset.csv', header=True)
