import torch
import pickle
import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ['ocean_proximity']
TARGET_COLUMN = 'median_house_value'

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 

        self.test_data = None
        self.test_target = None

        self.target_mean = 0
        self.target_std = 1

        self.model = None
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, df, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - df {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        

        for col in df.columns.difference(CATEGORICAL_COLUMNS):
            df[col] = df[col].fillna(df[col].mean())

        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)

        # convert to numpy array
        data = df.to_numpy().astype(np.float32)
        if y:
            target = y.to_numpy().astype(np.float32)
            # normalize target
            self.target_mean = target.mean()
            self.target_std = target.std()
            target = (target - self.target_mean) / self.target_std

        # normalize data
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        data = (data - data_mean) / data_std

        if training:
        # split into train and test
            np.random.seed(42)
            indices = np.random.permutation(data.shape[0])
            train_indices = indices[:int(0.8 * data.shape[0])]
            test_indices = indices[int(0.8 * data.shape[0]):]

            train_data = data[train_indices]
            if y:
                train_target = target[train_indices]
                self.test_target = target[test_indices]

            self.test_data = data[test_indices]

            return torch.tensor(train_data), torch.tensor(train_target).unsqueeze(1)

        else:
            return torch.tensor(data), None
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget4

        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.output_size)
        )

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(self.nb_epoch):
            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, Y)

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, loss {loss.item()}')

        self.model = model

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        if self.model is None:
            raise Exception("Model not trained")
        
        with torch.no_grad():
            return self.model(X).numpy() * self.target_std + self.target_mean
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        if self.model is None:
            raise Exception("Model not trained")
        
        with torch.no_grad():
            Y_pred = self.model(X).numpy() * self.target_std + self.target_mean
            return np.mean((Y_pred - self.test_target) ** 2)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    output_label = TARGET_COLUMN

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()

