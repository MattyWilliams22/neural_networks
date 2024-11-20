import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

CATEGORICAL_COLUMNS = ['ocean_proximity']
TARGET_COLUMN = 'median_house_value'

class Regressor():

    def __init__(
        self, 
        x, 
        nb_epoch=1000,
        learning_rate=0.001,
        batch_size=16,
        loss_fn=nn.MSELoss(),
        hidden_layers=[1024, 1024, 1024],
        activations=[nn.ReLU()] * 3
        ):
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

        # Scalers & encoder used when preprocessing
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.encoder = OneHotEncoder()

        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.column_names = []

        # Create model
        layers = []
        previous_layer = self.input_size
        for neurons, activation in zip(hidden_layers, activations):
            layers.append(nn.Linear(previous_layer, neurons))
            layers.append(activation)
            previous_layer = neurons
        
        layers.append(nn.Linear(hidden_layers[-1], self.output_size))

        self.model = nn.Sequential(*layers)


        # Create Optimiser
        self.optimiser = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate
            )


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
        
        # Fill missing values
        for col in df.columns.difference(CATEGORICAL_COLUMNS):
            df[col] = df[col].fillna(df[col].mean())
        
        # Get textual columns
        textual_cols = df.select_dtypes(include=["object"]).columns
        df_textual = df[textual_cols]

        
        # Get numerical columns
        numerial_cols = df.select_dtypes(include=[np.number]).columns
        df_numerical = df[numerial_cols]

        if training:
            # One-hot encode textual values
            df_textual = self.encoder.fit_transform(df_textual).toarray()

            # Normalise numerical operations
            df_numerical = self.input_scaler.fit_transform(df_numerical)

            # Preprocess (normalise) target dataset
            if y is not None:
                y = pd.DataFrame(self.output_scaler.fit_transform(y), 
                columns=y.columns)
        
        else:
            # One-hot encode textual values
            df_textual = self.encoder.transform(df_textual).toarray()

            # Normalise numerical operations
            df_numerical = self.input_scaler.transform(df_numerical)
        
        self.column_names = df.columns.tolist()
        df = pd.DataFrame(
            data = np.concatenate([df_numerical, df_textual], axis=1),
            columns = numerial_cols.tolist()
            + self.encoder.get_feature_names_out(textual_cols).tolist()
        )

        self.column_names = df.columns.tolist()

        df_preprocessed = torch.tensor(df.values, dtype=torch.float32)
        if y is not None:
            return (df_preprocessed, torch.tensor(y.values, dtype=torch.float32))
        else:
            return (df_preprocessed, None)

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

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        for epoch in range(self.nb_epoch):
            # We decided to shuffle the data and split into batches
            idxs = torch.randperm(X.size(0))
            inp_batches = torch.split(X[idxs], self.batch_size)
            out_batches = torch.split(Y[idxs], self.batch_size)

            for inp_batch, out_batch in zip(inp_batches, out_batches):
                # Zero gradients
                self.optimiser.zero_grad()

                # Forward pass
                y_pred = self.model(inp_batch)

                # Compute loss
                loss = self.loss_fn(y_pred, out_batch)

                # Backward pass
                loss.backward()

                # Optimize (gradient descent step)
                self.optimiser.step()

            # if epoch % 100 == 0:
            #     print(f'Epoch {epoch}, loss {loss.item()}')
            print(f'Epoch {epoch}, loss {loss.item()}')


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
            return self.output_scaler.inverse_transform(self.model(X).detach().numpy())
        

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
            Y_pred = self.predict(X)
            return mean_squared_error(Y_pred, Y.values, squared=False)

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



def perform_hyperparameter_search(x_train, y_train, x_test, y_test): 
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
    import concurrent.futures

    # Define hyper-parameter values to try
    learning_rates = [0.001, 0.01, 0.1]
    epochs = [200, 300, 400]

    def train_with_params(params):
        lr, nb_epoch = params
        regressor = Regressor(
            x_train,
            nb_epoch = nb_epoch,
            learning_rate = lr,
            batch_size = 8,
            loss_fn = nn.MSELoss(),
            hidden_layers = [1024, 1024, 1024],
            activations = [nn.ReLU()] * 3
        )
        regressor.fit(x_train, y_train)
        error = regressor.score(x_test, y_test)
        return lr, nb_epoch, error

    # Try all param combos
    param_combs = [(lr, nb_epoch) for lr in learning_rates for nb_epoch in epochs]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(train_with_params, param_combs))

    optimal_params = None

    lowest_error = float(inf)
    for lr, nb_epoch, error in results:
        if error < lowest_error:
            lowest_error = error
            optimal_params = lr, nb_epoch

    return optimal_params

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    output_label = TARGET_COLUMN

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    train, test = train_test_split(data, test_size=0.2)

    # Splitting input and output
    x_train = train.loc[:, train.columns != output_label]
    y_train = train.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(
        x_train, 
        learning_rate=0.1,
        nb_epoch=300,
        batch_size=8,
        hidden_layers=[1024, 1024, 1024],
        activations = [nn.ReLU()] * 3)

    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()



    # output_label = TARGET_COLUMN

    # # Use pandas to read CSV data as it contains various object types
    # # Feel free to use another CSV reader tool
    # # But remember that LabTS tests take Pandas DataFrame as inputs
    # data = pd.read_csv("housing.csv")

    # train, test = train_test_split(data, test_size=0.2)

    # # Splitting input and output
    # x_train = train.loc[:, train.columns != output_label]
    # y_train = train.loc[:, [output_label]]


    # x_test = test.loc[:, test.columns != output_label]
    # y_test = test.loc[:, [output_label]]

    # print(perform_hyperparameter_search(x_train, y_train, x_test, y_test))

