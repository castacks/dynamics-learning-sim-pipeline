import torch
import torch.nn as nn

class BaselineSequenceModel(nn.Module):
    def __init__(self, architecture_params, features_schema, norm_stats):
        super().__init__()

        self.state_dim = len(features_schema["observable_state"])
        self.action_dim = len(features_schema["actions"])
        self.target_dim = len(features_schema["targets"])

        encoder_hidden_dim = architecture_params.get("encoder_hidden_dim", 64)
        decoder_hidden_dim = architecture_params.get("decoder_hidden_dim", 64)
        
        self.encoder = nn.LSTM(
            input_size=self.state_dim + self.action_dim,
            hidden_size=encoder_hidden_dim,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, self.target_dim)
        )

    def forward(self, X):
        """_summary_: TODO

        Args:
            X (torch.tensor shape: [batch, sequence_len, feature_dim]): Expects normalized inputs
        """
        inputs = self.__extract_input_features(X)
        output, _ = self.encoder(inputs)
        final_output = output[:, -1, :]
        return self.decoder(final_output)
        
    def __extract_input_features(self, X):
        """Takes the large feature matrix and extracts only the features intended for the model

        Args:
            X (np.ndarray, shape:[batch, sequence_len, feature_dim]): 
                Normalized feature matrix. May include more state information than necessary
        """
        
        # This model takes observable states and actions as inputs
        return X[:,:,:self.state_dim + self.action_dim]