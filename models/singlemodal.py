import torch
import torch.nn as nn

class MambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, state_selection_threshold=0.):
        super(MambaModel, self).__init__()
        self.hidden_size = hidden_size
        self.state_selection_threshold = state_selection_threshold
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        mask = (hidden > self.state_selection_threshold).float()
        hidden = hidden * mask
        output = self.h2o(hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.rand(batch_size, self.hidden_size)
    
class SingleModal(torch.nn.Module):
    def __init__(self, args):
        super(SingleModal, self).__init__()
        self.args = args

        if self.args.modality == "text":
            modality_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
                self.args.text_embeddings_type
            ]

        if self.args.modality == "image":
            modality_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
                self.args.image_embeddings_type
            ]

        self.modality_projection = torch.nn.Linear(
            modality_embedding_size, self.args.final_encoder_args["embedding_size"]
        )

        self.final_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                self.args.final_encoder_args["embedding_size"],
                self.args.final_encoder_args["n_heads"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=self.args.final_encoder_args["dropout_prob"],
                dim_feedforward=4 * self.args.final_encoder_args["embedding_size"],
            ),
            self.args.final_encoder_args["n_layers"],
        )

        self.output_classification = torch.nn.Linear(
            self.args.final_encoder_args["embedding_size"], 1
        )
        
        self.mamba = MambaModel(self.args.final_encoder_args["embedding_size"], self.args.final_encoder_args["embedding_size"], self.args.final_encoder_args["embedding_size"])

    def _g(self, times):
        return 1 / (times + 1.0)

    def forward(self, batch):
        modality_feats = self.modality_projection(batch["modality"])

        position_embeddings = torch.zeros_like(modality_feats)

        modality_feats = modality_feats + position_embeddings

        final_vector = self.final_transformer(modality_feats)
        hidden = self.mamba.init_hidden(final_vector.size(0)).cuda()
        
        for i in range(final_vector.size(1)):
            output, hidden = self.mamba(final_vector[:,i], hidden)

        output = self.output_classification(output)
        output_proba = torch.sigmoid(output)

        return {"logits": output, "probas": output_proba}