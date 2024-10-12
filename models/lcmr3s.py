import torch
import torch.nn as nn
from models.time2vec import Time2Vec
from models.layers.attention import SOCML

torch.manual_seed(28)

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
    
class LCMR3S(torch.nn.Module):
    def __init__(self, args):
        super(LCMR3S, self).__init__()
        self.args = args

        image_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
            self.args.image_embeddings_type
        ]
        text_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
            self.args.text_embeddings_type
        ]
        
        if self.args.position_embeddings == "time2vec":
            self.position_embeddings = Time2Vec(args)
        elif self.args.position_embeddings == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(
                    1,
                    self.args.window_size,
                    self.args.cross_encoder_args["embedding_size"],
                )
            )

        self.image_projection = torch.nn.Linear(
            image_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )
        self.text_projection = torch.nn.Linear(
            text_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )

        self.layers = torch.nn.ModuleList(
            [SOCML(args) for _ in range(self.args.cross_encoder_args["n_layers"])]
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
    
    def forward(self, batch):
        extended_image_attention_mask = (1.0 - batch["image_mask"]) * -10000.0
        extended_text_attention_mask = (1.0 - batch["text_mask"]) * -10000.0

        all_lang_feats = self.text_projection(batch["text_embeddings"])      
        all_visn_feats = self.image_projection(batch["image_embeddings"])       
        
        if self.args.position_embeddings == "time2vec":
            position_embeddings = self.position_embeddings[batch["time"]]
        elif self.args.position_embeddings == "learned":
            position_embeddings = self.position_embeddings
        else:
            position_embeddings = torch.zeros_like(all_lang_feats)

        lang_feats = all_lang_feats + position_embeddings
        visn_feats = all_visn_feats + position_embeddings

        for layer_module in self.layers:
            lang_feats, visn_feats = layer_module(
                lang_feats,
                extended_text_attention_mask,
                visn_feats,
                extended_image_attention_mask,
            )
        
        cross_modal_vectors = lang_feats 
        final_vector = self.final_transformer(cross_modal_vectors)
        hidden = self.mamba.init_hidden(final_vector.size(0)).cuda()

        for i in range(final_vector.size(1)):
            output, hidden = self.mamba(final_vector[:,i], hidden)
        
        ssm = output
        output = self.output_classification(output)     
        output_proba = torch.sigmoid(output)

        return {"logits": output, "probas": output_proba, "cross": final_vector, "ssm": ssm}
