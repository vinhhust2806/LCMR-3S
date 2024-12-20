import math
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

torch.manual_seed(28)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421356))
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(hidden_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value) 
        return context, attn
    
class Guide_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(Guide_Block, self).__init__()
        self.att = ScaledDotProductAttention(hidden_dim)
        self.linear_observer = nn.Linear(hidden_dim,hidden_dim)
        self.linear_matrix = nn.Linear(hidden_dim,hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query_ = query[:key.size(0),:,:] 
        context, att_weight = self.att(query_,key, value)
        new_query_vec = self.dropout(self.layer_norm(self.activation(self.linear_observer(context))))
        new_key_matrix = self.dropout(self.layer_norm(self.activation(self.linear_matrix(key))))
        
        return new_query_vec, new_key_matrix, att_weight
    
class GuidedAttention(nn.Module):
    def __init__(self, args, ctx_dim=None):
        super().__init__()

        self.args = args

        self.num_attention_heads = self.args.cross_encoder_args["n_heads"]
        self.attention_head_size = int(
            self.args.cross_encoder_args["embedding_size"]
            / self.args.cross_encoder_args["n_heads"]
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.args.cross_encoder_args["embedding_size"]

        self.query = nn.Linear(
            self.args.cross_encoder_args["embedding_size"], self.all_head_size
        )
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(0.1)
        self.guide = Guide_Block(self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)  
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        
        mixed_query_layer, mixed_key_layer, _ = self.guide(mixed_query_layer, mixed_key_layer, mixed_value_layer)
    
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)       
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
        
class Attention(nn.Module):
    def __init__(self, args, ctx_dim=None):
        super().__init__()

        self.args = args

        self.num_attention_heads = self.args.cross_encoder_args["n_heads"]
        self.attention_head_size = int(
            self.args.cross_encoder_args["embedding_size"]
            / self.args.cross_encoder_args["n_heads"]
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.args.cross_encoder_args["embedding_size"]

        self.query = nn.Linear(
            self.args.cross_encoder_args["embedding_size"], self.all_head_size
        )
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer1 = self.query(hidden_states)
        mixed_key_layer1 = self.key(context)
        mixed_value_layer1 = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer1)
        key_layer = self.transpose_for_scores(mixed_key_layer1)
        value_layer = self.transpose_for_scores(mixed_value_layer1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class AttOutput(nn.Module):
    def __init__(self, args):
        super(AttOutput, self).__init__()

        self.args = args

        self.dense = nn.Linear(
            self.args.cross_encoder_args["embedding_size"],
            self.args.cross_encoder_args["embedding_size"],
        )
        self.LayerNorm = nn.LayerNorm(
            self.args.cross_encoder_args["embedding_size"], eps=1e-6
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SelfGuidedAtt(nn.Module):
    def __init__(self, args):
        super(SelfGuidedAtt, self).__init__()

        self.args = args

        self.self = GuidedAttention(args)
        self.output = AttOutput(args)
        
    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
    
class CrossAtt(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.att = Attention(args)
        self.output = AttOutput(args)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()

        self.args = args

        self.dense = nn.Linear(
            self.args.cross_encoder_args["embedding_size"],
            4 * self.args.cross_encoder_args["embedding_size"],
        )
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, args):
        super(Output, self).__init__()
        self.args = args

        self.dense = nn.Linear(
            4 * self.args.cross_encoder_args["embedding_size"],
            self.args.cross_encoder_args["embedding_size"],
        )
        self.LayerNorm = nn.LayerNorm(
            self.args.cross_encoder_args["embedding_size"], eps=1e-6
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SOCML(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        # Self-Guided Attention Layers
        self.lang_self_att = SelfGuidedAtt(args)
        self.visn_self_att = SelfGuidedAtt(args)

        # Cross Attention Layer
        self.cross_attention = CrossAtt(args)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = Intermediate(args)
        self.lang_output = Output(args)
        self.visn_inter = Intermediate(args)
        self.visn_output = Output(args)
    
    def self_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output
    
    def cross_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        lang_att_output = self.cross_attention(
            lang_input, visn_input, ctx_att_mask=visn_attention_mask
        )
        visn_att_output = self.cross_attention(
            visn_input, lang_input, ctx_att_mask=lang_attention_mask
        )
        return lang_att_output, visn_att_output 

    def output_fc(self, lang_input, visn_input):
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )

        lang_att_output, visn_att_output = self.cross_att(   
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output
