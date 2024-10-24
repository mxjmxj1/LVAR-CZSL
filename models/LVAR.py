import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
from .word_embedding import load_word_embeddings
from .VARM import VARM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.multiprocessing.set_sharing_strategy('file_system')

class LVAR(nn.Module):
    def __init__(self, dataset, args):
        super(LVAR, self).__init__()
        self.args = args
        self.dataset = dataset
        self.val_forward = self.val_forward
        self.train_forward = self.train_forward

        self.num_attrs, self.num_objs, self.num_pairs = len(dataset.attrs), len(dataset.objs), len(dataset.pairs)
        self.pairs = dataset.pairs

        if self.args.train_only:
            train_idx = []
            for current in dataset.train_pairs:
                train_idx.append(dataset.all_pair2idx[current]+self.num_attrs+self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        '''====================== boj, attr, pair MLP-head ======================'''
        self.obj_head = MLP(dataset.feat_dim, args.emb_dim, num_layers=args.obj_fclayers, relu=args.relu,
                            dropout=args.dropout, norm=args.norm, layers=args.obj_emb)
        self.attr_head = MLP(dataset.feat_dim, args.emb_dim, num_layers=args.attr_fclayers, relu=args.relu,
                             dropout=args.dropout, norm=args.norm, layers=args.attr_emb)
        self.pair_head = MLP(dataset.feat_dim, args.emb_dim, num_layers=args.pair_fclayers, relu=args.relu,
                            dropout=args.dropout, norm=args.norm, layers=args.pair_emb)

        obj_words = list(dataset.objs)
        attr_words = list(dataset.attrs)

        self.obj_to_idx = {word: idx for idx, word in enumerate(dataset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(dataset.attrs)}

        obj_embeddings = load_word_embeddings(args.emb_init, obj_words).to(device)
        attr_embeddings = load_word_embeddings(args.emb_init, attr_words).to(device)
        pair_embeddings = self.get_compositional_embeddings(attr_embeddings, obj_embeddings, self.pairs)

        '''====================== pair_embedder ======================'''
        self.pairs_idx = torch.arange(len(self.pairs)).long().to(device)
        self.pairs_embedder = nn.Embedding(len(self.dataset.pairs), args.emb_dim).to(device)
        self.pairs_embedder.weight.data.copy_(pair_embeddings)

        self.obj_embeddings = obj_embeddings.unsqueeze(0)
        self.attr_embeddings = attr_embeddings.unsqueeze(0)

        '''====================== Encoder_attr_obj ======================'''
        '''Encoder-obj'''
        obj_encoder_layer = nn.TransformerEncoderLayer(d_model=args.emb_dim, nhead=args.obj_nhead)
        obj_transformer_encoder = nn.TransformerEncoder(obj_encoder_layer, num_layers=args.obj_nlayer)
        self.trans_obj = obj_transformer_encoder
        '''Encoder-attr'''
        attr_encoder_layer = nn.TransformerEncoderLayer(d_model=args.emb_dim, nhead=args.attr_nhead)
        attr_transformer_encoder = nn.TransformerEncoder(attr_encoder_layer, num_layers=args.attr_nlayer)
        self.trans_attr = attr_transformer_encoder

        '''====================== VARM_attr_obj ======================'''
        self.Attr_Extractor = VARM(512, 512).to(device)
        self.Obj_Extractor = VARM(512, 512).to(device)

    def compute_loss(self, preds, labels):
        loss = F.cross_entropy(preds, labels)
        return loss

    def get_compositional_embeddings(self, embeddings_attr, embeddings_obj, pairs):
        # Getting compositional embeddings from base embeddings
        composition_embeds = []
        for (attr, obj) in pairs:
            attr_embed = embeddings_attr[self.attr_to_idx[attr]]
            obj_embed = embeddings_obj[self.obj_to_idx[obj]]
            composed_embed = (attr_embed + obj_embed) / 2
            composition_embeds.append(composed_embed)
        composition_embeds = torch.stack(composition_embeds)
        return composition_embeds

    def train_forward(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_attr = img
        img_obj = img
        img = F.adaptive_avg_pool2d(img, output_size=1).flatten(1)
        attr_feats = self.Attr_Extractor(img_attr)
        attr_feats = self.attr_head(attr_feats)
        obj_feats = self.Obj_Extractor(img_obj)
        obj_feats = self.obj_head(obj_feats)
        pair_feats = self.pair_head(img)

        obj_embeddings = self.trans_obj(self.obj_embeddings).squeeze(0)
        attr_embeddings = self.trans_attr(self.attr_embeddings).squeeze(0)
        compose_embeddings = self.pairs_embedder(self.pairs_idx)
        pair_embeddings = torch.cat([attr_embeddings, obj_embeddings, compose_embeddings], dim=0)

        obj_embed = obj_embeddings.permute(1, 0)
        attr_embed = attr_embeddings.permute(1, 0)
        pair_embed = pair_embeddings[self.train_idx].permute(1, 0)

        obj_pred = torch.matmul(obj_feats, obj_embed)
        attr_pred = torch.matmul(attr_feats, attr_embed)
        pair_pred = torch.matmul(pair_feats, pair_embed)

        loss_obj = self.compute_loss(obj_pred, objs)
        loss_attr = self.compute_loss(attr_pred, attrs)
        loss_pair = self.compute_loss(pair_pred, pairs)
        loss = loss_pair * self.args.beta + (loss_obj + loss_attr) * (1-self.args.beta)
        return loss, None

    def val_forward(self, x):
        img = x[0]

        img_attr = img
        img_obj = img
        img = F.adaptive_avg_pool2d(img, output_size=1).flatten(1)
        attr_feats = self.Attr_Extractor(img_attr)
        attr_feats = self.attr_head(attr_feats)
        obj_feats = self.Obj_Extractor(img_obj)
        obj_feats = self.obj_head(obj_feats)

        pair_feats = self.pair_head(img)

        obj_embeddings = self.trans_obj(self.obj_embeddings).squeeze(0)
        attr_embeddings = self.trans_attr(self.attr_embeddings).squeeze(0)
        compose_embeddings = self.pairs_embedder(self.pairs_idx)
        pair_embeddings = torch.cat([attr_embeddings, obj_embeddings, compose_embeddings], dim=0)

        obj_embed = obj_embeddings.permute(1, 0)
        attr_embed = attr_embeddings.permute(1, 0)
        pair_embed = pair_embeddings[
                      self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :].permute(1, 0)

        score_obj = torch.matmul(obj_feats, obj_embed)
        score_attr = torch.matmul(attr_feats, attr_embed)
        score_pair = torch.matmul(pair_feats, pair_embed)

        scores = {}
        for _, pair in enumerate(self.dataset.pairs):
            attr, obj = pair
            scores[pair] = score_pair[:, self.dataset.all_pair2idx[pair]] * self.args.alpha + (score_attr[:, self.dataset.attr2idx[attr]] + score_obj[:,
                                                                                                             self.dataset.obj2idx[
                                                                                                                 obj]]) * (1 - self.args.alpha)
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred
