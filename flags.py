import argparse

DATA_FOLDER = ""

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/args.yml', help='path of the config file (training only)')
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='mit-states', help='local path to data root dir from ' + DATA_FOLDER)
parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
parser.add_argument('--name', default='temp', help='Name of exp used to name models')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--image_extractor', default='resnet18', help='Feature extractor model')
parser.add_argument('--norm_family', default='imagenet', help='Normalization values from dataset')
parser.add_argument('--num_negs', type=int, default=1, help='Number of negatives to sample per positive (triplet loss)')
parser.add_argument('--pair_dropout', type=float, default=0.0, help='Each epoch drop this fraction of train pairs')
parser.add_argument('--test_set', default='val', help='val|test mode')
parser.add_argument('--clean_only', action='store_true', default=False,
                    help='use only clean subset of data (mitstates)')
parser.add_argument('--subset', action='store_true', default=False, help='test on a 1000 image subset (debug purpose)')
parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
parser.add_argument('--seed', type=int, default=2048)

# Model parameters
parser.add_argument('--model', default='graphfull', help='visprodNN|redwine|labelembed+|attributeop|tmn|compcos')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of share embedding space')

parser.add_argument('--nmods', type=int, default=24, help='number of mods per layer for TMN')
parser.add_argument('--embed_rank', type=int, default=64, help='intermediate dimension in the gating model for TMN')
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument('--update_features', action='store_true', default=False,
                    help='If specified, train feature extractor')
parser.add_argument('--freeze_features', action='store_true', default=False,
                    help='If specified, put extractor in eval mode')
parser.add_argument('--emb_init', default=None,
                    help='w2v|ft|gl|glove|word2vec|fasttext, name of embeddings to use for initializing the primitives')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False,
                    help='do not optimize primitives representations')
parser.add_argument('--composition', default='mlp_add', help='add|mul|mlp|mlp_add, how to compose primitives')
parser.add_argument('--relu', action='store_true', default=False, help='Use relu in image embedder')
parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout in image embedder')
parser.add_argument('--norm', action='store_true', default=False, help='Use normalization in image embedder')
parser.add_argument('--train_only', action='store_true', default=False, help='Optimize only for train pairs')
parser.add_argument('--obj_nlayer', type=int, default=6, help=' the number of sub-encoder-layers in the encoder.')
parser.add_argument('--obj_nhead', type=int, default=6, help='')
parser.add_argument('--attr_nlayer', type=int, default=6, help=' the number of sub-encoder-layers in the encoder.')
parser.add_argument('--attr_nhead', type=int, default=6, help='')
parser.add_argument("--obj_emb", default='768,1024,1200', help="Image embedder layer config")
parser.add_argument("--attr_emb", default='768,1024,1200', help="Image embedder layer config")
parser.add_argument("--pair_emb", default='768,1024,1200', help="Image embedder layer config")
parser.add_argument('--obj_fclayers', type=int, default=3, help='Layers in the image embedder')
parser.add_argument('--attr_fclayers', type=int, default=3, help='Layers in the image embedder')
parser.add_argument('--pair_fclayers', type=int, default=3, help='Layers in the image embedder')

# Forward
parser.add_argument('--eval_type', default='dist', help='dist|prod|direct, function for computing the predictions')

# Hyperparameters
parser.add_argument('--topk', type=int, default=1, help="Compute topk accuracy")
parser.add_argument('--margin', type=float, default=2, help="Margin for triplet loss or feasibility scores in CompCos")
parser.add_argument('--workers', type=int, default=8, help="Number of workers")
parser.add_argument('--batch_size', type=int, default=512, help="Training batch size")
parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
parser.add_argument('--lrg', type=float, default=1e-3, help="Learning rate feature extractor")
parser.add_argument('--wd', type=float, default=5e-5, help="Weight decay")
parser.add_argument('--save_every', type=int, default=10000, help="Frequency of snapshots in epochs")
parser.add_argument('--eval_val_every', type=int, default=1, help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=800, help="Max number of epochs")
parser.add_argument('--alpha', type=float, default=0.5, help="")
parser.add_argument('--beta', type=float, default=0.5, help="")
