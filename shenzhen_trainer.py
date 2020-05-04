from trajnetbaselines import lstm
from trajnetbaselines.lstm import trainer
import argparse
import logging
import socket
import sys
import time
import random
import os
import torch
import trajnettools

from trajnetbaselines import augmentation
from trajnetbaselines.lstm.loss import PredictionLoss, L2Loss
from trajnetbaselines.lstm.lstm import LSTM, LSTMPredictor, drop_distant
from trajnetbaselines.lstm.pooling import Pooling, HiddenStateMLPPooling, FastPooling
from trajnetbaselines import __version__ as VERSION


class TrainerSZN(trainer.Trainer):
    def prepare_data(self):
        pass
    pass


def main(epochs=50, prepare_data=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--prepare_data', default=prepare_data, type=bool)
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp',
                                 'directionalmlp'),
                        help='type of LSTM to train')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling')
    parser.add_argument('--fast', action='store_true',
                        help='Fast pooling (Under devpt)')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--loss', default='L2',
                        help='loss function')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=1.0,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=10,
                                 help='number of cells per side')

    args = parser.parse_args()

    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.output:
        args.output = 'OUTPUT_BLOCK/{}/{}_{}.pkl'.format(args.path, args.type, args.output)
    else:
        args.output = 'OUTPUT_BLOCK/{}/{}.pkl'.format(args.path, args.type)

    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    # read in datasets
    args.path = 'DATA_BLOCK/' + args.path

    train_scenes = list(trajnettools.load_all(args.path + '/train/**/*.ndjson'))
    val_scenes = list(trajnettools.load_all(args.path + '/val/**/*.ndjson'))

    # create model
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim)
    elif args.type != 'vanilla':
        if args.fast:
            pool = FastPooling(type_=args.type, hidden_dim=args.hidden_dim,
                               cell_side=args.cell_side, n=args.n, front=args.front)
        else:
            pool = Pooling(type_=args.type, hidden_dim=args.hidden_dim,
                           cell_side=args.cell_side, n=args.n, front=args.front)
    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim)
    # Default Load
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # 1e-4
    lr_scheduler = None
    start_epoch = 0

    # train
    if args.load_state:
        # load pretrained model.
        # useful for tranfer learning
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        if args.load_full_state:
        # load optimizers from last training
        # useful to continue training
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # , weight_decay=1e-4
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    #trainer
    trainer_szn = TrainerSZN(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      criterion=args.loss, batch_size=args.batch_size, obs_length=args.obs_length,
                      pred_length=args.pred_length)
    trainer_szn.loop(train_scenes, val_scenes, args.output, epochs=args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()