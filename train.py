import os
import argparse

from dataset import get_implant_dataset, get_location_dataset
from engine import Engine
from models.UNet import UNet


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_every', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--dataset_dir', type=str, default='ImplantData/datasets/ImplantLocation/UpperAnterior')
    parser.add_argument('--task', type=str, default='ImplantLocation_UpperAnterior')

    return parser


def main(args) :

    model = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 1,
        features = (32, 32, 64, 128, 256, 32),
    )
    engine = Engine(
        model = model,
        max_epochs = args.num_epochs,
        batch_size = args.batch_size,
        val_every = args.val_every,
        results_dir = args.results_dir,
    )
    
    train_ds, val_ds = get_implant_dataset(data_dir=args.dataset_dir)
    engine.train(train_dataset=train_ds, val_dataset=val_ds)


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    
    args.results_dir = os.path.join(args.results_dir, args.task)
    os.makedirs(args.results_dir, exist_ok=True)
    
    main(args)
