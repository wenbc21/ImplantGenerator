import os
import argparse

from dataset import get_implant_dataset, get_location_dataset
from engine import Engine
from models.UNet import UNet


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--dataset_dir', type=str, default='ImplantData/datasets/ImplantGeneration/UpperAnterior')
    parser.add_argument('--task', type=str, default='ImplantGeneration_UpperAnterior')

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
        results_dir = args.results_dir,
        device = "cpu"
    )
    engine.load_state_dict(os.path.join(args.results_dir, "weight", "best.pt"))
    
    if "Generation" in args.task :
        test_ds = get_implant_dataset(data_dir=args.dataset_dir, is_train=False)
        engine.inference_generation(test_dataset=test_ds)
    elif "Location" in args.task :
        test_ds = get_location_dataset(data_dir=args.dataset_dir, is_train=False)
        engine.inference_location(test_dataset=test_ds)


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    
    args.results_dir = os.path.join(args.results_dir, args.task)
    os.makedirs(args.results_dir, exist_ok=True)
    
    main(args)
