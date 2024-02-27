from train import train
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str, default="DDPM_Unconditional", help='Name of the run')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--dataset_path', type=str, default="./data", help='Path to the dataset')
    parser.add_argument('--device', type=str, default="cuda", help='Device for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()