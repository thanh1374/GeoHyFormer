import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from load_data import GHPDataset
from model import GHP
from utils import train_one_epoch, evaluate
import os
import os.path as osp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--name', type=str, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = GHPDataset(root=args.root, name=args.name)
    train_idx = getattr(ds, 'train_idx', None)
    val_idx = getattr(ds, 'val_idx', None)
    test_idx = getattr(ds, 'test_idx', None)

    train_loader = DataLoader(ds[train_idx.tolist()], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx.tolist()], batch_size=args.batch_size)
    test_loader = DataLoader(ds[test_idx.tolist()], batch_size=args.batch_size)

    print("Train size:", len(ds.train_idx))
    print("Val size:", len(ds.val_idx))
    print("Test size:", len(ds.test_idx))

    in_dim = ds.num_node_features
    # pass num_heads param name to match model signature
    model = GHP(in_dim=in_dim, d_model=args.d_model, num_layers=args.num_layers, n_heads=args.num_heads).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_test = 0.0
    save_dir = osp.join(args.root, args.name, 'processed')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'GHP_best.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        val_acc = val_metrics.get('acc', 0.0)
        test_acc = test_metrics.get('acc', 0.0)

        # save best
        if (val_acc > best_val) or (val_acc == best_val and test_acc > best_test):
            best_val = val_acc
            best_test = test_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'args': vars(args)
            }, save_path)

        print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Train Acc {train_metrics['acc']:.4f} | "
              f"Val Acc {val_metrics['acc']:.4f} | Val F1 {val_metrics['f1_macro']:.4f} | "
              f"Test Acc {test_metrics['acc']:.4f} | Test F1 {test_metrics['f1_macro']:.4f}")

    print("BEST Val Acc:", best_val, "Test at best:", best_test)

if __name__ == "__main__":
    main()
