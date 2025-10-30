import argparse, torch, os
from torch_geometric.loader import DataLoader
from load_data import GHPDataset
from model import GHP
from utils import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='./data'); p.add_argument('--name', required=True)
    p.add_argument('--ckpt', default=None)
    args = p.parse_args()

    ds = GHPDataset(root=args.root, name=args.name)
    test_idx = getattr(ds, 'test_idx', None)
    if test_idx is None:
        test_idx = torch.arange(len(ds))
    loader = DataLoader(ds[test_idx.tolist()], batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GHP(in_dim=ds.num_node_features).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        # checkpoint might be dict or state_dict
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)

    metrics = evaluate(model, loader, device)
    print(f"Test Acc: {metrics['acc']:.4f} | F1_macro: {metrics['f1_macro']:.4f} | F1_micro: {metrics['f1_micro']:.4f}")
    print("Precision (macro):", metrics['precision_macro'])
    print("Recall (macro):", metrics['recall_macro'])
    print("Classification report:\n", metrics['classification_report'])
    print("Confusion matrix:\n", metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
