import torch

# Load the checkpoint
checkpoint = torch.load(
    '/home/pn2694/ghn3/checkpoints/resnet50_ghn3_predicted.pt',
    map_location='cpu'  # load on CPU so no GPU needed
)

# Show the top level keys
print("Top level keys:", checkpoint.keys())

# Show epoch and step
print("Epoch:", checkpoint['epoch'])
print("Step:", checkpoint['step'])

# Show all layer names and their weight shapes
print("\nLayer names and shapes:")
for name, tensor in checkpoint['state_dict'].items():
    print(f"  {name:50s}  {str(tensor.shape):30s}  dtype={tensor.dtype}")

# Look at the first few values of the first conv layer
w = checkpoint['state_dict']['conv1.weight']
print("\nFirst conv layer weight sample:")
print("  Shape:", w.shape)
print("  Min:", w.min().item())
print("  Max:", w.max().item())
print("  Mean:", w.mean().item())
print("  Std:", w.std().item())

# Show total number of parameters
total = sum(t.numel() for t in checkpoint['state_dict'].values())
print(f"\nTotal parameters: {total:,} ({total/1e6:.2f}M)")