import torch
import time
from geotransolver import GeoTransolver

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = GeoTransolver(
    space_dim=3, geom_dim=3, cond_dim=None, out_channels=4, num_constants=None,
    dim=512, depth=12, heads=8, dim_head=64, num_slices=96
).to(device)

pos = torch.randn(8, 4096, 3, device=device)
geom = torch.randn(8, 4096, 3, device=device)

# Warmup
with torch.autocast(device, dtype=torch.float16, enabled=(device!='cpu')):
    for _ in range(2):
        model(pos, geom)

if device == 'mps':
    torch.mps.synchronize()
start = time.time()
with torch.autocast(device, dtype=torch.float16, enabled=(device!='cpu')):
    for _ in range(5):
        model(pos, geom)
if device == 'mps':
    torch.mps.synchronize()

print(f"Time per forward pass on {device}: {(time.time() - start) / 5 * 1000:.2f} ms")
