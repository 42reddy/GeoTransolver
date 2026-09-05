import torch
import time
from geotransolver import GeoTransolver

model = GeoTransolver(
    space_dim=3, geom_dim=3, cond_dim=None, out_channels=4, num_constants=None,
    dim=512, depth=12, heads=8, dim_head=64, num_slices=96
).cuda()

pos = torch.randn(8, 4096, 3, device='cuda')
geom = torch.randn(8, 4096, 3, device='cuda')

# Warmup
with torch.autocast('cuda', dtype=torch.float16):
    for _ in range(5):
        model(pos, geom)

torch.cuda.synchronize()
start = time.time()
with torch.autocast('cuda', dtype=torch.float16):
    for _ in range(20):
        model(pos, geom)
torch.cuda.synchronize()

print(f"Time per forward pass: {(time.time() - start) / 20 * 1000:.2f} ms")
