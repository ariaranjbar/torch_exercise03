import torch
import torch.utils.data
from instance_segmentation import get_model_instance_segmentation
from matplotlib import pyplot as plt

from src.dataset import PennFudanDataset
from src.helpers import get_transform

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


    
dataset = PennFudanDataset('penn_fudan_ped', get_transform(train=True))
dataset_test = PennFudanDataset('penn_fudan_ped', get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

model = get_model_instance_segmentation(2).to(device)
model.load_state_dict(torch.load("coco.pth"))
model.eval()
rows = 2
cols = rows * 2
max_test_count = int(rows ** 2)
tested = 0
misses = []
(A, b) = dataset_test[0]
print(f"Shape of X [N, C, H, W]: {A.shape}, y: {b}")
for n, i in enumerate(torch.randint(0, len(dataset_test) - 1, (max_test_count,))):
    (X, y) = dataset_test[i]
    with torch.no_grad():
        X = torch.unsqueeze(X, 0).to(device)
        pred = model(X)[0]
        misses.append((i, pred))
        tested = n + 1

print(
    f"total tested: {tested}, missed: {len(misses)} ({(len(misses)/tested * 100):>1f}%)"
)

figure = plt.figure(figsize=(16, 10))
for i in range(0, len(misses)):
    sample_idx, msk = misses[i]
    figure.add_subplot(rows, cols, 2*i + 1)
    plt.title("predicted")
    plt.axis("off")
    plt.imshow(msk["masks"].sum(0).clamp(0, 1).cpu().squeeze(), cmap="gray")
    img = dataset_test[sample_idx][0]
    figure.add_subplot(rows, cols, 2*i + 2)
    plt.title("actual")
    plt.axis("off")
    plt.imshow(img.cpu().squeeze().permute(1,2,0))
plt.show()
