import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, r".\external\ChangeFormer")

from models.networks import define_G


CHECKPOINT = (
    "./external/ChangeFormer/checkpoints/"
    r"CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_"
    r"multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256\best_ckpt.pt"
)

BEFORE = r".\data\LEVIR-CD\test\A\test_1.png"
AFTER = r".\data\LEVIR-CD\test\B\test_1.png"
GT_PATH = r".\data\LEVIR-CD\test\label\test_1.png"

OUTPUT = r".\outputs\changeformer_test1_correct.png"

DEVICE = torch.device("cpu")
TILE = 256


def preprocess(tile):
    tensor = TF.to_tensor(tile)
    tensor = TF.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    return tensor.unsqueeze(0)


args = SimpleNamespace(
    net_G="ChangeFormerV6",
    gpu_ids=[],
    n_class=2,
    embed_dim=256,
)

model = define_G(args=args, gpu_ids=[])

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_G_state_dict"])
model.to(DEVICE)
model.eval()

before = Image.open(BEFORE).convert("RGB")
after = Image.open(AFTER).convert("RGB")

if before.size != after.size:
    raise ValueError("Before and after image sizes are different.")

width, height = before.size

prediction = np.zeros((height, width), dtype=np.uint8)

with torch.no_grad():

    for y in range(0, height, TILE):

        for x in range(0, width, TILE):

            x2 = min(x + TILE, width)
            y2 = min(y + TILE, height)

            before_tile = before.crop((x, y, x2, y2))
            after_tile = after.crop((x, y, x2, y2))

            original_width = before_tile.width
            original_height = before_tile.height

            if before_tile.size != (TILE, TILE):
                before_tile = before_tile.resize(
                    (TILE, TILE),
                    Image.BICUBIC,
                )
                after_tile = after_tile.resize(
                    (TILE, TILE),
                    Image.BICUBIC,
                )

            input_a = preprocess(before_tile).to(DEVICE)
            input_b = preprocess(after_tile).to(DEVICE)

            output = model(input_a, input_b)

            logits = output[-1] if isinstance(output, (list, tuple)) else output

            mask = (
                torch.argmax(logits, dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            if (original_width, original_height) != (TILE, TILE):

                mask = np.asarray(
                    Image.fromarray(mask).resize(
                        (original_width, original_height),
                        Image.NEAREST,
                    )
                )

            prediction[y:y2, x:x2] = mask[: y2 - y, : x2 - x]

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

Image.fromarray(prediction * 255).save(OUTPUT)

gt = (
    np.asarray(
        Image.open(GT_PATH).convert("L")
    ) > 0
).astype(np.uint8)

pred = prediction

tp = np.sum((pred == 1) & (gt == 1))
fp = np.sum((pred == 1) & (gt == 0))
fn = np.sum((pred == 0) & (gt == 1))
tn = np.sum((pred == 0) & (gt == 0))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
iou = tp / (tp + fp + fn + 1e-8)

print()
print("RESULTS")
print("----------------------------")
print("Predicted changed pixels :", int(pred.sum()))
print("Ground truth pixels      :", int(gt.sum()))
print("Predicted change %       :", round(pred.mean() * 100, 4))
print("Ground-truth change %    :", round(gt.mean() * 100, 4))
print("IoU                      :", round(iou, 4))
print("Precision                :", round(precision, 4))
print("Recall                   :", round(recall, 4))
print("F1                       :", round(f1, 4))
print("TP                       :", int(tp))
print("FP                       :", int(fp))
print("FN                       :", int(fn))
print("TN                       :", int(tn))
print("Saved                    :", OUTPUT)
