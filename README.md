# 🌭 Hot Dog / Not Hot Dog

Clasificador de imágenes que distingue si una foto es un **hot dog** o **no**.
Usa transfer learning sobre una **ConvNeXt-Tiny** preentrenada en ImageNet-22k (vía [timm](https://github.com/huggingface/pytorch-image-models)).

**Accuracy en test: 98%** (150 imágenes no vistas durante el entrenamiento).

|              | precision | recall | f1    |
|--------------|-----------|--------|-------|
| not_hot_dog  | 1.000     | 0.960  | 0.980 |
| hot_dog      | 0.962     | 1.000  | 0.980 |

## Estructura

```
.
├── data/
│   ├── hot_dog/        # 499 imágenes
│   └── not_hot_dog/    # 499 imágenes
├── train.py            # entrenamiento (con barra de progreso)
├── predict.py          # clasificar imágenes nuevas
├── metrics.json        # resultados del último entrenamiento
└── requirements.txt
```

## Origen de las imágenes

**Las imágenes de `data/` no son mías** y no tengo registro exacto de dónde las obtuve. Los orígenes más probables son:

- El dataset [**Hot Dog - Not Hot Dog** de Kaggle](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog), que tiene la misma estructura (`hot_dog/` y `not_hot_dog/`, ~500 imágenes por clase).
- [**Food-101**](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard et al., ETH Zürich, 2014), del que parece derivar el anterior. Los nombres de los archivos (p. ej. `1000288.jpg`) coinciden con su formato. Las imágenes de Food-101 se recolectaron de foodspotting.com y sus autores aclaran que no son dueños de ellas.

Se incluyen solo con fines educativos y no comerciales. Si sos titular de alguna imagen y querés que la quite, abrí un issue.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usar el modelo ya entrenado

Descargá `hotdog_model.pt` desde la sección [Releases](../../releases/latest) y ponelo en la raíz del repo:

```bash
gh release download v1.0 -p hotdog_model.pt
python predict.py mi_foto.jpg
# mi_foto.jpg: 🌭 HOT DOG  (p(hot dog) = 0.967)
```

## Entrenar

```bash
python train.py --epochs 10
```

```
Época  3/10: 100%|██████████████████████████████| 21/21 [00:26<00:00, 1.18s/batch, loss=0.3412]
   ↳ train_loss 0.3412 | val_loss 0.1345 | val_acc 0.9533
```

Usa automáticamente GPU CUDA, Apple Silicon (MPS) o CPU. En un Mac M4 tarda ~25 s por época.

Opciones: `--epochs`, `--lr`, `--batch-size`, `--seed` y `--model` (cualquier modelo de timm, p. ej. un ViT:
`--model vit_small_patch16_224.augreg_in21k_ft_in1k`).

## Cómo funciona

- **Split estratificado** 70 / 15 / 15 (train / val / test), semilla fija.
- **Data augmentation**: random resized crop, flip, rotación, color jitter y random erasing.
- **Fine-tuning completo** con AdamW: learning rate 10× mayor en la cabeza nueva que en el backbone, OneCycle schedule y label smoothing.
- Se guarda el checkpoint con mejor accuracy de validación y se evalúa una única vez en test.
