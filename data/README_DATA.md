# Data Placement

Place your dataset here.

## Option A: Folder-based
```
data/
  train/
    0/
    1/
  val/
    0/
    1/
  test/
    0/
    1/
```

## Option B: CSV-based (e.g., Kaggle HCD)
- `images/` contains .tif/.png files
- CSVs contain `id` (filename without extension) and `label` (0/1)

Update `configs/default.yaml` accordingly.