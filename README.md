# Download the dataset

```
1. curl -L -o ./dataset/animals10.zip https://www.kaggle.com/api/v1/datasets/download/alessiocorrado99/animals10
```

# Train the model
```
1. datasets structure (create a directory named datasets)
 * [test](./test)
   * [class_a](./test/image.jpg)
   * [class_b](./test/image.jpg)
 * [train](./train)
   * [class_a](./train/image.jpg)
   * [class_b](./train/image.jpg)
 * [val](./val)
   * [class_a](./val/image.jpg)
   * [class_b](./val/image.jpg)
2. python prepare_dataset.py
3. train.py
```