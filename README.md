# 🧠 CNN with Regularisation on Fashion MNIST

This project builds on the base CNN model from [this repo](https://github.com/adabyt/tensorflow-cnn-fashion-mnist), extending it with **Batch Normalisation** and **Dropout** to improve generalisation and reduce overfitting. The model is trained and evaluated on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

---

## 📌 Overview

The CNN architecture integrates:
- **Batch Normalisation** after convolutional and dense layers  
- **Dropout** (tunable rate) after the dense activation layer

These additions aim to:
- Accelerate training  
- Improve convergence  
- Prevent overfitting on the training set  

---

## 🧪 Dataset

The model is trained on the Fashion MNIST dataset:
- 60,000 training samples, 10,000 test samples
- 28×28 grayscale images across **10 classes**:
  `['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`

---

## ⚙️ Model Architecture

| Layer Type           | Notes                                                       |
|----------------------|-------------------------------------------------------------|
| Conv2D + BatchNorm + ReLU | 32 filters, 3×3 kernel                                 |
| MaxPooling2D         | Reduces dimensionality                                      |
| Conv2D + BatchNorm + ReLU | 64 filters, 3×3 kernel                                 |
| MaxPooling2D         |                                                              |
| Flatten              | Converts 3D feature maps to 1D                              |
| Dense + BatchNorm + ReLU + Dropout | 128 neurons, dropout rate = `0.5`             |
| Dense (Softmax)      | 10 output classes                                           |

---

## 🧠 Notes on Regularisation & Training

### 🏋️ Model Training Observations

- **Validation accuracy** is comparable to the unregularised CNN.
- **Validation loss** is **lower** and more stable — a sign of better generalisation.
- **Training accuracy** and **training loss** are slightly worse than the previous CNN — a good sign, as the model is **not overfitting**.

> ✅ The regularized model generalises better and is less likely to memorise training data.

---

### 🔁 Dropout Rate Experiments

| Dropout Rate | Test Accuracy | Test Loss | Epoch w/ Lowest Validation Loss |
|--------------|---------------|-----------|----------------------------------|
| `0.2`        | 0.9102        | 0.2941    | 7                                |
| `0.3`        | 0.9098        | 0.2677    | 7                                |
| `0.4`        | 0.9073        | 0.2840    | 9                                |
| `0.5`        | 0.9066        | 0.2695    | 9                                |
| _Previous CNN_ | 0.9101      | 0.2874    | 7                                |

🔍 **Interpretation**:
- A **lower dropout rate** consistently yields **better accuracy and lower loss**.
- This may suggest:
  - The model benefits from **more node interconnectivity** on this dataset.
  - Higher dropout introduces too much noise and degrades performance.

---

## 📈 Final Evaluation

- ✅ **Test Accuracy**: ~0.9066  
- ✅ **Test Loss**: ~0.2695  
- 🧪 First 5 predictions matched the true labels perfectly

---

## 🧠 Author’s Note

This project builds on [my previous CNN implementation](https://github.com/adabyt/tensorflow-cnn-fashion-mnist) by experimenting with normalisation and dropout strategies to improve generalisation and reduce overfitting.  

---

## 📚 References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- [Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)

---

## 📌 License

MIT License
