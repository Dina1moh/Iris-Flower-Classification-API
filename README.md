# 🌸 Iris Flower Classification API

This is a simple machine learning deployment project using **FastAPI** and **PyTorch** to classify iris flowers into one of three species: *Setosa*, *Versicolor*, or *Virginica*.  
The model is trained on the famous **Iris dataset** using a fully connected neural network.

---

## 🚀 Features

- Fast and lightweight REST API with FastAPI
- Model trained using PyTorch
- Accepts input features (sepal/petal length and width)
- Returns the predicted class of the flower
- Organized project structure

---

## 📊 Input Features

The API expects a JSON object with the following float values:

```json
{
  "f1": 5.1,
  "f2": 3.5,
  "f3": 1.4,
  "f4": 0.2
}
