import torch
from src.utils.model_loader import load_model 

model = load_model()

def predict(features : list[float]) -> int :
    """
    Predict the class of the input features using the loaded model.
    
    Args:
        features (list[float]): List of input features.
        
    Returns:
        int: Predicted class index.
    """
    input_tensors = torch.tensor(features , dtype=torch.float32 ).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensors)
        predicted_class = torch.argmax(output, dim =1)
        return int(predicted_class.item())
        