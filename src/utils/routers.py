from fastapi import APIRouter
from src.utils.schema import InputData , Prediction_Response
from src.inference import predict

router = APIRouter()

@router.post("/prediction" , response_model=Prediction_Response)
def get_prediction(input_data: InputData) -> Prediction_Response:
    """
    Predict the class of the input features using the loaded model.
    
    Args:
        input_data (InputData): Input data containing features.
        
    Returns:
        Prediction_Response: Predicted class index.
    """
    features = [input_data.f1 , input_data.f2 , input_data.f3 , input_data.f4]
    predicted_class = predict(features)
    
    return Prediction_Response(prediction_class=predicted_class)
