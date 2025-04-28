from pydantic import BaseModel

class InputData(BaseModel):
    f1 : float
    f2 : float
    f3 : float
    f4 : float
    
    
class Prediction_Response(BaseModel):
    
    prediction_class :int
    