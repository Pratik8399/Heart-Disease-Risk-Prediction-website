from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import code
app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class form (BaseModel):  
    age: int
    chestPain:int
    gender:str
    MaxHeartRate:int
    ExerciseInducedAngina:str
    oldpeak:float
    slope:int
    vessels:int
    thalassemia:int


@app.post('/HDP')
def check(obj:form):
    print(code.disease(obj))
    return code.disease(obj)

