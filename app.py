import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import predict_one_sample
from src import config
MODEL_NAME = 'rf'
def predict(df):
    predicted_class, _, _, _ = predict_one_sample(df.values.squeeze(), MODEL_NAME, config.SAVED_MODELS)
    df['Potability'] = predicted_class
    return df

iface = gr.Interface(predict, 
    gr.inputs.Dataframe(
        headers=['ph',
 'Hardness',
 'Solids',
 'Chloramines',
 'Sulfate',
 'Conductivity',
 'Organic_carbon',
 'Trihalomethanes',
 'Turbidity'],
        default=[[6.06, 160.77, 14775.15, 7.48, 305.83, 327.27, 12.31, 69.04, 3.47]]
    ),
    [
        "dataframe",
        # "plot",
        # "numpy"
    ],
    description="Enter data to predict water potability."
)
iface.launch()