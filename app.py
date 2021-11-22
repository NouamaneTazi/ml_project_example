""" gradio app """
import time
import gradio as gr
import pandas as pd
from src import config, predict_one_sample

MODEL_NAME = 'stacking'


def predict(*args):
    df = pd.DataFrame([args], columns=['ph',
                                       'Hardness',
                                       'Solids',
                                       'Chloramines',
                                       'Sulfate',
                                       'Conductivity',
                                       'Organic_carbon',
                                       'Trihalomethanes',
                                       'Turbidity']
                      )
    tik = time.time()
    predicted_class, pred_probs, _, _ = predict_one_sample(
        args, MODEL_NAME, config.SAVED_MODELS)
    print("Inference time: ", time.time() - tik)
    df['Potability'] = predicted_class
    return dict(zip(['Not Potable', 'Potable'], pred_probs)), df


iface = gr.Interface(
    predict,
    [
        gr.inputs.Slider(0, 14, label="Ph", default=7),
        gr.inputs.Slider(40, 330, label="Hardness", default=196),
        gr.inputs.Slider(320, 61230, label="Solids", default=20927),
        gr.inputs.Slider(0.3, 15, label="Chloramines", default=7.13),
        gr.inputs.Slider(120, 490, label="Sulfate", default=333),
        gr.inputs.Slider(181, 755, label="Conductivity", default=421),
        gr.inputs.Slider(2, 30, label="Organic carbon", default=14),
        gr.inputs.Slider(0.7, 130, label="Trihalomethanes", default=66),
        gr.inputs.Slider(1.4, 7, label="Turbidity", default=3.95),
    ],
    [
        gr.outputs.Label(type="auto", label="Water Potability"),
        "dataframe"
    ],
    interpretation="default",
    server_port=7860
)

if __name__ == "__main__":
    iface.launch()
