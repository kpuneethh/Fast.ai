import gradio as gr
from fastai.vision.all import *
import numpy as np
from PIL.Image import Image as PILImageOriginal

def is_cat(x): return x[0].isupper() 
    
# Load the exported model
learn_inf = load_learner('model.pkl')

# Define prediction function
def classify_image(img):
    if isinstance(img, PILImageOriginal):
        img = PILImage.create(np.array(img))
    else:
        img = PILImage.create(img)

    pred, idx, probs = learn_inf.predict(img)
    labels = learn_inf.dls.vocab
    return {label: float(prob) for label, prob in zip(labels, probs)}

# Launch Gradio interface
gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(),
    title="Cat vs Dog Classifier"
).launch()
