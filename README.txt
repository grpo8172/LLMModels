To run the code, it is best to use a virtual environment:
python3 -m venv .venv
source .venv/bin/activate
pip install transformers
python DownloadPhi2HuggingFace.py

But you should be able to just go ahead with your C Drive if you have enough space on it for the models as they are are quite sizeable. 

My code is using the phi2-local model. As this model is large, it can be downloaded onto your local drive by running: 
python3 DownloadPhi2HuggingFace.py

Then you can run trainQuant.py to train the ML model.

You can test out the model with tryModel.py

If you want to use the larger Mistral model then you would need to change trainQuant to read from mistral-local instead of phi2-local on line 6 which looks like this:

model_path = "./phi2-local"

So you would change it to "./mistral-local"

You could really use any model from hugging face you like as long as you tell the file that downloads the models (The first step of this ReadMe) which model you want to use by copying the model name on your
hugging face UI as you search for models.
