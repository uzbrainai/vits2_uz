# vits2_uz
#TTS model for Uzbek language

1. Create audio folder and put wav files under dataset_uz folder. # "dataset_uz/audio"
2. Download logs folder, put it under this path "datasets/custom_base" and unzip it. [Download here:](https://drive.google.com/file/d/1o1ksl9B3uWcOBzy97eYuOPVY2ysEyzcB/view?usp=drive_link)


3. install required packages (for pytorch 2.0)
   
   conda create -n vits2 python=3.11
   
   conda activate vits2
   
   pip install -r requirements.txt

   conda env config vars set PYTHONPATH="/path/to/vits2"


4. For continue training:
python train.py -c datasets/custom_base/config.yaml  -m custom_base


5. Inference Examples:
   
   see inference_uz.ipynb
