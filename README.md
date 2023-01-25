# Lithium-ion Battery Remaining Useful Life Estimation Using Deep Learning
Complete source code for the undergraduate student project by John Rufino Macasaet and Christian Jay Adducul under the Ubiquitous Computing Laboratory, Electrical and Electronics Engineering Institute, University of the Philippines.

## Training Notebooks
You can use [Google Colab](colab.research.google.com) to run the training notebooks. Alternatively, you can set up your own local Jupyter notebook server with the dependencies in the `requirements.txt` file.

## Inference Script
The scripts come in `.py` files intended for the Raspberry Pi 3 Model B. By default, the scripts come with the pretrained models and generated test tensors. If you have trained your own models, please use the same file names to avoid issues with the inference scripts.

Run the script using Python 3. You can use the `-h` or `--help` flag to display help. If you have the 16x2 LCD set up, you can also use `-p` or `--print` flag to print on the LCD. **Note that the Adafruit Character LCD library used in this project might be deprecated in the future, so you may have to edit the code to make interfacing wit the LCD work.**

## Results Processing
You can use Google Colab to run results processing. On Colab, place the generated `.pkl` files in the `/content/` folder. For local Jupyter Notebook server users, place the files in the same folder as the results processing `.ipynb` file.