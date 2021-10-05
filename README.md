# Volcanic Deformation Detection using InSAR data

------------------------------------
To run detection
------------------------------------
Usage for testing all interferograms (converted to grayscale and save in png format) in data folder:

```python
DATA_DIR="data/"
OUT_DIR="output/"
MODEL_NAME="models/Model1.pd"

python getProbmap_fn.py --out_dir="$OUT_DIR" --data_dir="$DATA_DIR" --model_name="$MODEL_NAME"
```
Download models:

- [Paper1] <a href="https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/Ef-Z187hrNBInYzNIhjihkIBA4g47w93zDtXgk-jkHrl9Q?e=KslBWQ">Model1</a>
- [Paper2] <a href="https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/EcQeotn8ogxNpvvQtTmv3MUBzxHn6cm1Ob6ybmHKyWhxZA?e=akrWpi">Model2</a>


------------------------------------
To retrain
------------------------------------
We trained our models with Matlab. You can download the pretrained model here: <a href="https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/EXH66HZ2rxlDrZBEoo5fgIABXYWYNcZl6N723jKesLdA9w?e=4QBoSj">model2.mat</a>.
Dataset is saved in two folders named 'deform' and stratified' under the main data folder 'data/deform' and 'data/stratified'.
For more details, please see comments in <a href="https://github.com/pui-nantheera/volcano_deform_detection/blob/main/runTrain.m">runTrain.m</a>.

The new model in mat file can be converted to newmodel.pd to use with Python Tensorflow as follows.

```matlab
% In Matlab
mode_name = 'new_model'
modeldir = 'results/'
modeldir = 'models/'
load([modeldir, modelname, '.mat']);
exportONNXNetwork(netFineTune, [modeldir, modelname, '.onnx']);
```
```python
# In Python, read onnx model and convert to pd graph
import onnx
from onnx_tf.backend import prepare

modeldir = 'results/'
mode_name = 'new_model'
onnx_model = onnx.load(modeldir + model_name + ".onnx") 
tf_rep = prepare(onnx_model)  
tf_rep.export_graph(modeldir + model_name + ".pd") 
```

------------------------------------
References:
------------------------------------
[<a href="https://research-information.bris.ac.uk/ws/portalfiles/portal/168247520/Full_text_PDF_final_published_version_.pdf">Paper1</a>] Application of Machine Learning to Classification of Volcanic Deformation in Routinely Generated InSAR Data, N Anantrasirichai, J Biggs, F Albino, P Hill, D Bull
Journal of Geophysical Research: Solid Earth, 2018. [<a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JB015911">https://doi.org/10.1029/2018JB015911</a>]

[<a href="https://arxiv.org/abs/1905.07286">Paper1</a>] A deep learning approach to detecting volcano deformation from satellite imagery using synthetic datasets, N Anantrasirichai, J Biggs, F Albino, D Bull
Remote Sensing of Environment 230, 2019. [<a href="https://www.sciencedirect.com/science/article/pii/S003442571930183X">https://doi.org/10.1016/j.rse.2019.04.032</a>]

<a href="https://doi.org/10.5281/zenodo.5550815"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5550815.svg" alt="DOI"></a>
