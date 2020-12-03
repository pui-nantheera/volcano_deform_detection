# Volcanic Deformation Detection using InSAR data

Usage for testing all interferograms in data folder:

DATA_DIR="data/"
OUT_DIR="output/"
MODEL_NAME="models/Model1.pd"

python getProbmap_fn.py --out_dir="$OUT_DIR" --data_dir="$DATA_DIR" --model_name="$MODEL_NAME"

Download models:

- [Paper1] <a href="https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/Ef-Z187hrNBInYzNIhjihkIBA4g47w93zDtXgk-jkHrl9Q?e=3MACfI">Model1</a>
- [Paper2] <a href="https://uob-my.sharepoint.com/:u:/g/personal/eexna_bristol_ac_uk/EcQeotn8ogxNpvvQtTmv3MUBzxHn6cm1Ob6ybmHKyWhxZA?e=4ypzBR">Model2</a>

------------------------------------
References:

[<a href="https://research-information.bris.ac.uk/ws/portalfiles/portal/168247520/Full_text_PDF_final_published_version_.pdf">Paper1</a>] Application of Machine Learning to Classification of Volcanic Deformation in Routinely Generated InSAR Data, N Anantrasirichai, J Biggs, F Albino, P Hill, D Bull
Journal of Geophysical Research: Solid Earth, 2018. [<a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JB015911">https://doi.org/10.1029/2018JB015911</a>]

[<a href="https://arxiv.org/abs/1905.07286">Paper1</a>] A deep learning approach to detecting volcano deformation from satellite imagery using synthetic datasets, N Anantrasirichai, J Biggs, F Albino, D Bull
Remote Sensing of Environment 230, 2019. [<a href="https://www.sciencedirect.com/science/article/pii/S003442571930183X">https://doi.org/10.1016/j.rse.2019.04.032</a>]
