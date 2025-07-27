# Focused view CT urography: towards a randomized trial investigating the relevance of incidental findings in patients with hematuria
This repository contains the code used to train the deep learning models used in the following publication:

```
Focused view CT urography: towards a randomized trial investigating the relevance of incidental findings in patients with hematuria. Sluijter TE, Roest C, Yakar D, Kwee TC. Diseases, 2025.
```

# Description of contents
The contents of the repository are as follows: 
- `train.py`
  Contains the main deep learning training loop.
- `config.py`
  Contains project specific configurations, including folder paths, image processing settings, and segmentation class definitions.
- `method12.py`
  Contains the postprocessing functionality to go from segmentation to focused view CTU using method 1 & 2 as described in the paper.
- `method3.py`
  Contains the postprocessing funcitonality to go from segmentation to focused view CTU using method 3, as described in the paper.
- `dual_attention_unet.py`
  Contains the code that defines the U-Net architecture.
- `expand_lines.py`
  Contains image postprocessing helpers for dilation of the generated masks.
- `predict.py`
  Contains inference functionality to generate segmentations and binary masks using trained models.

# Requirements
- `TensorFlow`
- `SimpleITK==2.2.0`
- `Scikit-learn`
- `Scipy`

# Model training
To train a new model, first adapt `config.py` to your environment settings.
Put your `.nii.gz` scans in your `IMAGE_DIR`, and the segmentations in your `SEGMENTATION_DIR`.
Then run `train.py` from the command line with the appropriate cross validation arguments. 
E.g.: `python train.py 0 5`, will run fold 0 out of 5 total folds for your dataset.

# Generate segmentations
Inference using trained models is done using `predict.py`.
First, prepare a text file containing file paths to the nifti's you want to predict on on each line.
Second, run `python predict.py -i [path to your text file] -m models/m_fold1.h5 models/m_fold2.h5 [etc]`
The generated segmentations will be exported to your `WORK_DIR`

# Apply post-processing to generated focused-view images
First, prepare a text file containing the paths to the scans you want to generate focused view scans for (you can use the txt file from the previous step).
Second, open either `method12.py` or `method3.py`, depending on which method you want to apply, and find the following line:

`scan_paths = [l.strip() for l in open("final_scans.txt")]`

Replace `final_scans.txt` with the path to your text file.

For method 1 & 2, simply run `python method12.py`.

For method 3, which is more resource intensive, there is the possibility to run it in parallel to reduce runtime.
To run method 3 for all scans, run `python method3.py 0 10`, to run te first batch of 10 total batches.
To run all scans in a single batch, run `python method3.py 0 1`.
