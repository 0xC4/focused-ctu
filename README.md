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
