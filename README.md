<div align="center">

# AI-based_GN_Diagnostic_Assistance_Tool

</div>

<div align="center">

## 🔥🔥🔥

#### Updated on June 30, 2025

</div>


## ✨Paper

This repository provides the official implementation of AI-based_GN_Diagnostic_Assistance_Tool。


**Improving Diagnostic Efficiency in Glomerular Nephritis through an Integrated AI-based Pathological Image Analysis Approach** (*Under Review*)


### Key Features
An AI-based GN diagnostic assistance tool is developed and the diagnostic pipeline comprises three sequential steps: glomerulus segmentation, glomerulus lesion feature extraction and patient-level diagnosis.

The tool consists of three core components: 

(1) a glomerular localization module for precise glomerulus segmentation;       

(2) two multi-classification module for identifying glomerular lesions; 

(3) a patient-level classification module for diagnosing four GN subtypes.




## ✨Installation & Preliminary
1. Clone the repository.
    ```
    git clone https://github.com/Git-HB-CHEN/AI-based_GN_Diagnostic_Assistance_Tool.git
    cd AI-based_GN_Diagnostic_Assistance_Tool
    ```

2. Create a virtual environment for AI-based_GN_Diagnostic_Assistance_Tool and activate the environment.
    ```
    conda create -n GNDAT python=3.9
    conda activate GNDAT
```
    
3. Install Pytorch and torchvision.
   (You can follow the instructions [here](https://pytorch.org/get-started/locally/))

4. Install other dependencies.
   ```
    pip install -r requirements.txt
   ```

## ✨Direct Inference with the AI-based_GN_Diagnostic_Assistance_Tool

1. Inference for Light Microscopy Images
   ```
    python running_GNDA_tool_LM_Image.py
   ```
2. Inference for Light Microscopy and Immunofluorescence Images 
   ```
    python running_GNDA_tool_LM_IF_Image.py
   ```

## ✨Training the AI-based_GN_Diagnostic_Assistance_Tool

1. Training the glomerulus segmentation model
   ```
    python running_training_glomerulus_segmentation.py
   ```
2. Training the glomerular lesion classification model
   ```
    python running_training_glomerular_lesion_classification.py
   ```
3. Training the patient-level classification model
   ```
    python running_training_patient_classification.py
   ```


*Details of the model architecture and its associated weights are being curated and will be released following the acceptance of this manuscript.*