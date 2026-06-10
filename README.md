# Accessible Melanoma Detection Across Diverse Skin Tones

## Project Purpose
Bias and inaccessibility are two significant problems in the healthcare industry that are detrimental to patient health outcomes. Skin cancer diagnosis is particularly affected, as dermatology models have been shown to underperform on darker skin tones due to underrepresentation in training data. Additionally, access to specialist care remains unequal across geographic and socioeconomic lines.

This project develops a Multiple Instance Learning (MIL) model to classify skin lesions as cancerous or non-cancerous from patient image bags, with a focus on maintaining high recall to minimize missed diagnoses. The model is trained on both dermascope and smartphone imagery with the intention for smartphone-only usage for accessability purposes. We also perform LAB-based skin tone augmentation to train the model on a wider range of skin tones, and analyze model fairness using TCAV (Testing with Concept Activation Vectors) to evaluate whether skin tone influences model predictions.

## Dataset
We use two datasets:
- **MIDAS** - primary training/validation/test dataset of skin lesion images, provides both dermascope and smartphone imagery for each patient
- **PAD-UFES-20** - external test dataset for generalization evaluation, contains smartphone imagery only

Data is organized as patient "bags" — each patient has multiple images of their lesion. Labels are binary: 0 (non-cancerous) and 1 (cancerous). Both datasets provide patient metadata such as age, gender, lesion diameter, and medical history. Metadata was intentionally excluded from the final model to improve accessibility, as a model that relies only on images can be used by anyone with a smartphone camera without requiring clinical measurements or patient history. Age and gender metadata were tested during development but found to reduce model performance.

## How to Train
### Installation
```bash
pip install git+https://github.com/livgleason/deep-learning-project.git 
git clone https://github.com/livgleason/deep-learning-project.git
cd deep-learning-project
pip install -e .
pip install -r requirements.txt
```

### Training
```bash
cd src/midas/training
python train_models.py
```
The best model checkpoint will be saved as `best_model.pth`.

## Results

## Limitations
- not a ton of data
- lab-space augmentation relatively simplified
- overall lack of avaliable resources for implementation of LAB-space augmentation
- very few dark skin images to do TCAV analysis with
- had to limit layers of methodology of TCAV due to memory issues (not sure if this is limitation)