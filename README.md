# Accessible Melanoma Detection Across Diverse Skin Tones

## Project Purpose
Bias and inaccessibility are two significant issues in healthcare that negatively impact patient outcomes. Skin cancer diagnosis is particularly affected, as dermatology models often underperform on darker skin tones due to underrepresentation in training data. Additionally, access to dermatological care is limited by geographic and socioeconomic barriers.

This project develops a Multiple Instance Learning (MIL) model to classify skin lesions as cancerous or non-cancerous from sets (“bags”) of patient images. The model prioritizes high recall to minimize missed diagnoses. It is trained on both dermascope and smartphone imagery, with the long-term goal of enabling smartphone-only deployment.

To improve representation across skin tones, we apply LAB color space augmentation. We also analyze fairness using TCAV (Testing with Concept Activation Vectors) to evaluate whether skin tone influences model predictions.

## Dataset
We use two datasets:
- **MIDAS** - primary training/validation/test dataset of skin lesion images, provides both dermascope and smartphone imagery for each patient
- **PAD-UFES-20** - external test dataset for generalization evaluation, contains smartphone imagery only

Both datasets provide patient metadata such as age, gender, lesion diameter, and medical history, with binary labels: 0 (non-cancerous) and 1 (cancerous). Metadata was intentionally excluded from the final model to improve accessibility, as a model that relies only on images can be used by anyone with a smartphone camera without requiring clinical measurements or patient history. Age and gender metadata were tested during development but found to reduce model performance. For model use, all images corresponding to one patient were grouped with their corresponding cancer/no cancer label.

## How to Train
### Installation
```bash
git clone https://github.com/livgleason/deep-learning-project.git
cd deep-learning-project
pip install -r requirements.txt
pip install -e .
```
Due to the size of both datasets, they are not included in this repository and are instead located on Talapas.  
```bash
export DATA_ROOT="/gpfs/home/ogleason/deep-learning-project/full_data"
```
/gpfs/home/ogleason/deep-learning-project/full_data/
├── MIDAS/
│   ├── MIDAS_images/
│   └── midas.csv
├── PAD-UFES-20/
│   ├── PAD_images/
│   └── metadata.csv

### Training
After setting DATA_ROOT, run
```bash
cd src/midas/training
python train_models.py
```
The best model checkpoint will be saved as `best_model.pth`, at "/gpfs/home/ogleason/deep-learning-project/src/midas/training/best_model.pth"

## Model Details
- Model: DetectionModel (MIL-based architecture)
- Loss: Binary Cross Entropy with Logits
- Optimizer: AdamW
- Learning rate scheduler: ReduceLROnPlateau
- Encoder partially frozen (fine-tuning higher layers)
- Early stopping enabled

## Results
**MIDAS**
- AUC: 0.586
- Recall: 0.94
- Precision: 0.20
- F1: 0.327

![MIDAS ROC](notebooks/images/midas_roc.png)
![MIDAS Confusion](notebooks/images/midas_confusion.png)

The model achieves very high recall (0.94), indicating that it successfully identifies most cancerous lesions. This aligns with the primary design goal of minimizing false negatives in a medical screening context. However, this comes at a substantial cost. The model exhibits very low precision (0.20), meaning that the majority of positive predictions are incorrect. In practice, this would lead to a large number of unnecessary follow-ups, reducing the model’s usefulness in real-world deployment. The AUC of 0.586 is only slightly better than random guessing, suggesting that the model has weak overall discriminative ability. This indicates that while the model is biased toward predicting positives (driving high recall), it has not learned strong features that reliably distinguish between cancerous and non-cancerous lesions.

**PAD-UFES-20**
- AUC: 0.465
- Recall: 0.89
- Precision: 0.019
- F1: 0.038

![PAD ROC](notebooks/images/pad_roc.png)
![PAD Confusion](notebooks/images/pad_confusion.png)

Performance on the external PAD-UFES-20 dataset is poor across nearly all metrics. While recall remains high (0.89), this is largely due to the model predicting positive for a significant portion of samples rather than demonstrating meaningful generalization. Precision drops to 0.019, indicating that nearly all positive predictions are false positives. This suggests the model has very limited ability to distinguish true cancerous cases in a new domain. The AUC of 0.465 is below 0.5, meaning the model performs worse than random guessing in ranking predictions. This is a strong indicator that the model has failed to generalize beyond the training distribution and may be overfitting to dataset-specific artifacts in MIDAS.

**TCAV**

TCAV (Testing with Concept Activation Vectors) was used to assess whether the model is sensitive to skin tone. This was done by collecting conecpt groups from our PAD-UFES-20 test set since the model is not trained on these images and the metadata includes fitzpatrick scale labels (classification system that describes a person's skin type based on its natural melanin content and how it reacts to ultraviolet radiation by burning or tanning). Concept groups included darker skin tones (Fitzpatrick 4–6), lighter skin tones (Fitzpatrick 1–3), and random baseline images that are not based on a specific skin tone. Through this evaluation, it was found that model predictions are not influenced by skin tone representations with a magnitude of -0.0073 for darker skins tones and 0.0041 for ligher. Since TCAV scores near 0 indicate no directional influence, these results suggest that the model’s predictions are not meaningfully influenced by skin tone representations. This is a positive outcome relative to the project’s initial goal of mitigating bias in melanoma detection. Despite the model’s overall performance limitations, it does not appear to rely heavily on skin tone as a predictive feature, which is an encouraging signal for fairness. However, it is important to interpret these results cautiously due to dataset limitations (see more below).

Overall, the model demonstrates a clear tradeoff between recall and overall predictive quality. While it consistently achieves high recall—successfully identifying most cancer cases—it does so by over-predicting the positive class, resulting in extremely low precision and weak discriminative performance. Additionally, the model performs poorly on external data, indicating limited generalization and a reliance on dataset-specific features rather than robust patterns of melanoma. Although fairness results are promising, the underlying model performance is not strong enough for real-world deployment.

## Limitations
- Limited dataset size: the total amount of training data is relatively small for a deep learning task, restricting the model’s ability to learn robust features and generalize effectively.
- Simplified LAB color augmentation: the LAB-space augmentation used to simulate diverse skin tones is relatively basic and may not fully capture realistic variations in skin appearance.
- Implementation constraints for augmentation: there was limited available resources for implementaing LAB-space augmentation, making the results less accurate and noisy. As such, the model was likely impacted and could have benefited from a more robust approach. 
- Limited dark skin samples for TCAV: ironically, there was not many images of darker skin tones to complete TCAV properly. The small number of darker skin tone images reduces confidence in the fairness analysis and limits the ability to draw stronger conclusions.
- Restricted TCAV methodology: TCAV analysis was applied to a limited set of layers due to memory constraints, which may reduce the depth and reliability of the interpretation.
- Poor external generalization: the model performs significantly worse on the PAD-UFES-20 dataset, suggesting overfitting to MIDAS-specific features.