# XRayNPC

**A chest X-ray tool that sorts scans into normal, pneumonia or COVID-19, and checks whether the AI is really looking at the lungs.**
> This project is a prototype, not medical advice.

FIVE models available, from smartest to dumbest.
During development, FOUR or them failed.
But the lessons were invaluable.

| | |
|---|---|
| Try it | https://healthcare.yoanhristovevents.xyz |
| Write-up | https://drive.google.com/file/d/1HS0gdlYvSpGOVfdsz8X97QSlFtvRStP8/view?usp=sharing |
| Colab notebook (all experiments) | https://colab.research.google.com/drive/1zBhthB3UbJwKH77j6N3dlWdr12vbMqQP?usp=sharing |
| Video | https://youtu.be/zRC3c4LqKks |

## How to try it
1. Open the site and pick **Model 5**.
2. Click a sample X-ray on the left, or drag it into the middle.
3. Watch the five steps run, then read the decision.
4. Click "View benchmarks" to see how well that model did on test images.


## The five models
Five models, from dumbest to most careful. Four of them failed. On the website you can see where they failed.

| Model | What it is | Lesson |
|---|---|---|
| 1 | Baseline | Model cheats by looking at other parts of the image |
| 2 | Geometric augmentation + Targeted augmentation | Targeted augmentation improves Specificity |
| 3 | Multiclass model with COVID-19/Pneumonia/Normal | Deicision capabilities remained, but decision boundary became aggressive, generating more false alarms on healthy scans |
| 4 | Trained on images that Gemini Decision 1 labeld as "Clean" | The overconfidence problem, the blackbox problem - described below |
| 5 | ResNet-18 + Gemini Decision 1 as a second reader | Set proper tresholds, calibrate, and never let an LLM classify medical images alone. |


## The important lessons for everyone
**1. Improper augmentations do not prevent a model from cheating.** The baseline model was looking at shoulders, throat, medical lables and even aspect ratio. Geometric augmentation boosted metrics, but covering the frame and side areas proved more effective

**2. Do not treat Gemini as a black box.**
- On 200 children's X-rays it got about 0.89 balanced accuracy, but found only 71% of viral pneumonia.
- On COVID-19 it found only 23% of cases!
- It never said it was "unsure" (0 of 200), even when offered the option.
- Removing images Gemini called "bad" before training did not help (+0.002 macro-F1). But only 1.5% of images were removed, so that test was weak. Its quality ratings were also tied to the class (175 of 600 COVID images "unacceptable" vs 23 of 600 normal).

> It is another piece of proof thar RLHF on Vision-Language models is adverse in healthare pipelines. Too overconfident.

**3.** Safe model does not mean benchmark maxxing. It's about telling "I dont know".
 In the 3-class test, when the ResNet only answers its most confident 60% of images, only 1% are wrong. If forced to answer everything, about 7% are wrong.

**4.** One metric hides problems. Always consult at least 5-10!

**5.** Mixing datasets can teach the dataset, not the disease. 

... and many more. Thorough analysis in the Colab.


## Limits
- The main model was trained on children aged 1 to 5 from one hospital. 
- I had no lung outlines for the children's scans, so "the middle of the picture" was assumed for "the lungs".
- I only ran 3 seeds per model.
- COVID images mostly come from a single source.
- Gemini may have seen public datasets like these during its own training.
- The site sends your image to Google, since their API is in use. Use public images only.

## Brief description of backend

The backend is a Python service built with FastAPI and Gradio. It provides five
ResNet-18 model variants for chest X-ray analysis, returning class probabilities,
a risk decision, and Grad-CAM visual explanations. Model 5 can additionally use
Google Gemini for image-scope and quality checks and as a second reader. The
service exposes `GET /health` and `GET /models`, plus `POST /analyze` (Model 5
by default) and `POST /analyze/{model_id}` (Models 1–5). Gemini use requires
backend configuration, and images processed by that feature are sent to
Google's API.

ResNet-18 models are hosted on HuggingFace, Gemini API key -  through Google AI Studio. Website hosted on render.

Frontend is vanilla HTML/CSS.

# Sources
- **Primary Chest X-Ray Dataset (Kermany):** Kermany, Goldbaum, Cai et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018. Kaggle copy by Paul Mooney (CC BY 4.0): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Multiclass Extension Dataset (COVID-QU-Ex):** Anas Mohammed Tahir et al., *COVID-QU-Ex dataset (33,920 images; classes: COVID-19, Non-COVID, Normal)* from 10 public databases: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
- **External COVID-19 Validation Dataset (COVID-19 Radiography Database):** M.E.H. Chowdhury, T. Rahman, A. Khandakar et al., *COVID-19 Radiography Database*: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **External Normal Chest X-Ray Dataset (Montgomery County):** Jaeger S, Karargyris A, Antani S et al., *NLM Montgomery County CXR Set* (subset of normal frontal chest radiographs): https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- **External Normal Chest X-Ray Dataset (Shenzhen):** National Library of Medicine, *ChinaSet_AllFiles* (Shenzhen No. 3 People's Hospital normal frontal chest radiographs): https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

## Tools and models
- **PyTorch Model Architecture:** ResNet-18 (He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016), pretrained on ImageNet, sourced from `torchvision.models`.
- **Google AI Model:** Gemini Decision 1 (released November 2026), utilized for quality gating, scope verification, and clinical multi-class consensus: https://aistudio.google.com/docs/models/gemini-decision-1
- **Visual Explanations (Grad-CAM):** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- **AI Assistants & Tools:** Claude (Anthropic), Google Antigravity, Gemini in Google Colab.
---


Submission of YoanHristov for the Veritas AI Hackaton, 2026