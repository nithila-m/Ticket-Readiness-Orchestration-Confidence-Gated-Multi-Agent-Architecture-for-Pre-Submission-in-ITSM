# Lightweight trained fallback classifier — proof of concept

TF-IDF (1-2 grams) + HistGradientBoostingClassifier, 5-fold stratified CV

n = 150 tickets, 5 categories

Cross-validated accuracy: **0.767**

The pipeline uses TF-IDF text features followed by a sparse-to-dense conversion because HistGradientBoostingClassifier requires dense input.

This is a separate, classically-trained model — not a fine-tuned version of the Gemini or Groq LLMs used elsewhere in the pipeline. It is a first step toward the XLNet/HGB fallback item on the research roadmap, evaluated honestly on a small (150-example) synthetic set — a proof-of-concept, not a production-ready model.

## Classification Report

```
                     precision    recall  f1-score   support

ad_account_creation       0.74      0.68      0.71        34
           ms_teams       0.87      0.87      0.87        30
    printer_support       0.83      0.70      0.76        27
          vit_email       0.67      0.79      0.72        33
      wifi_internet       0.78      0.81      0.79        26

           accuracy                           0.77       150
          macro avg       0.78      0.77      0.77       150
       weighted avg       0.77      0.77      0.77       150

```

## Confusion Matrix

Rows = true labels, columns = predicted labels.

```
                     ad_account_creation  ms_teams  printer_support  vit_email  wifi_internet
ad_account_creation                   23         2                1          7              1
ms_teams                               1        26                1          2              0
printer_support                        2         0               19          2              4
vit_email                              4         2                0         26              1
wifi_internet                          1         0                2          2             21
```