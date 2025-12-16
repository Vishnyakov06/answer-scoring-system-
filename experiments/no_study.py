# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
os.environ["WANDB_DISABLED"] = "true"

test_df = pd.read_csv("../datasets/test.csv", encoding="cp1251", sep=";", engine='python')
test_df = test_df.dropna(
    subset=["Вопрос", "Эталонный ответ преподавателя", "Ответ студента", "Оценка"]
)

print("Распределение оценок:")
print(test_df["Оценка"].value_counts().sort_index())
label_encoder = LabelEncoder()

label_encoder.fit(["2", "3", "4", "5"])

test_df["label"] = label_encoder.transform(
    test_df["Оценка"].astype(str)
)

print(f"Classes: {label_encoder.classes_}")
print(f"Test samples: {len(test_df)}")

MODEL_NAME = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
class SemanticSimilarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        question = row['Вопрос']
        reference_answer = row['Эталонный ответ преподавателя']
        student_answer = row['Ответ студента']
        text_a = f"Вопрос: {question} Ответ студента: {student_answer}"
        text_b = f"Эталонный ответ: {reference_answer}"
        encoded = self.tokenizer(
            text_a,
            text_b,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return encoded
test_dataset = SemanticSimilarityDataset(test_df, tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
print("\n" + "=" * 50)
print("ТЕСТ МОДЕЛИ БЕЗ ОБУЧЕНИЯ")
print("=" * 50)
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        inputs = test_dataset[i]
        outputs = model(
            input_ids=inputs['input_ids'].unsqueeze(0),
            attention_mask=inputs['attention_mask'].unsqueeze(0)
        )
        pred = outputs.logits.argmax(-1).item()
        predictions.append(pred)
        true_labels.append(inputs['labels'].item())
        print(i)
accuracy = accuracy_score(true_labels, predictions)
f1_macro = f1_score(true_labels, predictions, average='macro')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-macro: {f1_macro:.4f}")
print(f"Случайное угадывание: {1.0 / len(label_encoder.classes_):.4f}")
print("\n" + "=" * 40)
print("СРАВНЕНИЕ С ОБУЧЕННОЙ МОДЕЛЬЮ")
print("=" * 40)
print(f"Без обучения: Accuracy={accuracy:.4f}, F1={f1_macro:.4f}")
print(f"После обучения: Accuracy=0.6255, F1=0.6262")
print(f"Улучшение: Accuracy=+{0.6255 - accuracy:.4f}, F1=+{0.6262 - f1_macro:.4f}")
results_dir = "research_results"
os.makedirs(results_dir, exist_ok=True)
results = {
    "baseline_accuracy": float(accuracy),
    "baseline_f1_macro": float(f1_macro),
    "trained_accuracy": 0.6255,
    "trained_f1_macro": 0.6262,
    "improvement_accuracy": float(0.6255 - accuracy),
    "improvement_f1_macro": float(0.6262 - f1_macro)
}
with open(f"{results_dir}/baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nРезультаты сохранены в: {results_dir}/baseline_results.json")