# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Точные данные с ПРАВИЛЬНОЙ бинарной моделью
models_data = {
    "Базовая": {
        "accuracy": 0.3045,
        "f1_macro": 0.1622,
        "type": "baseline"
    },
    "v1_63": {
        "accuracy": 0.6337,
        "f1_macro": 0.6428,
        "class_accuracies": [0.820, 0.459, 0.574, 0.683],
        "confusion_matrix": np.array([
            [50, 7, 4, 0],
            [7, 28, 24, 2],
            [3, 11, 35, 12],
            [0, 1, 18, 41]
        ]),
        "type": "4-class"
    },
    "v2_65": {
        "accuracy": 0.6379,
        "f1_macro": 0.6425,
        "class_accuracies": [0.803, 0.508, 0.574, 0.667],
        "confusion_matrix": np.array([
            [49, 7, 5, 0],
            [9, 31, 20, 1],
            [3, 12, 35, 11],
            [0, 1, 19, 40]
        ]),
        "type": "4-class"
    },
    "Бинарная": {
        "accuracy": 0.984,
        "f1_macro": 0.98,
        "class_accuracies": [1.00, 0.97],  # Класс 2: 100%, Класс 5: 97%
        "confusion_matrix": np.array([
            [63, 0],
            [2, 60]
        ]),
        "type": "binary",
        "classes": ["2", "5"]  # Бинарная модель работала ТОЛЬКО с классами 2 и 5
    },
    "v3_81": {
        "accuracy": 0.8079,
        "f1_macro": 0.8107,
        "class_accuracies": [0.877, 0.819, 0.780, 0.756],
        "confusion_matrix": np.array([
            [71, 7, 3, 0],
            [5, 68, 9, 1],
            [0, 5, 64, 13],
            [0, 0, 20, 62]
        ]),
        "type": "4-class"
    },
    "v4_83": {
        "accuracy": 0.8262,
        "f1_macro": 0.8277,
        "class_accuracies": [0.877, 0.855, 0.805, 0.768],
        "confusion_matrix": np.array([
            [71, 9, 1, 0],
            [5, 71, 6, 1],
            [0, 5, 66, 11],
            [0, 0, 19, 63]
        ]),
        "type": "4-class"
    }
}

# ПОЛОТНО 1: ОБЩИЙ ПРОГРЕСС И АНАЛИЗ
fig1 = plt.figure(1, figsize=(18, 12))
gs1 = GridSpec(2, 3, figure=fig1)

# 1.1 Основной прогресс accuracy и F1
ax1 = fig1.add_subplot(gs1[0, :])
models_order = ["Базовая", "v1_63", "v2_65", "Бинарная", "v3_81", "v4_83"]
accuracies = [models_data[model]["accuracy"] for model in models_order]
f1_scores = [models_data[model].get("f1_macro", 0) for model in models_order]

x = np.arange(len(models_order))
width = 0.35

bars1 = ax1.bar(x - width / 2, [acc * 100 for acc in accuracies], width,
                label='Accuracy', alpha=0.7, color='blue')
bars2 = ax1.bar(x + width / 2, [f1 * 100 for f1 in f1_scores], width,
                label='F1-macro', alpha=0.7, color='red')

ax1.set_title('ЭВОЛЮЦИЯ КАЧЕСТВА МОДЕЛЕЙ (Accuracy и F1-macro)', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Метрика (%)', fontsize=12)
ax1.set_ylim(0, 110)
ax1.set_xticks(x)
ax1.set_xticklabels(models_order)

# Добавляем значения на столбцы
for bars, values in [(bars1, accuracies), (bars2, f1_scores)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 2,
                 f'{val * 100:.1f}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=9)

ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1.2 Accuracy по классам для 4-классовых моделей
ax2 = fig1.add_subplot(gs1[1, 0])
four_class_models = ["v1_63", "v2_65", "v3_81", "v4_83"]
class_names = ['Класс 2', 'Класс 3', 'Класс 4', 'Класс 5']

x_class = np.arange(len(four_class_models))
width_class = 0.2

for i, class_name in enumerate(class_names):
    class_accuracies = [models_data[model]["class_accuracies"][i] * 100 for model in four_class_models]
    bars_class = ax2.bar(x_class + i * width_class, class_accuracies, width_class, label=class_name, alpha=0.8)

    for bar, acc in zip(bars_class, class_accuracies):
        height = bar.get_height()
        if acc > 5:  # Не показываем текст на очень маленьких столбцах
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{acc:.1f}%', ha='center', va='bottom',
                     fontsize=6, fontweight='bold')

ax2.set_title('ТОЧНОСТЬ ПО КЛАССАМ (4-классовые)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)')
ax2.set_xticks(x_class + width_class * 1.5)
ax2.set_xticklabels(['v1', 'v2', 'v3', 'v4'], fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)

# 1.3 Анализ улучшений
ax3 = fig1.add_subplot(gs1[1, 1])
improvements_acc = [
    ("Базовая→v1", 32.9),
    ("v1→v2", 0.4),
    ("v2→v3", 17.0),
    ("v3→v4", 1.8)
]

improvements_f1 = [
    ("Базовая→v1", 48.0),
    ("v1→v2", -0.3),
    ("v2→v3", 16.8),
    ("v3→v4", 1.7)
]

labels = [imp[0] for imp in improvements_acc]
values_acc = [imp[1] for imp in improvements_acc]
values_f1 = [imp[1] for imp in improvements_f1]

x_imp = np.arange(len(labels))
width_imp = 0.35

bars_acc = ax3.bar(x_imp - width_imp / 2, values_acc, width_imp, label='Accuracy', alpha=0.7, color='blue')
bars_f1 = ax3.bar(x_imp + width_imp / 2, values_f1, width_imp, label='F1-macro', alpha=0.7, color='red')

ax3.set_title('ПРИРОСТ КАЧЕСТВА', fontweight='bold', fontsize=12)
ax3.set_ylabel('Δ Метрика (%)')
ax3.set_ylim(0, 50)
ax3.set_xticks(x_imp)
ax3.set_xticklabels(labels, fontsize=8)
ax3.legend(fontsize=8)

for bars, values in [(bars_acc, values_acc), (bars_f1, values_f1)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f'+{val}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=8)

ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 1.4 Сводная информация
ax4 = fig1.add_subplot(gs1[1, 2])
ax4.axis('off')

summary_text = (
    "КЛЮЧЕВЫЕ ВЫВОДЫ:\n\n"
    "• Базовая → v1: +32.9% Acc\n  +48.0% F1 (обучение)\n\n"
    "• v2 → v3: +17.0% Acc\n  +16.8% F1 (+900 данных)\n\n"
    "• Бинарная: 98.4% Acc\n  Только классы 2 и 5\n\n"
    "• F1-macro показывает\n  сбалансированность\n  по всем классам"
)

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
         fontsize=10, va='top', ha='left', linespacing=1.5,
         bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow"))

plt.tight_layout()

# ПОЛОТНО 2: CONFUSION MATRICES v1 и v2
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('CONFUSION MATRICES: v1 и v2', fontsize=16, fontweight='bold')

for i, (model, ax) in enumerate(zip(["v1_63", "v2_65"], [ax2a, ax2b])):
    cm = models_data[model]["confusion_matrix"]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annotations = []
    for i_row in range(cm.shape[0]):
        row = []
        for j_col in range(cm.shape[1]):
            abs_val = cm[i_row, j_col]
            pct_val = cm_normalized[i_row, j_col] * 100
            if i_row == j_col:
                row.append(f'{abs_val}\n({pct_val:.0f}%)')
            else:
                row.append(f'{abs_val}\n({pct_val:.1f}%)')
        annotations.append(row)

    sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['2', '3', '4', '5'],
                yticklabels=['2', '3', '4', '5'],
                cbar_kws={'label': 'Нормировано'},
                ax=ax, annot_kws={"size": 9})

    ax.set_title(
        f'{model}: {models_data[model]["accuracy"] * 100:.1f}% Acc, {models_data[model]["f1_macro"] * 100:.1f}% F1',
        fontweight='bold', fontsize=11)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истинно')

plt.tight_layout()

# ПОЛОТНО 3: CONFUSION MATRICES v3 и v4
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('CONFUSION MATRICES: v3 и v4', fontsize=16, fontweight='bold')

for i, (model, ax) in enumerate(zip(["v3_81", "v4_83"], [ax3a, ax3b])):
    cm = models_data[model]["confusion_matrix"]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annotations = []
    for i_row in range(cm.shape[0]):
        row = []
        for j_col in range(cm.shape[1]):
            abs_val = cm[i_row, j_col]
            pct_val = cm_normalized[i_row, j_col] * 100
            if i_row == j_col:
                row.append(f'{abs_val}\n({pct_val:.0f}%)')
            else:
                row.append(f'{abs_val}\n({pct_val:.1f}%)')
        annotations.append(row)

    sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['2', '3', '4', '5'],
                yticklabels=['2', '3', '4', '5'],
                cbar_kws={'label': 'Нормировано'},
                ax=ax, annot_kws={"size": 9})

    ax.set_title(
        f'{model}: {models_data[model]["accuracy"] * 100:.1f}% Acc, {models_data[model]["f1_macro"] * 100:.1f}% F1',
        fontweight='bold', fontsize=11)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истинно')

plt.tight_layout()

# ПОЛОТНО 4: БИНАРНАЯ МОДЕЛЬ (ТОЛЬКО КЛАССЫ 2 и 5)
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))

cm_binary = models_data["Бинарная"]["confusion_matrix"]
cm_binary_normalized = cm_binary.astype('float') / cm_binary.sum(axis=1)[:, np.newaxis]

binary_annotations = []
for i in range(2):
    row = []
    for j in range(2):
        abs_val = cm_binary[i, j]
        pct_val = cm_binary_normalized[i, j] * 100
        row.append(f'{abs_val}\n({pct_val:.1f}%)')
    binary_annotations.append(row)

sns.heatmap(cm_binary_normalized, annot=binary_annotations, fmt='', cmap='Greens',
            xticklabels=['2', '5'],  # ТОЛЬКО классы 2 и 5
            yticklabels=['2', '5'],
            cbar_kws={'label': 'Нормировано'},
            ax=ax4)

ax4.set_title(f'БИНАРНАЯ МОДЕЛЬ: 98.4% Accuracy\n(классификация только 2 и 5)',
              fontweight='bold', fontsize=14)
ax4.set_xlabel('Предсказано')
ax4.set_ylabel('Истинно')

# Добавляем статистику для классов 2 и 5
stats_text = f"Class 2 Accuracy: 100.0%\nClass 5 Accuracy: 96.8%"
ax4.text(0.5, -0.15, stats_text, transform=ax4.transAxes, ha='center', va='top',
         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()

# Показываем все полотна
plt.show()

# Статистика в консоли
print("=" * 60)
print("СТАТИСТИКА ПРОГРЕССА МОДЕЛЕЙ")
print("=" * 60)
print(
    f"Базовая модель: {models_data['Базовая']['accuracy'] * 100:.1f}% Accuracy, {models_data['Базовая']['f1_macro'] * 100:.1f}% F1")
print(
    f"Лучшая 4-классовая (v4): {models_data['v4_83']['accuracy'] * 100:.1f}% Accuracy, {models_data['v4_83']['f1_macro'] * 100:.1f}% F1")
print(f"Бинарная модель (только 2 и 5): {models_data['Бинарная']['accuracy'] * 100:.1f}% Accuracy")
print(f"Class 2 Accuracy: {models_data['Бинарная']['class_accuracies'][0] * 100:.1f}%")
print(f"Class 5 Accuracy: {models_data['Бинарная']['class_accuracies'][1] * 100:.1f}%")
print(f"Общий прогресс Accuracy: +{(models_data['v4_83']['accuracy'] - models_data['Базовая']['accuracy']) * 100:.1f}%")
print(f"Общий прогресс F1-macro: +{(models_data['v4_83']['f1_macro'] - models_data['Базовая']['f1_macro']) * 100:.1f}%")