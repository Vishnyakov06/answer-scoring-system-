# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Данные моделей
models_data = {
    "v1_63": {
        "class_accuracies": [0.820, 0.459, 0.574, 0.683],
    },
    "v2_65": {
        "class_accuracies": [0.803, 0.508, 0.574, 0.667],
    },
    "v3_81": {
        "class_accuracies": [0.877, 0.819, 0.780, 0.756],
    },
    "v4_83": {
        "class_accuracies": [0.877, 0.855, 0.805, 0.768],
    }
}

# Создаем фигуру
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Классы и цвета
class_names = ['Класс 2', 'Класс 3', 'Класс 4', 'Класс 5']
class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']  # Разные маркеры для классов

# Версии моделей
model_versions = ["v1", "v2", "v3", "v4"]
x_positions = np.arange(len(model_versions))

# === ЛЕВАЯ ЧАСТЬ: Основной график ===
for i, (class_name, color, marker) in enumerate(zip(class_names, class_colors, markers)):
    # Собираем accuracy для этого класса по всем версиям
    accuracies = [models_data[model]["class_accuracies"][i] * 100
                  for model in ['v1_63', 'v2_65', 'v3_81', 'v4_83']]

    # Линия с маркерами
    line = ax.plot(x_positions, accuracies,
                   label=class_name,
                   color=color,
                   marker=marker,
                   markersize=10,
                   linewidth=3,
                   alpha=0.8)

    # Подписи значений в точках
    for j, (x, y) in enumerate(zip(x_positions, accuracies)):
        ax.text(x, y + (1.5 if i != 3 else -3), f'{y:.1f}%',
                ha='center', va='bottom' if i != 3 else 'top',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          alpha=0.8))

# Настройки основного графика
ax.set_title('ДИНАМИКА ТОЧНОСТИ ПО КЛАССАМ',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Версия модели', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(30, 100)
ax.set_xlim(-0.2, len(model_versions) - 0.8)
ax.set_xticks(x_positions)
ax.set_xticklabels(model_versions, fontsize=11)

# Сетка
ax.grid(True, alpha=0.3, linestyle='--')

# Легенда
ax.legend(title='Классы оценки', fontsize=10, title_fontsize=11,
          loc='upper left', bbox_to_anchor=(1.02, 1))

# === ПРАВАЯ ЧАСТЬ: График прироста между версиями ===
# Подготовка данных для графика прироста
improvement_data = {class_name: [] for class_name in class_names}

for i, class_name in enumerate(class_names):
    accuracies = [models_data[model]["class_accuracies"][i] * 100
                  for model in ['v1_63', 'v2_65', 'v3_81', 'v4_83']]

    # Рассчитываем прирост между соседними версиями
    improvements = []
    for j in range(len(accuracies) - 1):
        improvement = accuracies[j + 1] - accuracies[j]
        improvements.append(improvement)

    improvement_data[class_name] = improvements

# X-позиции для графика прироста (между версиями)
x_improvements = np.arange(len(model_versions) - 1)
width = 0.2

# Рисуем столбцы прироста
for i, (class_name, color) in enumerate(zip(class_names, class_colors)):
    improvements = improvement_data[class_name]
    positions = x_improvements + (i - 1.5) * width

    bars = ax2.bar(positions, improvements, width,
                   label=class_name, color=color, alpha=0.7)

    # Подписи значений
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va_pos = 'bottom' if height >= 0 else 'top'
        y_offset = 0.2 if height >= 0 else -0.5
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 height + y_offset,
                 f'{imp:+.1f}%',
                 ha='center', va=va_pos,
                 fontsize=9, fontweight='bold')

# Настройки графика прироста
ax2.set_title('ПРИРОСТ ТОЧНОСТИ МЕЖДУ ВЕРСИЯМИ',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Переход между версиями', fontsize=12)
ax2.set_ylabel('Δ Accuracy (%)', fontsize=12)
ax2.set_xticks(x_improvements)
ax2.set_xticklabels(['v1→v2', 'v2→v3', 'v3→v4'], fontsize=11)
ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Вычисляем и отображаем общую статистику
stats_text = ""
for i, class_name in enumerate(class_names):
    start_acc = models_data['v1_63']["class_accuracies"][i] * 100
    end_acc = models_data['v4_83']["class_accuracies"][i] * 100
    total_improvement = end_acc - start_acc

    stats_text += f"{class_name}:\n"
    stats_text += f"  Начало: {start_acc:.1f}%\n"
    stats_text += f"  Конец: {end_acc:.1f}%\n"
    stats_text += f"  Общий прирост: {total_improvement:+.1f}%\n\n"

# Добавляем статистику на график
fig.text(0.99, 0.5, stats_text,
         transform=fig.transFigure,
         fontsize=9,
         verticalalignment='center',
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=1",
                   facecolor="lightyellow",
                   alpha=0.9))

plt.tight_layout(rect=[0, 0, 0.88, 1])  # Оставляем место справа для текста
plt.show()

# Дополнительная статистика в консоли
print("=" * 60)
print("АНАЛИЗ ПРОГРЕССА ПО КЛАССАМ")
print("=" * 60)
print("\n1. Общий прогресс от v1 до v4:")
for i, class_name in enumerate(class_names):
    start = models_data['v1_63']["class_accuracies"][i] * 100
    end = models_data['v4_83']["class_accuracies"][i] * 100
    progress = end - start
    print(f"   {class_name}: {start:.1f}% → {end:.1f}% (Δ{progress:+.1f}%)")

print("\n2. Ключевые улучшения между версиями:")
for i, transition in enumerate(['v1→v2', 'v2→v3', 'v3→v4']):
    print(f"\n   {transition}:")
    for j, class_name in enumerate(class_names):
        if i == 0:
            start = models_data['v1_63']["class_accuracies"][j]
            end = models_data['v2_65']["class_accuracies"][j]
        elif i == 1:
            start = models_data['v2_65']["class_accuracies"][j]
            end = models_data['v3_81']["class_accuracies"][j]
        else:
            start = models_data['v3_81']["class_accuracies"][j]
            end = models_data['v4_83']["class_accuracies"][j]

        improvement = (end - start) * 100
        if abs(improvement) > 0.1:  # Показываем только значимые изменения
            print(f"     {class_name}: {improvement:+.1f}%")