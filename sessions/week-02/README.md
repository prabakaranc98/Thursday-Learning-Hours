# Week 02 – Data Visualization with Matplotlib & Seaborn

> **Thursday Learning Hours · Session 02**

## 🎯 Learning Objectives

By the end of this session you will be able to:

1. Build the full range of **Matplotlib** chart types (line, bar, scatter, histogram, heatmap).
2. Produce statistical visualisations with **Seaborn** (distribution plots, pairplot, boxplot, violin).
3. Customise figures: titles, labels, tick marks, colour palettes, and layouts.
4. Choose the right chart type for a given analytical question.

## 🗂️ Contents

| File | Description |
|------|-------------|
| `README.md` | Session overview, objectives & references |
| `demo.ipynb` | Hands-on Jupyter notebook *(coming soon)* |

## 📖 Overview

Visualisation is the first step in any exploratory data analysis. A well-chosen chart can reveal patterns, outliers, and relationships that tables of numbers hide.

**Matplotlib** is the foundational plotting library — it gives precise, low-level control over every element of a figure. **Seaborn** is built on top of Matplotlib and provides high-level functions tied to statistical concepts, producing attractive charts in fewer lines of code.

In this session we work through both libraries using the same student dataset from Week 01, so we can focus on the charting skills rather than data wrangling.

## 🔑 Key Concepts

- **Figure / Axes / Artist** – the three-layer Matplotlib object model
- **Subplots** – arranging multiple charts in a grid with `plt.subplots()`
- **Statistical plots** – distribution, box, violin, pair, and heatmap
- **Colour palettes** – choosing accessible, print-friendly colours
- **Saving figures** – `savefig()` with DPI and format control

## 🛠️ Prerequisites

- [x] Week 01 – NumPy & Pandas Essentials

## 📦 Dependencies

```bash
pip install matplotlib seaborn pandas numpy
```

## 📚 References & Further Reading

1. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) – matplotlib.org
2. [Seaborn Documentation](https://seaborn.pydata.org/) – seaborn.pydata.org
3. [Python Data Science Handbook – Ch. 4](https://jakevdp.github.io/PythonDataScienceHandbook/) – Jake VanderPlas (free online)
4. [From Data to Viz](https://www.data-to-viz.com/) – Chart type decision tree
