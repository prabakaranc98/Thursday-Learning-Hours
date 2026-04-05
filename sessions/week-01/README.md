# Week 01 – Python for Data Science: NumPy & Pandas Essentials

> **Thursday Learning Hours · Session 01**

## 🎯 Learning Objectives

By the end of this session you will be able to:

1. Create and manipulate **NumPy arrays** (indexing, slicing, broadcasting, vectorised operations).
2. Load, explore and clean tabular data with **Pandas DataFrames**.
3. Apply common data-wrangling patterns (filtering, groupby, merge, pivot).
4. Understand the performance differences between pure Python loops and NumPy/Pandas vectorisation.

## 🗂️ Contents

| File | Description |
|------|-------------|
| `README.md` | Session overview, objectives & references |
| `demo.ipynb` | Hands-on Jupyter notebook with working experiments |

## 📖 Overview

NumPy and Pandas are the two foundational libraries for any data-science or machine-learning workflow in Python.

**NumPy** provides an efficient N-dimensional array object and a comprehensive collection of mathematical functions that operate on arrays without explicit Python loops. Under the hood, the heavy lifting is done in optimised C code, making operations orders of magnitude faster than equivalent pure-Python code.

**Pandas** is built on top of NumPy and introduces the `Series` (1-D) and `DataFrame` (2-D) abstractions that make working with heterogeneous, labelled, tabular data intuitive. It covers the full ETL workflow: reading data from CSV/Excel/SQL, cleaning missing values, transforming columns, aggregating with `groupby`, and joining multiple tables.

In this session we work through both libraries end-to-end using a real-world inspired dataset so every concept has an immediately runnable example.

## 🔑 Key Concepts

- **ndarray** – NumPy's core N-dimensional array object
- **Broadcasting** – how NumPy automatically expands shapes for element-wise ops
- **Vectorisation** – replacing loops with array-level operations for speed
- **DataFrame / Series** – Pandas labelled 1-D and 2-D data containers
- **GroupBy / Aggregation** – split-apply-combine pattern
- **Merge / Join** – combining DataFrames on shared keys

## 🛠️ Prerequisites

- [x] Basic Python (lists, dicts, functions, comprehensions)
- [ ] No prior NumPy or Pandas experience required

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib
```

## 📚 References & Further Reading

1. [NumPy Official Documentation](https://numpy.org/doc/stable/) – numpy.org
2. [Pandas Official Documentation](https://pandas.pydata.org/docs/) – pandas.pydata.org
3. [Python Data Science Handbook – Ch. 2 & 3](https://jakevdp.github.io/PythonDataScienceHandbook/) – Jake VanderPlas (free online)
4. [NumPy Broadcasting Visual Guide](https://numpy.org/doc/stable/user/basics.broadcasting.html) – numpy.org
5. [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html) – pandas.pydata.org
