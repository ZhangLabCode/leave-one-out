# UV-Vis Data Analysis Tool

A Python-based desktop application for **UV-Vis spectroscopic data analysis**, built with **PySide6 (Qt)** and **matplotlib**.  
The tool provides a workflow for importing, stacking, filtering, fitting (quadratic + Leave-One-Out), and predicting spectral data with a clean high-contrast UI.

---

## ✨ Features
- **Import & stack** A_ISARS and UV-Vis matrices from Excel/CSV.  
- **Filter & clean** data (drop NaNs, apply numeric ranges, exclude wavelengths).  
- **Fit & LOO** (leave-one-out) quadratic models per wavelength.  
- **Outlier handling** with pick, box, or lasso selection.  
- **Prediction & visualization** of UV-Vis spectra based on fitted coefficients.  
- **Export** results and plots to Excel, CSV, PNG, SVG, or PDF.  
- **Modern high-contrast UI** with light gray theme and blue accent.  

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Run the main script:

```bash
python version6.5.py
```

The application will launch a desktop window with the following workflow:

1. **Import & Stack**: Load A_ISARS and UV-Vis matrices, then combine them.  
2. **Fit & LOO**: Run a quadratic fit across wavelengths and perform leave-one-out modeling.  
3. **Predict & Plots**: Predict UV-Vis values, export predictions, and generate comparison plots.  

---


