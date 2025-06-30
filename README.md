# 📘 Pre-Trade Share Buyback Tool

## 🔍 Purpose
This tool helps simulate and evaluate corporate share buyback programs **before execution**. It is designed for use by finance professionals and quantitative researchers who want to:

- Estimate expected buyback costs  
- Compare execution benchmarks (VWAP, TWAP, harmonic mean)  
- Evaluate execution strategies (e.g., uniform daily execution)  
- Use historical or live data (via Yahoo Finance)  
- Export results in CSV or PDF format  

---

## 🧱 Project Structure

```
pretradeTool/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── config.py               # Centralized base path and folders
├── core/
│   ├── simulator.py        # Buyback execution logic
│   ├── benchmarks.py       # Benchmark computations
│   ├── utils.py            # Data validation/loading
│   └── exporter.py         # Export to CSV / PDF
├── data/
│   └── sample_data.csv     # Example input
├── exports/                # Output files
├── live/
│   └── yfinance_loader.py  # Live data (Yahoo Finance)
└── tests/
    ├── test_simulator.py
    └── test_benchmarks.py
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```

### 3. Provide Input Data
Upload your own CSV file with:

- `Date`, `Close`, `Volume`

Or use the provided example: `data/sample_data.csv`.

### 4. Configure Parameters
- Total buyback value (e.g., $10M)
- Execution horizon (e.g., 60 trading days)
- Benchmark type

### 5. Review Results
The app displays:
- Simulated average execution price
- Benchmark comparison
- Export options (CSV, PDF)

---

## ⚙️ Optional: Use Live Data

```python
from live.yfinance_loader import download_data
df = download_data("AAPL", "2023-01-01", "2023-12-31")
```

---

## 🧪 Running Tests
```bash
pytest tests/
```

---

## 🔄 Planned Extensions

| Feature | Status |
|--------|--------|
| Front-/back-loaded execution strategies | 🔲 Planned |
| Monte Carlo simulation | 🔲 Planned |
| Compliance features (10b-18) | 🔲 Planned |
| REST API for automation | 🔲 Planned |

---

## 👤 Author

**Prof. Dr. Joerg R. Osterrieder**  

University of Twente · Bern Business School  

[joergosterrieder.com](https://joergosterrieder.com)
