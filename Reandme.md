# Narrative Consistency Checker - Ollama Version

A local, cost-free solution for detecting contradictions in character backstories using Ollama LLMs and semantic search.

**Cost**: $0.00 (100% free, runs locally)  
**Time**: ~40 minutes for 60 test cases  
**Accuracy**: Depends on model size (llama3.1:8b recommended)

---

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## ✨ Features

- ✅ **Free & Local**: No API costs, runs entirely on your machine
- ✅ **Temporal Verification**: Checks dates, ages, and timeline math (±1 year tolerance)
- ✅ **Multi-Pass Retrieval**: 3-pass semantic search for comprehensive evidence gathering
- ✅ **Structured Claims**: Categorizes claims by type (TEMPORAL, SPATIAL, RELATIONAL, etc.)
- ✅ **Confidence Scoring**: Each prediction includes confidence level
- ✅ **Training Evaluation**: Calculates accuracy with confusion matrix
- ✅ **Windows Compatible**: Fallback vector store (no Pathway dependency required)

---

## 🔧 Prerequisites

- **Python**: 3.8 - 3.12 (Python 3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: ~5GB for Ollama + models
- **OS**: Windows, macOS, or Linux

---

## 📥 Installation

### Step 1: Install Ollama

#### **Windows**

1. Download Ollama installer:
   ```
   https://ollama.ai/download/windows
   ```

2. Run the installer (`OllamaSetup.exe`)

3. Verify installation:
   ```powershell
   ollama --version
   ```

#### **macOS**

1. Download Ollama:
   ```bash
   https://ollama.ai/download/mac
   ```

2. Or install via Homebrew:
   ```bash
   brew install ollama
   ```

3. Verify:
   ```bash
   ollama --version
   ```

#### **Linux**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Verify:
```bash
ollama --version
```

---

### Step 2: Download LLM Model

Choose a model based on your hardware:

#### **Recommended: Llama 3.1 8B** (Best balance of speed & accuracy)
```bash
ollama pull llama3.1:8b
```

**Other Options:**

| Model | Size | Speed | Accuracy | RAM Required |
|-------|------|-------|----------|--------------|
| `llama3.2:3b` | 3B | ⚡⚡⚡ Fast | ⭐⭐ Good | 4GB |
| `llama3.1:8b` | 8B | ⚡⚡ Medium | ⭐⭐⭐ Better | 8GB |
| `llama3.1:70b` | 70B | ⚡ Slow | ⭐⭐⭐⭐ Best | 64GB |

**Verify model is downloaded:**
```bash
ollama list
```

You should see:
```
NAME              ID              SIZE    MODIFIED
llama3.1:8b       abc123...       4.7 GB  2 hours ago
```

---

### Step 3: Clone Repository & Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/narrative-consistency-checker.git
cd narrative-consistency-checker

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4: Prepare Data Files

#### **Required Files:**

1. **train.csv** (for training evaluation)
   ```csv
   id,content,book_name,char,label
   1,"Character backstory text...","The Count of Monte Cristo","Edmond Dantes","consistent"
   2,"Another backstory...","In Search of the Castaways","Jacques Paganel","contradict"
   ```

2. **test.csv** (for predictions)
   ```csv
   id,content,book_name,char
   101,"Test backstory...","The Count of Monte Cristo","Fernand Mondego"
   102,"Another test...","In Search of the Castaways","Mary Grant"
   ```

3. **Novel files** (place in project root):
   - `The_Count_of_Monte_Cristo.txt`
   - `In_search_of_the_castaways.txt`

#### **Create .env file** (optional):
```bash
# .env
OLLAMA_MODEL=llama3.1:8b
SKIP_TRAINING_EVAL=false
```

---

## 📁 Project Structure

```
narrative-consistency-checker/
├── main.py                              # Main solution script
├── requirements.txt                     # Python dependencies
├── .env                                 # Configuration (optional)
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
│
├── train.csv                           # Training data
├── test.csv                            # Test data
│
├── The_Count_of_Monte_Cristo.txt       # Novel 1
├── In_search_of_the_castaways.txt      # Novel 2
│
└── outputs/                            # Generated after running
    ├── results.csv                     # Submission file (id, prediction)
    ├── results_detailed_ollama.csv     # With rationales
    └── training_evaluation_ollama.csv  # Training accuracy report
```

---

## 🚀 Usage

### Basic Usage

```bash
# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# Run the solution
python main.py
```

### Expected Output

```
======================================================================
HACKATHON 2026 - Ollama Version (Train/Test Format)
Model: llama3.1:8b | Cost: $0.00
======================================================================

Checking Ollama setup...
✓ Ollama working with model: llama3.1:8b

[STEP 1] Building vector stores...

📚 Building for: The Count of Monte Cristo
  Building vector store for: The_Count_of_Monte_Cristo.txt
  Loading novel: The_Count_of_Monte_Cristo.txt
  Created 2847 chunks
  Encoding chunks...

[STEP 2] Evaluating on training data...
======================================================================
EVALUATING ON TRAINING DATA
======================================================================

[1/40] ID=1 | Actual: consistent
    Extracting claims...
✓ Extracted 8 structured claims
    Found 8 claims to verify
    [1/8] Checking: Born in 1815...
      Category: TEMPORAL | Importance: HIGH
      Retrieved 12 evidence chunks
      ✓ Consistent (confidence: 0.85)
    ...

======================================================================
TRAINING ACCURACY: 72.50% (29/40)

Confusion Matrix:
  True Positives (correct consistent):  15
  True Negatives (correct contradict):  14
  False Positives (missed contradict):  6
  False Negatives (wrong contradict):   5

Total Ollama Requests: 320
Cost: $0.00 (Local model)
======================================================================

[STEP 3] Generating test predictions...
...

✓ PREDICTIONS SAVED
  Submission file: results.csv
  Detailed file: results_detailed_ollama.csv
  Total: 60
  Consistent: 4
  Contradict: 56

Total Ollama Requests: 498
Total Cost: $0.00 (Free!)
======================================================================
```

---

## ⚙️ Configuration

### Environment Variables (.env file)

```bash
# Model selection
OLLAMA_MODEL=llama3.1:8b

# Skip training evaluation (faster, test-only mode)
SKIP_TRAINING_EVAL=false

# For debugging
DEBUG=false
```

### In-Code Configuration

Edit `main.py` to customize:

```python
# Model configuration
OLLAMA_MODEL = "llama3.1:8b"

# Novel mapping
BOOK_MAPPING = {
    'In Search of the Castaways': 'In_search_of_the_castaways.txt',
    'The Count of Monte Cristo': 'The_Count_of_Monte_Cristo.txt'
}

# Retrieval settings
k_per_pass = 5  # Chunks per retrieval pass (reduce to 3 for speed)

# Token limits
num_predict = 1000  # Max tokens per LLM call (reduce to 500 for speed)
```

---

## 🐛 Troubleshooting

### Issue: "Ollama not found"

**Solution:**
```bash
# Check if Ollama is installed
ollama --version

# If not installed, download from:
https://ollama.ai/download
```

---

### Issue: "Model not found: llama3.1:8b"

**Solution:**
```bash
# Download the model
ollama pull llama3.1:8b

# Verify it's downloaded
ollama list
```

---

### Issue: "Connection refused" or "Ollama server not running"

**Solution:**

**Windows/macOS:** Ollama runs as a background service automatically. Restart your computer.

**Linux:**
```bash
# Start Ollama service
ollama serve

# Or run in background
nohup ollama serve &
```

---

### Issue: "Out of memory"

**Solutions:**

1. **Use smaller model:**
   ```bash
   ollama pull llama3.2:3b
   ```
   Update `.env`:
   ```
   OLLAMA_MODEL=llama3.2:3b
   ```

2. **Reduce chunk retrieval:**
   In `main.py`, line ~350:
   ```python
   evidence_chunks = multi_pass_retrieval(vector_store, claim, k_per_pass=3)  # Was 5
   ```

3. **Close other applications** to free up RAM

---

### Issue: "Novel file not found"

**Solution:**

Check file names match exactly (case-sensitive):
```bash
# Windows PowerShell
Get-ChildItem *.txt

# macOS/Linux
ls -l *.txt
```

Ensure:
- `The_Count_of_Monte_Cristo.txt` (underscores, not spaces)
- `In_search_of_the_castaways.txt` (lowercase "search")

---

### Issue: "Very slow performance (>1 hour)"

**Solutions:**

1. **Use GPU acceleration** (if available):
   
   Check if Ollama is using GPU:
   ```bash
   ollama ps
   ```

2. **Reduce token generation:**
   ```python
   # In main.py, reduce num_predict
   options={'temperature': 0, 'num_predict': 500}  # Was 1000
   ```

3. **Skip training evaluation:**
   ```bash
   # .env
   SKIP_TRAINING_EVAL=true
   ```

4. **Reduce retrieval passes:**
   ```python
   # Change k_per_pass from 5 to 3
   evidence_chunks = multi_pass_retrieval(vector_store, claim, k_per_pass=3)
   ```

---

## ⚡ Performance Optimization

### Speed vs Accuracy Trade-offs

| Optimization | Speed Gain | Accuracy Impact | How to Apply |
|--------------|------------|-----------------|--------------|
| Use `llama3.2:3b` | 2-3x faster | -5-10% accuracy | `ollama pull llama3.2:3b` |
| Reduce `k_per_pass` to 3 | 1.5x faster | -2-5% accuracy | Edit line ~350 in `main.py` |
| Reduce `num_predict` to 500 | 1.3x faster | -1-3% accuracy | Edit all `ollama.generate()` calls |
| Skip training eval | 2x faster | No impact on test | Set `SKIP_TRAINING_EVAL=true` |

### Recommended Fast Setup (for quick testing):

```bash
# Use smaller, faster model
ollama pull llama3.2:3b
```

**.env file:**
```
OLLAMA_MODEL=llama3.2:3b
SKIP_TRAINING_EVAL=true
```

**Edit main.py:**
```python
# Line ~350
evidence_chunks = multi_pass_retrieval(vector_store, claim, k_per_pass=3)

# All ollama.generate() calls
options={'temperature': 0, 'num_predict': 500}
```

**Expected time**: ~15-20 minutes for 60 test cases

---

## 📊 Output Files

### results.csv (Submission File)
```csv
id,prediction
1,0
2,1
3,0
```

### results_detailed_ollama.csv (With Rationales)
```csv
id,prediction,rationale
1,0,"Born in 1820... - Evidence shows born in 1815, timeline contradiction"
2,1,"All 7 claims verified consistent (avg confidence: 0.82)"
```

### training_evaluation_ollama.csv
```csv
id,predicted,actual,correct,rationale
1,1,1,True,"All claims consistent..."
2,0,0,True,"Claim about childhood contradicted..."
```

---

## 📝 Important Notes

### 1. **Model Quality Matters**

- `llama3.2:3b`: Fast but may miss subtle contradictions
- `llama3.1:8b`: **Recommended** - good balance
- `llama3.1:70b`: Best accuracy but requires 64GB RAM and very slow

### 2. **First Run is Slower**

First run includes:
- Loading embedding model (~500MB)
- Encoding novel chunks (~2-3 minutes per novel)

Subsequent runs are faster.

### 3. **No Internet Required**

Once Ollama and models are downloaded, everything runs offline.

### 4. **Reproducibility**

Set `temperature=0` for deterministic results (already configured).

### 5. **Cost Comparison**

| Solution | Cost (60 test cases) | Time |
|----------|---------------------|------|
| Gemini API | ~$2-5 | 5-10 min |
| OpenAI GPT-4 | ~$10-20 | 5-10 min |
| **Ollama (This)** | **$0.00** | **40 min** |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🆘 Support

**Issues?** Open a GitHub issue with:
- Python version: `python --version`
- Ollama version: `ollama --version`
- Model used: Check `.env` or output
- Error message (full traceback)

---

## 🎯 Quick Start Checklist

- [ ] Install Ollama
- [ ] Download model: `ollama pull llama3.1:8b`
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Add `train.csv`, `test.csv`, and novel `.txt` files
- [ ] Run: `python main.py`
- [ ] Check `results.csv` output

---

**Total setup time**: ~15 minutes  
**Total cost**: $0.00  
**Happy consistency checking!** 🚀