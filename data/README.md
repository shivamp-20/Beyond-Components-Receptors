# Data Directory

Place your dataset CSV files in this directory.

## Expected Files

### Gender Pronoun (GP) Task
- `train_1k_gp.csv` - Training data (1000 samples)
- `val_gp.csv` - Validation data
- `test_gp.csv` - Test data

**Required columns:**
- `prefix` - Input sentence prefix (e.g., "The doctor said that")
- `pronoun` - Correct pronoun (" he" or " she")
- `name` - Name in the sentence
- `corr_prefix` - Corrupted prefix
- `corr_pronoun` - Corrupted pronoun
- `corr_name` - Corrupted name

### Indirect Object Identification (IOI) Task
- `train_1k_ioi.csv` - Training data
- `val_ioi.csv` - Validation data
- `test_1k_ioi.csv` - Test data

**Required columns:**
- `ioi_sentences_input` - Clean input sentence
- `ioi_sentences_labels` - Correct label (indirect object name)
- `corr_ioi_sentences_input` - Corrupted input
- `ioi_sentences_labels_wrong` - Incorrect label (subject name)

### Greater Than (GT) Task
- `train_gt_2k.csv` - Training data
- `val_gt_500.csv` - Validation data
- `test_gt.csv` - Test data

## Data Format

All files should be CSV format with headers. Example GP data:

```csv
prefix,pronoun,name,corr_prefix,corr_pronoun,corr_name
"The nurse said that", she,"Mary","The nurse said that", he,"John"
"The doctor explained that", he,"Robert","The doctor explained that", she,"Lisa"
```

Note: Pronouns should include the leading space (e.g., " he" not "he").
