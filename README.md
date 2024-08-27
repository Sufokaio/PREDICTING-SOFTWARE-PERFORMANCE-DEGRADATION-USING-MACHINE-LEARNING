# PREDICTING-SOFTWARE-PERFORMANCE-DEGRADATION-USING-MACHINE-LEARNING

El Hihi Taha 0567414 Taha.El.Hihi@vub.be


This package uses the Tailor framework: https://github.com/jun-zeng/Tailor/tree/main

Some functions have been renamed to avoid context confusion (e.g., code clone -> performance degradation).
We do not claim that Tailor is our work, in fact we only made a couple of modifications. 

This package also uses data from the Code4Bench dataset: https://github.com/code4bench/Code4Bench 
The Java submissions used and the scripts that prepare the data are in this package.


## Training Logs and Results

The training logs and results are available in the `cpgnn/log` folder.

### Results for Comparing Combination Methods on "Within" Split

**Table 4.2:**

- **Euclidean Distance:** [cpgnn/log/new_dist_new.txt](cpgnn/log/new_dist_new.txt)
- **Compare Regressions:** [cpgnn/log/new_regs_new.txt](cpgnn/log/new_regs_new.txt)
- **Concatenation:** [cpgnn/log/new_conc_new.txt](cpgnn/log/new_conc_new.txt)

### Results for "Compare Regressions" on Unseen Split

**Table 4.3:**

- **Compare Regressions:** [cpgnn/log/new_regs_unseen.txt](cpgnn/log/new_regs_unseen.txt)

### Results of Ablation Test (Excluding CFG and DF Edges)

**Table 4.4:**

- **Ablation Test:** [cpgnn/log/new_regs_without_cfg_dfg.txt](cpgnn/log/new_regs_without_cfg_dfg.txt)

## Datasets

Due to the large amount of data, the datasets have been stored externally on Google Drive. The corresponding links for each part can be found in the following text files:

- `datasets.txt`
- `cpgnn/data.txt`
- `cpgnn/data2.txt`

Please download them and unzip them in the corresponding paths.

### Datasets Content

- **`datasets/`:**
  - `newpairs.py`: Script to build and label pairs for Unseen and Within splits.
  - `keep.py`: Script to retain only the Java submissions that appear in at least one pair.
  - `sent.json`: Contains all the extracted Java tuples from the Code4bench dataset.
  - `code4bench_within`: Folder containing the Within split before CPG.
  - `code4bench`: Folder containing the Unseen split before CPG.

- **`cpgnn/data.txt`:**
  - Contains the CPG dict for each different split method (input for the model).
  - `c4b_perf_encoding_within`: The Within split.
  - `c4b_perf_encoding_unseen`: The Unseen split.
  - `c4b_perf_encoding_within_without_cfg_dfg`: The AST split.


## Steps to Reproduce

### Results of Comparing Combination Methods on "Within" Split
Navigate to `cpgnn/`.

#### For Table 4.2: 
- **Euclidean Distance:**
    ```bash
    python main_c4b.py --perf_test_supervised --epoch 30 --degradation_threshold 0.5 --dataset c4b_perf_encoding_within --type_dim 16 --layer_size [32,32,32,32] --batch_size_perf 384 --gpu_id 0,1 --model_end DIST --report dist_report
    ```

- **Compare Regressions:**
    ```bash
    python main_c4b.py --perf_test_supervised --epoch 30 --degradation_threshold 0.5 --dataset c4b_perf_encoding_within --type_dim 16 --layer_size [32,32,32,32] --batch_size_perf 384 --gpu_id 0,1 --model_end REGS --report regs_report
    ```

- **Concatenation:**
    ```bash
    python main_c4b.py --perf_test_supervised --epoch 30 --degradation_threshold 0.5 --dataset c4b_perf_encoding_within --type_dim 16 --layer_size [32,32,32,32] --batch_size_perf 384 --gpu_id 0,1 --model_end CONC --report conc_report
    ```

### Results of "Compare Regressions" on Unseen Split

#### For Table 4.3:
- **Compare Regressions:**
    ```bash
    python main_c4b.py --perf_test_supervised --epoch 30 --degradation_threshold 0.5 --dataset c4b_perf_encoding_unseen --type_dim 16 --layer_size [32,32,32,32] --batch_size_perf 384 --gpu_id 0,1 --model_end REGS --report regs_unseen_report
    ```

### Results of Ablation Test Excluding CFG and DF Edges

#### For Table 4.4:
- **Ablation Test:**
    ```bash
    python main_c4b.py --perf_test_supervised --epoch 30 --degradation_threshold 0.5 --dataset c4b_perf_encoding_within_without_cfg_dfg --type_dim 16 --layer_size [32,32,32,32] --batch_size_perf 384 --gpu_id 0,1 --model_end REGS --report regs_AST_report
    ```

## Building Pairs and Encodings / CPG Dicts

### Within Split
1. Navigate to `datasets/`.
2. Run the following command after commenting lines 57 to 99 in `newpairs.py`:
    ```bash
    python newpairs.py
    ```

### Unseen Split
1. In `datasets/`, run the following command after commenting lines 51 to 56, 92-93, and 103-104 in `newpairs.py`:
    ```bash
    python newpairs.py
    ```

2. Manually remove the last empty line from the generated `perf_labels.txt`. (behaviour depends on OS)

### Common Steps for Within and Unseen Splits
1. Run the following script:
    ```bash
    python keep.py
    ```

2. This will create the folder `programs/code4bench`.
3. Copy `perf_labels.txt` and paste it into `programs/code4bench`.
4. Copy the entire `programs/code4bench` folder and paste it into `datasets/`.

### Generating Encodings

#### For Both Within and Unseen Splits
1. Navigate to `cpg/`.
2. Run the following command:
    ```bash
    python driver.py --lang java --src_path ../datasets/code4bench --statistics --encoding --encode_path ../cpgnn/data/c4b_perf_encoding
    ```

### AST Encoding
1. Modify the file `cpg/javacpg/encoding/encoding.py` by commenting line 136 and uncommenting lines 132 - 133.
2. Run the same command as above to generate the AST encodings:
    ```bash
    python driver.py --lang java --src_path ../datasets/code4bench --statistics --encoding --encode_path ../cpgnn/data/c4b_perf_encoding
    ```



