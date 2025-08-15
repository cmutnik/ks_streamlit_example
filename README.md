<!-- Copyright (c) 2025 takotime808 -->
# Two Sample Comparator App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://distributions-and-metrics.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.10.17%2B-blue.svg)](https://www.python.org/)
<!-- [![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://takotime808.github.io/multioutreg/) -->


<!-- ![](docs/distquant.png) -->

<p align="center">
  <img width="300" height="300" src="docs/_static/images/distquant_logo.png">
</p>

This Streamlit app compares two uploaded samples using statistical tests like **Kolmogorov-Smirnov** and **Anderson-Darling**. It includes:
- Histogram and ECDF visualizations
- Preprocessing options (Log transform & Standardization)
- Statistical test results with automatic interpretation


## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/cmutnik/ks_streamlit_example.git
   ```
2. Install the requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. Launch the streamlit app:
   ```sh
   streamlit run Compare_Distributions.py
   ```

----
## Requirements

**Python 3.7 - 3.10**

```txt
streamlit==1.16.0
pandas==1.5.3
numpy==1.23.5
scipy==1.10.1
matplotlib==3.6.3
scikit-learn==1.0.2  # compatible with Python 3.10.17
# specify Altair version 4.x for compatibility
# altair<5
altair==4.2.0
```

**Python 3.12.10**

```txt
streamlit==1.20.0
pandas==2.1.0
numpy==1.24.2
scipy==1.11.0
matplotlib==3.7.1
```

----
## Metrics Checklist

- [ ] Validation Metrics:
    - [ ] Deterministic Validation Metric:
        - [ ] Root Mean Square Error
        - [ ] Minkowski Distance ($l_{p}$ Distance)
    - [ ] Probability-Based Validation Metrics
        - [ ] Normalized Euclidean Metrid
        - [ ] Mahalanobis Distance
        - [ ] Kullback-Leibler Divergence
        - [ ] Symmetrized Divergence
        - [ ] Jensen-Shannon Divergence
        - [ ] Hellinger Metric
        - [ ] Kolmogorov-Smirnov Test
        - [ ] Total Variation Distance
    - [ ] Signal Processing Validation Metrics
        - [ ] Simple Cross Correlation
        - [ ] Normalized Cross Correlationl
        - [ ] Normalized Zero-Mean Sum of Squared Distances
        - [ ] Moravec Correlationl
        - [ ] Index of Agreement
        - [ ] Sprague-Geers Metric

All of the metrics above come from the Sandia Report titled "Validation Metrics for Deterministic and Probabilistic Data". The full report is:

SANDIA REPORT
SAND2016-1421
Unlimited Release
Printed January 2017
Validation Metrics for Deterministic and Probabilistic Data
Kathryn A. Maupin and Laura P. Swiler

----
----

## **Random Distribution Generator and Validation Metrics**

This project provides a Streamlit-based web application that allows users to generate random distributions, visualize them with histograms, and download the data as CSV files. Additionally, the app calculates and displays various **validation metrics** based on the **Sandia 2016-1421 paper** for comparing observed and predicted data. These metrics include **Mean Bias (MB)**, **Root Mean Squared Error (RMSE)**, **Percent Bias (PBIAS)**, **R-squared (R²)**, and others. 

### **Features**
1. **Generate Random Distributions**: Users can choose from four different types of distributions:
   - **Normal** (Gaussian)
   - **Uniform**
   - **Exponential**
   - **Binomial**
   
2. **Download Data**: After generating a distribution, users can download the generated data in CSV format.

3. **Validation Metrics**: Displays various metrics to compare observed and predicted data:
   - **Mean Bias (MB)**
   - **Root Mean Squared Error (RMSE)**
   - **Mean Absolute Error (MAE)**
   - **Percent Bias (PBIAS)**
   - **R-squared (R²)**
   - **Kolmogorov-Smirnov Statistic (KS)**
   - **Anderson-Darling Test Statistic**

4. **Visualization**: A histogram of the generated distribution is displayed for visual analysis.

### **Requirements**
This app is built using Python and several libraries, which are specified in the `requirements.txt` file.

The `requirements.txt` includes the following libraries:

- **Streamlit**: For creating the interactive web application.
- **Pandas**: For handling data and CSV export functionality.
- **NumPy**: For generating random data for different distributions.
- **Matplotlib**: For visualizing histograms of the generated distributions.
- **SciPy**: For statistical tests such as KS and Anderson-Darling.
- **Altair**: For advanced data visualization (optional for this example).
- **Scikit-learn**: For calculating validation metrics such as R².

Here is the content of the `requirements.txt`:

```txt
streamlit==1.16.0
pandas==1.5.3
numpy==1.23.5
scipy==1.10.1
matplotlib==3.6.3
altair==4.2.0  # specify Altair version 4.x for compatibility
scikit-learn==1.0.2  # compatible with Python 3.10.17
```

### **Installation Instructions**

#### Step 1: Set Up Your Environment
To begin, you’ll need Python 3.10.17 or a compatible version. You can set up a virtual environment to keep your dependencies isolated from the system Python.

1. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

#### Step 2: Install Dependencies
Use the `requirements.txt` to install the necessary libraries for the app. You can install all the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

#### Step 3: Run the Application
Once the dependencies are installed, you can run the application using Streamlit:

```bash
streamlit run Compare_Distributions.py
```

After running the command, Streamlit will start the app, and you can open it in your browser (typically at `http://localhost:8501`).

### **Usage Instructions**

1. **Generate a Distribution**:
   - Select the distribution type from the sidebar: Normal, Uniform, Exponential, or Binomial.
   - Adjust the parameters such as mean, standard deviation, sample size, etc., according to the selected distribution type.
   - Click the **Generate** button to generate the data.

2. **Visualize the Distribution**:
   - A histogram will automatically be displayed to visualize the generated data.
   
3. **Download the Data**:
   - After the distribution is generated, you can download the data as a CSV file by clicking the **Download CSV** button.

4. **View Validation Metrics**:
   - The app will show validation metrics comparing the generated data against a **standard normal distribution**.
   - Metrics displayed include:
     - **Mean Bias (MB)**: Average difference between generated data and observed data.
     - **Root Mean Squared Error (RMSE)**: A measure of the differences between the predicted and observed data.
     - **Percent Bias (PBIAS)**: A measure of the relative bias of the generated data.
     - **R-squared (R²)**: The proportion of variance in the data explained by the model.
     - **Kolmogorov-Smirnov (KS) Test**: A statistical test to compare the distribution of the generated data to a normal distribution.
     - **Anderson-Darling Test**: A statistical test for the goodness of fit between the generated data and the normal distribution.

### **Sample Calculations**
For demonstration purposes, the app calculates the following metrics by comparing the generated data to a **standard normal distribution**:

- **Mean Bias (MB)**: The mean of the differences between the generated data and the observed (standard normal) data.
- **RMSE**: The square root of the mean squared errors between the generated and observed data.
- **PBIAS**: The percentage bias of the data compared to the observed values.
- **R²**: How well the generated data fits the observed data.
- **KS Statistic**: Used to determine if the generated data follows a specific distribution (e.g., normal).
- **Anderson-Darling Statistic**: Another method for testing how well the generated data fits a normal distribution.

<!-- ## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contact**
For any issues or feature requests, please open an issue on this repository. Alternatively, you can reach out to the project maintainers via email at [email@example.com]. -->