## Personalized-Product-Recommendation System
This project implements a machine learning-based personalized product recommendation system using collaborative filtering (SVD). It includes data cleaning, model training, and visualization of top product recommendations for customers. The system is designed to improve customer retention and sales by suggesting relevant products based on past purchase behavior.

---

## Features
- Data cleaning and preprocessing to ensure high-quality input data.
- Collaborative filtering using SVD for personalized recommendations.
- Hyperparameter tuning with GridSearchCV to optimize model performance.
- Visualization of recommendations for improved presentation.

---

## Dataset Overview
This project uses a sample dataset that simulates transaction-level purchase data from an online retail store. The sample dataset contains the following columns:

- **CustomerID**: Unique identifier for each customer.
- **StockCode**: Unique identifier for each product.
- **Description**: Text description of the product.
- **Quantity**: Number of units purchased in a single transaction.
- **UnitPrice**: Price per unit of the product.
- **InvoiceDate**: Date and time of the transaction.

A small, anonymized sample dataset (`General customer data.csv`) is included in the repository for demonstration purposes.

### Data Preprocessing
The following preprocessing steps were applied to the dataset:
1. Removed rows with missing `CustomerID` or invalid `Quantity` and `UnitPrice` values (e.g., negative or zero values).
2. Converted `CustomerID` to integer type and `InvoiceDate` to datetime format.
3. Created new features:
   - **TotalPrice**: `Quantity` × `UnitPrice`.
   - **NormalizedQuantity**: Scaled `Quantity` to a range of 1 to 5 using MinMaxScaler.
4. Prepared the data in a format suitable for collaborative filtering using the `Surprise` library.

---

## Model Overview
The system uses collaborative filtering with the **Singular Value Decomposition (SVD)** algorithm for personalized recommendations.

### Hyperparameter Tuning
Hyperparameter tuning was performed using GridSearchCV with the following parameter grid:
- **n_factors**: [50, 100]
- **lr_all**: [0.002, 0.005]
- **reg_all**: [0.02, 0.1]

### Results
- **Best RMSE on Training Data**: `0.0071`
- **Best Hyperparameters**:
  - `n_factors`: 50
  - `lr_all`: 0.005
  - `reg_all`: 0.1
- **Final RMSE on Test Data**: `0.0070`

- ### Sample Recommendations for Customer `17850`
| Product ID | Description                                   | Predicted Rating |
|------------|-----------------------------------------------|------------------|
| 23843      | PAPER CRAFT , LITTLE BIRDIE                  | 1.35             |
| 35916A     | YELLOW FELT HANGING HEART W FLOWER           | 1.03             |
| 23166      | MEDIUM CERAMIC TOP STORAGE JAR               | 1.03             |
| 16259      | PIECE OF CAMO STATIONERY SET                 | 1.03             |
| 90184B     | AMETHYST CHUNKY BEAD BRACELET W STR          | 1.03             |





<img width="648" alt="product recom results" src="https://github.com/user-attachments/assets/1ad9be2c-e750-4f09-99e1-dea7b74103d4" />





![product recom graph](https://github.com/user-attachments/assets/de5fb9bb-5d1b-4bd2-9bfd-20ed95bad0ce)



---
