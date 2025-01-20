import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
def load_and_clean_data(file_path):
    """
    Load and clean the dataset for use in the recommendation system.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        DataFrame: Cleaned dataset.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying ISO-8859-1...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Drop rows with missing CustomerID or negative/zero quantities and prices
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Convert CustomerID to integer and InvoiceDate to datetime
    df['CustomerID'] = df['CustomerID'].astype('int')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Create TotalPrice for reference
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    print("Data cleaned and ready for processing!")
    return df

# Step 2: Prepare the Data for Collaborative Filtering
def prepare_data_for_model(df):
    """
    Prepare the cleaned data for collaborative filtering.
    Args:
        df (DataFrame): Cleaned dataset.
    Returns:
        Dataset: Prepared Surprise dataset for collaborative filtering.
    """
    scaler = MinMaxScaler(feature_range=(1, 5))
    df['NormalizedQuantity'] = scaler.fit_transform(df[['Quantity']])
    
    # Use Surprise's Reader to format the data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['CustomerID', 'StockCode', 'NormalizedQuantity']], reader)
    return data

# Step 3: Train Collaborative Filtering Model
def train_collaborative_model(data):
    """
    Train a collaborative filtering model using SVD.
    Args:
        data (Dataset): Surprise dataset for collaborative filtering.
    Returns:
        SVD: Trained SVD model.
    """
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Perform Grid Search for hyperparameter tuning
    param_grid = {'n_factors': [50, 100], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.1]}
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)
    
    print(f"Best RMSE: {grid_search.best_score['rmse']}")
    print(f"Best Parameters: {grid_search.best_params['rmse']}")
    
    model = grid_search.best_estimator['rmse']
    model.fit(trainset)
    
    predictions = model.test(testset)
    print(f"Final RMSE on test data: {rmse(predictions)}")
    return model

# Step 4: Recommend Products for a Customer
def recommend_products(model, customer_id, df, n_recommendations=5):
    """
    Recommend top N products for a customer based on predicted ratings.
    Args:
        model (SVD): Trained collaborative filtering model.
        customer_id (int): Customer ID for whom recommendations are generated.
        df (DataFrame): Cleaned dataset.
        n_recommendations (int): Number of recommendations to return.
    Returns:
        List[Tuple]: List of recommended products with descriptions and predicted ratings.
    """
    if customer_id not in df['CustomerID'].unique():
        print(f"Customer ID {customer_id} not found in the dataset!")
        return []

    all_products = df['StockCode'].unique()
    purchased_products = df[df['CustomerID'] == customer_id]['StockCode'].unique()
    unrated_products = [p for p in all_products if p not in purchased_products]
    
    predictions = [model.predict(customer_id, product) for product in unrated_products]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    top_recommendations = predictions[:n_recommendations]
    recommended_products = [(pred.iid, pred.est) for pred in top_recommendations]
    
    recommended_with_names = []
    for product_id, predicted_rating in recommended_products:
        description = (
            df[df['StockCode'] == product_id]['Description'].iloc[0]
            if not df[df['StockCode'] == product_id]['Description'].empty
            else "No description available"
        )
        recommended_with_names.append((product_id, description, predicted_rating))
    
    return recommended_with_names

# Step 5: Visualize Recommendations
def visualize_recommendations(recommendations, customer_id):
    """
    Visualize the top recommendations for a customer using a bar chart.
    Args:
        recommendations (List[Tuple]): Recommended products with descriptions and ratings.
        customer_id (int): Customer ID for whom recommendations are generated.
    """
    product_ids = [r[0] for r in recommendations]
    descriptions = [r[1] for r in recommendations]
    ratings = [r[2] for r in recommendations]
    
    plt.figure(figsize=(10, 6))
    plt.barh(descriptions, ratings, color='skyblue')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Products')
    plt.title(f'Top Recommendations for Customer {customer_id}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Main Function
if __name__ == "__main__":
    file_path = 'General customer data.csv'  # Replace with your file path
    
    df = load_and_clean_data(file_path)
    data = prepare_data_for_model(df)
    model = train_collaborative_model(data)
    
    customer_id = 17850  # Replace with an existing customer ID
    recommendations = recommend_products(model, customer_id, df)
    
    print(f"Top Recommendations for Customer {customer_id}:")
    print(f"{'Product ID':<15} {'Description':<50} {'Predicted Rating':<15}")
    print("-" * 80)
    for product_id, description, predicted_rating in recommendations:
        print(f"{product_id:<15} {description:<50} {predicted_rating:.2f}")
    
    visualize_recommendations(recommendations, customer_id)
