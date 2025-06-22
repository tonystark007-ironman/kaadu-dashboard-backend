# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from datetime import datetime, date
import re # Import regex module for standardize_category_name

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
# This is crucial for allowing your frontend (running on a different port/origin)
# to communicate with your backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Global variable to store the DataFrame. In a real-world scenario,
# you might use a database or a more persistent storage solution.
# For now, data will be in memory and reset if the server restarts.
df_store = {}

# Function to standardize category names
def standardize_category_name(product_name):
    """
    Extracts the first word of a product name and maps similar words to a standard category.
    This helps in grouping similar products under one main category.
    You can extend this mapping as needed.
    """
    if pd.isna(product_name) or not isinstance(product_name, str) or not product_name.strip():
        return "Other" # Default category for empty or invalid product names

    # Split by space, underscore, or hyphen to get initial keyword
    first_word = re.split(r'[ _-]', product_name.strip())[0]

    # Define a mapping for similar categories to a single main category
    category_mapping = {
        "jaggery": "Jaggery",
        "rice": "Rice",
        "dal": "Dal",
        "oil": "Oils",      # Adjusted for 'oil' to map to 'Oils'
        "oils": "Oils",
        "millets": "Millets",
        "spices": "Spices",
        "flour": "Flour",
        "atta": "Flour", # Mapping atta to Flour
        "honey": "Honey",
        "ghee": "Ghee",
        "powder": "Powder",
        "soap": "Soap",
        "sugar": "Sugar",
        "salt": "Salt",
        "nuts": "Nuts",
        "tea": "Tea",
        "coffee": "Coffee",
        "grains": "Grains",
        "cereals": "Grains",
        "seeds": "Seeds",
        "snacks": "Snacks",
        "sweets": "Sweets",
        "herbs": "Herbs",
        "drinks": "Drinks",
        "beverages": "Drinks",
        "vegetables": "Vegetables",
        "fruits": "Fruits",
        "dairy": "Dairy",
        "bread": "Bakery",
        "bakery": "Bakery",
        "meat": "Meat & Fish",
        "fish": "Meat & Fish",
        "cosmetics": "Cosmetics", # Added as per frontend colors
        "personal": "Personal Care", # Added as per frontend colors (e.g., "Personal Hygiene")
    }

    # Convert to lowercase for consistent mapping
    first_word_lower = first_word.lower()

    # Return the standardized category, or capitalize the first word if no specific mapping exists
    return category_mapping.get(first_word_lower, first_word.capitalize() if first_word else "Other")


@app.get("/")
async def read_root():
    """
    Root endpoint for a simple health check.
    """
    return {"message": "Welcome to the Dashboard Backend! Upload a CSV file."}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint to handle CSV file uploads.
    Parses the CSV content into a Pandas DataFrame, creates a 'category' column
    based on the 'Product' column, and standardizes category names.
    Now more robust to variations in column names and presence of optional columns.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        contents = await file.read()
        data = io.StringIO(contents.decode("utf-8"))

        # Define the expected columns and their possible variations, mapping to internal names
        # Format: {'internal_name': ['Possible CSV Column Name 1', 'Possible CSV Column Name 2']}
        column_mapping = {
            "date": ["Date", "Sale Date", "Transaction Date", "Order Date", "Invoice Date"],
            "product": ["Product", "Product Name", "Item", "Item Name", "Description"], # Added "Description"
            "unit": ["Unit", "UOM", "Measurement Unit", "Unit Of Measure"],
            "sales": ["Amount", "Sale Amount", "Total Price", "Revenue", "Total Sales", "Net Amount"], # Added more variations
            "quantity": ["Quantity", "Qty", "Units Sold", "Num Units"], # Added more variations
            "price_per_unit": ["Price/Unit", "Price / Unit", "Price/UOM", "Rate", "Unit Price"], # Added "Unit Price"
            "invoice_no": ["Invoice No.", "Invoice No", "Invoice Number", "Bill No", "Bill Number"], # Added more variations
            "party_name": ["Party Name", "Party", "Customer Name", "Customer"], # Added "Customer"
            "item_code": ["Item code", "Item Code", "Product Code", "SKU"], # Added "SKU"
            "hsn_sac": ["HSN/SAC", "HSN Code", "SAC Code", "HSN"], # Added "HSN"
            "discount": ["Discount", "Discount Amount", "Disc."],
            "tax": ["Tax", "Tax Amount", "VAT", "GST"],
            "store_name": ["Store", "Store Name", "Location", "Branch"],
            "customer_id": ["Customer ID", "CustomerID", "Customer Identifier", "Client ID"],
            "payment_method": ["Payment Method", "PaymentType", "Method of Payment", "Payment Mode"],
        }
        
        # Read the CSV, skipping bad lines for robustness
        df = pd.read_csv(data, on_bad_lines='skip')
        print(f"Initial CSV columns: {df.columns.tolist()}")

        # Identify actual columns from the CSV that match our expected list (case-insensitive)
        found_columns_map = {}
        df_lower_cols = {col.lower(): col for col in df.columns} # Map lowercased col to original col name
        
        for internal_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_lower_cols:
                    found_columns_map[df_lower_cols[possible_name.lower()]] = internal_name
                    break # Found a match, move to the next internal name

        # Create a new DataFrame with only the found columns and their standardized names
        df_processed = pd.DataFrame()
        for original_col, new_col in found_columns_map.items():
            df_processed[new_col] = df[original_col]
        df = df_processed # Replace original df with the processed one
        print(f"Columns after initial mapping: {df.columns.tolist()}")

        # Define which internal names are absolutely required for basic functionality
        # Removed 'quantity' from required columns
        required_internal_columns = ["date", "product", "sales"] 
        missing_required = [col for col in required_internal_columns if col not in df.columns]
        if missing_required:
            raise HTTPException(
                status_code=400,
                detail=f"Missing critical columns after parsing: {', '.join(missing_required)}. "
                       f"Please ensure your CSV contains variants of these (e.g., 'Date', 'Product', 'Amount'). "
                       f"Available columns in processed file: {', '.join(df.columns.tolist())}"
            )
        
        # Handle optional columns: if not found, add them with a default value
        # This loop now covers ALL possible internal names from column_mapping
        for internal_name in column_mapping.keys():
            if internal_name not in df.columns:
                # Assign sensible defaults based on expected type
                if internal_name in ["sales", "quantity", "discount", "tax", "price_per_unit"]:
                    df[internal_name] = 0.0 # Numeric default
                else: # For other string-based optional columns
                    df[internal_name] = "N/A" # String default
                print(f"Note: '{internal_name}' column not found in CSV. Added with default values.")


        # --- START: Robust parsing logic for Discount and GST values ---
        # Apply regex to extract only the numeric part before parentheses for 'discount' and 'tax'
        for col_name in ["discount", "tax"]:
            if col_name in df.columns and df[col_name].dtype == 'object': # Only process if it's a string/object column
                # Use a regex to find numbers that might be followed by parentheses
                # This will extract "1.20" from "1.20(15.0%)"
                # The regex looks for an optional hyphen, then one or more digits, an optional decimal point, and zero or more digits.
                df[col_name] = df[col_name].astype(str).str.extract(r'^(-?\d+\.?\d*)', expand=False)
                print(f"Note: Applied regex extraction to '{col_name}' column.")
        # --- END: Robust parsing logic for Discount and GST values ---

        # Ensure numeric columns are indeed numeric AFTER any string manipulations (like regex extraction)
        # Include all columns that *could* be numeric and convert them
        numeric_cols = ["sales", "quantity", "discount", "tax", "price_per_unit"]
        for col in numeric_cols:
            if col in df.columns: # Check if the column exists after optional column handling
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                print(f"Converted '{col}' to numeric, filled NaNs with 0.")
            else:
                # This case should ideally be caught by the "Handle optional columns" block above,
                # but as a failsafe, ensure it's numeric if added later.
                df[col] = 0.0
                print(f"Warning: '{col}' not found, set to 0.0 (failsafe).")

        # Convert 'date' column to datetime objects with multiple format attempts
        # First, ensure 'date' column is treated as string before parsing to avoid issues with mixed types
        df['date'] = df['date'].astype(str)
        initial_valid_dates = df['date'].notna().sum()
        date_formats_to_try = [
            '%d-%m-%Y',    # e.g., 21-06-2025
            '%Y-%m-%d',    # e.g., 2025-06-21
            '%m/%d/%Y',    # e.g., 06/21/2025
            '%d/%m/%Y',    # e.g., 21/06/2025
            '%Y/%m/%d',    # e.g., 2025/06/21
            '%d-%m-%y',    # e.g., 21-06-25 (two-digit year)
            '%m/%d/%y',    # e.g., 06/21/25 (two-digit year)
            '%d %b %Y',    # e.g., 21 Jun 2025
            '%d %B %Y'     # e.g., 21 June 2025
        ]
        
        original_date_series = df['date'].copy() # Preserve original for retries
        
        parsed_successfully = False
        for fmt in date_formats_to_try:
            df['date'] = pd.to_datetime(original_date_series, format=fmt, errors='coerce')
            if df['date'].notna().sum() > 0: # Check if any dates were successfully parsed with this format
                print(f"Successfully parsed dates with format: '{fmt}'. Valid dates: {df['date'].notna().sum()}")
                parsed_successfully = True
                break
        
        # If specific formats didn't work and there were non-null dates initially, try general inference
        if not parsed_successfully and initial_valid_dates > 0:
            df['date'] = pd.to_datetime(original_date_series, errors='coerce', infer_datetime_format=True)
            if df['date'].notna().sum() > 0:
                print(f"Successfully parsed dates with inferred format. Valid dates: {df['date'].notna().sum()}")
            else:
                print("Warning: Date parsing failed even with inference. All dates might be invalid.")
        
        # If after all attempts, the 'date' column is still all NaT (Not a Time), raise an error
        # Now, if 'date' is fully unparseable AND was initially present, raise error. If 'date' wasn't there, it would be N/A and then set to default.
        if df['date'].isnull().all() and initial_valid_dates > 0:
             raise HTTPException(
                status_code=400,
                detail="Date column could not be parsed into a valid date format. Please check date format in your CSV."
            )

        df.dropna(subset=['date'], inplace=True) # Drop rows where date parsing ultimately failed
        print(f"DataFrame after date processing and dropping NaNs: {len(df)} rows.")


        # Create 'category' column by taking the first word of 'product' and standardizing it
        # Ensure 'product' column is string type before applying standardize_category_name
        df['product'] = df['product'].astype(str)
        df['category'] = df['product'].apply(standardize_category_name)
        print("Note: 'category' column created and standardized based on 'product'.")

        # Ensure 'unit' column exists and is string type
        if 'unit' not in df.columns:
            df['unit'] = "units" # Default unit if not present
        df['unit'] = df['unit'].astype(str).fillna("units") # Ensure it's string and fill NaN with default
        
        # Ensure 'store_name' column exists and is string type
        if 'store_name' not in df.columns:
            df['store_name'] = "N/A" # Default store name if not present
        df['store_name'] = df['store_name'].astype(str).fillna("N/A")

        # Store the DataFrame in our in-memory store
        df_store["current_dashboard_data"] = df
        print(f"Successfully uploaded and parsed: {file.filename}")
        print(f"DataFrame Head:\n{df.head()}")
        print(f"DataFrame Info:\n{df.info()}")
        print(f"Final DataFrame Columns: {df.columns.tolist()}")

        # Return unique standardized categories, products, and store names to the frontend
        categories = df["category"].unique().tolist()
        products = df["product"].unique().tolist()
        store_names = df["store_name"].unique().tolist() if "store_name" in df.columns else []

        # Before returning, ensure 'quantity' is treated correctly if it was missing
        # The frontend expects total_quantity. If 'quantity' was missing, its sum will be 0 due to fillna(0)
        # So we can calculate total_quantity here based on the processed df.
        total_quantity_calculated = df["quantity"].sum() if "quantity" in df.columns else 0.0
        total_sales_calculated = df["sales"].sum() if "sales" in df.columns else 0.0

        return {
            "message": "CSV uploaded and processed successfully!",
            "filename": file.filename,
            "rows_processed": len(df),
            "categories": categories,
            "products": products,
            "store_names": store_names,
            # Pass these calculated values back for immediate display if needed
            "total_quantity": float(total_quantity_calculated),
            "total_sales": float(total_sales_calculated)
        }

    except pd.errors.EmptyDataError:
        print("Error: Empty CSV file.")
        raise HTTPException(status_code=400, detail="Empty CSV file.")
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file. Check delimiter or file format: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"An unexpected error occurred during CSV upload: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during file processing: {e}. If the issue persists, please contact support with your CSV file.")


def get_current_df():
    """Helper function to retrieve the DataFrame or raise an error."""
    df = df_store.get("current_dashboard_data")
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded yet. Please upload a CSV file.")
    return df.copy() # Return a copy to avoid modifying the original stored DataFrame

@app.get("/data/categories")
async def get_categories():
    """Returns a list of unique, standardized categories."""
    df = get_current_df()
    return {"categories": df["category"].unique().tolist()}

@app.get("/data/products")
async def get_products(category: str = Query(None)):
    """
    Returns a list of unique products, optionally filtered by standardized category.
    """
    df = get_current_df()
    if category:
        # Filter by the standardized category
        df = df[df["category"] == category]
    return {"products": df["product"].unique().tolist()}

# New endpoint to get unique store names
@app.get("/data/stores")
async def get_stores():
    """Returns a list of unique store names."""
    df = get_current_df()
    if "store_name" in df.columns:
        return {"store_names": df["store_name"].unique().tolist()}
    return {"store_names": []} # Return empty if column not present


@app.get("/data/summary")
async def get_summary_data(
    category: str = Query(None),
    product: str = Query(None),
    start_date: date = Query(None),
    end_date: date = Query(None),
    store_name: str = Query(None) # New filter for store name
):
    """
    Returns a summary of sales data, filtered by category, product, date range, and store.
    Aggregates total quantity, total sales, and count of transactions.
    Also includes sales summaries by category, by product (with quantity and unit), and by unit.
    """
    df = get_current_df()

    # Apply filters
    if category:
        df = df[df["category"] == category]
    if product:
        df = df[df["product"] == product]
    if start_date:
        df = df[df["date"].dt.date >= start_date]
    if end_date:
        df = df[df["date"].dt.date <= end_date]
    if store_name and "store_name" in df.columns: # Apply store filter only if column exists
        df = df[df["store_name"] == store_name]

    if df.empty:
        return {
            "total_quantity": 0,
            "total_sales": 0,
            "transaction_count": 0,
            "message": "No data found for the selected filters.",
            "summary_by_category": [],
            "summary_by_product": [],
            "summary_by_unit": [],
            "daily_sales_trend": [] # Ensure empty list for charts if no data
        }

    # Conditional calculation of total_quantity
    total_quantity = df["quantity"].sum() if "quantity" in df.columns else 0.0
    total_sales = df["sales"].sum()
    transaction_count = len(df)

    total_quantity_output = float(total_quantity) if pd.notna(total_quantity) else 0.0
    total_sales_output = float(total_sales) if pd.notna(total_sales) else 0.0

    # Summary by Category (for Pie Chart) - uses the 'category' column
    summary_by_category_data = df.groupby('category')['sales'].sum().reset_index()
    summary_by_category_output = summary_by_category_data.to_dict(orient="records")

    # Summary by Product (for Bar Chart) - now includes category, quantity, and unit
    # Group by product and category to retain category information, and sum both sales and quantity
    # For unit, take the most frequent unit for that product in the filtered data.
    summary_by_product_data = df.groupby(['product', 'category']).agg(
        sales=('sales', 'sum'),
        # Conditionally aggregate quantity or default to 0 if column is missing
        quantity=('quantity', lambda x: x.sum() if "quantity" in df.columns else 0.0),
        # Get the most frequent unit for the product. Handles cases where a product might have multiple units.
        unit=('unit', lambda x: x.mode()[0] if not x.empty and "unit" in df.columns else 'N/A') # Use N/A if unit column not found
    ).reset_index()

    # Sort by sales in descending order
    summary_by_product_data = summary_by_product_data.sort_values(by='sales', ascending=False)

    # Apply top 5 filter if no specific product is selected
    if product is None or product == "": # This means 'All Products' is selected in the frontend dropdown
        summary_by_product_data = summary_by_product_data.head(5)

    summary_by_product_output = summary_by_product_data.to_dict(orient="records")


    # Group by unit and sum sales and quantity
    unit_sales_summary = df.groupby('unit').agg(
        # Conditionally aggregate quantity or default to 0 if column is missing
        total_quantity=('quantity', lambda x: x.sum() if "quantity" in df.columns else 0.0),
        total_sales=('sales', 'sum')
    ).reset_index()
    summary_by_unit_output = unit_sales_summary.to_dict(orient="records")

    # Daily Sales Trend for the main sales chart (Line Chart)
    # This aggregates sales by date for the selected filters
    daily_sales_trend_data = df.groupby(df['date'].dt.date)['sales'].sum().reset_index()
    daily_sales_trend_data.columns = ['date', 'sales']
    # Convert date objects to string for JSON serialization
    daily_sales_trend_data['date'] = daily_sales_trend_data['date'].astype(str)
    daily_sales_trend_output = daily_sales_trend_data.to_dict(orient="records")


    return {
        "total_quantity": total_quantity_output,
        "total_sales": total_sales_output,
        "transaction_count": transaction_count,
        "summary_by_category": summary_by_category_output,
        "summary_by_product": summary_by_product_output,
        "summary_by_unit": summary_by_unit_output,
        "daily_sales_trend": daily_sales_trend_output # New: Daily sales trend
    }

@app.get("/data/monthly-sales-trend")
async def get_monthly_sales_trend(
    start_date: date = Query(None),
    end_date: date = Query(None),
    store_name: str = Query(None) # New filter for store name
):
    """
    Returns month-wise sales trend for the top 5 products, optionally filtered by date range and store.
    """
    df = get_current_df()

    # Apply date and store filters
    if start_date:
        df = df[df["date"].dt.date >= start_date]
    if end_date:
        df = df[df["date"].dt.date <= end_date]
    if store_name and "store_name" in df.columns:
        df = df[df["store_name"] == store_name]

    if df.empty:
        return {
            "months": [],
            "products_sales_data": [],
            "message": "No data found for the selected filters."
        }

    # Ensure 'date' column is datetime and extract month-year for grouping
    # Convert Period to string for consistent JSON serialization and frontend display
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    # Calculate total sales for each product to determine top 5 within the filtered timeframe
    product_total_sales = df.groupby('product')['sales'].sum().reset_index()
    product_total_sales = product_total_sales.sort_values(by='sales', ascending=False)
    top_5_products_names = product_total_sales.head(5)['product'].tolist()

    # Filter original (date-filtered) DataFrame for only the top 5 products
    df_top_5 = df[df['product'].isin(top_5_products_names)]

    # Get all unique month_year periods in the filtered data to use as labels
    all_months = sorted(df_top_5['month_year'].unique().tolist())

    products_sales_data = []
    # For each of the top 5 products, get their sales for each month
    for product_name in top_5_products_names:
        product_df = df_top_5[df_top_5['product'] == product_name]
        # Get its category (assuming one category per product for simplicity)
        category = product_df['category'].iloc[0] if not product_df.empty else "Other"

        # Group by month and sum sales
        # Create a temporary Series with all_months as index to ensure all months are present
        temp_series = product_df.groupby('month_year')['sales'].sum().reindex(all_months, fill_value=0)

        products_sales_data.append({
            "product_name": product_name,
            "category": category,
            "sales_data": temp_series.tolist()
        })

    return {
        "months": all_months,
        "products_sales_data": products_sales_data
    }
