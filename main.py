# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from datetime import datetime, date
import re # Import regex module for standardize_category_name
import logging # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    Extracts the first meaningful word from a product name and capitalizes it as the category.
    This provides a simpler, direct categorization based on the initial part of the product name.
    It also includes a small map for common aliases or plural forms to ensure consistency.
    """
    if pd.isna(product_name) or not isinstance(product_name, str) or not product_name.strip():
        logging.info(f"Product name '{product_name}' is invalid or empty, returning 'Other'.")
        return "Other"

    product_name_lower = product_name.lower().strip()
    logging.info(f"Standardizing category for product: '{product_name}'")

    # Define common prefixes or words to ignore if they appear at the very beginning
    ignore_prefixes = [
        r'^(?:kaadu\s+organics?|pure|organic|fresh|premium|natural|indian|farm|garden|daily)\s*',
        r'^\d+\s*(?:kg|g|gm|ml|l|liter|pack|pkt|piece|pc|units?|bottles?)\s*' # Ignore leading quantity/unit
    ]

    cleaned_name = product_name_lower
    for prefix_pattern in ignore_prefixes:
        cleaned_name = re.sub(prefix_pattern, '', cleaned_name).strip()

    # Split by space, underscore, or hyphen to get the first significant word
    # Only split by the first occurrence of these delimiters to preserve multi-word names
    # e.g., "Rice - Aathur Kichali Samba" -> "Rice"
    parts = re.split(r'[ _-]', cleaned_name, 1) # Split only once

    first_word = parts[0].strip() if parts else "Other"

    # Minimal explicit mapping for common aliases or pluralization
    simple_category_map = {
        "jaggery": "Jaggery",
        "rice": "Rice",
        "dal": "Dal",
        "oil": "Oils",      # Alias for 'oils'
        "oils": "Oils",
        "millet": "Millets", # Singular to plural
        "millets": "Millets",
        "spice": "Spices",   # Singular to plural
        "spices": "Spices",
        "flour": "Flour",
        "atta": "Flour",
        "honey": "Honey",
        "ghee": "Ghee",
        "powder": "Powder",
        "soap": "Soap",
        "sugar": "Sugar",
        "salt": "Salt",
        "nut": "Nuts",       # Singular to plural
        "nuts": "Nuts",
        "tea": "Tea",
        "coffee": "Coffee",
        "grain": "Grains",   # Singular to plural
        "grains": "Grains",
        "cereal": "Cereals", # Singular to plural
        "cereals": "Cereals",
        "seed": "Seeds",     # Singular to plural
        "seeds": "Seeds",
        "snack": "Snacks",   # Singular to plural
        "snacks": "Snacks",
        "sweet": "Sweets",   # Singular to plural
        "sweets": "Sweets",
        "herb": "Herbs",     # Singular to plural
        "herbs": "Herbs",
        "cosmetic": "Cosmetics", # Singular to plural
        "cosmetics": "Cosmetics",
        "personal": "Personal Care", # For "Personal Care"
        "mix": "Mixes",
        "shipping": "Shipping",
        "miscellaneous": "Miscellaneous",
        "poha": "Rice Products",
        "aval": "Rice Products",
        "paneer": "Dairy & Cheese",
        "dates": "Fruits & Dry Fruits",
    }
    
    # Use the mapping if an exact match for the first word is found, otherwise capitalize it
    category = simple_category_map.get(first_word, first_word.capitalize())
    if not category: # Fallback if first_word was empty after cleaning
        category = "Other"

    logging.info(f"Categorized '{product_name}' as '{category}' using first meaningful word/map.")
    return category

# Function to extract variety from product names
def extract_variety(product_name, category_name):
    """
    Extracts the specific variety from a product name after the main category is identified.
    It attempts to remove the category name and common suffixes like 'Raw', 'Boiled', 'Full Skin'
    to isolate the true variety descriptor.
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return "Unknown"

    product_name_lower = product_name.lower().strip()
    category_name_lower = category_name.lower().strip()
    logging.info(f"Extracting variety for '{product_name}' (Category: {category_name})")

    # 1. Remove the main category word (and common aliases/variations) from the start of the product name.
    # This regex is designed to catch "Category - Variety", "Category_Variety", "Category Variety"
    # It accounts for possible "s" at the end of category if product name uses singular form.
    # Make sure to handle categories that are multiple words too if they were mapped that way (e.g., "Rice Products")
    
    # Generate a more dynamic pattern for category removal based on the actual category name and its singular/plural forms
    category_forms = [re.escape(category_name_lower)]
    if category_name_lower.endswith('s'):
        category_forms.append(re.escape(category_name_lower[:-1])) # e.g., 'oils' -> 'oil'
    else:
        category_forms.append(re.escape(category_name_lower + 's')) # e.g., 'rice' -> 'rices' (though 'rices' is less common)
    

    # Create a single regex pattern to match any of the category forms at the beginning
    category_removal_pattern = r'^(?:' + '|'.join(category_forms) + r')(?:\s*[-_:]?\s*)?'
    
    variety_part = re.sub(category_removal_pattern, '', product_name_lower).strip()

    # If the variety_part is empty after initial removal, try a less aggressive removal
    # This can happen if the product name is just the category (e.g., "Jaggery")
    if not variety_part and product_name_lower == category_name_lower:
        logging.info(f"Product name is just the category '{product_name}'. Returning 'Standard'.")
        return "Standard" # No specific variety if product name is just its category

    # 2. Remove common generic descriptors/suffixes from the *end* or middle that are not part of the variety.
    generic_descriptors_to_remove = [
        r'\s*[-_:]?\s*(?:raw|boiled|full skin|semi skin|scented)\s*$', # End of string processing
        r'\s*\d+(?:\.\d+)?\s*(?:kg|g|gm|ml|l|liter|pack|pkt|piece|pc|units?|bottles?)\s*$', # Units at end
        r'\s*(?:kaadu|organic|pure|fresh|premium|natural|indian|farm|garden|daily)\b\s*' # Generic brand/quality words
    ]

    for pattern in generic_descriptors_to_remove:
        variety_part = re.sub(pattern, ' ', variety_part).strip()
    
    # Remove any remaining extra spaces
    variety_part = re.sub(r'\s+', ' ', variety_part).strip()

    if variety_part:
        # Capitalize each word for cleaner presentation
        variety = ' '.join([word.title() for word in variety_part.split()])
        logging.info(f"Extracted variety: '{variety}' for product '{product_name}'.")
        return variety
    
    logging.info(f"Variety for '{product_name}' (Category: {category_name}) could not be specifically extracted, returning 'Generic Variety'.")
    return "Generic Variety"

# Function to extract pack size label and its derived numeric value
def extract_pack_size(product_name):
    """
    Extracts the pack size (e.g., 1kg, 500g, 2L, 10 Pcs) and its derived numeric value from a product name.
    Prioritizes common units.
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        logging.info(f"Pack size extraction: Product name '{product_name}' is invalid or empty, returning 'Unknown', 0.0.")
        return "Unknown", 0.0

    product_name_lower = product_name.lower()
    logging.info(f"Extracting pack size for product: '{product_name}' (lower: '{product_name_lower}')")

    # Regex to find numbers (integers or floats) followed by common weight/volume/count units
    # This pattern is robust for various number formats (e.g., 0.5kg, 1kg, 2.5l)
    # Added "pkt" and "box" explicitly, and ensured "pack" captures "pack of X"
    match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gm|ml|l|liter|pc|pkt|pack(?:s)?|box(?:es)?|units?)', product_name_lower)
    if match:
        value = float(match.group(1))
        unit = match.group(2)

        # Standardize units and convert to a base unit (e.g., kg for weight, liter for volume, 'count' for discrete items)
        if unit in ['kg', 'liter', 'l']:
            logging.info(f"Extracted pack size: {value}{unit} (Derived: {value})")
            return f"{value}{unit}", value # Return value as is (e.g., 1.0 for 1kg/1L)
        elif unit in ['g', 'gm']:
            converted_value = value / 1000.0
            logging.info(f"Extracted pack size: {value}{unit} (Derived: {converted_value})")
            return f"{value}{unit}", converted_value # Convert grams to kg (e.g., 0.5 for 500g)
        elif unit == 'ml':
            converted_value = value / 1000.0
            logging.info(f"Extracted pack size: {value}{unit} (Derived: {converted_value})")
            return f"{value}{unit}", converted_value # Convert ml to liters (e.g., 0.25 for 250ml)
        elif unit in ['pc', 'pkt', 'pack', 'packs', 'box', 'boxes', 'unit', 'units']:
            logging.info(f"Extracted pack size: {int(value)} {unit.replace('s', '')} (Derived: {value})")
            return f"{int(value)} {unit.replace('s', '')}", value # Return as is for count-based units, integer for label, singular unit
    
    # Handle cases like "x2", "pack of 4" where unit is implied or less explicit
    count_match = re.search(r'(?:x|pack of|set of)\s*(\d+)', product_name_lower)
    if count_match:
        value = float(count_match.group(1))
        logging.info(f"Extracted count-based pack size: {int(value)} Count (Derived: {value})")
        return f"{int(value)} Count", value # Treat as unit count

    # If no specific size/unit found, assume it's a "Standard Pack" or "Single Item"
    # Assign a derived_unit_value of 1.0 for calculation consistency
    logging.info(f"Pack size for '{product_name}' extracted as 'Standard Pack', derived value 1.0.")
    return "Standard Pack", 1.0

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are allowed.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        logging.info(f"Initial CSV columns: {df.columns.tolist()}")

        # Expected columns (case-insensitive and flexible)
        required_cols_map = {
            'date': ['date', 'order date', 'transaction date'],
            'product': ['product', 'product name', 'item'],
            'quantity': ['quantity', 'qty', 'units'],
            'sales': ['sales', 'revenue', 'amount'],
            'unit': ['unit', 'uom', 'measure'] # Optional, but good to have for advanced analysis
        }

        # Rename columns to standardized names
        df_cols = [col.lower() for col in df.columns]
        standard_df_cols = {}
        for standard_name, possible_names in required_cols_map.items():
            found = False
            for p_name in possible_names:
                if p_name in df_cols:
                    df.rename(columns={df.columns[df_cols.index(p_name)]: standard_name}, inplace=True)
                    standard_df_cols[standard_name] = True
                    found = True
                    break
            # Modified: 'quantity' is no longer strictly mandatory at upload
            if not found and standard_name in ['date', 'product', 'sales']: # Make core columns mandatory
                logging.error(f"Missing required column: {standard_name}. Please ensure your CSV has one of: {', '.join(possible_names)}.")
                raise HTTPException(status_code=400, detail=f"Missing required column: {standard_name}. Please ensure your CSV has one of: {', '.join(possible_names)}.")
        
        logging.info(f"Columns after initial mapping: {df.columns.tolist()}")

        # Ensure numeric types for quantity and sales
        # Use .get() for 'quantity' to safely handle its absence
        if 'quantity' in df.columns:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            logging.info("Quantity column processed.")
        else:
            df['quantity'] = 0 # Add a quantity column with default 0 if it was missing
            logging.info("Quantity column not found in CSV. Added with default value 0.")

        df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
        logging.info("Sales column processed.")


        # Convert 'date' column to datetime objects
        # Try multiple date formats for robustness
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        if df['date'].isnull().all():
            logging.error("Date column could not be parsed.")
            raise HTTPException(status_code=400, detail="Date column could not be parsed. Please ensure dates are in a recognizable format (e.g.,YYYY-MM-DD, MM/DD/YYYY).")
        
        # Drop rows where essential data (date, product, sales) is missing after coercion
        # Quantity is now optional for dropping subset, as it defaults to 0
        df.dropna(subset=['date', 'product', 'sales'], inplace=True)
        logging.info(f"DataFrame after dropping rows with missing essential data: {len(df)} rows.")


        # Generate 'month_year' for monthly trends
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        logging.info("Month_year column generated.")


        # Add 'category' column based on product name standardization
        df['category'] = df['product'].apply(standardize_category_name)
        logging.info("Category column added and standardized.")

        # Ensure 'unit' column exists and is string type
        if 'unit' not in df.columns:
            df['unit'] = "units" # Default unit if not present
            logging.info("Unit column not found in CSV. Added with default 'units'.")
        df['unit'] = df['unit'].astype(str).fillna("units") # Ensure it's string and fill NaN with default
        logging.info("Unit column processed.")

        # Store DataFrame globally for subsequent queries
        df_store['data'] = df # IMPORTANT: This key 'data' is the one used by get_current_df() below.
        
        # Get unique categories and products for dropdowns
        categories = sorted(df['category'].dropna().unique().tolist())
        products = sorted(df['product'].dropna().unique().tolist())

        logging.info(f"CSV uploaded and processed. Categories found: {categories}, Products found: {products}")

        return {"message": "CSV uploaded and processed successfully!",
                "rows_processed": len(df),
                "categories": categories,
                "products": products}

    except KeyError as e:
        logging.error(f"Missing expected column after initial CSV read: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Missing expected column: {e}. Please check your CSV header.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {e}")

def get_current_df():
    """Helper function to retrieve the DataFrame or raise an error."""
    # This function expects the data to be stored under the key 'data'
    df = df_store.get('data')
    if df is None:
        raise HTTPException(status_code=404, detail="No CSV data uploaded yet. Please upload a CSV file first.")
    return df.copy() # Return a copy to avoid modifying the original stored DataFrame


@app.get("/data/summary") # Removed trailing slash
async def get_summary_data(
    category: str = Query(None),
    product: str = Query(None),
    start_date: date = Query(None),
    end_date: date = Query(None)
):
    df = get_current_df() # Correctly retrieves from 'data'
    filtered_df = df.copy()
    logging.info(f"Fetching summary data with filters: category={category}, product={product}, start_date={start_date}, end_date={end_date}")

    if category:
        # Use str.contains with case=False for case-insensitive matching
        filtered_df = filtered_df[filtered_df['category'].str.contains(category, case=False, na=False)]
    if product:
        # FIX: Changed 'False' to 'na=False' to resolve positional argument error
        filtered_df = filtered_df[filtered_df['product'].str.contains(product, case=False, na=False)]
    if start_date:
        filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]

    if filtered_df.empty:
        logging.info(f"No data found for the selected filters for summary: category={category}, product={product}, start_date={start_date}, end_date={end_date}")
        return {"message": "No data found for the selected filters."}

    total_sales = filtered_df['sales'].sum()
    total_quantity = filtered_df['quantity'].sum() # This will be 0 if 'quantity' column was missing
    transaction_count = len(filtered_df) # Assuming each row is a transaction or part of one

    # Summary by Category
    summary_by_category = filtered_df.groupby('category').agg(
        sales=('sales', 'sum'),
        quantity=('quantity', 'sum')
    ).reset_index().sort_values(by='sales', ascending=False).to_dict(orient='records')
    logging.info(f"Summary by Category: {summary_by_category}")


    # Summary by Product (Top 10)
    summary_by_product = filtered_df.groupby('product').agg(
        sales=('sales', 'sum'),
        quantity=('quantity', 'sum'),
        # Get the most frequent unit for the product. Handles cases where a product might have multiple units.
        # Use .mode()[0] for the most frequent, or 'N/A' if empty
        unit=('unit', lambda x: x.mode()[0] if not x.empty else 'N/A')
    ).reset_index().sort_values(by='sales', ascending=False)
    # Re-add category to product summary for consistent coloring on frontend
    summary_by_product = summary_by_product.merge(df[['product', 'category']].drop_duplicates(), on='product', how='left')
    summary_by_product = summary_by_product.head(10).to_dict(orient='records')
    logging.info(f"Summary by Product (Top 10): {summary_by_product}")


    # Summary by Unit (if 'unit' column exists)
    summary_by_unit = []
    if 'unit' in filtered_df.columns:
        # Group by the original 'unit' column if it exists and is meaningful
        # Only include rows where 'unit' is not NaN or empty
        unit_df = filtered_df.dropna(subset=['unit']).copy()
        if not unit_df.empty:
            summary_by_unit = unit_df.groupby('unit').agg(
                total_sales=('sales', 'sum'),
                total_quantity=('quantity', 'sum')
            ).reset_index().sort_values(by='total_sales', ascending=False).to_dict(orient='records')
        else:
            # Fallback if no explicit 'unit' column or it's all empty
            # Use extract_pack_size to derive units if product names have them
            filtered_df[['pack_size_label', 'derived_unit_value']] = filtered_df['product'].apply(extract_pack_size).apply(pd.Series)
            # Filter out 'Unknown' or 'Standard Pack' if they are too generic
            derived_unit_df = filtered_df[~filtered_df['pack_size_label'].isin(["Unknown", "Standard Pack"])].copy() # Updated filter
            if not derived_unit_df.empty:
                summary_by_unit = derived_unit_df.groupby('pack_size_label').agg(
                    total_sales=('sales', 'sum'),
                    total_quantity=('quantity', 'sum') # This would be total packs, not derived units for this grouping
                ).reset_index().sort_values(by='total_sales', ascending=False).to_dict(orient='records')
    logging.info(f"Summary by Unit: {summary_by_unit}")


    return {
        "total_sales": total_sales,
        "total_quantity": total_quantity,
        "transaction_count": transaction_count,
        "summary_by_category": summary_by_category,
        "summary_by_product": summary_by_product,
        "summary_by_unit": summary_by_unit
    }

@app.get("/data/products") # Removed trailing slash
async def get_products(category: str = Query(None)):
    df = get_current_df() # Correctly retrieves from 'data'
    filtered_df = df.copy()
    if category:
        # Use str.contains with case=False for case-insensitive matching
        filtered_df = filtered_df[filtered_df['category'].str.contains(category, case=False, na=False)]

    products = sorted(filtered_df['product'].dropna().unique().tolist())
    logging.info(f"Fetched products for category '{category}': {products}")
    return {"products": products}


@app.get("/data/monthly-sales-trend") # Removed trailing slash
async def get_monthly_sales_trend(
    start_date: date = Query(None),
    end_date: date = Query(None)
):
    df = get_current_df() # Correctly retrieves from 'data'
    filtered_df = df.copy()
    logging.info(f"Fetching monthly sales trend with filters: start_date={start_date}, end_date={end_date}")


    if start_date:
        filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]

    if filtered_df.empty:
        logging.info("No data found for the selected filters for monthly sales trend.")
        return {"months": [], "products_sales_data": []}

    # Identify top 5 products by total sales within the filtered timeframe
    product_total_sales = filtered_df.groupby('product')['sales'].sum().reset_index()
    product_total_sales = product_total_sales.sort_values(by='sales', ascending=False)
    top_5_products_names = product_total_sales.head(5)['product'].tolist()
    logging.info(f"Top 5 products for monthly trend: {top_5_products_names}")


    # Filter original (date-filtered) DataFrame for only the top 5 products
    df_top_5 = filtered_df[filtered_df['product'].isin(top_5_products_names)]

    # Get all unique month_year periods in the filtered data to use as labels
    all_months = sorted(df_top_5['month_year'].unique().tolist())
    logging.info(f"All months in filtered data: {all_months}")

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
            "category": category, # Include category for frontend coloring
            "sales_data": temp_series.tolist()
        })
    logging.info("Monthly sales data prepared.")

    return {"months": all_months, "products_sales_data": products_sales_data}


@app.get("/data/category-specific-summary") # Removed trailing slash
async def get_category_specific_summary(
    category: str = Query(...), # Category is now a mandatory parameter
    start_date: date = Query(None),
    end_date: date = Query(None)
):
    df = get_current_df() # FIX: Changed from df_store.get('data') to get_current_df()
    logging.info(f"Fetching category-specific summary for '{category}' with filters: start_date={start_date}, end_date={end_date}")

    # Filter by the requested category using case-insensitive contains
    category_df = df[df['category'].str.contains(category, case=False, na=False)].copy() 

    if start_date:
        category_df = category_df[category_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        category_df = category_df[category_df['date'] <= pd.to_datetime(end_date)]

    if category_df.empty:
        logging.info(f"Category-specific summary: No data found for category '{category}' with selected date filters after filtering by case-insensitive category match.")
        return {"variety_summary": [], "pack_size_summary": []}

    # Apply variety and pack size extraction
    category_df['parsed_variety'] = category_df.apply(lambda row: extract_variety(row['product'], row['category']), axis=1)
    # Apply pack size extraction and expand to two new columns
    category_df[['pack_size_label', 'derived_unit_value']] = category_df['product'].apply(extract_pack_size).apply(pd.Series)

    # Calculate "Derived Units Sold" for Variety Summary: quantity * derived_unit_value
    # Ensure 'quantity' and 'derived_unit_value' are numeric and handle potential non-numeric values
    category_df['quantity'] = pd.to_numeric(category_df['quantity'], errors='coerce').fillna(0)
    category_df['derived_unit_value'] = pd.to_numeric(category_df['derived_unit_value'], errors='coerce').fillna(0)
    
    # If derived_unit_value is 0 (meaning no specific quantifiable unit was found from product name),
    # assume each 'quantity' represents 1 derived unit. This is crucial for "pack" or "piece" items.
    category_df['total_derived_units_sold'] = category_df.apply(
        lambda row: row['quantity'] * row['derived_unit_value'] if row['derived_unit_value'] > 0 else row['quantity'],
        axis=1
    )
    
    # Summary by Variety
    variety_summary = category_df.groupby('parsed_variety').agg(
        total_sales=('sales', 'sum'),
        total_derived_units_sold=('total_derived_units_sold', 'sum')
    ).reset_index().sort_values(by='total_sales', ascending=False).to_dict(orient='records')

    # Summary by Pack Size
    # For pack size, 'total_packs_sold' is just the sum of 'quantity' for that pack_size_label
    pack_size_summary = category_df.groupby('pack_size_label').agg(
        total_sales=('sales', 'sum'),
        total_packs_sold=('quantity', 'sum'),
        total_derived_units_from_packs=('total_derived_units_sold', 'sum')
    ).reset_index().sort_values(by='total_sales', ascending=False).to_dict(orient='records')

    logging.info(f"Category '{category}' variety summary: {variety_summary}")
    logging.info(f"Category '{category}' pack size summary: {pack_size_summary}")

    return {
        "variety_summary": variety_summary,
        "pack_size_summary": pack_size_summary
    }

