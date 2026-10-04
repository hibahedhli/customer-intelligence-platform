"""
The canonical (internal) schema. The modelling pipeline ONLY knows these names.
Dataset-specific column names live in the adapter configuration (configs/*.yaml).
"""

# ---- transaction table: one row per order line ---------------------------------------------
REQUIRED_TX = ["customer_id", "order_id", "date", "quantity", "unit_price"]
OPTIONAL_TX = ["transaction_id", "product_id", "product_name", "category", "discount", "payment_method"]
TX_COLUMNS = ["transaction_id", "order_id", "customer_id", "date", "product_id", "product_name", "category",
              "quantity", "unit_price", "discount", "payment_method"]

# ---- customer table: one row per customer (derived from transactions if not provided) ----------
REQUIRED_CUST = ["customer_id"]
OPTIONAL_CUST = ["customer_registration_date", "customer_age", "customer_gender", "customer_location"]
CUST_COLUMNS = ["customer_id"] + OPTIONAL_CUST

DESCRIPTIONS = {
    "customer_id": "Unique customer identifier (required). Rows without one cannot be analysed per customer and are excluded.",
    "order_id": "Identifier of one purchase event / basket / invoice (required). Several lines share it. "
                "If only a per-row id exists, map it to order_id as well (each row is then one order).",
    "transaction_id": "Unique id of an order LINE (optional; synthesised if absent).",
    "date": "Date (or datetime) of the purchase (required). Time of day is dropped.",
    "quantity": "Units on the line (required). Negative values are returns only if the return rule says so.",
    "unit_price": "Price per unit (required), any currency.",
    "product_id": "Product identifier (optional; enables the 'distinct products' feature).",
    "product_name": "Free-text product name (optional, descriptive).",
    "category": "Product category (optional; enables category features and favourite category).",
    "discount": "Discount as a fraction in [0, 1] (optional; use discount.scale: percent for 0-100).",
    "payment_method": "Payment method (optional, descriptive).",
    "customer_registration_date": "Account creation date (optional; falls back to the first purchase date).",
    "customer_age": "Age in years (optional, descriptive only, not used by the model).",
    "customer_gender": "Gender (optional, descriptive only, not used by the model).",
    "customer_location": "City / region / country (optional, descriptive only, not used by the model).",
}
