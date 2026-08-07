import argparse
import json
import os
import numpy as np
import pandas as pd
from config import Config
from common import utils

COL_REAL_DAYS = "Days for shipping (real)"
COL_SCHED_DAYS = "Days for shipment (scheduled)"
COL_LATE_RISK = "Late_delivery_risk"
COL_DELIVERY_STATUS = "Delivery Status"
COL_ORDER_STATUS = "Order Status"
COL_ORDER_ID = "Order Id"
COL_ORDER_DATE = "order date (DateOrders)"
COL_SHIP_DATE = "shipping date (DateOrders)"
COL_SHIP_MODE = "Shipping Mode"
COL_REGION = "Order Region"
COL_MARKET = "Market"
COL_QUANTITY = "Order Item Quantity"
COL_SALES = "Sales"
COL_DISCOUNT_RATE = "Order Item Discount Rate"
COL_LATITUDE = "Latitude"
COL_LONGITUDE = "Longitude"

REQUIRED_COLUMNS = [
    COL_REAL_DAYS, COL_SCHED_DAYS, COL_LATE_RISK, COL_DELIVERY_STATUS,
    COL_ORDER_STATUS, COL_ORDER_ID, COL_ORDER_DATE, COL_SHIP_DATE,
    COL_SHIP_MODE, COL_REGION, COL_MARKET, COL_QUANTITY, COL_SALES,
    COL_DISCOUNT_RATE, COL_LATITUDE, COL_LONGITUDE,
]

DROP_ORDER_STATUSES = ["CANCELED", "SUSPECTED_FRAUD"]
DROP_DELIVERY_STATUSES = ["Shipping canceled"]

SHIPPING_MODE_SPEED = {
    "Same Day": 0,
    "First Class": 1,
    "Second Class": 2,
    "Standard Class": 3,
}

CAUSAL_COLUMNS = [
    "shipping_mode_speed",
    "scheduled_ship_days",
    "order_processing_time",
    "geographic_distance",
    "order_item_count",
    "total_quantity",
    "total_sales",
    "avg_discount_rate",
    "daily_order_volume",
    "region_weekly_orders",
    "delivery_delay",
]

METADATA_COLUMNS = [
    "order_id",
    "order_date",
    "order_region",
    "order_market",
    "region_month",
    "late_flag", 
]

CONSTRUCTED_VARIABLES = {
    "daily_order_volume": (
        "Count of orders sharing the order's calendar date. A direct measure "
        "of how much work entered the fulfilment pipeline that day."
    ),
    "region_weekly_orders": (
        "Count of orders sharing the order's destination region and ISO week. "
        "A direct measure of local demand pressure on that lane."
    ),
    "order_processing_time": (
        "Days between order placement and dispatch, derived from the order "
        "and shipping date columns. Internal handling time before the carrier."
    ),
    "geographic_distance": (
        "Great-circle distance from the customer coordinates to the network's "
        "mean location. A distance-to-serve proxy; not a true customer-to-"
        "origin distance, since DataCo gives the order side only as strings."
    ),
}

NATURAL_DICTIONARY = {
    "shipping_mode_speed": "Ordinal shipping service level chosen for the order (0 = same day, 3 = standard slow service).",
    "scheduled_ship_days": "Number of days the carrier promised for delivery when the order was placed.",
    "order_processing_time": "Days between the order being placed and dispatched (internal handling time before the carrier takes over).",
    "geographic_distance": "Great-circle distance from the customer location to the network's mean hub location, in km (a distance-to-serve proxy).",
    "order_item_count": "Number of distinct line items in the order.",
    "total_quantity": "Total units of product across all line items of the order.",
    "total_sales": "Total sales value of the order in dollars.",
    "avg_discount_rate": "Average discount rate applied across the order's line items.",
    "daily_order_volume": "Number of orders the retailer received on the same calendar day (fulfilment workload proxy).",
    "region_weekly_orders": "Number of orders from the same destination region in the same week (regional demand proxy).",
    "delivery_delay": "Actual delivery days minus promised delivery days; positive means the order arrived late (outcome).",
}

RENAMED_DICTIONARY = {
    "shipping_mode_speed": ("service_tier", "Ordinal tier of the transport service booked for a consignment (0 = fastest promise, 3 = slowest)."),
    "scheduled_ship_days": ("promised_transit_days", "Days quoted to the client for the consignment to arrive."),
    "order_processing_time": ("handling_lead_days", "Days between booking and dispatch (internal handling before carrier pickup)."),
    "geographic_distance": ("serve_distance_km", "Great-circle distance from destination to the network hub, in km."),
    "order_item_count": ("line_count", "Number of distinct product lines in the consignment."),
    "total_quantity": ("unit_total", "Total units across all lines of the consignment."),
    "total_sales": ("consignment_value", "Monetary value of the consignment."),
    "avg_discount_rate": ("mean_markdown", "Average markdown applied to the consignment's lines."),
    "daily_order_volume": ("same_day_workload", "Consignments booked by the seller on the same day (workload proxy)."),
    "region_weekly_orders": ("zone_weekly_bookings", "Consignments headed to the same zone in the same week (local demand proxy)."),
    "delivery_delay": ("arrival_slip_days", "Actual transit days minus promised transit days; positive = late (outcome)."),
}

def load_raw(path):
    df = pd.read_csv(path, encoding="latin-1")
    missing = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(
            "Raw CSV is missing expected columns: " + ", ".join(missing) + "\nFound columns: " + ", ".join(df.columns)
        )
    return df

def basic_clean(df, report):
    n_start = len(df)

    keep = ~df[COL_ORDER_STATUS].isin(DROP_ORDER_STATUSES)
    keep = keep & ~df[COL_DELIVERY_STATUS].isin(DROP_DELIVERY_STATUSES)
    df = df[keep].copy()
    report["rows_dropped_cancelled_or_fraud"] = n_start - len(df)

    df["order_date"] = pd.to_datetime(df[COL_ORDER_DATE], errors="coerce")
    df["ship_date"] = pd.to_datetime(df[COL_SHIP_DATE], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["order_date", "ship_date"]).copy()
    report["rows_dropped_bad_dates"] = n_before - len(df)

    core = [COL_REAL_DAYS, COL_SCHED_DAYS, COL_QUANTITY, COL_SALES,
            COL_DISCOUNT_RATE, COL_LATITUDE, COL_LONGITUDE]
    n_before = len(df)
    df = df.dropna(subset=core).copy()
    report["rows_dropped_missing_numeric"] = n_before - len(df)

    n_before = len(df)
    df = df[df[COL_SHIP_MODE].isin(SHIPPING_MODE_SPEED.keys())].copy()
    report["rows_dropped_unknown_ship_mode"] = n_before - len(df)

    report["item_rows_after_clean"] = len(df)
    return df

def aggregate_to_orders(df, report):
    grouped = df.groupby(COL_ORDER_ID)

    orders = grouped.agg(
        real_days=(COL_REAL_DAYS, "first"),
        scheduled_ship_days=(COL_SCHED_DAYS, "first"),
        late_flag=(COL_LATE_RISK, "first"),
        ship_mode=(COL_SHIP_MODE, "first"),
        order_region=(COL_REGION, "first"),
        order_market=(COL_MARKET, "first"),
        order_date=("order_date", "min"),
        ship_date=("ship_date", "min"),
        latitude=(COL_LATITUDE, "first"),
        longitude=(COL_LONGITUDE, "first"),
        order_item_count=(COL_QUANTITY, "size"),
        total_quantity=(COL_QUANTITY, "sum"),
        total_sales=(COL_SALES, "sum"),
        avg_discount_rate=(COL_DISCOUNT_RATE, "mean"),
    ).reset_index()
    orders = orders.rename(columns={COL_ORDER_ID: "order_id"})

    report["order_rows"] = len(orders)
    return orders

def add_outcome(orders):
    orders["delivery_delay"] = orders["real_days"] - orders["scheduled_ship_days"]
    return orders

def add_processing_time(orders):
    orders["order_processing_time"] = (orders["ship_date"] - orders["order_date"]).dt.days
    return orders

def add_shipping_mode_speed(orders):
    speeds = []
    for mode in orders["ship_mode"]:
        speeds.append(SHIPPING_MODE_SPEED[mode])
    orders["shipping_mode_speed"] = speeds
    return orders

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return radius * 2 * np.arcsin(np.sqrt(a))

def add_geographic_distance(orders):
    hub_lat = orders["latitude"].mean()
    hub_lon = orders["longitude"].mean()
    orders["geographic_distance"] = haversine_km(
        orders["latitude"].values, orders["longitude"].values,
        hub_lat, hub_lon)
    return orders


def add_congestion_proxy(orders):
    order_day = orders["order_date"].dt.normalize()
    counts_per_day = order_day.value_counts()

    volumes = []
    for day in order_day:
        volumes.append(int(counts_per_day[day]))
    orders["daily_order_volume"] = volumes
    return orders


def add_demand_proxy(orders):
    iso = orders["order_date"].dt.isocalendar()
    region_week = (
        orders["order_region"].astype(str)
        + "|" + iso["year"].astype(str)
        + "-W" + iso["week"].astype(str)
    )
    counts_per_region_week = region_week.value_counts()

    volumes = []
    for key in region_week:
        volumes.append(int(counts_per_region_week[key]))
    orders["region_weekly_orders"] = volumes
    return orders


def add_region_month(orders):
    orders["region_month"] = (orders["order_region"].astype(str) + "|" + orders["order_date"].dt.strftime("%Y-%m"))
    return orders

def add_shipping_mode_speed(orders):
    speeds = []
    for mode in orders["ship_mode"]:
        speeds.append(SHIPPING_MODE_SPEED[mode])
    orders["shipping_mode_speed"] = speeds
    return orders


def build_data_dictionary(renamed=False):
    data_dict = {"leg": "dataco", "renamed": renamed, "natural": []}
    source = RENAMED_DICTIONARY if renamed else NATURAL_DICTIONARY
    for col in NATURAL_DICTIONARY:
        if renamed:
            new_name, desc = RENAMED_DICTIONARY[col]
            data_dict["natural"].append({"name": new_name,
                                         "description": desc})
        else:
            data_dict["natural"].append({"name": col,
                                         "description": NATURAL_DICTIONARY[col]})
    return data_dict

def _renamed(col, renamed):
    if renamed and col in RENAMED_DICTIONARY:
        return RENAMED_DICTIONARY[col][0]
    return col


def build_downstream_config(renamed=False):
    target = "delivery_delay"
    causes = []
    actionables = [
        "shipping_mode_speed",
        "scheduled_ship_days",
        "order_processing_time",
    ]
    return {
        "renamed": renamed,
        "target": _renamed(target, renamed),
        "causes": [_renamed(c, renamed) for c in causes],
        "actionables": [_renamed(a, renamed) for a in actionables],
    }

def rename_frame(df):
    mapping = {col: RENAMED_DICTIONARY[col][0] for col in df.columns
               if col in RENAMED_DICTIONARY}
    return df.rename(columns=mapping)


def clean(df):
    report = {}

    print("Cleaning ...")
    df = basic_clean(df, report)

    print("Aggregating to orders ...")
    orders = aggregate_to_orders(df, report)

    print("Building causal variables ...")
    orders = add_outcome(orders)
    orders = add_shipping_mode_speed(orders)
    orders = add_processing_time(orders)
    orders = add_geographic_distance(orders)
    orders = add_congestion_proxy(orders)
    orders = add_demand_proxy(orders)
    orders = add_region_month(orders)

    n_before = len(orders)
    orders = orders[orders["order_processing_time"] >= 0].copy()
    report["rows_dropped_negative_processing"] = n_before - len(orders)

    df = orders[CAUSAL_COLUMNS].copy()
    for col in CAUSAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    for key, value in report.items():
        print(f"  {key}: {value}")
    return df
    
def process(output_dir, renamed=False):

    print("Loading raw CSV ...")
    df = load_raw(Config.REAL_DATA_DIR + '/DataCoSupplyChainDataset.csv')
    print(f"  {len(df)} item rows, {len(df.columns)} columns")

    data_output_dir = output_dir + '/data'
    utils.create_dir_if_not_exists(data_output_dir)

    df_final = clean(df)
    if renamed:
        df_final = rename_frame(df_final)
    n = len(df_final)

    data_file_name = f'data_{n}.csv'
    df_final.to_csv(os.path.join(data_output_dir, data_file_name),
                    index=False)

    metadata_output_dir = output_dir + '/metadata'
    utils.create_dir_if_not_exists(metadata_output_dir)

    data_files_list = [
        {'name': data_file_name, 'seed': None, 'num_rows': n}
    ]
    utils.write_json(data_files_list, metadata_output_dir + '/data_files_list.json')

    data_dict = build_data_dictionary(renamed=renamed)
    utils.write_json(data_dict, metadata_output_dir + '/data_dictionary.json')

    cfg = build_downstream_config(renamed=renamed)
    utils.write_json(cfg, metadata_output_dir + '/downstream_config.json')

    print(f"Wrote {n} rows to {data_file_name} "
          f"({'renamed' if renamed else 'natural'} variables).")
    
    
    

