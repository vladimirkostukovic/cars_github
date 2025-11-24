# Auto Project

A data analytics system for the Czech automotive market built in Python.
Scrapes listings from sauto.cz, processes them through an ETL pipeline,
and visualizes the results in Apache Superset.

## Pipeline Steps

1. **Merge** - Daily snapshots into unified table. Tracks new listings, sold cars, and reactivations.
2. **Transform** - Raw data to clean format. Extracts fuel, gearbox, manufacturer from JSON fields.
3. **Price Tracking** - Records price changes over time for each listing.
4. **Analytics** - Calculates market metrics per car segment.

## Analytics Metrics

Segments are grouped by: manufacturer, model, year, fuel, gearbox, mileage range.

For each segment:

- **Price categories** - Split into 3 price tiers (cheap, mid, expensive) based on min/max in segment
- **Sales by category** - How many sold in each price tier, percentage breakdown
- **Days on market** - Average time from listing to sale
- **Liquidity 30d** - How many listed vs sold in last 30 days, sell ratio
- **Price drops** - Average price reduction before sale (absolute and percent)
- **Recommended price** - Suggested listing price based on what actually sells
- **Max buy price** - Maximum price to pay if buying for resale
- **Risk score** - 0-100 score based on how fast cheap cars sell in this segment

## Stack

- Python, Pandas, SQLAlchemy
- PostgreSQL
- Apache Superset
