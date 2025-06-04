# Supermarket Sales Analytics Dashboard

An interactive dashboard built with Python and Streamlit to analyze and visualize supermarket sales data, track KPIs, and forecast trends.

## Features
- **Dynamic Data Filtering**: Slice data by city, customer type, gender, product line, and date range
- **Real-time KPIs**: Track total sales, average ratings, transactions, and profit margins
- **Interactive Visualizations**:
  - Sales trends by hour, weekday, and product line
  - Heatmaps, pie/donut charts, and polar plots for multidimensional analysis
- **Time Series Forecasting**: Integrated Facebook Prophet for predictive sales modeling
- **Export Options**: Download filtered data as CSV or Excel

## Tech Stack
- **Python** (Pandas, Plotly, Streamlit)
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Forecasting**: Facebook Prophet
- **Data Handling**: Pandas, Excel/CSV I/O

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supermarket-sales-dashboard.git
   cd supermarket-sales-dashboard

2. Install dependencies:

3. Run the Streamlit app: streamlit run app.py

## Usage

Filter Data: Use the sidebar to select cities, customer types, etc.

Explore Visualizations: Navigate through tabs to view charts and forecasts

Export Data: Download filtered datasets via the "Download Options" section

Forecasting: Upload a CSV with ds (date) and y (value) columns to train a Prophet model

## License: MIT
