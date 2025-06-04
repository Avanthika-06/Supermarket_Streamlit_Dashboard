import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from io import BytesIO

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

@st.cache_data
def get_data_from_excel():
    df = pd.read_excel(
        io="supermarkt_sales.xlsx",
        engine="openpyxl",
        sheet_name="Sales",
        skiprows=3,
        usecols="B:R",
        nrows=1000,
    )
    df["City"] = df["City"].replace({
        "Yangon": "Madurai",
        "Mandalay": "Coimbatore",
        "Naypyitaw": "Salem"
    })
    
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    df["Date"] = pd.to_datetime(df["Date"])
    df["weekday"] = df["Date"].dt.day_name()
    return df

df = get_data_from_excel()

# Sidebar filters
st.sidebar.header("Please Filter Here:")
city = st.sidebar.multiselect(
    "Select the City:",
    options=["Madurai", "Coimbatore", "Salem"],
    default=["Madurai", "Coimbatore", "Salem"]
)
customer_type = st.sidebar.multiselect(
    "Select the Customer Type:", 
    options=df["Customer_type"].unique(), 
    default=df["Customer_type"].unique()
)
gender = st.sidebar.multiselect(
    "Select the Gender:", 
    options=df["Gender"].unique(), 
    default=df["Gender"].unique()
)

df_selection = df.query("City == @city & Customer_type == @customer_type & Gender == @gender")
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# KPI Calculation
total_sales = df_selection["Total"].sum()
average_rating = round(df_selection["Rating"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_sale_by_transaction = df_selection["Total"].mean()
total_transactions = df_selection["Invoice ID"].nunique()
total_quantity = int(df_selection["Quantity"].sum())
total_gross_income = df_selection["gross income"].sum()
unique_products_sold = df_selection["Product line"].nunique()
gross_margin_percentage = 30
total_gross_margin = (gross_margin_percentage / 100) * total_sales
profit_margin_pct = (total_gross_margin / total_sales) * 100 if total_sales != 0 else 0

# Currency formatting function
def format_currency(value):
    return "₹{:,.2f}".format(value)

# Title
st.title(":bar_chart: SALES DASHBOARD FOR SUPERMARKET")
st.markdown("##")

# KPIs (top row)
left_col, middle_col, right_col = st.columns(3)
with left_col:
    st.markdown('<h4 style="color:red;">TOTAL SALES:</h4>', unsafe_allow_html=True)
    st.markdown(f"<h3>{format_currency(total_sales)}</h3>", unsafe_allow_html=True)
with middle_col:
    st.markdown('<h4 style="color:red;">AVERAGE RATING:</h4>', unsafe_allow_html=True)
    st.subheader(f"{average_rating} {star_rating}")
with right_col:
    st.markdown('<h5 style="color:red;">AVERAGE SALES PER TRANSACTION:</h5>', unsafe_allow_html=True)
    st.markdown(f"<h3>{format_currency(average_sale_by_transaction)}</h3>", unsafe_allow_html=True)

# Sidebar KPIs
with st.sidebar:
    st.markdown('<h4 style="color:yellow;">TOTAL TRANSACTIONS:</h4>', unsafe_allow_html=True)
    st.markdown(f"<h3>{total_transactions:,}</h3>", unsafe_allow_html=True)
    st.markdown('<h4 style="color:yellow;">TOTAL QUANTITY SOLD:</h4>', unsafe_allow_html=True)
    st.markdown(f"<h3>{total_quantity:,}</h3>", unsafe_allow_html=True)
    st.markdown('<h4 style="color:yellow;">TOTAL GROSS INCOME:</h4>', unsafe_allow_html=True)
    st.markdown(f"<h3>{format_currency(total_gross_income)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:yellow;'>UNIQUE PRODUCTS SOLD:</h4>", unsafe_allow_html=True)
    st.markdown(f"<h3>{unique_products_sold}</h3>", unsafe_allow_html=True)
    st.markdown('<h5 style="color:yellow;">PROFIT MARGIN %:</h5>', unsafe_allow_html=True)
    st.markdown(f"<h3>{profit_margin_pct:.0f}%</h3>", unsafe_allow_html=True)

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.header("📝 Notes Section")
user_notes = st.sidebar.text_area("Add your notes here:", height=200, 
                                 help="Type any observations or comments about the data")

# Data Table Section
st.header("📊 Detailed Transaction Data")

col1, col2, col3 = st.columns(3)
with col1:
    product_filter = st.multiselect(
        "Filter by Product Line:",
        options=df_selection["Product line"].unique(),
        default=df_selection["Product line"].unique()
    )
with col2:
    payment_filter = st.multiselect(
        "Filter by Payment Method:",
        options=df_selection["Payment"].unique(),
        default=df_selection["Payment"].unique()
    )
with col3:
    date_range = st.date_input(
        "Filter by Date Range:",
        value=(df_selection["Date"].min(), df_selection["Date"].max()),
        min_value=df_selection["Date"].min(),
        max_value=df_selection["Date"].max()
    )

# Apply filters
filtered_df = df_selection[
    (df_selection["Product line"].isin(product_filter)) &
    (df_selection["Payment"].isin(payment_filter)) &
    (df_selection["Date"] >= pd.to_datetime(date_range[0])) &
    (df_selection["Date"] <= pd.to_datetime(date_range[1]))
]

# Display data with currency formatting
st.dataframe(filtered_df.style.format({
    "Total": "₹{:,.2f}",
    "gross income": "₹{:,.2f}",
    "Rating": "{:.1f}"
}), height=500)

# Download buttons
st.markdown("### Download Options")

csv = filtered_df.to_csv(index=False).encode('utf-8')
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    filtered_df.to_excel(writer, index=False, sheet_name='SalesData')
excel_data = output.getvalue()

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        label="Download as Excel",
        data=excel_data,
        file_name="filtered_sales_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")

# Sales Overview
st.header("Sales Overview")

col1, col2 = st.columns(2)
with col1:
    sales_by_product_line = df_selection.groupby("Product line")[["Total"]].sum().sort_values(by="Total").reset_index()
    fig_product_sales_bar = px.bar(
        sales_by_product_line,
        x="Total",
        y="Product line",
        orientation="h",
        color="Total",
        color_continuous_scale=px.colors.sequential.Viridis,
        template="plotly_white"
    )
    fig_product_sales_bar.update_layout(
        title=dict(text="Sales by Product Line", font=dict(size=20, color="violet")),
        plot_bgcolor="rgba(0,0,0,0)", 
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickprefix="₹", tickformat=",.2f")
    )
    st.plotly_chart(fig_product_sales_bar, use_container_width=True)

with col2:
    sales_by_hour = df_selection.groupby('hour')['Total'].sum().reset_index()
    fig_scatter_bright = px.scatter(
        sales_by_hour,
        x='hour',
        y='Total',
        color='Total',
        size='Total',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        template='plotly_white'
    )
    fig_scatter_bright.update_traces(marker=dict(size=15, line=dict(width=1, color='black'), symbol='circle'))
    fig_scatter_bright.update_layout(
        title=dict(text="Sales by Hour", font=dict(size=20, color="violet")),
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False, tickprefix="₹", tickformat=",.2f"),
        xaxis_title="Hour of Day",
        yaxis_title="Total Sales",
    )
    st.plotly_chart(fig_scatter_bright, use_container_width=True)

# City and Payment Analysis
st.header("City and Payment Analysis")

col3, col4 = st.columns(2)
with col3:
    sales_by_city = df_selection.groupby(by=["City"])[["Total"]].sum().reset_index()
    fig_sales_city = px.pie(
        sales_by_city,
        values="Total",
        names="City",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_sales_city.update_traces(texttemplate='₹%{value:,.2f}<br>%{percent:.2%}', textposition='auto', textfont=dict(color='white', size=14))
    fig_sales_city.update_layout(title=dict(text="Sales Distribution by City", font=dict(size=20, color="violet")))
    st.plotly_chart(fig_sales_city, use_container_width=True)

with col4:
    payment_counts = df_selection["Payment"].value_counts().reset_index()
    payment_counts.columns = ["Payment Method", "Count"]
    fig_payment_donut = px.pie(
        payment_counts,
        values="Count",
        names="Payment Method",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_payment_donut.update_traces(textinfo='value+percent', texttemplate='%{label}<br>%{value}<br>(%{percent})')
    fig_payment_donut.update_layout(title=dict(text="Payment Method Distribution", font=dict(size=20, color="violet")))
    st.plotly_chart(fig_payment_donut, use_container_width=True)

# Weekly and Daily Trends
st.header("Weekly and Daily Sales Trends")

col5, col6 = st.columns(2)
with col5:
    transactions_by_weekday = df_selection.groupby("weekday")["Invoice ID"].count().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index()
    fig_weekday_polar = px.line_polar(
        transactions_by_weekday,
        r="Invoice ID",
        theta="weekday",
        line_close=True,
        color_discrete_sequence=["#636EFA"]
    )
    fig_weekday_polar.update_traces(fill="toself")
    fig_weekday_polar.update_layout(
        title=dict(text="Number of Transactions by Day of the Week", font=dict(size=20, color="violet")),
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, ticks=""),
            angularaxis=dict(direction="clockwise")
        )
    )
    st.plotly_chart(fig_weekday_polar, use_container_width=True)

with col6:
    daily_sales = df_selection.groupby(by=["Date"])[["Total"]].sum().reset_index()
    fig_daily_sales = px.line(
        daily_sales,
        x="Date",
        y="Total",
        title="Daily Sales Trend",
        markers=True,
        color_discrete_sequence=["#00CC96"]
    )
    fig_daily_sales.update_layout(
        title=dict(text="Daily Sales Trend", font=dict(size=20, color="violet")),
        xaxis_title="Date",
        yaxis_title="Total Sales",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False, tickprefix="₹", tickformat=",.2f")
    )
    st.plotly_chart(fig_daily_sales, use_container_width=True)

# Advanced Visualizations
st.header("Advanced Visualizations")

col7, col8 = st.columns(2)
with col7:
    sales_pivot = df_selection.pivot_table(index="Product line", columns="hour", values="Total", aggfunc="sum").fillna(0)
    fig_heatmap = px.imshow(
        sales_pivot,
        labels=dict(x="Hour of Day", y="Product Line", color="Total Sales"),
        x=sales_pivot.columns,
        y=sales_pivot.index,
        color_continuous_scale="Viridis",
    )
    fig_heatmap.update_traces(hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Sales: ₹%{z:,.2f}<extra></extra>")
    fig_heatmap.update_layout(title=dict(text="Heatmap of Sales by Hour and Product Line", font=dict(size=20, color="violet")))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col8:
    transactions_wk_prod = df_selection.groupby(["weekday", "Product line"]).size().reset_index(name="count")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    transactions_wk_prod["weekday"] = pd.Categorical(transactions_wk_prod["weekday"], categories=weekday_order, ordered=True)
    transactions_wk_prod = transactions_wk_prod.sort_values("weekday")

    fig = px.bar(
        transactions_wk_prod,
        x="weekday",
        y="count",
        color="Product line",
        labels={"count": "Number of Transactions", "weekday": "Day of Week"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        title=dict(text="Transactions by Weekday and Product Line", font=dict(size=20, color="violet")),
        xaxis_title="Day of Week",
        yaxis_title="Number of Transactions",
        barmode="stack",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Forecasting Section
st.header("📈 Time Series Forecasting with Prophet")

st.markdown("""
Upload your CSV file with **'ds'** (date) and **'y'** (value) columns to train and forecast.
Use the slider to select forecast horizon.
Use the date filter to select training data period.
""")

uploaded_file = st.file_uploader("Upload CSV with columns 'ds' and 'y'", type=["csv"])

if uploaded_file is not None:
    df_forecast = pd.read_csv(uploaded_file)
    if not {'ds', 'y'}.issubset(df_forecast.columns):
        st.error("CSV must contain 'ds' and 'y' columns.")
    else:
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
        min_date = df_forecast['ds'].min()
        max_date = df_forecast['ds'].max()

        st.write(f"Date range in data: {min_date.date()} to {max_date.date()}")
        date_filter = st.date_input("Select training data date range:",
                                    value=(min_date.date(), max_date.date()),
                                    min_value=min_date.date(),
                                    max_value=max_date.date())

        start_date, end_date = pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])
        df_train = df_forecast[(df_forecast['ds'] >= start_date) & (df_forecast['ds'] <= end_date)]

        max_horizon = 365 if len(df_forecast) < 365 else 2 * 365
        forecast_horizon = st.slider("Select forecast horizon (days):", min_value=1, max_value=max_horizon, value=30)

        @st.cache_data(show_spinner=False)
        def train_prophet_model(df):
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(df)
            return m

        if len(df_train) < 2:
            st.warning("Not enough data points in the selected date range to train the model.")
        else:
            model = train_prophet_model(df_train)
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)

            st.subheader("Forecast Plot")
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(yaxis=dict(tickprefix="₹", tickformat=",.2f"))
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("Actual vs Forecast")
            merged_df = pd.merge(df_forecast, forecast[['ds', 'yhat']], on='ds', how='outer')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged_df['ds'], y=merged_df['y'], mode='markers+lines', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=merged_df['ds'], y=merged_df['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
            fig.update_layout(
                title='Actual vs Forecast', 
                xaxis_title='Date', 
                yaxis_title='Value', 
                template='plotly_white',
                yaxis=dict(tickprefix="₹", tickformat=",.2f")
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a CSV file above to get started with forecasting.")