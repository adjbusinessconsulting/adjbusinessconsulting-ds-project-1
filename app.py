import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer Clustering — DS Project 1",
    page_icon="🔵",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0D1B2E; }
    .stApp { background-color: #0D1B2E; color: white; }
    h1, h2, h3 { color: white; }
    .metric-card {
        background: #162440;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid #1E3A5F;
    }
    .cluster-0 { background: #1a3a5c; border-left: 4px solid #60A5FA; }
    .cluster-1 { background: #1a3a2a; border-left: 4px solid #34D399; }
    .cluster-2 { background: #3a2a1a; border-left: 4px solid #FBBF24; }
    .cluster-3 { background: #3a1a1a; border-left: 4px solid #F87171; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_cluster():
    df = pd.read_parquet("online_retail_cleaned.parquet")
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    df = df[df['customer_id'].notna()]
    df = df[(df['price'] > 0) & (df['quantity'] > 0)]
    df['customer_id'] = df['customer_id'].astype(str).replace('nan', np.nan)
    df = df.dropna(subset=['customer_id'])

    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        Recency=('invoicedate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('invoice', 'nunique'),
        Monetary=('total_price', 'sum'),
        Total_Quantity=('quantity', 'sum')
    ).reset_index()
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Avg_Quantity_per_Invoice'] = rfm['Total_Quantity'] / rfm['Frequency']

    cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Quantity_per_Invoice']
    log_df = rfm.copy()
    log_df[cols] = np.log1p(log_df[cols])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_df[cols])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    cluster_names = {0: 'Seasonal Buyers', 1: 'MVP Customers', 2: 'Mid Value Customers', 3: 'Low Value Customers'}
    rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_names)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    rfm['PC1'] = pca_result[:, 0]
    rfm['PC2'] = pca_result[:, 1]

    return rfm

st.title("🔵 Customer Clustering with K-Means ML")
st.markdown("**DS Project 1 — ADJ Business Consulting** · UCI Online Retail Dataset · 805,549 rows → 5,878 customers")
st.markdown("---")

with st.spinner("Running K-Means clustering on your data..."):
    rfm = load_and_cluster()

cluster_colors = {
    'Seasonal Buyers': '#60A5FA',
    'MVP Customers': '#34D399',
    'Mid Value Customers': '#FBBF24',
    'Low Value Customers': '#F87171'
}

# ── METRICS ROW ──
col1, col2, col3, col4 = st.columns(4)
for col, cluster_id, name, color in zip(
    [col1, col2, col3, col4],
    [0, 1, 2, 3],
    ['Seasonal Buyers', 'MVP Customers', 'Mid Value Customers', 'Low Value Customers'],
    ['#60A5FA', '#34D399', '#FBBF24', '#F87171']
):
    subset = rfm[rfm['Cluster'] == cluster_id]
    pct = len(subset) / len(rfm) * 100
    monetary = subset['Monetary'].median()
    col.metric(f"Cluster {cluster_id} — {name}", f"{pct:.1f}% ({len(subset):,})", f"Median spend: ${monetary:,.0f}")

st.markdown("---")

# ── PCA SCATTER ──
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Customer Clusters — PCA Visualization")
    fig_pca = px.scatter(
        rfm, x='PC1', y='PC2', color='Cluster_Name',
        color_discrete_map=cluster_colors,
        hover_data={'Recency': True, 'Frequency': True, 'Monetary': ':.0f', 'PC1': False, 'PC2': False},
        opacity=0.6, height=450,
        labels={'Cluster_Name': 'Cluster'}
    )
    fig_pca.update_layout(
        paper_bgcolor='#0D1B2E', plot_bgcolor='#0D1B2E',
        font_color='white', legend_title_text='Cluster',
    )
    fig_pca.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_pca, use_container_width=True)

with col_right:
    st.subheader("Revenue Share by Cluster")
    revenue = rfm.groupby('Cluster_Name')['Monetary'].sum().reset_index()
    revenue['pct'] = revenue['Monetary'] / revenue['Monetary'].sum() * 100
    fig_pie = px.pie(
        revenue, values='Monetary', names='Cluster_Name',
        color='Cluster_Name', color_discrete_map=cluster_colors,
        height=450
    )
    fig_pie.update_layout(
        paper_bgcolor='#0D1B2E', font_color='white',
        legend_title_text='Cluster'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── RFM BOX PLOTS ──
st.subheader("RFM Distribution by Cluster")
tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])
for tab, metric in zip([tab1, tab2, tab3], ['Recency', 'Frequency', 'Monetary']):
    with tab:
        fig = px.box(
            rfm, x='Cluster_Name', y=metric, color='Cluster_Name',
            color_discrete_map=cluster_colors, height=380
        )
        fig.update_layout(
            paper_bgcolor='#0D1B2E', plot_bgcolor='#0D1B2E',
            font_color='white', showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ── CLUSTER PROFILES ──
st.subheader("Cluster Profile Table")
profile = rfm.groupby(['Cluster', 'Cluster_Name']).agg(
    Customers=('customer_id', 'nunique'),
    Recency_median=('Recency', 'median'),
    Frequency_median=('Frequency', 'median'),
    Monetary_median=('Monetary', 'median'),
    Revenue_Total=('Monetary', 'sum'),
).reset_index()
profile['Revenue_%'] = (profile['Revenue_Total'] / profile['Revenue_Total'].sum() * 100).round(2)
profile['Customer_%'] = (profile['Customers'] / profile['Customers'].sum() * 100).round(2)
profile = profile[['Cluster', 'Cluster_Name', 'Customers', 'Customer_%', 'Recency_median', 'Frequency_median', 'Monetary_median', 'Revenue_%']]
profile.columns = ['Cluster', 'Name', 'Customers', 'Customer %', 'Recency (days)', 'Frequency', 'Monetary ($)', 'Revenue %']
st.dataframe(profile, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("**ADJ Business Consulting** · [Portfolio](https://adjbusinessconsulting.github.io/adj-consulting/portfolio.html)")
