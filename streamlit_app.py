import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set up Streamlit page
st.set_page_config(
    page_title='Model Comparison Dashboard',
    page_icon=':bar_chart:',
)

# Title and description
st.title(':bar_chart: Used Cars Model Comparison')
st.markdown("""
#### MWS_WDM_S24  
Supervisor: Dr. Bassel Alkhatib  
Created by: Elias Dahdal (elias_335295) & Natalie Alkalaf (natalie_336924) 
""")
st.write("""
This dashboard compares the performance of three different machine learning algorithms:
- **Logistic Regression**: A linear model widely used for classification tasks.
- **Decision Tree**: A model that makes predictions by splitting data into branches.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
""")

# Initialize the comparison table data
Comparing_Table = {
    "Logistic Regression": {"Accuracy": 80.9, "Precision": 80.44, "Recall": 80.9, "F1 Score": 80.24},
    "Decision Tree": {"Accuracy": 81.13, "Precision": 80.96, "Recall": 81.13, "F1 Score": 81.01},
    "Naive Bayes": {"Accuracy": 37.21, "Precision": 76.55, "Recall": 37.21, "F1 Score": 41.16}
}

# Convert Comparing_Table to a DataFrame for easier plotting
df_comparison = pd.DataFrame(Comparing_Table)

# Sidebar Interactivity
st.sidebar.subheader("Interactive Options")
chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Radar Chart", "Heatmap"])

# Allow users to set a threshold for model accuracy
accuracy_threshold = st.sidebar.slider("Minimum Accuracy (%)", 0, 100, 50)
highlighted_models = df_comparison.loc['Accuracy'] >= accuracy_threshold
highlighted_text = ", ".join(df_comparison.columns[highlighted_models])
st.sidebar.write(f"Models with Accuracy ≥ {accuracy_threshold}%: {highlighted_text if highlighted_text else 'None'}")

# Allow users to select which metrics to display
selected_metrics = st.sidebar.multiselect("Select Metrics to Display", options=df_comparison.index, default=df_comparison.index)
filtered_df_comparison = df_comparison.loc[selected_metrics]


# Display selected chart type based on user input
if chart_type == "Bar Chart":
    st.subheader("Bar Chart of Model Performance")
    fig = px.bar(
        filtered_df_comparison,
        x=filtered_df_comparison.index,
        y=filtered_df_comparison.columns,
        title="Selected Model Performance Metrics",
        labels={'x': "Metrics", 'value': "Percentage (%)", 'variable': "Model"}
    )
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)

elif chart_type == "Radar Chart":
    st.subheader("Radar Chart of Model Performance")
    fig = go.Figure()
    for model_name in filtered_df_comparison.columns:
        fig.add_trace(go.Scatterpolar(
            r=filtered_df_comparison[model_name].tolist(),
            theta=filtered_df_comparison.index,
            fill='toself',
            name=model_name
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="Radar Chart of Selected Metrics",
        showlegend=True
    )
    st.plotly_chart(fig)

elif chart_type == "Heatmap":
    st.subheader("Heatmap of Model Performance")
    fig = px.imshow(
        filtered_df_comparison.values,
        labels=dict(x="Model", y="Metric", color="Percentage (%)"),
        x=filtered_df_comparison.columns,
        y=filtered_df_comparison.index,
        title="Heatmap of Selected Metrics",
        color_continuous_scale=color_theme
    )
    st.plotly_chart(fig)

# Explanations and Analysis
st.subheader("Discussion and Analysis")
st.write("""
Based on the performance metrics:
- **Decision Tree** is the top performer across all metrics.
- **Logistic Regression** performs well but slightly below the Decision Tree.
- **Naive Bayes** shows high Precision but struggles in other metrics due to its strong independence assumption.

### Conclusion
The Decision Tree model is preferred for this dataset due to its balanced and high performance across metrics.
""")
