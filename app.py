import streamlit as st
import pandas as pd
from utils import run_prediction, plot_sales_trend, plot_feature_importance, plot_residual_distribution

st.title('销售预测 Web App')

uploaded_file = st.file_uploader('上传 CSV 数据', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    days = st.slider('预测未来天数', 1, 30, 7)
    if st.button('运行预测'):
        with st.spinner('处理中...'):
            model, results, future_predictions, df_features, feature_cols, y_test, diagnosis = run_prediction(df, days)
        st.subheader('评估指标')
        st.write(f"MAE: {results['mae']:.2f}")
        st.write(f"RMSE: {results['rmse']:.2f}")
        st.write(f"MAPE: {results['mape']:.2f}%")
        st.subheader('可视化结果')
        st.plotly_chart(plot_sales_trend(df_features, future_predictions))
        st.plotly_chart(plot_feature_importance(model, feature_cols))
        st.plotly_chart(plot_residual_distribution(y_test, results['predictions']))
        st.subheader('残差诊断')
        st.table(diagnosis)
        st.subheader('未来预测')
        st.dataframe(future_predictions)
        csv = future_predictions.to_csv(index=False).encode('utf-8')
        st.download_button('下载预测 CSV', csv, 'future_predictions.csv', 'text/csv')
else:
    st.info('上传 CSV 开始预测。 示例: Dataset_train.csv')