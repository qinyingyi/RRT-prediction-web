import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
from joblib import load
import matplotlib
from sklearn.preprocessing import StandardScaler
import os


# 设置中文字体支持和负号显示
# 提供更多的字体选项，增加找到可用字体的概率
matplotlib.rcParams.update({
    "font.family": [
        "SimHei", "WenQuanYi Micro Hei", "Heiti TC",  # 常用中文字体
        "Microsoft YaHei", "MS Gothic", "Arial Unicode MS",  # 更多字体选项
        "sans-serif"  # 最后回退到sans-serif
    ],
    "axes.unicode_minus": False,  # 禁用Unicode负号，使用ASCII负号
    "font.sans-serif": ["SimHei"] + matplotlib.rcParams["font.sans-serif"]  # 确保SimHei在前
})

# 双重确保负号设置
plt.rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="RRT预测计算器",
    page_icon=":hospital:",
    layout="wide"
)

# 标题和介绍
st.title("RRT预测计算器")
st.markdown("""
本应用使用CatBoost模型预测病人发生RRT的风险，并通过SHAP值解释模型决策。
输入病人特征后，点击"预测"按钮获取结果。
""")

# 加载模型
@st.cache_resource
def load_model():
    try:
        model = load('grid_cat.pkl')
        return model
    except FileNotFoundError:
        st.error("未找到模型文件！请确保grid_cat.pkl在正确的路径下。")
        return None

def load_rf_model():
    try:
        model = load('grid_rfc.pkl')
        return model
    except FileNotFoundError:
        st.error("未找到模型文件！请确保grid_rfc.pkl在正确的路径下。")
        return None        

def load_gbm_model():
    try:
        model = load('grid_gbm.pkl')
        return model
    except FileNotFoundError:
        st.error("未找到模型文件！请确保grid_gbm.pkl在正确的路径下。")
        return None          

def load_shap_explainer():
    try:
        explainer = load( 'explainer_test.pkl')
        return explainer
    except FileNotFoundError:
        st.error("未找到模型文件！请确保explainer_test.pkl在正确的路径下。")
        return None

def load_scaler():
    try:
        scaler = load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("未找到模型文件！请确保scaler.pkl在正确的路径下。")
        return None


# 特征名称（严格按照训练时的顺序和大小写）
FEATURE_NAMES = ['egfr', 'anion_gap', 'platelet', 'bun','glucocorticoid', 'Antifungal']

# 特征中文显示名称（用于界面）
FEATURE_DISPLAY_NAMES = {
    'egfr': 'EGFR',
    'anion_gap': 'Anion gap',
    'platelet': 'Platelet(10^9)',
    'bun': 'BUN(mmol/L)',
    'glucocorticoid': 'Glucocorticoid',
    'Antifungal': 'Antifungal'
}

# 初始化会话状态的函数
def init_session_state():
    """初始化所有需要的会话状态变量，使用浮点数类型"""
    if 'egfr_value' not in st.session_state:
        st.session_state.egfr_value = 90.0
    if 'anion_gap_value' not in st.session_state:
        st.session_state.anion_gap_value = 14.0
    if 'platelet_value' not in st.session_state:
        st.session_state.platelet_value = 200.0
    if 'bun_value' not in st.session_state:
        st.session_state.bun_value = 20.0

# 连续变量的更新函数
def update_value_from_text(var_name, min_val, max_val):
    """从文本框更新变量值到会话状态"""
    try:
        value = float(st.session_state[f"{var_name}_text"])
        if min_val <= value <= max_val:
            st.session_state[f"{var_name}_value"] = value
    except ValueError:
        pass

# 创建输入表单
def create_input_form(scaler):
    """创建包含双向绑定输入控件的表单"""
    # 初始化会话状态
    init_session_state()
    
    # 创建输入字段（按训练时的顺序排列）
    input_data = {}
    
    # 不在表单内的双向绑定控件 - 先显示
    st.subheader("病人特征输入")
    
    # EGFR - 双向绑定的滑杆和文本输入框
    col1_slider, col1_text = st.columns([3, 1])
    with col1_slider:
        st.session_state.egfr_value = st.slider(
            FEATURE_DISPLAY_NAMES['egfr'], 
            min_value=0.0, 
            max_value=300.0, 
            value=st.session_state.egfr_value,
            key='egfr_slider'
        )
    with col1_text:
        # 移除on_change回调，使用另一种方式同步
        st.text_input(
            "", 
            value=str(st.session_state.egfr_value),
            key='egfr_text'
        )
    
    # Anion gap - 双向绑定的滑杆和文本输入框
    col2_slider, col2_text = st.columns([3, 1])
    with col2_slider:
        st.session_state.anion_gap_value = st.slider(
            FEATURE_DISPLAY_NAMES['anion_gap'], 
            min_value=0.0, 
            max_value=200.0, 
            value=st.session_state.anion_gap_value,
            key='anion_gap_slider'
        )
    with col2_text:
        st.text_input(
            "", 
            value=str(st.session_state.anion_gap_value),
            key='anion_gap_text'
        )
    
    # Platelet - 双向绑定的滑杆和文本输入框
    col3_slider, col3_text = st.columns([3, 1])
    with col3_slider:
        st.session_state.platelet_value = st.slider(
            FEATURE_DISPLAY_NAMES['platelet'], 
            min_value=0.0, 
            max_value=1000.0, 
            value=st.session_state.platelet_value,
            key='platelet_slider'
        )
    with col3_text:
        st.text_input(
            "", 
            value=str(st.session_state.platelet_value),
            key='platelet_text'
        )
    
    # BUN - 双向绑定的滑杆和文本输入框
    col4_slider, col4_text = st.columns([3, 1])
    with col4_slider:
        st.session_state.bun_value = st.slider(
            FEATURE_DISPLAY_NAMES['bun'], 
            min_value=0.0, 
            max_value=300.0, 
            value=st.session_state.bun_value,
            key='bun_slider'
        )
    with col4_text:
        st.text_input(
            "", 
            value=str(st.session_state.bun_value),
            key='bun_text'
        )
    
    # 表单内的控件
    form = st.form("patient_input_form")
    
    with form:
        # Glucocorticoid
        input_data['glucocorticoid'] = st.selectbox(
            FEATURE_DISPLAY_NAMES['glucocorticoid'], 
            [0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        
        # Antifungal
        input_data['Antifungal'] = st.selectbox(
            FEATURE_DISPLAY_NAMES['Antifungal'], 
            [0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        
        # 提交按钮
        submitted = st.form_submit_button("预测")
    
    # 表单外处理提交逻辑
    if submitted:
        # 从文本框更新值到会话状态
        update_value_from_text('egfr', 0, 300)
        update_value_from_text('anion_gap', 0, 200)
        update_value_from_text('platelet', 0, 1000)
        update_value_from_text('bun', 0, 300)
        
        # 将session state中的值更新到input_data中
        input_data['egfr'] = st.session_state.egfr_value
        input_data['anion_gap'] = st.session_state.anion_gap_value
        input_data['platelet'] = st.session_state.platelet_value
        input_data['bun'] = st.session_state.bun_value
        
        # 创建DataFrame时严格按照FEATURE_NAMES的顺序排列列
        patient_orign = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        # patient_orign = patient_orign[['anion_gap', 'bun', 'egfr', 'platelet', 'glucocorticoid', 'Antifungal']]
        continuous_select_features = ['anion_gap', 'bun', 'egfr', 'platelet']
        patient_orign_cont = patient_orign[continuous_select_features]
        patient_orign_non_cont = patient_orign.drop(columns=continuous_select_features)
        patient_orign_scaled_cont = scaler.transform(patient_orign_cont)
        patient_df = pd.DataFrame(np.hstack((patient_orign_scaled_cont, patient_orign_non_cont)), 
                               columns=patient_orign_cont.columns.tolist() + patient_orign_non_cont.columns.tolist())
        return patient_df, patient_orign
    
    # 表单未提交时返回None
    return None

# 计算SHAP值并可视化（删除force plot，保留其他两种图）
def explain_prediction(explainer, patient_data,patient_orign):
    
    # 计算SHAP值
    shap_values = explainer.shap_values(patient_data)
    
    # 处理不同格式的shap_values
    # 如果是列表，且长度大于1，使用第二个元素（对应二分类的正类）
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_class = shap_values[1]
    else:
        # 否则直接使用shap_values（可能是单输出模型或不同版本的SHAP）
        shap_values_class = shap_values
    
    # 创建特征重要性条形图（保留）
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 在绘图前再次确保负号设置（避免被SHAP覆盖）
    plt.rcParams["axes.unicode_minus"] = False
    
    shap.summary_plot(
        shap_values, 
        patient_data,
        feature_names=patient_data.columns,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    
    # 创建特征影响方向图（保留）
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # 在绘图前再次确保负号设置（避免被SHAP覆盖）
    plt.rcParams["axes.unicode_minus"] = False
    
    # 确保只传递单个样本的SHAP值
    # 如果是矩阵格式，提取第一个样本
    if len(shap_values_class.shape) > 1:
        shap_values_single = shap_values_class[0]
        patient_data_single = patient_orign.iloc[0]
    else:
        shap_values_single = shap_values_class
        patient_data_single = patient_orign
    
    # 处理基准值（expected_value）的不同格式
    if isinstance(explainer.expected_value, (list, tuple)):
        base_value = explainer.expected_value[1]  # 对于二分类模型，使用正类的基准值
    else:
        base_value = explainer.expected_value
    
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single,  # 单个样本的正类SHAP值
            base_values=base_value,  # 正类的基准值
            data=patient_data_single.values,  # 单个样本的特征值
            feature_names=patient_data.columns,  # 特征名称        
        ),
        max_display=10,  # 显示的特征数量
        show=False  # 不自动显示，以便我们可以自定义
    )
    plt.tight_layout()
    
    return fig2, fig3, shap_values_single


# 主函数
def main():
    # 加载模型
    model = load_model()
    if model is None:
        return

    rf_model = load_rf_model()
    if rf_model is None:
        return
    gbm_model = load_gbm_model()
    if gbm_model is None:
        return

    # 加载SHAP解释器
    explainer = load_shap_explainer()
    if explainer is None:
        return
    # 加载缩放器
    scaler = load_scaler()
    if scaler is None:
        return        
    
    # 创建输入表单
    form_result = create_input_form(scaler)
    
    if form_result is not None:
        patient_data, patient_orign = form_result
        # 显示输入数据（使用中文名称）
        st.subheader("输入的病人数据")
        display_df = patient_orign.copy()
        display_df.columns = [FEATURE_DISPLAY_NAMES.get(col, col) for col in display_df.columns]
        st.dataframe(display_df)
        # st.dataframe(patient_data)
        
        # 预测
        probability = model.predict_proba(patient_data)[:, 1][0]
        
        # 显示预测结果
        st.subheader("Catboost模型预测结果（cutoff=10%）")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "RRT风险概率", 
                f"{probability:.4%}",
                delta="高风险" if probability > 0.1 else "低风险",
                delta_color="inverse" if probability > 0.1 else "normal"
            )
        with col2:
            st.write(f"预测类别: {'RRT' if probability > 0.1 else 'No RRT'}")
            

        st.subheader("Other models")
        # 计算随机森林模型预测结果
        rf_probability = rf_model.predict_proba(patient_data)[:, 1][0]
        
        # 计算GBM模型预测结果
        gbm_probability = gbm_model.predict_proba(patient_data)[:, 1][0]
 
        
        #将rf_model和gbm_model的预测结果列成表格展示，包含模型名称、预测概率、预测类别，用英文
        models_df = pd.DataFrame({
            'Model': ['Random Forest', 'GBM'],
            'Probability': [rf_probability, gbm_probability],
            'Prediction': ['RRT' if x > 0.1 else 'No RRT' for x in [rf_probability, gbm_probability]]
        })
        st.dataframe(models_df)
        
        
        # 为SHAP值计算准备训练数据摘要
        st.subheader("模型决策解释")
        with st.spinner("正在计算SHAP值..."):
            # 由于没有实际训练数据，我们用一个模拟数据代替
            
            # 计算并显示SHAP解释（删除force plot调用）
            bar_plot, direction_plot, shap_values = explain_prediction(explainer, patient_data,patient_orign)
            
            # 显示特征重要性条形图 - 展示各特征的相对重要性（保留）
            st.subheader("特征重要性")
            st.pyplot(bar_plot)
            
            # 显示特征影响方向图 - 展示特征值与SHAP值的关系（保留）
            st.subheader("Waterfall Plot")
            st.pyplot(direction_plot)
            
            # 显示SHAP值表格
            st.subheader("特征贡献值 (SHAP值)")
            shap_df = pd.DataFrame({
                'Names': [FEATURE_DISPLAY_NAMES.get(f, f) for f in patient_data.columns],
                'Feature': patient_data.columns,  # 显示原始英文名称用于调试
                'SHAP Value': shap_values,
                'Direction': ['Increase risk' if x > 0 else 'Decrease risk' for x in shap_values],
                'Impact': ['High' if abs(x) > np.mean(np.abs(shap_values)) else 'Low' for x in shap_values]
            })
            # 按SHAP值绝对值排序，显示绝对值较大的特征
            shap_df = shap_df.sort_values('SHAP Value', ascending=False, key=abs)
            
            st.dataframe(shap_df)

if __name__ == "__main__":
    main()


