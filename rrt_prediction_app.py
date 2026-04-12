import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import shap
from joblib import load
import matplotlib
import os
import base64
from io import BytesIO

# ============================================
# JAMA Color Scheme Configuration
# ============================================
JAMA_COLORS = {
    'primary': '#003366',      # JAMA Navy Blue
    'secondary': '#0072B2',    # JAMA Blue
    'accent1': '#D55E00',      # JAMA Orange-Red
    'accent2': '#009E73',      # JAMA Green
    'accent3': '#CC79A7',      # JAMA Pink
    'accent4': '#F0E442',      # JAMA Yellow
    'accent5': '#56B4E9',      # Light Blue
    'accent6': '#E69F00',      # Orange
    'accent7': '#999999',      # Gray
    'accent8': '#000000',      # Black
    'background': '#FAFAFA',   # Light Gray Background
    'text': '#1A1A1A',         # Dark Text
    'light_text': '#666666',   # Secondary Text
    'border': '#E0E0E0',       # Border Color
    'high_risk': '#D55E00',    # High Risk - Red
    'low_risk': '#009E73',     # Low Risk - Green
}

# Model colors for consistent visualization
MODEL_COLORS = {
    'CatBoost': '#003366',
    'MLP': '#0072B2',
    'GBM': '#D55E00',
    'XGBoost': '#009E73',
    'ADA': '#CC79A7',
    'LR': '#F0E442',
    'RF': '#56B4E9',
    'Bagging': '#E69F00',
    'NB': '#999999',
}

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="RRT Prediction Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# Custom CSS for JAMA Styling
# ============================================
def load_css():
    st.markdown(f"""
    <style>
        /* Global Styles */
        .stApp {{
            background-color: {JAMA_COLORS['background']};
        }}
        
        /* Header Styles */
        h1 {{
            color: {JAMA_COLORS['primary']} !important;
            font-family: 'Georgia', 'Times New Roman', serif !important;
            font-weight: 700 !important;
            border-bottom: 3px solid {JAMA_COLORS['primary']};
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: {JAMA_COLORS['primary']} !important;
            font-family: 'Georgia', 'Times New Roman', serif !important;
            font-weight: 600 !important;
            margin-top: 20px !important;
        }}
        
        h3 {{
            color: {JAMA_COLORS['secondary']} !important;
            font-family: 'Arial', sans-serif !important;
            font-weight: 600 !important;
        }}
        
        /* Tab Styles */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            border: 2px solid {JAMA_COLORS['border']};
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
            color: {JAMA_COLORS['light_text']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {JAMA_COLORS['primary']} !important;
            color: white !important;
            border-color: {JAMA_COLORS['primary']} !important;
        }}
        
        /* Button Styles */
        .stButton > button {{
            background-color: {JAMA_COLORS['primary']};
            color: white;
            font-weight: 600;
            padding: 12px 32px;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 51, 102, 0.2);
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {JAMA_COLORS['secondary']};
            box-shadow: 0 6px 12px rgba(0, 51, 102, 0.3);
            transform: translateY(-2px);
        }}
        
        /* Metric Cards */
        div[data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: {JAMA_COLORS['primary']};
        }}
        
        div[data-testid="stMetricDelta"] {{
            font-size: 1rem !important;
        }}
        
        /* Info Box */
        .info-box {{
            background-color: white;
            border-left: 4px solid {JAMA_COLORS['primary']};
            padding: 20px;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        /* Result Cards */
        .result-card {{
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin: 16px 0;
            border: 1px solid {JAMA_COLORS['border']};
        }}
        
        .high-risk {{
            border-left: 5px solid {JAMA_COLORS['high_risk']};
        }}
        
        .low-risk {{
            border-left: 5px solid {JAMA_COLORS['low_risk']};
        }}
        
        /* Table Styles */
        .dataframe {{
            font-size: 14px !important;
        }}
        
        .dataframe th {{
            background-color: {JAMA_COLORS['primary']} !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: center !important;
        }}
        
        .dataframe td {{
            text-align: center !important;
        }}
        
        /* Slider Styles */
        .stSlider > div > div > div[data-testid="stTickBar"] {{
            background-color: rgba(0, 51, 102, 0.3) !important;
        }}
        
        .stSlider > div > div > div > div {{
            background-color: {JAMA_COLORS['primary']} !important;
        }}
        
        .stSlider > div > div > div > div > div {{
            background-color: {JAMA_COLORS['primary']} !important;
        }}
        
        /* Slider track and progress */
        div[data-testid="stSlider"] > div[data-baseweb="slider"] > div:first-child {{
            background-color: rgba(0, 51, 102, 0.15) !important;
        }}
        
        div[data-testid="stSlider"] > div[data-baseweb="slider"] > div:first-child > div {{
            background-color: {JAMA_COLORS['primary']} !important;
            opacity: 1 !important;
        }}
        
        /* Input Styles */
        .stTextInput > div > div > input {{
            border-radius: 6px;
            border: 2px solid {JAMA_COLORS['border']};
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {JAMA_COLORS['primary']};
        }}
        
        /* Selectbox Styles */
        .stSelectbox > div > div > div {{
            border-radius: 6px;
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {JAMA_COLORS['primary']};
        }}
        
        /* Alert Boxes */
        .stAlert {{
            border-radius: 8px;
        }}
        
        /* Separator */
        hr {{
            border-color: {JAMA_COLORS['border']};
            margin: 30px 0;
        }}
        
        /* Arrow indicators */
        .arrow-up {{
            color: {JAMA_COLORS['high_risk']};
            font-size: 1.2rem;
        }}
        
        .arrow-down {{
            color: {JAMA_COLORS['low_risk']};
            font-size: 1.2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================
# Feature Configuration
# ============================================
FEATURE_NAMES = ['egfr', 'anion_gap', 'platelet', 'bun', 'glucocorticoid', 'Antifungal']

FEATURE_DISPLAY_NAMES = {
    'egfr': 'eGFR (mL/min/1.73m²)',
    'anion_gap': 'Anion Gap (mEq/L)',
    'platelet': 'Platelet (10⁹/L)',
    'bun': 'BUN (mg/dL)',
    'glucocorticoid': 'Glucocorticoid Use',
    'Antifungal': 'Antifungal Use'
}

FEATURE_RANGES = {
    'egfr': {'min': 0.0, 'max': 300.0, 'default': 90.0, 'step': 1.0},
    'anion_gap': {'min': 0.0, 'max': 200.0, 'default': 14.0, 'step': 0.5},
    'platelet': {'min': 0.0, 'max': 1000.0, 'default': 200.0, 'step': 1.0},
    'bun': {'min': 0.0, 'max': 300.0, 'default': 20.0, 'step': 1.0},
}

# ============================================
# Model Loading Functions
# ============================================
@st.cache_resource
def load_all_models():
    """Load all prediction models"""
    models = {}
    model_files = {
        'CatBoost': 'grid_cat.pkl',
        'MLP': 'grid_mlp.pkl',
        'GBM': 'grid_gbm.pkl',
        'XGBoost': 'grid_xgb.pkl',
        'ADA': 'grid_ada.pkl',
        'LR': 'grid_lr.pkl',
        'RF': 'grid_rfc.pkl',
        'Bagging': 'grid_bagging.pkl',
        'NB': 'grid_nb.pkl',
    }
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    loaded_count = 0
    for name, filename in model_files.items():
        try:
            file_path = os.path.join(current_dir, filename)
            if os.path.exists(file_path):
                model = load(file_path)
                # Handle GridSearchCV objects - get the best estimator
                if hasattr(model, 'best_estimator_'):
                    models[name] = model.best_estimator_
                else:
                    models[name] = model
                loaded_count += 1
            else:
                st.warning(f"Model file not found: {filename}")
                models[name] = None
        except Exception as e:
            st.warning(f"Failed to load {name} model: {e}")
            models[name] = None
    
    # Show success message
    if loaded_count > 0:
        st.success(f"✅ Successfully loaded {loaded_count} models")
    
    return models

@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'explainer_test.pkl')
        return load(file_path)
    except Exception as e:
        st.error(f"Failed to load SHAP explainer: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load data scaler"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'scaler.pkl')
        return load(file_path)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None

# ============================================
# Session State Initialization
# ============================================
def init_session_state():
    """Initialize session state variables"""
    for feature in FEATURE_NAMES[:4]:  # Continuous features
        if f'{feature}_value' not in st.session_state:
            st.session_state[f'{feature}_value'] = FEATURE_RANGES[feature]['default']
    
    # Binary features
    if 'glucocorticoid_value' not in st.session_state:
        st.session_state['glucocorticoid_value'] = 0
    if 'Antifungal_value' not in st.session_state:
        st.session_state['Antifungal_value'] = 0
    
    # Prediction results
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False
    if 'patient_data' not in st.session_state:
        st.session_state['patient_data'] = None
    if 'patient_original' not in st.session_state:
        st.session_state['patient_original'] = None
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None
    if 'shap_values' not in st.session_state:
        st.session_state['shap_values'] = None

# ============================================
# Input Synchronization Functions
# ============================================
def sync_slider_to_text(feature):
    """Sync slider value to text input"""
    st.session_state[f'{feature}_text'] = str(st.session_state[f'{feature}_slider'])
    st.session_state[f'{feature}_value'] = st.session_state[f'{feature}_slider']

def sync_text_to_slider(feature):
    """Sync text input value to slider"""
    try:
        value = float(st.session_state[f'{feature}_text'])
        min_val = FEATURE_RANGES[feature]['min']
        max_val = FEATURE_RANGES[feature]['max']
        if min_val <= value <= max_val:
            st.session_state[f'{feature}_value'] = value
            st.session_state[f'{feature}_slider'] = value
    except ValueError:
        pass

# ============================================
# Data Collection Functions
# ============================================
def collect_patient_data():
    """Collect patient data from inputs"""
    data = {}
    for feature in FEATURE_NAMES[:4]:  # Continuous features
        data[feature] = st.session_state[f'{feature}_value']
    
    # Binary features
    data['glucocorticoid'] = st.session_state['glucocorticoid_value']
    data['Antifungal'] = st.session_state['Antifungal_value']
    
    return pd.DataFrame([data])

def preprocess_data(patient_original, scaler):
    """Preprocess patient data for model input"""
    # Expected order: ['anion_gap', 'bun', 'egfr', 'platelet', 'Antifungal', 'glucocorticoid']
    continuous_features = ['anion_gap', 'bun', 'egfr', 'platelet']
    patient_cont = patient_original[continuous_features]
    
    # Get binary features in the correct order
    binary_features = ['Antifungal', 'glucocorticoid']
    patient_binary = patient_original[binary_features]
    
    # Scale continuous features
    patient_scaled_cont = scaler.transform(patient_cont)
    
    # Combine in the correct order
    patient_processed = pd.DataFrame(
        np.hstack((patient_scaled_cont, patient_binary)),
        columns=continuous_features + binary_features
    )
    
    return patient_processed

# ============================================
# Prediction Functions
# ============================================
def make_predictions(patient_data, models):
    """Make predictions with all models"""
    predictions = {}
    for name, model in models.items():
        if model is not None:
            try:
                prob = model.predict_proba(patient_data)[:, 1][0]
                predictions[name] = {
                    'probability': prob,
                    'prediction': 'RRT' if prob > 0.1 else 'No RRT',
                    'high_risk': prob > 0.1
                }
            except Exception as e:
                predictions[name] = {'probability': None, 'prediction': 'Error', 'high_risk': False}
    return predictions

# ============================================
# Visualization Functions
# ============================================
def create_probability_gauge(probability, title="Risk Probability"):
    """Create a gauge-style probability visualization"""
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection='polar'))
    
    # Gauge background
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 100))
    
    for i in range(len(theta)-1):
        ax.fill_between([theta[i], theta[i+1]], 0, 1, color=colors[i], alpha=0.3)
    
    # Indicator
    angle = probability * np.pi
    ax.annotate('', xy=(angle, 0.9), xytext=(angle, 0.3),
                arrowprops=dict(arrowstyle='->', color=JAMA_COLORS['primary'], lw=4))
    
    # Center text
    ax.text(np.pi/2, 0.15, f'{probability:.1%}', 
            ha='center', va='center', fontsize=36, fontweight='bold',
            color=JAMA_COLORS['high_risk'] if probability > 0.1 else JAMA_COLORS['low_risk'])
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color=JAMA_COLORS['primary'])
    
    plt.tight_layout()
    return fig

def create_shap_bar_plot(shap_values, feature_names, patient_data):
    """Create SHAP feature importance bar plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values)
    }).sort_values('importance', ascending=True)
    
    # Colors based on direction
    colors = [JAMA_COLORS['high_risk'] if shap_values[feature_names.index(f)] > 0 
              else JAMA_COLORS['low_risk'] for f in shap_df['feature']]
    
    bars = ax.barh(shap_df['feature'], shap_df['importance'], color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Styling
    ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Feature Importance', fontsize=16, fontweight='bold', color=JAMA_COLORS['primary'], pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(JAMA_COLORS['border'])
    ax.spines['bottom'].set_color(JAMA_COLORS['border'])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, shap_df['importance'])):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10, color=JAMA_COLORS['text'])
    
    plt.tight_layout()
    return fig

def create_shap_waterfall(shap_values, base_value, feature_names, patient_values, explainer):
    """Create SHAP waterfall plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle base value format
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=patient_values,
        feature_names=feature_names
    )
    
    # Use SHAP's waterfall plot
    shap.plots.waterfall(explanation, max_display=10, show=False)
    
    plt.title('SHAP Waterfall Plot - Feature Contributions', 
              fontsize=14, fontweight='bold', color=JAMA_COLORS['primary'], pad=20)
    plt.tight_layout()
    return fig

def create_shap_force_plot(shap_values, base_value, feature_names, patient_values):
    """Create SHAP force plot using matplotlib (JAMA style)"""
    import matplotlib.colors as mcolors
    
    # Handle base value format
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    # Convert to numpy arrays
    shap_values = np.array(shap_values)
    patient_values = np.array(patient_values)
    
    # JAMA配色
    pos_color = '#BC3C29'  # JAMA红色
    neg_color = '#0072B5'  # JAMA蓝色
    
    # SHAP默认颜色（用于匹配替换）
    target_pos = mcolors.to_rgb('#FF0051')  # SHAP默认红色
    target_neg = mcolors.to_rgb('#008BFB')  # SHAP默认蓝色
    
    def match_color_jama(c):
        """匹配并替换为JAMA配色"""
        if c is None:
            return None
        try:
            c_rgb = mcolors.to_rgb(c)
            # 计算色差，匹配红色
            if sum((a - b)**2 for a, b in zip(c_rgb, target_pos)) < 0.05:
                return pos_color
            # 计算色差，匹配蓝色
            if sum((a - b)**2 for a, b in zip(c_rgb, target_neg)) < 0.05:
                return neg_color
        except:
            pass
        return None
    
    # Create force plot with matplotlib
    shap_fig = shap.force_plot(
        base_value,
        shap_values,
        patient_values,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    # Get current figure
    fig = shap_fig if shap_fig is not None else plt.gcf()
    fig.set_size_inches(16, 4)
    
    # Get axis
    ax = fig.gca()
    
    # 遍历所有绘图对象进行颜色替换
    for obj in ax.findobj():
        # 替换线条颜色
        if hasattr(obj, 'get_color') and hasattr(obj, 'set_color'):
            new_c = match_color_jama(obj.get_color())
            if new_c:
                obj.set_color(new_c)
        
        # 替换填充颜色
        if hasattr(obj, 'get_facecolor') and hasattr(obj, 'set_facecolor'):
            fc = obj.get_facecolor()
            if isinstance(fc, np.ndarray) and fc.size >= 3:
                fc = fc[0] if fc.ndim == 2 else fc
            new_c = match_color_jama(fc)
            if new_c:
                obj.set_facecolor(new_c)
        
        # 替换边缘颜色
        if hasattr(obj, 'get_edgecolor') and hasattr(obj, 'set_edgecolor'):
            ec = obj.get_edgecolor()
            if isinstance(ec, np.ndarray) and ec.size >= 3:
                ec = ec[0] if ec.ndim == 2 else ec
            new_c = match_color_jama(ec)
            if new_c:
                obj.set_edgecolor(new_c)
    
    # 处理底部特征文本重叠（自适应错层）
    feature_texts = [t for t in ax.texts if '=' in t.get_text()]
    feature_texts.sort(key=lambda t: t.get_position()[0])
    
    lines = ax.get_lines()
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_threshold = x_span * 0.15
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    step = y_span * 0.12
    
    levels = [0.0, -step, -step*2, -step*3, -step*4, -step*5]
    last_x_at_level = {lvl: -float('inf') for lvl in levels}
    min_y_attained = ax.get_ylim()[0]
    
    for text_obj in feature_texts:
        x, y = text_obj.get_position()
        chosen_level = levels[0]
        
        for lvl in levels:
            if x - last_x_at_level[lvl] > x_threshold:
                chosen_level = lvl
                break
        
        last_x_at_level[chosen_level] = x
        new_y = y + chosen_level
        min_y_attained = min(min_y_attained, new_y)
        
        if chosen_level != 0.0:
            text_obj.set_position((x, new_y))
            for line in lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 2 and abs(xdata[0] - x) < 1e-3 and abs(xdata[1] - x) < 1e-3:
                    if abs(ydata[0] - y) < abs(ydata[1] - y):
                        ydata[0] = new_y
                    else:
                        ydata[1] = new_y
                    line.set_ydata(ydata)
    
    # 调整底部边界
    ax.set_ylim(bottom=min_y_attained - step)
    
    plt.tight_layout()
    return fig

# ============================================
# UI Components
# ============================================
def render_header():
    """Render application header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem;">🏥 RRT Prediction Calculator</h1>
        <p style="font-size: 1.1rem; color: #666; margin-top: 10px;">
            A Clinical Decision Support Tool for Renal Replacement Therapy Risk Assessment
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

def render_data_input_tab(scaler):
    """Render the data input tab"""
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 1rem;">
            <strong>Instructions:</strong> This application utilizes the CatBoost machine learning model to predict 
            a patient's risk of requiring Renal Replacement Therapy (RRT). The model interprets its decisions through 
            SHAP (SHapley Additive exPlanations) values to provide clinical insights. Enter the patient's laboratory 
            values below and click "Calculate" to obtain the risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Patient Feature Input")
    
    # Create two columns for layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("##### Continuous Variables")
        
        # EGFR
        with st.container():
            st.markdown(f"**{FEATURE_DISPLAY_NAMES['egfr']}**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "", 
                    min_value=FEATURE_RANGES['egfr']['min'],
                    max_value=FEATURE_RANGES['egfr']['max'],
                    value=st.session_state['egfr_value'],
                    step=FEATURE_RANGES['egfr']['step'],
                    key='egfr_slider',
                    on_change=sync_slider_to_text,
                    args=('egfr',)
                )
                st.markdown(f"<p style='font-size: 0.8rem; color: #333; font-weight: 500; margin-top: -5px;'>Range: 0 - 300</p>", unsafe_allow_html=True)
            with c2:
                st.text_input(
                    "",
                    value=str(st.session_state['egfr_value']),
                    key='egfr_text',
                    on_change=sync_text_to_slider,
                    args=('egfr',)
                )
        
        # Anion Gap
        with st.container():
            st.markdown(f"**{FEATURE_DISPLAY_NAMES['anion_gap']}**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "", 
                    min_value=FEATURE_RANGES['anion_gap']['min'],
                    max_value=FEATURE_RANGES['anion_gap']['max'],
                    value=st.session_state['anion_gap_value'],
                    step=FEATURE_RANGES['anion_gap']['step'],
                    key='anion_gap_slider',
                    on_change=sync_slider_to_text,
                    args=('anion_gap',)
                )
                st.markdown(f"<p style='font-size: 0.8rem; color: #333; font-weight: 500; margin-top: -5px;'>Range: 0 - 200</p>", unsafe_allow_html=True)
            with c2:
                st.text_input(
                    "",
                    value=str(st.session_state['anion_gap_value']),
                    key='anion_gap_text',
                    on_change=sync_text_to_slider,
                    args=('anion_gap',)
                )
        
        # Platelet
        with st.container():
            st.markdown(f"**{FEATURE_DISPLAY_NAMES['platelet']}**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "", 
                    min_value=FEATURE_RANGES['platelet']['min'],
                    max_value=FEATURE_RANGES['platelet']['max'],
                    value=st.session_state['platelet_value'],
                    step=FEATURE_RANGES['platelet']['step'],
                    key='platelet_slider',
                    on_change=sync_slider_to_text,
                    args=('platelet',)
                )
                st.markdown(f"<p style='font-size: 0.8rem; color: #333; font-weight: 500; margin-top: -5px;'>Range: 0 - 1000</p>", unsafe_allow_html=True)
            with c2:
                st.text_input(
                    "",
                    value=str(st.session_state['platelet_value']),
                    key='platelet_text',
                    on_change=sync_text_to_slider,
                    args=('platelet',)
                )
        
        # BUN
        with st.container():
            st.markdown(f"**{FEATURE_DISPLAY_NAMES['bun']}**")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "", 
                    min_value=FEATURE_RANGES['bun']['min'],
                    max_value=FEATURE_RANGES['bun']['max'],
                    value=st.session_state['bun_value'],
                    step=FEATURE_RANGES['bun']['step'],
                    key='bun_slider',
                    on_change=sync_slider_to_text,
                    args=('bun',)
                )
                st.markdown(f"<p style='font-size: 0.8rem; color: #333; font-weight: 500; margin-top: -5px;'>Range: 0 - 300</p>", unsafe_allow_html=True)
            with c2:
                st.text_input(
                    "",
                    value=str(st.session_state['bun_value']),
                    key='bun_text',
                    on_change=sync_text_to_slider,
                    args=('bun',)
                )
    
    with col_right:
        st.markdown("##### Binary Variables")
        
        # Glucocorticoid
        st.session_state['glucocorticoid_value'] = st.selectbox(
            FEATURE_DISPLAY_NAMES['glucocorticoid'],
            options=[0, 1],
            format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
            index=int(st.session_state['glucocorticoid_value'])
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Antifungal
        st.session_state['Antifungal_value'] = st.selectbox(
            FEATURE_DISPLAY_NAMES['Antifungal'],
            options=[0, 1],
            format_func=lambda x: "❌ No" if x == 0 else "✅ Yes",
            index=int(st.session_state['Antifungal_value'])
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Summary of current inputs
        st.markdown("##### Current Input Summary")
        summary_data = {
            'Feature': [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_NAMES],
            'Value': [
                f"{st.session_state['egfr_value']:.1f}",
                f"{st.session_state['anion_gap_value']:.1f}",
                f"{st.session_state['platelet_value']:.1f}",
                f"{st.session_state['bun_value']:.1f}",
                "Yes" if st.session_state['glucocorticoid_value'] else "No",
                "Yes" if st.session_state['Antifungal_value'] else "No",
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # Calculate button
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        if st.button("🧮 Calculate RRT Risk", use_container_width=True):
            with st.spinner("Processing..."):
                # Collect and process data
                patient_original = collect_patient_data()
                patient_processed = preprocess_data(patient_original, scaler)
                
                # Store in session state
                st.session_state['patient_original'] = patient_original
                st.session_state['patient_data'] = patient_processed
                st.session_state['prediction_made'] = True
                
                # Make predictions
                models = load_all_models()
                predictions = make_predictions(patient_processed, models)
                st.session_state['predictions'] = predictions
                
                # Calculate SHAP values
                explainer = load_shap_explainer()
                if explainer is not None:
                    try:
                        shap_vals = explainer.shap_values(patient_processed)
                        if isinstance(shap_vals, list) and len(shap_vals) > 1:
                            st.session_state['shap_values'] = shap_vals[1][0]
                        else:
                            st.session_state['shap_values'] = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                    except Exception as e:
                        st.error(f"SHAP calculation failed: {e}")
                        st.session_state['shap_values'] = None
                
                st.success("✅ Calculation completed! Please navigate to the 'Model Predictions' or 'Model Interpretability' tabs to view results.")

def render_predictions_tab():
    """Render the predictions tab"""
    if not st.session_state['prediction_made']:
        st.warning("⚠️ Please enter patient data and click 'Calculate' in the Data Input tab first.")
        return
    
    patient_original = st.session_state['patient_original']
    predictions = st.session_state['predictions']
    
    # Display patient data
    st.subheader("📊 Patient Data")
    display_df = patient_original.copy()
    display_df.columns = [FEATURE_DISPLAY_NAMES.get(col, col) for col in display_df.columns]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # CatBoost Prediction (Primary Model)
    st.subheader("🎯 CatBoost Model Prediction (Primary)")
    catboost_pred = predictions.get('CatBoost', {})
    
    if catboost_pred.get('probability') is not None:
        prob = catboost_pred['probability']
        is_high_risk = prob > 0.1
        
        # Main prediction display
        cols = st.columns([1, 2, 1])
        with cols[1]:
            risk_color = JAMA_COLORS['high_risk'] if is_high_risk else JAMA_COLORS['low_risk']
            risk_class = "high-risk" if is_high_risk else "low-risk"
            risk_icon = "🔴" if is_high_risk else "🟢"
            risk_text = "HIGH RISK" if is_high_risk else "LOW RISK"
            
            st.markdown(f"""
            <div class="result-card {risk_class}" style="text-align: center;">
                <h3 style="margin: 0; color: {risk_color};">{risk_icon} {risk_text}</h3>
                <p style="font-size: 3rem; font-weight: bold; color: {risk_color}; margin: 10px 0;">
                    {prob:.1%}
                </p>
                <p style="color: #666; margin: 0;">RRT Risk Probability (Cutoff: 10%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar visualization
        st.markdown("##### Risk Level Visualization")
        st.progress(min(prob, 1.0), text=f"Risk Score: {prob:.1%}")
        
        # Risk interpretation
        if is_high_risk:
            st.error("""
            ⚠️ **High Risk Alert**: The patient is predicted to be at HIGH risk for requiring Renal Replacement Therapy (RRT).
            Consider immediate nephrology consultation and close monitoring.
            """)
        else:
            st.success("""
            ✅ **Low Risk**: The patient is predicted to be at LOW risk for requiring RRT.
            Continue standard care and monitoring of renal function.
            """)
    
    st.markdown("---")
    
    # Other Models Comparison
    st.subheader("📈 Comparison with Other Models")
    
    model_order = ['MLP', 'GBM', 'XGBoost', 'ADA', 'LR', 'RF', 'Bagging', 'NB']
    
    comparison_data = []
    failed_models = []
    
    for model_name in model_order:
        pred = predictions.get(model_name, {})
        if pred.get('probability') is not None:
            is_high_risk = pred['high_risk']
            arrow = "🔴 ↑" if is_high_risk else "🟢 ↓"
            comparison_data.append({
                'Model': model_name,
                'Probability': f"{pred['probability']:.2%}",
                'Prediction': f"{arrow} {pred['prediction']}",
                'Risk Level': 'High' if is_high_risk else 'Low',
                '_sort_prob': pred['probability']
            })
        else:
            failed_models.append(model_name)
    
    # Debug info
    if failed_models:
        st.info(f"⚠️ Models not loaded or prediction failed: {', '.join(failed_models)}")
    
    if comparison_data:
        # Sort by probability (keep _sort_prob for chart)
        comparison_data_sorted = sorted(comparison_data, key=lambda x: x['_sort_prob'], reverse=True)
        
        # Create DataFrame for display (without _sort_prob column)
        display_data = [{k: v for k, v in d.items() if k != '_sort_prob'} for d in comparison_data_sorted]
        comparison_df = pd.DataFrame(display_data)
        
        # Styled dataframe
        def highlight_risk(val):
            if 'RRT' in str(val):
                return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
            return ''
        
        styled_df = comparison_df.style.applymap(highlight_risk, subset=['Prediction'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Visualization bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = [d['Model'] for d in comparison_data_sorted]
        probs = [d['_sort_prob'] for d in comparison_data_sorted]
        colors = [JAMA_COLORS['high_risk'] if p > 0.1 else JAMA_COLORS['low_risk'] for p in probs]
        
        bars = ax.barh(models, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add cutoff line
        ax.axvline(x=0.1, color=JAMA_COLORS['primary'], linestyle='--', linewidth=2, label='Cutoff (10%)')
        
        # Styling
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison - RRT Risk Predictions', fontsize=14, fontweight='bold', color=JAMA_COLORS['primary'])
        ax.set_xlim(0, max(probs) * 1.2 if probs else 1.0)
        ax.legend(loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add probability labels
        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1%}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No other model predictions available. Please ensure all model files are present.")

def render_interpretability_tab():
    """Render the model interpretability tab"""
    if not st.session_state['prediction_made']:
        st.warning("⚠️ Please enter patient data and click 'Calculate' in the Data Input tab first.")
        return
    
    if st.session_state['shap_values'] is None:
        st.error("❌ SHAP values could not be calculated. Please check the model files.")
        return
    
    patient_data = st.session_state['patient_data']
    patient_original = st.session_state['patient_original']
    shap_values = st.session_state['shap_values']
    explainer = load_shap_explainer()
    
    feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in patient_data.columns]
    
    st.subheader("🔍 Model Interpretability with SHAP")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>About SHAP Values:</strong> SHAP (SHapley Additive exPlanations) values explain how each feature 
            contributes to pushing the model's prediction away from the baseline (average prediction). 
            Positive values (red) increase the predicted risk, while negative values (blue) decrease it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance Bar Plot
    st.markdown("### 📊 Feature Importance (Bar Plot)")
    with st.spinner("Generating feature importance plot..."):
        fig_bar = create_shap_bar_plot(shap_values, feature_names, patient_data)
        st.pyplot(fig_bar)
    
    st.markdown("---")
    
    # Waterfall Plot
    st.markdown("### 🌊 Waterfall Plot")
    st.markdown("""
    <p style="color: #666; font-size: 0.9rem;">
    The waterfall plot shows how each feature contributes to shifting the prediction from the base value 
    (average model output) to the final predicted probability. Red arrows push the prediction higher, 
    blue arrows push it lower.
    </p>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generating waterfall plot..."):
        try:
            base_value = explainer.expected_value
            patient_values = patient_original.iloc[0].values
            fig_waterfall = create_shap_waterfall(shap_values, base_value, feature_names, patient_values, explainer)
            st.pyplot(fig_waterfall)
        except Exception as e:
            st.error(f"Could not generate waterfall plot: {e}")
    
    st.markdown("---")
    
    # Force Plot
    st.markdown("### 💥 Force Plot")
    st.markdown("""
    <p style="color: #666; font-size: 0.9rem;">
    The force plot visualizes the push and pull of each feature on the prediction. 
    Features pushing the prediction higher (increasing risk) are shown in red, 
    while those pushing lower are shown in blue.
    </p>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generating force plot..."):
        try:
            base_value = explainer.expected_value
            patient_values = patient_original.iloc[0].values
            fig_force = create_shap_force_plot(shap_values, base_value, feature_names, patient_values)
            st.pyplot(fig_force)
        except Exception as e:
            st.error(f"Could not generate force plot: {e}")
    
    st.markdown("---")
    
    # SHAP Values Table
    st.markdown("### 📋 Feature Contribution Values (SHAP Values)")
    
    shap_table_data = []
    for i, (feat, val, shap_val) in enumerate(zip(feature_names, patient_original.iloc[0].values, shap_values)):
        shap_table_data.append({
            'Feature': feat,
            'Input Value': f"{val:.2f}" if isinstance(val, (int, float)) else str(val),
            'SHAP Value': f"{shap_val:.4f}",
            '|SHAP Value|': abs(shap_val),
            'Direction': '↑ Increase Risk' if shap_val > 0 else '↓ Decrease Risk',
            'Impact Level': 'High' if abs(shap_val) > np.mean(np.abs(shap_values)) else 'Low'
        })
    
    shap_df = pd.DataFrame(shap_table_data)
    shap_df = shap_df.sort_values('|SHAP Value|', ascending=False).drop('|SHAP Value|', axis=1)
    
    # Apply styling
    def color_direction(val):
        if 'Increase' in str(val):
            return 'color: #D55E00; font-weight: bold'
        elif 'Decrease' in str(val):
            return 'color: #009E73; font-weight: bold'
        return ''
    
    def highlight_impact(val):
        if val == 'High':
            return 'background-color: #fff3cd'
        return ''
    
    styled_shap = shap_df.style.applymap(color_direction, subset=['Direction']).applymap(highlight_impact, subset=['Impact Level'])
    st.dataframe(styled_shap, use_container_width=True, hide_index=True)
    
    # Summary interpretation
    st.markdown("---")
    st.markdown("### 📝 Interpretation Summary")
    
    top_increasing = shap_df[shap_df['Direction'].str.contains('Increase')].head(2)
    top_decreasing = shap_df[shap_df['Direction'].str.contains('Decrease')].head(2)
    
    summary_text = f"""
    **Key Risk Factors (Increasing RRT Risk):**
    {chr(10).join([f"- **{row['Feature']}** ({row['Input Value']}): SHAP = {row['SHAP Value']}" for _, row in top_increasing.iterrows()]) if len(top_increasing) > 0 else '- None identified'}
    
    **Protective Factors (Decreasing RRT Risk):**
    {chr(10).join([f"- **{row['Feature']}** ({row['Input Value']}): SHAP = {row['SHAP Value']}" for _, row in top_decreasing.iterrows()]) if len(top_decreasing) > 0 else '- None identified'}
    """
    
    st.markdown(summary_text)

# ============================================
# Main Application
# ============================================
def main():
    # Initialize
    init_session_state()
    
    # Load models
    models = load_all_models()
    scaler = load_scaler()
    
    if scaler is None:
        st.error("❌ Failed to load required model files. Please ensure all model files are present in the RRT-prediction-web directory.")
        return
    
    # Render header
    render_header()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Data Input", 
        "📊 Model Predictions", 
        "🔍 Model Interpretability"
    ])
    
    with tab1:
        render_data_input_tab(scaler)
    
    with tab2:
        render_predictions_tab()
    
    with tab3:
        render_interpretability_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.85rem; padding: 20px;">
        <p>RRT Prediction Calculator | Powered by CatBoost & SHAP</p>
        <p style="font-size: 0.75rem; color: #bbb;">
            Disclaimer: This tool is for research and educational purposes only. 
            Clinical decisions should not be based solely on this prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
