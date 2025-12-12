"""
Streamlit Web Application for Semi-Supervised Grocery Classification

A beautiful and interactive web application for classifying grocery images
using semi-supervised learning algorithms.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_utils import CLASS_NAMES, IDX_TO_CLASS, get_dataset_stats
from src.feature_extraction import FeatureExtractor
from src.semi_supervised import SemiSupervisedClassifier
from src.recipe_utils import RecipeRecommender


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="🛒 Grocery Classifier",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for Premium Design
# ============================================================================

st.markdown("""
<style>
    /* Main page styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(0, 212, 255, 0.2) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(124, 58, 237, 0.3);
        margin: 1rem 0;
    }
    
    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        text-transform: capitalize;
    }
    
    .confidence-score {
        font-size: 1.2rem;
        color: #a78bfa;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(249, 115, 22, 0.2);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f97316;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e30 0%, #2d2d44 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Algorithm cards */
    .algo-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .algo-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Class chips */
    .class-chip {
        display: inline-block;
        background: rgba(124, 58, 237, 0.2);
        color: #a78bfa;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = None
    
if 'models' not in st.session_state:
    st.session_state.models = {}

if 'training_results' not in st.session_state:
    st.session_state.training_results = None

if 'recipe_recommender' not in st.session_state:
    st.session_state.recipe_recommender = None

if 'grocery_list' not in st.session_state:
    st.session_state.grocery_list = []  # List of dicts: {'name': str, 'image': PIL.Image}


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def load_feature_extractor():
    """Load the feature extractor (cached)."""
    model_path = "models/feature_extractor.pth"
    if os.path.exists(model_path):
        return FeatureExtractor(model_path=model_path)
    return FeatureExtractor()


@st.cache_resource
def load_model(model_path: str):
    """Load a trained model (cached)."""
    return SemiSupervisedClassifier.load(model_path)


def load_training_results():
    """Load training results if available."""
    results_path = "models/training_results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_recipe_recommender():
    """Load the recipe recommender (cached)."""
    data_path = "DS3RECIPES/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    if os.path.exists(data_path):
        return RecipeRecommender(data_path)
    return None


def get_available_models():
    """Get list of available trained models."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    return [f.stem.replace("_model", "") for f in models_dir.glob("*_model.joblib")]


def predict_image(image: Image.Image, model_name: str):
    """Make prediction on an uploaded image."""
    # Load feature extractor
    if st.session_state.feature_extractor is None:
        st.session_state.feature_extractor = load_feature_extractor()
    
    extractor = st.session_state.feature_extractor
    
    # Load model
    if model_name not in st.session_state.models:
        model_path = f"models/{model_name}_model.joblib"
        st.session_state.models[model_name] = load_model(model_path)
    
    model = st.session_state.models[model_name]
    
    # Preprocess image
    from src.data_utils import get_image_transform
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0)
    
    # Extract features
    features = extractor.extract_single(image_tensor)
    features = features.reshape(1, -1)
    
    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return prediction, probabilities


def get_sample_images(class_name: str, n_samples: int = 5):
    """Get sample images from a class."""
    class_dir = Path(f"DS2GROCERIES/train/{class_name}")
    if not class_dir.exists():
        return []
    
    images = list(class_dir.glob("*.jpg"))[:n_samples]
    return [str(img) for img in images]


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Navigation")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Predict", "🍳 Recipe Recommender", "📊 Dataset Explorer", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model selection for prediction
    if page == "🔍 Predict":
        st.markdown("### 🤖 Model Selection")
        available_models = get_available_models()
        
        if available_models:
            selected_model = st.selectbox(
                "Choose Algorithm",
                available_models,
                format_func=lambda x: x.replace("_", " ").title()
            )
        else:
            st.warning("No trained models found. Please train models first.")
            selected_model = None
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">20</div>
        <div class="metric-label">Classes</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">SSL Algorithms</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Made with ❤️ for ML Mini-Project
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Page: Home
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🛒 Grocery Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h2>Welcome to the Semi-Supervised Grocery Image Classifier!</h2>
        <p style="font-size: 1.1rem; color: #94a3b8; line-height: 1.8;">
            This application uses state-of-the-art <strong>semi-supervised learning</strong> algorithms 
            to classify grocery images into 20 different categories. The system is trained using only 
            a small portion of labeled data, demonstrating the power of SSL techniques in scenarios 
            where labeled data is scarce.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>🏷️ Label Propagation</h3>
            <p style="color: #94a3b8;">
                Graph-based method that spreads labels through K-NN similarity structure
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>📊 Label Spreading</h3>
            <p style="color: #94a3b8;">
                Uses normalized graph Laplacian for more robust label propagation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3>🔄 Self-Training</h3>
            <p style="color: #94a3b8;">
                Iteratively adds high-confidence predictions to the training set
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Classes display
    st.markdown("### 🥬 Grocery Categories")
    
    classes_html = ""
    for class_name in CLASS_NAMES:
        emoji = {
            'bacon': '🥓', 'banana': '🍌', 'bread': '🍞', 'broccoli': '🥦',
            'butter': '🧈', 'carrots': '🥕', 'cheese': '🧀', 'chicken': '🍗',
            'cucumber': '🥒', 'eggs': '🥚', 'fish': '🐟', 'lettuce': '🥬',
            'milk': '🥛', 'onions': '🧅', 'peppers': '🫑', 'potatoes': '🥔',
            'sausages': '🌭', 'spinach': '🥬', 'tomato': '🍅', 'yogurt': '🥛'
        }.get(class_name, '🛒')
        classes_html += f'<span class="class-chip">{emoji} {class_name.title()}</span>'
    
    st.markdown(f"""
    <div class="glass-card">
        {classes_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    <div class="info-box">
        <strong>Step 1:</strong> Train the models using the command: <code>python src/train.py</code><br>
        <strong>Step 2:</strong> Navigate to the <strong>Predict</strong> page<br>
        <strong>Step 3:</strong> Upload a grocery image and get predictions!
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Page: Predict
# ============================================================================

elif page == "🔍 Predict":
    st.markdown('<h1 class="main-header">🔍 Image Prediction</h1>', unsafe_allow_html=True)
    
    available_models = get_available_models()
    
    if not available_models:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h2>⚠️ No Trained Models Found</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                Please train the models first by running:
            </p>
            <code style="background: rgba(0,0,0,0.3); padding: 0.5rem 1rem; border-radius: 8px;">
                python src/train.py
            </code>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a grocery image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image of a grocery item to classify"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Prediction Results")
            
            if uploaded_file is not None and 'selected_model' in dir():
                if selected_model:
                    with st.spinner("🔮 Analyzing image..."):
                        try:
                            prediction, probabilities = predict_image(image, selected_model)
                            predicted_class = IDX_TO_CLASS[prediction]
                            confidence = probabilities[prediction] * 100
                            
                            emoji = {
                                'bacon': '🥓', 'banana': '🍌', 'bread': '🍞', 'broccoli': '🥦',
                                'butter': '🧈', 'carrots': '🥕', 'cheese': '🧀', 'chicken': '🍗',
                                'cucumber': '🥒', 'eggs': '🥚', 'fish': '🐟', 'lettuce': '🥬',
                                'milk': '🥛', 'onions': '🧅', 'peppers': '🫑', 'potatoes': '🥔',
                                'sausages': '🌭', 'spinach': '🥬', 'tomato': '🍅', 'yogurt': '🥛'
                            }.get(predicted_class, '🛒')
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <div style="font-size: 4rem;">{emoji}</div>
                                <div class="prediction-label">{predicted_class}</div>
                                <div class="confidence-score">Confidence: {confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Top 5 predictions chart
                            st.markdown("#### 📊 Top Predictions")
                            top_indices = np.argsort(probabilities)[-5:][::-1]
                            top_classes = [IDX_TO_CLASS[i] for i in top_indices]
                            top_probs = [probabilities[i] * 100 for i in top_indices]
                            
                            fig = go.Figure(go.Bar(
                                x=top_probs,
                                y=top_classes,
                                orientation='h',
                                marker=dict(
                                    color=top_probs,
                                    colorscale=[[0, '#7c3aed'], [1, '#00d4ff']]
                                )
                            ))
                            
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                xaxis=dict(title='Confidence (%)', showgrid=False),
                                yaxis=dict(showgrid=False),
                                height=300,
                                margin=dict(l=0, r=0, t=0, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to grocery list
                            # Check if item already exists to avoid duplicates (optional, but good UX)
                            # We allow duplicates if user wants to add multiple items, but maybe unique is better?
                            # Let's just append for now as requested "fill after each image prediction"
                            
                            # Store a thumbnail for the list
                            thumb = image.copy()
                            thumb.thumbnail((100, 100))
                            
                            st.session_state.grocery_list.append({
                                'name': predicted_class,
                                'image': thumb
                            })
                            st.toast(f"Added {predicted_class} to your grocery list!", icon="🛒")
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 3rem;">
                    <p style="color: #94a3b8; font-size: 1.2rem;">
                        👈 Upload an image to get started
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Grocery List Display (Full width below columns)
        st.markdown("---")
        st.markdown("### 📝 My Grocery List")
        
        if st.session_state.grocery_list:
            if st.button("🗑️ Clear List"):
                st.session_state.grocery_list = []
                st.rerun()
                
            cols = st.columns(6)
            for i, item in enumerate(st.session_state.grocery_list):
                col_idx = i % 6
                with cols[col_idx]:
                    st.image(item['image'], caption=item['name'].title(), use_container_width=True)
        else:
            st.info("Your grocery list is empty. Predict items to add them here!")


# ============================================================================
# Page: Dataset Explorer
# ============================================================================

elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="main-header">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    
    # Get dataset stats
    stats = get_dataset_stats("DS2GROCERIES")
    
    if not stats['splits']:
        st.warning("Dataset not found. Please ensure DS2GROCERIES directory exists.")
    else:
        # Overall stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{stats['total_images']:,}</div>
                <div class="metric-label">Total Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{len(CLASS_NAMES)}</div>
                <div class="metric-label">Categories</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{len(stats['splits'])}</div>
                <div class="metric-label">Data Splits</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Split distribution
        st.markdown("### 📈 Data Distribution by Split")
        
        split_data = []
        for split, split_stats in stats['splits'].items():
            split_data.append({
                'Split': split.upper(),
                'Images': split_stats['total']
            })
        
        if split_data:
            fig = px.pie(
                split_data,
                values='Images',
                names='Split',
                color_discrete_sequence=['#7c3aed', '#00d4ff', '#f97316']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Class distribution
        st.markdown("### 📊 Images per Category (Training Set)")
        
        if 'train' in stats['splits']:
            class_data = []
            for class_name, count in stats['splits']['train']['classes'].items():
                class_data.append({
                    'Category': class_name.title(),
                    'Images': count
                })
            
            class_data.sort(key=lambda x: x['Images'], reverse=True)
            
            fig = go.Figure(go.Bar(
                x=[d['Category'] for d in class_data],
                y=[d['Images'] for d in class_data],
                marker=dict(
                    color=[d['Images'] for d in class_data],
                    colorscale=[[0, '#7c3aed'], [0.5, '#00d4ff'], [1, '#f97316']]
                )
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(tickangle=45, showgrid=False),
                yaxis=dict(title='Number of Images', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample images viewer
        st.markdown("### 🖼️ Sample Images")
        
        selected_class = st.selectbox(
            "Select Category",
            CLASS_NAMES,
            format_func=lambda x: x.title()
        )
        
        sample_images = get_sample_images(selected_class, n_samples=5)
        
        if sample_images:
            cols = st.columns(5)
            for i, img_path in enumerate(sample_images):
                with cols[i]:
                    img = Image.open(img_path)
                    st.image(img, caption=f"Sample {i+1}", use_container_width=True)
        else:
            st.info("No sample images found for this category.")


# ============================================================================
# Page: Model Performance
# ============================================================================

elif page == "📈 Model Performance":
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    results = load_training_results()
    
    if results is None:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h2>⚠️ No Training Results Found</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                Please train the models first to see performance metrics.
            </p>
            <code style="background: rgba(0,0,0,0.3); padding: 0.5rem 1rem; border-radius: 8px;">
                python src/train.py
            </code>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Metadata
        metadata = results.get('metadata', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{metadata.get('n_train_samples', 'N/A'):,}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{metadata.get('n_test_samples', 'N/A'):,}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            labeled_ratio = metadata.get('labeled_ratio', 0) * 100
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{labeled_ratio:.0f}%</div>
                <div class="metric-label">Labeled Data</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="metric-value">{metadata.get('feature_dim', 'N/A')}</div>
                <div class="metric-label">Feature Dimension</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm comparison
        st.markdown("### 🏆 Algorithm Comparison")
        
        algo_data = []
        for algo in ['label_propagation', 'label_spreading', 'self_training']:
            if algo in results and 'error' not in results[algo]:
                algo_data.append({
                    'Algorithm': algo.replace('_', ' ').title(),
                    'Train Accuracy': results[algo]['train_accuracy'] * 100,
                    'Test Accuracy': results[algo]['test_accuracy'] * 100
                })
        
        if algo_data:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Train Accuracy',
                x=[d['Algorithm'] for d in algo_data],
                y=[d['Train Accuracy'] for d in algo_data],
                marker_color='#7c3aed'
            ))
            
            fig.add_trace(go.Bar(
                name='Test Accuracy',
                x=[d['Algorithm'] for d in algo_data],
                y=[d['Test Accuracy'] for d in algo_data],
                marker_color='#00d4ff'
            ))
            
            fig.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(
                    title='Accuracy (%)',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    range=[0, 100]
                ),
                xaxis=dict(showgrid=False),
                legend=dict(font=dict(color='white')),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed cards
            st.markdown("### 📋 Detailed Results")
            
            cols = st.columns(len(algo_data))
            for i, data in enumerate(algo_data):
                with cols[i]:
                    algo_name = data['Algorithm']
                    train_acc = data['Train Accuracy']
                    test_acc = data['Test Accuracy']
                    
                    st.markdown(f"""
                    <div class="algo-card">
                        <h4 style="color: #00d4ff;">{algo_name}</h4>
                        <p><strong>Train:</strong> <span style="color: #10b981;">{train_acc:.1f}%</span></p>
                        <p><strong>Test:</strong> <span style="color: #f97316;">{test_acc:.1f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================================
# Page: Recipe Recommender
# ============================================================================

elif page == "🍳 Recipe Recommender":
    st.markdown('<h1 class="main-header">🍳 Recipe Recommender</h1>', unsafe_allow_html=True)
    
    # Load recommender
    if st.session_state.recipe_recommender is None:
        with st.spinner("Loading recipe database..."):
            st.session_state.recipe_recommender = load_recipe_recommender()
            
    if st.session_state.recipe_recommender is None:
        st.error("Recipe dataset not found. Please ensure the DS3RECIPES directory exists.")
    else:
        st.markdown("""
        <div class="glass-card">
            <h3>Find Recipes by Ingredients</h3>
            <p style="color: #94a3b8;">
                Enter the ingredients you have, and we'll suggest recipes you can make!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ingredient input
        
        # Pre-fill with grocery list items if available
        grocery_list_names = [item['name'] for item in st.session_state.grocery_list]
        grocery_list_str = ", ".join(grocery_list_names)
        
        if grocery_list_names:
            st.info(f"✨ Including {len(grocery_list_names)} items from your grocery list: {grocery_list_str}")
        
        ingredients_input = st.text_area(
            "Enter additional ingredients (comma separated)",
            placeholder="e.g., salt, pepper",
            height=100
        )
        
        if st.button("🍳 Find Recipes"):
            all_ingredients = grocery_list_names.copy()
            
            if ingredients_input.strip():
                manual_ingredients = [i.strip() for i in ingredients_input.split(',')]
                all_ingredients.extend(manual_ingredients)
            
            if not all_ingredients:
                st.warning("Please enter some ingredients or add items to your grocery list first.")
            else:
                with st.spinner("Searching for recipes..."):
                    recommendations = st.session_state.recipe_recommender.search_recipes(
                        all_ingredients, 
                        top_k=5
                    )
                
                if not recommendations:
                    st.info("No matching recipes found. Try adding more ingredients!")
                else:
                    st.markdown("### 🥗 Recommended Recipes")
                    
                    for i, rec in enumerate(recommendations):
                        match_pct = rec['Match_Percentage'] * 100
                        missing_count = len(rec['Missing_Ingredients'])
                        
                        # Determine card border color based on match
                        border_color = "#10b981" if match_pct > 75 else "#f97316" if match_pct > 40 else "#64748b"
                        
                        with st.expander(f"{rec['Title']} (Match: {match_pct:.0f}%)"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Try to load image if available
                                img_name = rec['Image_Name']
                                # FIX: Update path to nested folder
                                img_path = Path("DS3RECIPES/Food Images/Food Images") / f"{img_name}.jpg"
                                
                                if img_path.exists():
                                    st.image(str(img_path), use_container_width=True)
                                else:
                                    st.markdown("📷 No image available")
                                    
                                st.markdown(f"""
                                <div style="margin-top: 1rem; text-align: center;">
                                    <div style="font-size: 2rem; font-weight: bold; color: {border_color};">
                                        {match_pct:.0f}%
                                    </div>
                                    <div style="color: #94a3b8; font-size: 0.8rem;">Match Score</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("#### 📝 Instructions")
                                st.write(rec['Instructions'])
                                
                                st.markdown("#### ✅ Matching Ingredients")
                                matching_html = ""
                                for ing in rec['Matching_Ingredients']:
                                    matching_html += f'<span class="class-chip" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border-color: rgba(16, 185, 129, 0.3);">{ing}</span>'
                                st.markdown(matching_html, unsafe_allow_html=True)
                                
                                if missing_count > 0:
                                    st.markdown(f"#### 🛒 Missing Ingredients ({missing_count})")
                                    missing_html = ""
                                    for ing in rec['Missing_Ingredients']:
                                        missing_html += f'<span class="class-chip" style="background: rgba(239, 68, 68, 0.2); color: #ef4444; border-color: rgba(239, 68, 68, 0.3);">{ing}</span>'
                                    st.markdown(missing_html, unsafe_allow_html=True)


# ============================================================================
# Page: About
# ============================================================================

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h2>🎯 Project Overview</h2>
        <p style="color: #94a3b8; line-height: 1.8;">
            This is a machine learning mini-project that demonstrates the power of 
            <strong>semi-supervised learning</strong> for image classification. The system 
            classifies grocery images into 20 categories using only a small fraction of 
            labeled data, making it ideal for scenarios where labeling is expensive or 
            time-consuming.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🔬 Semi-Supervised Learning</h3>
            <p style="color: #94a3b8; line-height: 1.8;">
                Semi-supervised learning bridges the gap between supervised and 
                unsupervised learning. It uses a small amount of labeled data along 
                with a large amount of unlabeled data during training.
            </p>
            <h4 style="color: #00d4ff;">Algorithms Used:</h4>
            <ul style="color: #94a3b8;">
                <li><strong>Label Propagation:</strong> Spreads labels through a similarity graph</li>
                <li><strong>Label Spreading:</strong> Uses normalized graph Laplacian</li>
                <li><strong>Self-Training:</strong> Iteratively adds confident predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🏗️ Technical Architecture</h3>
            <p style="color: #94a3b8; line-height: 1.8;">
                The system uses a two-stage approach:
            </p>
            <ol style="color: #94a3b8;">
                <li><strong>Feature Extraction:</strong> Pre-trained ResNet18 extracts 
                    512-dimensional feature vectors from images</li>
                <li><strong>Semi-Supervised Classification:</strong> SSL algorithms 
                    classify based on extracted features</li>
            </ol>
            <h4 style="color: #00d4ff;">Technologies:</h4>
            <ul style="color: #94a3b8;">
                <li>PyTorch (Feature Extraction)</li>
                <li>scikit-learn (SSL Algorithms)</li>
                <li>Streamlit (Web Interface)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>📚 How to Use</h3>
        <div style="color: #94a3b8;">
            <h4>1. Install Dependencies</h4>
            <code style="background: rgba(0,0,0,0.3); padding: 0.5rem 1rem; border-radius: 8px; display: block; margin: 0.5rem 0;">
                pip install -r requirements.txt
            </code>
            
            <h4>2. Train Models</h4>
            <code style="background: rgba(0,0,0,0.3); padding: 0.5rem 1rem; border-radius: 8px; display: block; margin: 0.5rem 0;">
                python src/train.py --labeled-ratio 0.1
            </code>
            
            <h4>3. Run the Application</h4>
            <code style="background: rgba(0,0,0,0.3); padding: 0.5rem 1rem; border-radius: 8px; display: block; margin: 0.5rem 0;">
                streamlit run app.py
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer" style="margin-top: 2rem;">
        <p>🎓 ML Mini-Project | Semi-Supervised Grocery Classification</p>
        <p style="font-size: 0.8rem;">Built with Streamlit, PyTorch, and scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)
