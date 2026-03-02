import streamlit as st
import requests
import os
import json
import base64
import time
import random
import math
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import plotly.graph_objects as go

# ==========================================
# 1. Core Configurations
# ==========================================

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
UNSPLASH_ACCESS_KEY = st.secrets["UNSPLASH_ACCESS_KEY"]
APS_CLIENT_ID = st.secrets["APS_CLIENT_ID"]
APS_CLIENT_SECRET = st.secrets["APS_CLIENT_SECRET"]

US_CITIES = ["Boston, MA", "New York, NY", "Seattle, WA", "San Francisco, CA", "Chicago, IL", "Los Angeles, CA", "Miami, FL"]

ENV_NAMES = [
    "Type A: 100% WWR (Curtain Wall)",
    "Type B: 60% WWR (Diffuse Glazing)",
    "Type C: 40% WWR (Punched Windows)"
]

# Strictly lock the three image queries as requested
FIXED_QUERIES = [
    "office window",
    "modern office architecture working area",
    "modern office with diffuse daylight dominance"
]

REQUIRED_SHAPES = ["Narrow Bar", "Deep Block", "L-Shape", "U-Shape", "Cross-Shape", "O-Shape"]

# ==========================================
# 2. Data Structure Definitions
# ==========================================
class MassingVariant(BaseModel):
    name: str
    footprint: list[list[float]]
    height: float
    floor_count: int
    rationale: str

class MassingResponse(BaseModel):
    variants: list[MassingVariant]

class RationaleList(BaseModel):
    rationales: list[str] = Field(description="Exactly 6 design rationales, corresponding to the 6 predefined shapes.")

class ArchitectReport(BaseModel):
    recommended_variant: str = Field(description="Name of the best performing variant")
    executive_summary: str = Field(description="One-sentence executive summary")
    performance_analysis: str = Field(description="Detailed analysis weighing sDA vs ASE")
    mitigation_strategies: str = Field(description="Suggestions to mitigate negative effects")
    form_evolution_suggestions: str = Field(description="Architectural suggestions for evolving the geometry")

# ==========================================
# 3. Parametric Geometry Engine (Pure mathematical generation, zero broken faces)
# ==========================================
def generate_perfect_footprint(shape_name, target_area):
    """Generate absolutely perfect, non-self-intersecting building polygons using mathematical matrices, scaled precisely to the target area."""
    templates = {
        "Narrow Bar": [[0,0], [4,0], [4,1], [0,1], [0,0]],
        "Deep Block": [[0,0], [1.5,0], [1.5,1.5], [0,1.5], [0,0]],
        "L-Shape": [[0,0], [3,0], [3,1], [1,1], [1,3], [0,3], [0,0]],
        "U-Shape": [[0,0], [3,0], [3,3], [2,3], [2,1], [1,1], [1,3], [0,3], [0,0]],
        "Cross-Shape": [[1,0], [2,0], [2,1], [3,1], [3,2], [2,2], [2,3], [1,3], [1,2], [0,2], [0,1], [1,1], [1,0]],
        # Use a tiny 0.2 gap to draw an O-shaped courtyard in one continuous stroke, visually completely closed
        "O-Shape": [[0,0], [4,0], [4,4], [0,4], [0,2.1], [1,2.1], [1,3], [3,3], [3,1], [1,1], [1,1.9], [0,1.9], [0,0]]
    }
    
    t = templates.get(shape_name, templates["Narrow Bar"])
    
    # Calculate template area arithmetically (Shoelace Formula)
    base_area = 0.5 * abs(sum(t[i][0]*t[i+1][1] - t[i+1][0]*t[i][1] for i in range(len(t)-1)))
    # Calculate scaling factor based on target area
    scale = math.sqrt(target_area / base_area)
    
    # Center the coordinates so the 3D rotation axis is perfectly centered
    cx = sum(p[0] for p in t[:-1]) / (len(t)-1)
    cy = sum(p[1] for p in t[:-1]) / (len(t)-1)
    
    return [[round((x - cx)*scale, 2), round((y - cy)*scale, 2)] for x, y in t]

# ==========================================
# 4. Agents & Tools
# ==========================================
def get_aps_token():
    url = "https://developer.api.autodesk.com/authentication/v2/token"
    auth_str = f"{APS_CLIENT_ID}:{APS_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, headers=headers, data={"grant_type": "client_credentials", "scope": "data:read data:write data:create"})
        return response.json().get("access_token")
    except: return None

def get_unsplash_image(query):
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page=1&orientation=landscape"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results: return results[0]['urls']['regular']
    except: pass
    return None

def agent_2_generate_massing(city, area, floors, vibe):
    """Agent 2 is now only responsible for the design rationale; complex geometry generation is handled by the Python math engine to prevent distortion."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    structured_llm = llm.with_structured_output(RationaleList)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert computational architect. 
        We have 6 specific architectural massing typologies: {', '.join(REQUIRED_SHAPES)}.
        Write a concise, professional 2-sentence design rationale for EACH of the 6 shapes, 
        explaining how its geometry relates to the {vibe} lighting vibe in {city}.
        """),
        ("human", "Generate the 6 rationales in exact order.")
    ])
    
    rationale_response = (prompt | structured_llm).invoke({})
    
    variants = []
    target_footprint_area = area / floors
    height = floors * 15.0
    
    for i, shape_name in enumerate(REQUIRED_SHAPES):
        rationale_text = rationale_response.rationales[i] if i < len(rationale_response.rationales) else f"Parametric optimization for {shape_name}."
        footprint = generate_perfect_footprint(shape_name, target_footprint_area)
        
        variants.append(MassingVariant(
            name=shape_name, footprint=footprint, height=height, floor_count=floors, rationale=rationale_text
        ))
        
    return MassingResponse(variants=variants)

def plot_3d_wireframe(variant):
    pts = variant.footprint
    if not pts: return go.Figure()

    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    z_base = [0] * len(pts); z_top = [variant.height] * len(pts)

    fig = go.Figure()
    glass_color = 'rgba(0, 113, 227, 0.08)'
    frame_color = '#86868B'
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z_base, mode='lines', surfaceaxis=2, surfacecolor=glass_color, name='Base', line=dict(color=frame_color, width=3)))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z_top, mode='lines', surfaceaxis=2, surfacecolor=glass_color, name='Roof', line=dict(color=frame_color, width=3)))
    for i in range(len(pts) - 1):
        fig.add_trace(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[0, variant.height], mode='lines', showlegend=False, line=dict(color=frame_color, width=2)))

    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# [Important Fix]: Restored the missing docstring to resolve errors
@tool
def forma_daylight_evaluator(variant_name: str, vibe_type: str, city: str, floors: int) -> dict:
    """
    Agent 3 Tool: Simulates Autodesk Forma API to evaluate Spatial Daylight Autonomy (sDA) and Annual Sunlight Exposure (ASE).
    """
    time.sleep(0.4) 
    sda = 50.0; ase = 5.0  
    climate_sun = {"Miami, FL": 1.4, "Los Angeles, CA": 1.3, "Chicago, IL": 1.0, "New York, NY": 1.0, "Boston, MA": 0.9, "Seattle, WA": 0.6, "San Francisco, CA": 1.1}
    sun_factor = climate_sun.get(city, 1.0)
    
    if "100%" in vibe_type: sda = 90.0 * sun_factor; ase = 45.0 * sun_factor
    elif "60%" in vibe_type: sda = 72.0 * sun_factor; ase = 12.0 * sun_factor
    elif "40%" in vibe_type: sda = 45.0 * sun_factor; ase = 2.0 * sun_factor
        
    name_lower = variant_name.lower()
    if "narrow" in name_lower or "cross" in name_lower: sda += 8.0; ase += 5.0
    elif "deep" in name_lower: sda -= 15.0; ase -= 2.0
    elif "l-shape" in name_lower: sda += 5.0; ase += 2.0
    elif "u-shape" in name_lower or "o-shape" in name_lower: sda -= 2.0; ase -= 4.0 
        
    return {"sDA": round(min(100.0, max(0.0, sda + random.uniform(-2, 2))), 1), "ASE": round(min(100.0, max(0.0, ase + random.uniform(-1, 1))), 1)}

def agent_4_chief_architect(city, area, floors, vibe, variants, analysis_results):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    structured_llm = llm.with_structured_output(ArchitectReport)
    context_data = "".join([f"- [{var.name}] | sDA: {res['sDA']}% | ASE: {res['ASE']}%\n" for var, res in zip(variants, analysis_results)])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Chief Architect. Produce a highly rigorous, yet accessible architectural evaluation.
        Write in a clean, professional tone. Focus on spatial performance, daylight autonomy, and geometric strategies.
        """),
        ("human", f"Context: {city} | {area} sqft | {floors}F | {vibe}\nData:\n{context_data}")
    ])
    return (prompt | structured_llm).invoke({})

# ==========================================
# 5. Streamlit UI & Custom CSS (Apple Minimalist Vibe)
# ==========================================
st.set_page_config(layout="wide", page_title="Forma AI Optimizer", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F5F5F7; color: #1D1D1F; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    h1, h2, h3, h4, h5 { font-weight: 600 !important; color: #1D1D1F; letter-spacing: -0.02em; }
    h1 { font-size: 2.5rem !important; border: none; padding-bottom: 0px; margin-bottom: 0.5rem; text-align: center; }
    .block-container { padding-top: 3rem; max-width: 1200px; }
    header { visibility: hidden; }
    
    /* Core visuals: All action buttons use a premium light gray background with dark text, turning blue on hover */
    .stButton > button {
        background-color: #F2F2F7 !important; color: #1D1D1F !important; border: 1px solid #E5E5EA !important;
        border-radius: 12px !important; font-weight: 500; font-size: 0.95rem;
        padding: 10px 24px; transition: all 0.2s ease-in-out; box-shadow: none;
    }
    .stButton > button:hover { 
        background-color: #0071E3 !important; color: #FFFFFF !important; border-color: #0071E3 !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.2) !important;
    }
    
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E5EA; }
    
    /* Text box style below the images */
    .rationale-box {
        background-color: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 12px;
        padding: 12px; margin-bottom: 10px; font-size: 0.85rem; color: #86868B;
        height: 85px; overflow: hidden; line-height: 1.4;
    }
    
    hr { border-top: 1px solid #E5E5EA; margin: 3rem 0; }
</style>
""", unsafe_allow_html=True)

if 'selected_vibe' not in st.session_state: st.session_state.selected_vibe = None
if 'vibe_images' not in st.session_state: st.session_state.vibe_images = []
if 'massing_variants' not in st.session_state: st.session_state.massing_variants = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

st.markdown("<h1>Agentic Forma Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#86868B; font-size:1.05rem; margin-bottom:3rem;'>AI-driven generative design & environmental analysis</p>", unsafe_allow_html=True)

st.sidebar.markdown("### Project Parameters")
selected_city = st.sidebar.selectbox("Site Location", US_CITIES)
target_area = st.sidebar.number_input("Total GFA (sq ft)", value=30000, step=1000)
target_floors = st.sidebar.number_input("Floor Count", min_value=1, max_value=100, value=3)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Explore Design Intents"):
    token = get_aps_token()
    with st.spinner("Curating lighting environments..."):
        images = [get_unsplash_image(q) for q in FIXED_QUERIES]
        st.session_state.vibe_images = [img if img else "" for img in images]
        st.session_state.selected_vibe = None
        st.session_state.massing_variants = None
        st.session_state.analysis_results = None
        st.session_state.run_flow = True

if st.session_state.get("run_flow"):
    st.markdown("### 01. Visual Intent")
    if st.session_state.vibe_images:
        cols = st.columns(3)
        for i, col in enumerate(cols):
            with col:
                if st.session_state.vibe_images[i]: 
                    st.image(st.session_state.vibe_images[i], use_container_width=True)
                if st.button(ENV_NAMES[i], key=f"btn_env_{i}"):
                    st.session_state.selected_vibe = ENV_NAMES[i]
                    st.session_state.massing_variants = None
                    st.session_state.analysis_results = None
    
    if st.session_state.selected_vibe:
        st.markdown("---")
        st.markdown("### 02. Generative Massing")
        st.info(f"✨ **Locked Intent:** {st.session_state.selected_vibe}")
        
        if st.button("Generate Spatial Topologies"):
            with st.spinner("Parametric geometric engine running..."):
                response = agent_2_generate_massing(selected_city, target_area, target_floors, st.session_state.selected_vibe)
                st.session_state.massing_variants = response.variants
                st.session_state.analysis_results = None 

        if st.session_state.massing_variants:
            for row_idx in range(2): 
                grid_cols = st.columns(3)
                for col_idx in range(3):
                    var_idx = row_idx * 3 + col_idx
                    if var_idx < len(st.session_state.massing_variants):
                        var = st.session_state.massing_variants[var_idx]
                        
                        with grid_cols[col_idx]:
                            st.markdown(f"#### {var.name}")
                            st.plotly_chart(plot_3d_wireframe(var), use_container_width=True, key=f"3d_plot_{var_idx}")
                            
                            # Second image requirement: Text box design
                            st.markdown(f"<div class='rationale-box'>{var.rationale}</div>", unsafe_allow_html=True)
                            
                            if st.session_state.analysis_results:
                                res = st.session_state.analysis_results[var_idx]
                                
                                # Green sDA, Yellow ASE, absolute justified alignment
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 10px; padding-top: 15px; border-top: 1px solid #E5E5EA;">
                                    <div style="text-align: left;">
                                         <div style="font-size:0.75rem; color:#86868B; text-transform: uppercase; font-weight:600; margin-bottom:4px;">sDA (>55%)</div>
                                         <div style="font-size:1.8rem; color:#28CD41; font-weight:600; line-height:1;">{res['sDA']}%</div>
                                    </div>
                                    <div style="text-align: right;">
                                         <div style="font-size:0.75rem; color:#86868B; text-transform: uppercase; font-weight:600; margin-bottom:4px;">ASE (<10%)</div>
                                         <div style="font-size:1.8rem; color:#FF9500; font-weight:600; line-height:1;">{res['ASE']}%</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 03. Environmental Analysis")
            
            # Button defaults to light gray with black text, glows only on hover
            trigger_eval = st.button("Run Forma Simulations")
            
            if trigger_eval:
                st.session_state.analysis_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, var in enumerate(st.session_state.massing_variants):
                    status_text.text(f"Evaluating {var.name} metrics...")
                    score_dict = forma_daylight_evaluator.invoke({
                        "variant_name": var.name, "vibe_type": st.session_state.selected_vibe, "city": selected_city, "floors": target_floors
                    })
                    st.session_state.analysis_results.append(score_dict)
                    progress_bar.progress((idx + 1) / 6)
                
                status_text.empty()
                progress_bar.empty()
                st.rerun() 
                
            if st.session_state.analysis_results:
                with st.spinner("Synthesizing Chief Architect Report..."):
                    report = agent_4_chief_architect(
                        selected_city, target_area, target_floors, st.session_state.selected_vibe, 
                        st.session_state.massing_variants, st.session_state.analysis_results
                    )
                
                st.markdown("---")
                st.markdown("### 04. Executive Decision")
                with st.container(border=True):
                    st.success(f"🏆 **Recommended Topology:** {report.recommended_variant}")
                    st.markdown(f"<p style='font-size:1.1rem; color:#1D1D1F; font-weight:500;'>{report.executive_summary}</p>", unsafe_allow_html=True)
                    st.markdown("#### Performance Analysis")
                    st.markdown(f"<p style='color:#555; line-height: 1.6;'>{report.performance_analysis}</p>", unsafe_allow_html=True)
                    st.markdown("#### Mitigation Strategies")
                    st.markdown(f"<p style='color:#555; line-height: 1.6;'>{report.mitigation_strategies}</p>", unsafe_allow_html=True)
                    st.markdown("#### Form Evolution")
                    st.markdown(f"<p style='color:#555; line-height: 1.6;'>{report.form_evolution_suggestions}</p>", unsafe_allow_html=True)