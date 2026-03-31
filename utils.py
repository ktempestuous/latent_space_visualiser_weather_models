import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

"""
utils.py

Utility for:

1 - make_circle_points - makes a circle of points on the sphere in degrees
2 - mesh_features_to_latlon - converts Graphcast mesh node features to coordinates on mesh node grid
3 - plot_global_data_with_overlay - plots 2D geospatial data (lat x lon) on a global Cartopy map with optional overlay points.
4 - select_nodes_within_radius - select indices of mesh nodes within a given radius from a centre point
5 - plot_global_overlay_only - plot data (e.g. latent channel value) on mesh nodes
6 - apply_translator - apply translator to latent features for an individual timestep and processor step, for all mesh nodes and latent channels
7 - dataframe_to_figure - render a pandas DataFrame as a matplotlib figure for PDF export
8 - apply_theme_css - app theme helper
9 - section_card - adds heirarchy to app 
"""

# ----------------------------------------------------
# --- 1. Makes a circle of points on the sphere in degrees
# ----------------------------------------------------

def make_circle_points(center_lat, center_lon, radius_deg, n=200):
    """Makes a circle of points on the sphere in degrees."""
    theta = np.linspace(0, 2 * np.pi, n)
    dlat = radius_deg * np.cos(theta)
    # degrees defined at the equator; scale longitude degrees by cos(lat)
    dlon = (radius_deg * np.sin(theta)) / np.cos(np.deg2rad(center_lat))
    return center_lat + dlat, center_lon + dlon

# ----------------------------------------------------
# --- 2. Convert features to (lat, lon)
# ----------------------------------------------------
def mesh_features_to_latlon(mesh_nodes_file):
    """
    Convert GraphCast mesh node features (cosθ, cosφ, sinφ) to latitude and longitude.
    Args:
        mesh_nodes_file: string of file name. Contains np.ndarray of shape (num_nodes, >=3)
    Returns:
        latitudes, longitudes: each np.ndarray of shape (num_nodes,)
    """
    mesh_nodes = np.load(mesh_nodes_file)
    
    cos_theta = mesh_nodes[:, 0]
    cos_phi = mesh_nodes[:, 1]
    sin_phi = mesh_nodes[:, 2]

    theta = np.arccos(cos_theta)           # polar angle [0, π]
    phi = np.arctan2(sin_phi, cos_phi)     # azimuth [-π, π]

    lat = 90.0 - np.degrees(theta)         # convert to latitude [-90, 90]
    lon = np.degrees(phi)                  # convert to longitude [-180, 180]
    return lat, lon

# --------------------------------------------------- 
# --- 3. Plot gloabl era5 data with option to overlay at select lat/lon coordinates
# ---------------------------------------------------

def plot_global_data_with_overlay(
    data,
    lon,
    lat,
    title="Map",
    cmap="viridis",
    cbar_label=None,
    figsize=(10, 5),
    overlay_lats=None,
    overlay_lons=None,
    dpi=100,
):
    """
    Plot 2D geospatial data (lat x lon) on a global Cartopy map
    with optional overlay points.

    Args:
        data (np.ndarray): 2D array (lat x lon)
        lon (np.ndarray): 1D longitude array
        lat (np.ndarray): 1D latitude array
        title (str): plot title
        cmap (str): colormap for base field
        cbar_label (str): label for colorbar
        figsize (tuple): figure size
        overlay_lats (np.ndarray): optional latitudes for overlay points
        overlay_lons (np.ndarray): optional longitudes for overlay points
        dpi (int): figure resolution
    """
    # Create 2D lon/lat grid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create figure with Cartopy projection
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Base field using coordinate-aware plotting
    img = ax.pcolormesh(
        lon2d,
        lat2d,
        data,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        edgecolors=None,
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )

    # # Gridlines
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=0.5,
    #     color="white",
    #     alpha=1,
    #     linestyle="--",
    #     xlocs=np.arange(-180, 181, 30),
    #     ylocs=np.arange(-90, 91, 30),
    # )
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {"size": 10}
    # gl.ylabel_style = {"size": 10}

    # Add axis ticks and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", fraction=0.03, pad=0.04,shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    # Overlay points, no label
    if overlay_lats is not None and overlay_lons is not None:
        ax.scatter(
            overlay_lons,
            overlay_lats,
            s=1,
            c="black",
            transform=ccrs.PlateCarree(),
            alpha=0.75,
            zorder=5,
        )

    ax.set_title(title)
    return fig

# ---------------------------------------------------
# --- 4. Return indices around centre point for defined radius
# ---------------------------------------------------

def select_nodes_within_radius(latitudes: np.ndarray, longitudes: np.ndarray,
                               center_lat: float, center_lon: float,
                               radius_km: float):
    """
    Select indices of mesh nodes within a given radius from a center point.
    
    Args:
        latitudes: np.ndarray of shape (num_nodes,)
        longitudes: np.ndarray of shape (num_nodes,)
        center_lat: latitude of center point in degrees
        center_lon: longitude of center point in degrees
        radius_km: radius in kilometers
    
    Returns:
        indices: np.ndarray of selected indices
    """
    # Earth's radius in km
    R = 6371.0

    # Convert degrees to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)

    # Haversine formula
    dlat = lat_rad - center_lat_rad
    dlon = lon_rad - center_lon_rad
    a = np.sin(dlat/2)**2 + np.cos(center_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c  # distance in km

    # Select indices within radius
    indices = np.where(distances <= radius_km)[0]
    return indices

# ---------------------------------------------------
# --- 5. Plot data (e.g. latent channel value) on mesh nodes
# ---------------------------------------------------
def plot_global_overlay_only(
    title="",
    figsize=(10, 5),
    overlay_lats=None,
    overlay_lons=None,
    overlay_values=None,
    circle_lons=None,
    circle_lats=None,
    dpi=100,
    plot_theme_mode="app theme",
):
    """
    Plot overlay values on a global Cartopy map without a background field.
    Overlay colors are fully opaque and zero-centered.

    Args:
        overlay_lats (np.ndarray): latitudes of overlay points
        overlay_lons (np.ndarray): longitudes of overlay points
        overlay_values (np.ndarray): values to plot at overlay points
        circle_lons (np.ndarray): optional circle longitude coordinates
        circle_lats (np.ndarray): optional circle latitude coordinates
    """

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    ax.set_global()
    ax.coastlines(linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=6)

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="white",
        linestyle="--",
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, 91, 30),
    )
    gl.top_labels = False
    gl.right_labels = False

    # --- Overlay only ---
    if overlay_lats is not None and overlay_lons is not None and overlay_values is not None:
        overlay_values = np.asarray(overlay_values).flatten()

        max_abs = np.max(np.abs(overlay_values))

        norm = TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0.0,
            vmax=max_abs,
        )

        if plot_theme_mode == "app theme":
            resolved_mode = st.session_state.get("theme_mode", "light")
        elif plot_theme_mode in ["light", "dark"]:
            resolved_mode = plot_theme_mode
        else:
            resolved_mode = None

        if resolved_mode == "light":
            cmap = plt.get_cmap("PRGn")
        elif resolved_mode == "dark":
            cmap = plt.get_cmap("PRGn")
        else:
            cmap = plt.get_cmap("coolwarm")

        colors = cmap(norm(overlay_values))
        colors[:, -1] = 1.0  # fully opaque

        sc = ax.scatter(
            overlay_lons,
            overlay_lats,
            s=10,
            c=colors,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)

    # --- Circle overlay ---
    if circle_lats is not None and circle_lons is not None:
        circle_lons = np.where(circle_lons > 180, circle_lons - 360, circle_lons)

        ax.scatter(
            circle_lons,
            circle_lats,
            s=1,
            c="black",
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    ax.set_title(title)

    return fig

# ---------------------------------------------------
# --- 6. Apply translator to latent features for an individual timestep and processor step, for all mesh nodes and latent channels
# ---------------------------------------------------

def apply_translator(latent_t, translator_file):
    """
    Apply translator weights and bias to latent tensor.

    Parameters
    ----------
    latent_t : np.ndarray
        Shape (nodes, batch_num, latent_dim), e.g., latent_array[selected_timestep,:,0,:]
    translator_file : str
        Path to .npz file containing W and b
        W shape: [latent_dim, latent_dim], b shape: [latent_dim]

    Returns
    -------
    np.ndarray
        Transformed latent tensor, same shape as latent_t: (nodes, batch_num, latent_dim)
    """
    npz = np.load(translator_file)
    W = np.array(npz["W"])
    b = np.array(npz["b"])

    # Apply linear transformation to each node and batch
    # latent_t: (nodes, latent_dim)
    # W: (latent_dim, latent_dim)
    # Result: (nodes, latent_dim)
    transformed = np.einsum("nd,df->nf", latent_t, W) + b

    return transformed

# ---------------------------------------------------
# --- 7. Render a pandas DataFrame as a matplotlib figure for PDF export
# ---------------------------------------------------

def dataframe_to_figure(df, title="", max_rows=30, fontsize=8):
    """Render a pandas DataFrame as a matplotlib figure for PDF export."""
    theme = THEMES.get(st.session_state.get("theme_mode", "light"), THEMES["light"])

    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(8.27, 2.0))
        ax.axis("off")
        ax.text(
            0.02, 0.8,
            f"{title}\n\nNo data available.",
            fontsize=11,
            va="top",
            color=theme["text"],
        )
        fig.patch.set_facecolor(theme["background"])
        return fig

    df_show = df.head(max_rows).copy()

    nrows, ncols = df_show.shape
    fig_height = min(11.0, 1.2 + 0.35 * (nrows + 1))
    fig_width = min(11.5, max(8.0, 1.4 * ncols))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(theme["background"])
    ax.set_facecolor(theme["background"])
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, pad=12, color=theme["text"])

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(theme["border"])
        if row == 0:
            cell.set_facecolor(theme["secondary_background"])
            cell.set_text_props(color=theme["text"], weight="bold")
        else:
            cell.set_facecolor(theme["background"])
            cell.set_text_props(color=theme["text"])

    plt.tight_layout()
    return fig

# ---------------------------------------------------
# --- 8. App theme helper
# ---------------------------------------------------

THEMES = {
    "light": {
        "background": "#FBF8F1",
        "secondary_background": "#F3EEE3",
        "sidebar_background": "#F6F1E6",
        "text": "#1C241D",
        "muted_text": "#5E6A5F",
        "primary": "#5E7F65",
        "primary_text": "#F8F6EF",
        "border": "#DDD5C7",
        "code_bg": "#ECE7DC",
        "font_body": "Inter, sans-serif",
        "font_heading": "Inter, sans-serif",
        "font_code": "JetBrains Mono, monospace",
        "plot_seq": ["#F8F6EF", "#E7EFD9", "#C9D8B6", "#8FAA7A", "#5E7F65"],
        "plot_cat": ["#5E7F65", "#7FA37F", "#A3B18A", "#D9CBB3", "#8C6F5A"],
        "plot_div": "BrBG",
        "checkbox_tick": "#1C241D",
    },
    "dark": {
        "background": "#102019",
        "secondary_background": "#1A2B22",
        "sidebar_background": "#0D1813",
        "text": "#F3EBDD",
        "muted_text": "#CDBFA8",
        "primary": "#E6D8BE",
        "primary_text": "#102019",
        "border": "#314238",
        "code_bg": "#16241D",
        "font_body": "Inter, sans-serif",
        "font_heading": "Inter, sans-serif",
        "font_code": "JetBrains Mono, monospace",
        "plot_seq": ["#16241D", "#35523A", "#5E7F65", "#8FAA7A", "#E6D8BE"],
        "plot_cat": ["#E6D8BE", "#A3B18A", "#7FA37F", "#5E7F65", "#8C6F5A"],
        "plot_div": "BrBG",
        "checkbox_tick": "#102019",
    },
}

def apply_theme_css(theme: dict):
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            font-family: {theme["font_body"]};
        }}

        /* Main content area */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: {theme["background"]};
        }}

        /* Remove the dark top header/banner area */
        [data-testid="stHeader"] {{
            background: {theme["background"]};
            border-bottom: 1px solid {theme["border"]};
        }}

        /* Toolbar area near the top */
        [data-testid="stToolbar"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background-color: {theme["sidebar_background"]};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text"]};
            font-family: {theme["font_heading"]};
        }}

        p, label, .stMarkdown, .stText {{
            color: {theme["text"]};
            letter-spacing: 0.01em;
        }}

        code, pre {{
            font-family: {theme["font_code"]} !important;
            background-color: {theme["code_bg"]} !important;
            color: {theme["text"]} !important;
            border-radius: 0.4rem;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background-color: {theme["primary"]} !important;
            color: {theme["primary_text"]} !important;
            border: 1px solid {theme["primary"]} !important;
            border-radius: 0.5rem;
        }}

        /* Force ALL nested text elements */
        .stButton > button *,
        .stDownloadButton > button * {{
            color: {theme["primary_text"]} !important;
            fill: {theme["primary_text"]} !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            filter: brightness(0.95);
        }}

        /* Inputs / selects */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        /* Text inside select boxes */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: {theme["text"]} !important;
        }}

        input, textarea {{
            color: {theme["text"]} !important;
            background-color: {theme["secondary_background"]} !important;
        }}

        /* Dropdown menu popover */
        [role="listbox"] {{
            background-color: {theme["secondary_background"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        [role="option"] {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
        }}

        [role="option"]:hover {{
            background-color: {theme["background"]} !important;
        }}

        .stSlider, .stNumberInput, .stTextInput, .stSelectbox, .stMultiSelect {{
            color: {theme["text"]};
        }}

        /* Checkbox / radio clarity */
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {{
            color: {theme["text"]} !important;
        }}

        /* Checkbox box */
        [data-testid="stCheckbox"] div[role="checkbox"] {{
            background-color: {theme["secondary_background"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        /* Checked state */
        [data-testid="stCheckbox"] div[aria-checked="true"] {{
            background-color: {theme["primary"]} !important;
            border: 1px solid {theme["primary"]} !important;
        }}

        /* The tick (SVG) */
        [data-testid="stCheckbox"] svg {{
            stroke: {theme["checkbox_tick"]} !important;
            fill: {theme["checkbox_tick"]} !important;
        }}

        [data-testid="stCheckbox"] input,
        [data-testid="stRadio"] input {{
            accent-color: {theme["primary"]};
        }}

        /* Dataframe / table styling */
        .stDataFrame, .stTable {{
            border: 1px solid {theme["border"]};
            border-radius: 0.5rem;
            background-color: {theme["background"]};
        }}

        [data-testid="stDataFrame"] {{
            background-color: {theme["background"]} !important;
        }}

        [data-testid="stDataFrame"] div {{
            color: {theme["text"]} !important;
        }}

        table {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}

        thead tr th {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
        }}

        tbody tr td {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}

        hr {{
            border-color: {theme["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    

# ---------------------------------------------------
# --- 9. App heirarchy helper
# ---------------------------------------------------

def section_card(title, subtitle=None):
    theme = THEMES[st.session_state.get("theme_mode", "light")]
    border = theme["border"]
    bg = theme["secondary_background"]
    text = theme["text"]
    
    st.markdown(
        f"""
        <div style="
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background-color: rgba(128,128,128,0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        ">
            <h2 style="margin-bottom: 0.6rem;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(subtitle)
