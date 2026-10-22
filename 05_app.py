from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# 專案檔案均相對於本程式所在目錄，部署時請一併提供資料與模型檔。
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "copper_foil_all_1000.csv"
MODEL_PATH = APP_DIR / "processed" / "random_forest_anomaly_model.joblib"

st.set_page_config(
    page_title="銅箔小管家｜製程分析",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 柔和粉彩風格；中文字型由瀏覽器依序套用系統可用字型。
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #493f5c;
        --pink: #f5a8bd;
        --lavender: #c8b8f2;
        --mint: #a9ded2;
        --cream: #fffaf3;
    }
    html, body, [class*="css"] {
        font-family: 'Nunito', 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: linear-gradient(145deg, #fff9f5 0%, #fbf7ff 52%, #f2fbf9 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.45); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0f4 0%, #f5f0ff 100%);
        border-right: 1px solid #eee2f6;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px; }
    h1, h2, h3 { color: #514365; letter-spacing: .01em; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.82);
        border: 1px solid #eee5f5;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 5px 18px rgba(103, 81, 126, .06);
    }
    div.stButton > button, div.stFormSubmitButton > button {
        border: 0;
        border-radius: 14px;
        color: #fff;
        font-weight: 700;
        background: linear-gradient(100deg, #ee91ad, #a894e6);
        box-shadow: 0 5px 13px rgba(162, 132, 205, .2);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        color: white;
        border: 0;
        filter: brightness(1.04);
    }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,.78);
        padding: 1.25rem;
        border: 1px solid #eee5f5;
        border-radius: 20px;
    }
    .cute-note {
        border-radius: 16px;
        padding: 14px 18px;
        background: #fff;
        border: 1px solid #efe6f5;
        color: #665b78;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="正在準備製程資料…")
def load_data(path: str) -> pd.DataFrame:
    """讀取並快取完整銅箔資料，解析時間欄位供圖表排序。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    return frame


@st.cache_resource(show_spinner="正在喚醒異常偵測小助手…")
def load_model(path: str) -> dict:
    """載入訓練時封裝的 Pipeline 與特徵欄位資訊。"""
    return joblib.load(path)


def chart_style(fig, height: int = 390):
    """統一圖表字型、色彩與卡片留白。"""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Nunito, Noto Sans TC, Microsoft JhengHei, sans-serif", "color": "#514365"},
        paper_bgcolor="rgba(255,255,255,0.82)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        margin={"l": 28, "r": 20, "t": 65, "b": 35},
        legend_title_text="",
    )
    return fig


def show_sidebar() -> str:
    """左側功能選單。"""
    with st.sidebar:
        st.markdown("# 🐾 銅箔小管家")
        st.caption("陪你一起看懂製程的小助手")
        page = st.radio(
            "選擇功能",
            ["📊 資料視覺化", "🔮 異常預測"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("### 🍬 小提醒")
        st.caption("預測結果是模型輔助判讀，請搭配現場量測與製程知識一起確認。")
        st.caption("模型：Random Forest · Is_Anomaly")
    return page


def render_dashboard() -> None:
    """第 1 頁：查看整體製程分布、時間趨勢及關鍵變數關係。"""
    st.title("📊 製程觀察小花園")
    st.markdown("看看銅箔厚度與幾個重要製程特徵，找出值得留意的小訊號。")
    if not DATA_PATH.exists():
        st.error(f"找不到資料檔：`{DATA_PATH}`。請確認部署時已包含來源 CSV。")
        st.stop()

    data = load_data(str(DATA_PATH))
    shifts = ["全部班別"] + sorted(data["Shift"].dropna().unique().tolist())
    chosen_shift = st.selectbox("🌼 想查看哪個班別？", shifts)
    view = data if chosen_shift == "全部班別" else data[data["Shift"] == chosen_shift]

    total_count = len(view)
    anomaly_rate = float(view["Is_Anomaly"].mean()) if total_count else 0.0
    out_spec_rate = float(view["Is_Out_of_Spec"].mean()) if total_count else 0.0
    avg_thickness = float(view["Thick_Mean_um"].mean()) if total_count else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🍀 樣本數", f"{total_count:,}")
    k2.metric("🎀 異常比例", f"{anomaly_rate:.1%}")
    k3.metric("🧁 超規比例", f"{out_spec_rate:.1%}")
    k4.metric("📏 平均厚度", f"{avg_thickness:.3f} μm")

    st.markdown("### 厚度小觀察")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            view,
            x="Thick_Mean_um",
            color="Is_Out_of_Spec",
            nbins=32,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={0: "#9edbd0", 1: "#f29bb5"},
            labels={"Thick_Mean_um": "平均厚度（μm）", "count": "筆數", "Is_Out_of_Spec": "超出規格"},
            title="平均厚度分布與規格界線",
        )
        fig.add_vline(x=17, line_dash="dash", line_color="#7c6ca5", annotation_text="下限 17 μm")
        fig.add_vline(x=19, line_dash="dash", line_color="#e58ca9", annotation_text="上限 19 μm")
        st.plotly_chart(chart_style(fig), use_container_width=True)
    with right:
        fig = px.scatter(
            view.sort_values("Timestamp"),
            x="Timestamp",
            y="Thick_Mean_um",
            color="Is_Anomaly",
            color_discrete_map={0: "#9bc9dd", 1: "#ee829f"},
            hover_data=["Sample_ID", "Shift", "Thick_Range_um"],
            labels={"Timestamp": "時間", "Thick_Mean_um": "平均厚度（μm）", "Is_Anomaly": "標記異常"},
            title="平均厚度時間趨勢",
        )
        fig.add_hline(y=17, line_dash="dash", line_color="#7c6ca5")
        fig.add_hline(y=19, line_dash="dash", line_color="#e58ca9")
        st.plotly_chart(chart_style(fig), use_container_width=True)

    st.markdown("### 製程特徵探索")
    feature_names = {
        "Line_Speed_mpm": "線速（m/min）",
        "Current_ASD": "電流密度（ASD）",
        "Bath_Temp_C": "槽液溫度（°C）",
        "Cu_Conc_gL": "銅離子濃度（g/L）",
        "Additive_ppm": "添加劑（ppm）",
        "Jet_Pressure_bar": "噴流壓力（bar）",
        "Edge_Current_Bias": "邊緣電流偏差",
        "Thick_Range_um": "橫向厚度差（μm）",
    }
    feature_options = list(feature_names)
    selected_feature = st.selectbox(
        "選擇要查看的特徵",
        feature_options,
        format_func=lambda key: feature_names[key],
    )
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            view,
            x=selected_feature,
            color="Is_Anomaly",
            nbins=30,
            marginal="box",
            color_discrete_map={0: "#a9d9ce", 1: "#ef9bb4"},
            labels={selected_feature: feature_names[selected_feature], "Is_Anomaly": "標記異常", "count": "筆數"},
            title=f"{feature_names[selected_feature]}分布",
        )
        st.plotly_chart(chart_style(fig), use_container_width=True)
    with right:
        fig = px.scatter(
            view,
            x="Current_ASD",
            y="Edge_Current_Bias",
            color="Is_Anomaly",
            hover_data=["Sample_ID", "Shift", "Bath_Temp_C"],
            color_discrete_map={0: "#9bc9dd", 1: "#ee829f"},
            labels={"Current_ASD": "電流密度（ASD）", "Edge_Current_Bias": "邊緣電流偏差", "Is_Anomaly": "標記異常"},
            title="電流密度與邊緣電流偏差",
        )
        st.plotly_chart(chart_style(fig), use_container_width=True)

    st.markdown("### 關鍵數值關聯")
    corr_columns = [
        "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
        "Additive_ppm", "Jet_Pressure_bar", "Edge_Current_Bias",
        "Thick_Mean_um", "Thick_Range_um",
    ]
    corr_labels = [feature_names.get(c, {"Thick_Mean_um": "平均厚度（μm）"}.get(c, c)) for c in corr_columns]
    corr = view[corr_columns].corr()
    fig = px.imshow(
        corr,
        x=corr_labels,
        y=corr_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="相關係數熱圖（僅呈現線性關聯）",
    )
    st.plotly_chart(chart_style(fig, height=560), use_container_width=True)

    with st.expander("🧺 查看目前篩選後的資料"):
        st.dataframe(view, use_container_width=True, hide_index=True)


def render_predictor() -> None:
    """第 2 頁：表單收集製程輸入，呼叫 joblib Pipeline 預測異常。"""
    st.title("🔮 異常預測小魔法")
    st.markdown("填入一筆製程條件，小管家會回報模型判斷與異常分數。")
    st.markdown(
        '<div class="cute-note">🐱 <b>小提醒：</b>模型使用製程設定、班別與時間特徵。厚度量測結果不會當作輸入，以免把答案直接提供給模型。</div>',
        unsafe_allow_html=True,
    )
    if not DATA_PATH.exists():
        st.error(f"找不到參考資料：`{DATA_PATH}`。請確認部署時已包含來源 CSV。")
        st.stop()
    if not MODEL_PATH.exists():
        st.error(f"找不到模型檔：`{MODEL_PATH}`。請先執行 `03_random_forest_training.ipynb` 並部署 joblib 檔。")
        st.stop()

    try:
        data = load_data(str(DATA_PATH))
        bundle = load_model(str(MODEL_PATH))
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
    except Exception as exc:
        st.error(f"讀取資料或模型時發生問題：{exc}")
        st.stop()

    st.markdown("### 🧋 製程資料表單")
    st.caption("輸入範圍依訓練資料的觀測值設定；模型分數僅供輔助判讀。")
    with st.form("anomaly_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            shift_options = sorted(data["Shift"].dropna().unique().tolist())
            default_shift = data["Shift"].mode().iloc[0]
            shift = st.selectbox("班別", shift_options, index=shift_options.index(default_shift))
            line_speed = st.number_input(
                "線速（m/min）", min_value=float(data["Line_Speed_mpm"].min()),
                max_value=float(data["Line_Speed_mpm"].max()),
                value=float(data["Line_Speed_mpm"].median()), step=0.01, format="%.2f",
            )
            current_asd = st.number_input(
                "電流密度（ASD）", min_value=float(data["Current_ASD"].min()),
                max_value=float(data["Current_ASD"].max()),
                value=float(data["Current_ASD"].median()), step=0.1, format="%.2f",
            )
        with c2:
            bath_temp = st.number_input(
                "槽液溫度（°C）", min_value=float(data["Bath_Temp_C"].min()),
                max_value=float(data["Bath_Temp_C"].max()),
                value=float(data["Bath_Temp_C"].median()), step=0.1, format="%.2f",
            )
            cu_conc = st.number_input(
                "銅離子濃度（g/L）", min_value=float(data["Cu_Conc_gL"].min()),
                max_value=float(data["Cu_Conc_gL"].max()),
                value=float(data["Cu_Conc_gL"].median()), step=0.1, format="%.2f",
            )
            additive = st.number_input(
                "添加劑（ppm）", min_value=float(data["Additive_ppm"].min()),
                max_value=float(data["Additive_ppm"].max()),
                value=float(data["Additive_ppm"].median()), step=0.01, format="%.2f",
            )
        with c3:
            jet_pressure = st.number_input(
                "噴流壓力（bar）", min_value=float(data["Jet_Pressure_bar"].min()),
                max_value=float(data["Jet_Pressure_bar"].max()),
                value=float(data["Jet_Pressure_bar"].median()), step=0.01, format="%.2f",
            )
            edge_bias = st.number_input(
                "邊緣電流偏差", min_value=float(data["Edge_Current_Bias"].min()),
                max_value=float(data["Edge_Current_Bias"].max()),
                value=float(data["Edge_Current_Bias"].median()), step=0.01, format="%.3f",
            )
            hour = st.slider("量測小時", min_value=0, max_value=23, value=12, format="%d 時")
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        day_name = st.selectbox("星期幾", weekday_names, index=0)
        submitted = st.form_submit_button("✨ 開始預測", use_container_width=True)

    if submitted:
        # 時間欄位以訓練時相同公式轉成週期特徵。
        input_values = {
            "Line_Speed_mpm": float(line_speed),
            "Current_ASD": float(current_asd),
            "Bath_Temp_C": float(bath_temp),
            "Cu_Conc_gL": float(cu_conc),
            "Additive_ppm": float(additive),
            "Jet_Pressure_bar": float(jet_pressure),
            "Edge_Current_Bias": float(edge_bias),
            "Hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "Hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "DayOfWeek": int(weekday_names.index(day_name)),
            "Shift": shift,
        }
        # 依模型保存的欄位順序建立輸入，確保前處理 Pipeline 收到正確欄位。
        input_frame = pd.DataFrame([input_values], columns=feature_columns)
        predicted_class = int(model.predict(input_frame)[0])
        class_values = list(model.classes_)
        anomaly_probability = float(model.predict_proba(input_frame)[0][class_values.index(1)])

        st.markdown("### 📨 小管家的回覆")
        result_col, score_col = st.columns([1.15, 1])
        with result_col:
            if predicted_class == 1:
                st.error("🚨 模型判定：需要留意的異常訊號")
                st.markdown("建議確認製程條件，並搭配現場量測與品保流程進一步判讀。")
            else:
                st.success("🌷 模型判定：目前較接近正常狀態")
                st.markdown("仍請依既有品質規範監控製程，模型結果不能取代現場檢驗。")
        with score_col:
            st.metric("🎯 異常預測分數", f"{anomaly_probability:.1%}")
            st.progress(anomaly_probability, text="異常類別機率（分類門檻 50%）")
        st.caption("異常分數是模型輸出的預測機率，未必等同實際發生率；輸入超出訓練資料範圍時，結果可能不可靠。")
        with st.expander("🔎 檢視送入模型的特徵"):
            st.dataframe(input_frame, use_container_width=True, hide_index=True)


page = show_sidebar()
if page == "📊 資料視覺化":
    render_dashboard()
else:
    render_predictor()