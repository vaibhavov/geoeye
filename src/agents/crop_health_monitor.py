import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict, Literal

# ---- Earth Engine ----
import ee

# ---- LangChain / LangGraph ----
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

# =========================
# 0) EE INIT (same pattern you used)
# =========================
def init_ee():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if credentials_path and project_id:
            credentials = ee.ServiceAccountCredentials(None, credentials_path)
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize()
        print("🛰️ Earth Engine initialized")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize EE: {e}")

# =========================
# 1) Agent State + Models
# =========================
class Issue(BaseModel):
    type: Literal["disease","pest","weed","water","nutrient"]
    severity: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = []
    actions: List[str] = []

class AgentState(TypedDict):
    # context
    field_id: str
    crop: str
    stage_das: int
    latitude: float
    longitude: float
    radius_m: int
    # runtime
    context: Dict[str, Any]
    plan: List[str]
    artifacts: Dict[str, Any]
    issues: List[Issue]
    status: str   # 'ok'|'needs_followup'|'error'
    started_at: str
    finished_at: Optional[str]

# =========================
# 2) LLMs (Brain / Planner / (optional) Explainer)
# =========================
# You can swap to smaller models for planner/supervisor if you pulled them:
#   llm_planner = ChatOllama(model="qwen2.5:3b-instruct", base_url="http://localhost:11434", temperature=0)
# For simplicity, we keep all on mistral-nemo (fast+good 7B):
llm_brain = ChatOllama(model="mistral-nemo", base_url="http://localhost:11434",
                       temperature=0,
                       model_kwargs={"num_ctx": 4096, "num_predict": 256, "top_p": 0.9, "repeat_penalty": 1.05})
llm_planner = ChatOllama(model="mistral-nemo", base_url="http://localhost:11434", temperature=0)

# =========================
# 3) Orchestrator (simple, deterministic)
# =========================
PLANNER_SYS = "Return a Python list of tool names to run for crop health monitoring. No prose."

def orchestrator(state: AgentState) -> AgentState:
    """
    Decide which tool steps to run. Keep it simple & fast.
    """
    try:
        # Minimal prompt (we keep a deterministic default too)
        prompt = [
            {"role":"system","content":PLANNER_SYS},
            {"role":"user","content":"Tools: ['compute_indices','pest_assess','weed_detect','water_stress','nutrient_estimator']"}
        ]
        msg = llm_planner.invoke(prompt)
        try:
            plan = [s.strip().strip('"\'') for s in eval(msg.content)]
            if not isinstance(plan, list) or not all(isinstance(x,str) for x in plan):
                raise ValueError
        except Exception:
            plan = ["compute_indices","pest_assess","weed_detect","water_stress","nutrient_estimator"]
    except Exception:
        plan = ["compute_indices","pest_assess","weed_detect","water_stress","nutrient_estimator"]

    state["plan"] = plan
    state.setdefault("artifacts", {})
    state["status"] = "ok"
    return state

# =========================
# 4) Tools (EE indices + placeholders for others)
# =========================

def tool_compute_indices(state: AgentState) -> AgentState:
    """
    GEE Sentinel-2 indices (NDVI/NDRE/NDWI) stats for the farm buffer
    """
    try:
        lat, lon, radius = state["latitude"], state["longitude"], state["radius_m"]
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30)

        farm_point = ee.Geometry.Point([lon, lat])
        farm_area = farm_point.buffer(radius)

        collection = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(farm_area)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort("CLOUDY_PIXEL_PERCENTAGE"))

        image_count = collection.size().getInfo()
        if image_count == 0:
            state["artifacts"]["compute_indices"] = {"error":"No suitable images in period"}
            return state

        img = collection.median()

        # NDVI = (B8 - B4)/(B8 + B4)
        ndvi = img.normalizedDifference(["B8","B4"]).rename("NDVI")
        # NDRE = (B8 - B5)/(B8 + B5)  (B5 red-edge)
        ndre = img.normalizedDifference(["B8","B5"]).rename("NDRE")
        # NDWI (Gao) ~ (B3 - B8)/(B3 + B8)
        ndwi = img.normalizedDifference(["B3","B8"]).rename("NDWI")

        stack = ndvi.addBands(ndre).addBands(ndwi)

        # Stats
        reducer = (ee.Reducer.mean()
                    .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
                    .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True))

        stats = stack.reduceRegion(reducer=reducer, geometry=farm_area, scale=10, maxPixels=1e9).getInfo()

        def pick(prefix: str, key: str) -> float:
            return float(stats.get(f"{prefix}_{key}", 0) or 0)

        out = {
            "period": {"start": str(start_date), "end": str(end_date)},
            "image_count": image_count,
            "area_ha": float(3.14159 * (radius/1000.0)**2),
            "NDVI": {
                "mean": pick("NDVI","mean"),
                "min":  pick("NDVI","min"),
                "max":  pick("NDVI","max"),
                "std":  pick("NDVI","stdDev"),
            },
            "NDRE": {
                "mean": pick("NDRE","mean"),
                "min":  pick("NDRE","min"),
                "max":  pick("NDRE","max"),
                "std":  pick("NDRE","stdDev"),
            },
            "NDWI": {
                "mean": pick("NDWI","mean"),
                "min":  pick("NDWI","min"),
                "max":  pick("NDWI","max"),
                "std":  pick("NDWI","stdDev"),
            },
        }
        state["artifacts"]["compute_indices"] = out
        return state
    except Exception as e:
        state["artifacts"]["compute_indices"] = {"error": f"indices failed: {e}"}
        return state

def tool_pest_assess(state: AgentState) -> AgentState:
    """
    Placeholder: Replace with YOLO on trap-cam images → counts per class.
    For now, we pass through any provided counts from context to keep JSON small.
    """
    counts = state["context"].get("pest_counts", [])  # [{"class":"aphid","n":12,"ts":"..."}]
    out = {"counts": counts, "note":"placeholder"}
    state["artifacts"]["pest_assess"] = out
    return state

def tool_weed_detect(state: AgentState) -> AgentState:
    """
    Placeholder: Replace with U-Net/SegFormer on UAV orthos → weed mask & cover %.
    For now, we optionally accept a precomputed percent from context.
    """
    weed_cover_pct = state["context"].get("weed_cover_pct")
    out = {"weed_cover_pct": weed_cover_pct, "mask_path": state["context"].get("weed_mask_path"), "note":"placeholder"}
    state["artifacts"]["weed_detect"] = out
    return state

def tool_water_stress(state: AgentState) -> AgentState:
    """
    Water stress score from NDWI mean (proxy) + optional VPD from context.
    Higher NDWI usually => more water; so stress ~ inverse of NDWI (simple heuristic)
    """
    idx = state["artifacts"].get("compute_indices", {})
    ndwi_mean = idx.get("NDWI",{}).get("mean", None)
    vpd = state["context"].get("weather",{}).get("vpd")  # optional

    if ndwi_mean is None:
        out = {"error":"no NDWI available"}
    else:
        # Simple proxy score in [0..1], higher = more stress
        # map NDWI mean [-0.2..0.4] to stress [1..0] and blend with VPD
        clamped = max(-0.2, min(0.4, ndwi_mean))
        base = 1.0 - ((clamped + 0.2) / 0.6)  # NDWI -0.2 =>1.0 stress, 0.4 =>0.0 stress
        if isinstance(vpd,(int,float)):
            # normalize typical VPD [0..3] into [0..1] and blend
            vpd_norm = max(0.0, min(1.0, vpd/3.0))
            stress = 0.6*base + 0.4*vpd_norm
        else:
            stress = base
        out = {"stress_score": round(float(stress),3), "hotspots":[]}
    state["artifacts"]["water_stress"] = out
    return state

def tool_nutrient_estimator(state: AgentState) -> AgentState:
    """
    Nutrient stress proxy from NDRE mean (rough N proxy) + phenology.
    Lower NDRE often hints at N deficiency; this is a simple heuristic score.
    """
    idx = state["artifacts"].get("compute_indices", {})
    ndre_mean = idx.get("NDRE",{}).get("mean", None)
    das = state["stage_das"]

    if ndre_mean is None:
        out = {"error":"no NDRE available"}
    else:
        # Map NDRE mean [0.1..0.6] to N sufficiency [0..1], then invert => stress
        clamped = max(0.1, min(0.6, ndre_mean))
        suff = (clamped - 0.1) / 0.5      # 0..1
        stress = 1.0 - suff               # 1 = high stress (deficiency)
        # Early stages less reliable
        if das < 20:
            stress *= 0.7
        out = {
            "N_stress_score": round(float(stress),3),
            "notes":"proxy from NDRE mean; validate with SPAD/tissue test"
        }
    state["artifacts"]["nutrient_estimator"] = out
    return state

# =========================
# 5) Brain (LLM synthesis)
# =========================
BRAIN_SYS = """You are an agronomy expert.
Return a compact JSON array of issues:
[
  {"type":"weed|disease|pest|water|nutrient",
   "severity":0-3,
   "confidence":0-1,
   "evidence":[<max 3 short strings>],
   "actions":[<max 3 short strings>]
  }
]
No prose, JSON only.
"""

def brain(state: AgentState) -> AgentState:
    idx = state["artifacts"].get("compute_indices", {})
    weeds = state["artifacts"].get("weed_detect", {})
    pests = state["artifacts"].get("pest_assess", {})
    water = state["artifacts"].get("water_stress", {})
    nuts  = state["artifacts"].get("nutrient_estimator", {})

    evidence = {
        "indices": {
            "ndvi_mean": round(idx.get("NDVI",{}).get("mean", 0.0), 3),
            "ndre_mean": round(idx.get("NDRE",{}).get("mean", 0.0), 3),
            "ndwi_mean": round(idx.get("NDWI",{}).get("mean", 0.0), 3),
            "img_count": idx.get("image_count", 0),
        },
        "weeds": {"weed_cover_pct": weeds.get("weed_cover_pct")},
        "pests": {"counts": pests.get("counts", [])[:3]},
        "water": {"stress_score": water.get("stress_score")},
        "nutrients": {"N_stress_score": nuts.get("N_stress_score")},
        "phenology": {"das": state["stage_das"], "crop": state["crop"]}
    }

    prompt = [
        {"role":"system","content":BRAIN_SYS},
        {"role":"user","content":str(evidence)}
    ]

    try:
        msg = llm_brain.invoke(prompt)
        try:
            issues = [Issue(**i) for i in eval(msg.content)]
        except Exception:
            # Safe fallback (very compact default)
            issues = [
                Issue(type="water", severity=1, confidence=0.6, evidence=["NDWI proxy"], actions=["Irrigation check"]),
                Issue(type="nutrient", severity=1, confidence=0.5, evidence=["NDRE proxy"], actions=["SPAD/tissue test"]),
            ]
        state["issues"] = issues
        return state
    except Exception as e:
        state["issues"] = [
            Issue(type="weed", severity=0, confidence=0.4, evidence=["LLM error"], actions=["Manual scout"])
        ]
        return state

# =========================
# 6) Supervisor (guardrails + QA)
# =========================
def supervisor(state: AgentState) -> AgentState:
    issues: List[Issue] = state.get("issues", [])
    das = state["stage_das"]

    # Phenology: cap nutrient confidence early
    if das < 20:
        for i in issues:
            if i.type == "nutrient":
                i.confidence = min(i.confidence, 0.4)
                i.actions.append("Re-check after 5–7 DAS or tissue test.")

    # If very low image_count or missing indices → lower confidence on all
    idx = state["artifacts"].get("compute_indices", {})
    if idx.get("image_count", 0) < 1 or "error" in idx:
        for i in issues:
            i.confidence = min(i.confidence, 0.4)
            i.actions.append("Collect more cloud-free imagery.")

    state["status"] = "needs_followup" if any(i.confidence < 0.45 for i in issues) else "ok"
    return state

# =========================
# 7) Memory (persist & final report)
# =========================
def memory_and_report(state: AgentState) -> AgentState:
    # Build a final report similar to your earlier format
    idx = state["artifacts"].get("compute_indices", {})
    ndvi = idx.get("NDVI", {})
    ndvi_mean = float(ndvi.get("mean", 0.0) or 0.0)
    ndvi_std  = float(ndvi.get("std", 0.0) or 0.0)

    # Health score mapping (your earlier mapping)
    if ndvi_mean < 0:
        score = 0
    elif ndvi_mean < 0.2:
        score = int(ndvi_mean * 250)
    elif ndvi_mean < 0.6:
        score = int(50 + (ndvi_mean - 0.2) * 125)
    else:
        score = min(100, int(80 + (ndvi_mean - 0.6) * 50))

    if score >= 80:
        status = "Excellent"; emoji = "🟢"
    elif score >= 60:
        status = "Good"; emoji = "🟡"
    elif score >= 40:
        status = "Fair"; emoji = "🟠"
    else:
        status = "Poor"; emoji = "🔴"

    # Auto recommendations similar to your logic + issues-based actions
    recs = []
    if ndvi_mean < 0.3:
        recs.extend([
            "🚨 Low vegetation health detected - investigate potential stress factors",
            "💧 Check soil moisture levels and irrigation systems",
            "🐛 Scout for pest or disease issues"
        ])
    if ndvi_mean < 0.5:
        recs.append("🌱 Consider nutrient assessment - crops may benefit from fertilization")
    if ndvi_std > 0.15:
        recs.extend([
            "📊 High variability detected - some areas performing better than others",
            "🎯 Consider zone-specific management strategies"
        ])
    if ndvi_mean > 0.7:
        recs.append("✅ Excellent crop health - maintain current practices")
    if not recs:
        recs.append("📈 Crop health is stable - continue monitoring")

    # Merge in top actions from issues (dedup)
    for i in state["issues"]:
        recs.extend(i.actions)
    # Deduplicate while keeping order
    seen = set()
    recs = [x for x in recs if not (x in seen or seen.add(x))]

    state["finished_at"] = datetime.utcnow().isoformat()

    report = {
        "field_id": state["field_id"],
        "farm_name": state["field_id"],
        "analysis_timestamp": state["finished_at"],
        "location": {"latitude": state["latitude"], "longitude": state["longitude"]},
        "health_summary": {"overall_score": score, "status": status, "status_emoji": emoji},
        "ndvi_analysis": {
            "mean": round(ndvi_mean,3),
            "min": round(float(ndvi.get("min",0.0) or 0.0),3),
            "max": round(float(ndvi.get("max",0.0) or 0.0),3),
            "variability": round(ndvi_std,3)
        },
        "data_quality": {
            "images_used": idx.get("image_count", 0),
            "analysis_period_days": 30,
            "area_analyzed_hectares": round(float(idx.get("area_ha", 0.0) or 0.0), 1),
            "period": idx.get("period", {})
        },
        "issues": [i.dict() for i in state["issues"]],
        "recommendations": recs,
        "next_analysis": (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "artifacts": {k: v for k, v in state["artifacts"].items() if k != "compute_indices"}  # keep light
    }

    state["artifacts"]["final_report"] = report
    return state

# =========================
# 8) Build the LangGraph
# =========================
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("compute_indices", tool_compute_indices)
    graph.add_node("pest_assess", tool_pest_assess)
    graph.add_node("weed_detect", tool_weed_detect)
    graph.add_node("water_stress", tool_water_stress)
    graph.add_node("nutrient_estimator", tool_nutrient_estimator)
    graph.add_node("brain", brain)
    graph.add_node("supervisor", supervisor)
    graph.add_node("memory", memory_and_report)

    graph.set_entry_point("orchestrator")
    # Deterministic pipeline; orchestrator sets plan but we run all (cheap) tools
    graph.add_edge("orchestrator", "compute_indices")
    graph.add_edge("compute_indices", "pest_assess")
    graph.add_edge("pest_assess", "weed_detect")
    graph.add_edge("weed_detect", "water_stress")
    graph.add_edge("water_stress", "nutrient_estimator")
    graph.add_edge("nutrient_estimator", "brain")
    graph.add_edge("brain", "supervisor")
    graph.add_edge("supervisor", "memory")
    graph.add_edge("memory", END)
    return graph.compile()

# =========================
# 9) CLI entry
# =========================
def run_once(field_id: str, crop: str, lat: float, lon: float, radius_m: int = 1000, das: int = 28,
             context: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    if context is None:
        context = {}
    init_ee()
    app = build_graph()
    init_state: AgentState = {
        "field_id": field_id,
        "crop": crop,
        "stage_das": das,
        "latitude": lat,
        "longitude": lon,
        "radius_m": radius_m,
        "context": context,
        "plan": [],
        "artifacts": {},
        "issues": [],
        "status": "ok",
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None
    }
    t0 = time.time()
    final_state = app.invoke(init_state)
    dt = time.time() - t0
    report = final_state["artifacts"].get("final_report", {})
    print(f"\n=== Run finished in {dt:.2f}s | status={final_state['status']} ===")
    return report

if __name__ == "__main__":
    # Example: Iowa test (your previous coordinates)
    report = run_once(
        field_id="Iowa_Test_Farm",
        crop="corn",
        lat=41.878003,
        lon=-93.097702,
        radius_m=1000,
        das=28,
        context={
            # Optional hints while stubs are in place:
            "pest_counts": [{"class":"aphid","n":34,"ts":"2025-08-19"}],
            "weed_cover_pct": 18.2,
            "weather": {"vpd": 1.8}
        }
    )
    if report:
        print(f"\n✅ {report['health_summary']['status_emoji']} {report['farm_name']}: "
              f"Score {report['health_summary']['overall_score']}/100  | "
              f"NDVI mean {report['ndvi_analysis']['mean']}  | "
              f"Images {report['data_quality']['images_used']}")
        print("\nTop recommendations:")
        for r in report["recommendations"][:5]:
            print(" -", r)
    else:
        print("❌ No report produced (check EE auth / imagery availability).")
