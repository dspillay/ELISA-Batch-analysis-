# ELISA-Batch-analysis-
# .Py Script runs in RStudio

# 📌📖 READ ME 
# to run: library(reticulate) -> source_python("name_of_file.py") # 🚨save .Py file ahead of analysis
#         or, reticulate::source_python("name_of_file.py")        # 🚨update file name
#        🚨🚨 save file after changing toggle settings and execution block before running
# ATT: edit the Execution block to match file names
# ATT: toggle features include (1) blank correction is optional
#                              (2) AIC test is optional
#                              (3) force a particular model - in such a case, AIC must be toggled off
#                              (4) sample duplication correction is optional
# script looks for duplicate standards (post standard curve, pre-individual sample processing)
# script inverts OD values to concentration - ideal for competitive ELISA analysis
# for sandwich/direct ELISA: switch the toggle
# script includes confidence intervals (5%, 95%) and detection limits for standard curves


#!/usr/bin/env python3
"""
name_of_file.py          # 🚨add .Py script file name
Full ELISA pipeline with:
 - optional blank correction per-plate (APPLY_BLANK_CORRECTION)
 - AIC toggle (APPLY_AIC). Option A behavior when APPLY_AIC=False:
     * Use FORCED_MODEL to pick a single forced model ("3pl"/"4pl"/"5pl")
     * Generate unified curves only for that forced model (no extra comparison curves)
 - When APPLY_AIC=True the script behaves as before: selects BEST models by AIC and
   also produces forced-model comparison unified plots (3PL/4PL/5PL) for visual comparison.
 - Produces unified reference-only plots, includes R² on plots and sample final concentration range annotation.
 - Saves per-plate outputs and unified summaries (CSV)
 - NEW: Sample duplicate analysis toggle (SAMPLES_IN_DUPLICATE) with 10% CV warning

Note: ELISAtools vignette (uploaded):
    /mnt/data/elisatools-vignette.pdf
"""
import os
import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, root_scalar
from plotnine import ggplot, aes, geom_point, geom_line, labs, theme_bw
import traceback

VIGNETTE_PATH = "/mnt/data/elisatools-vignette.pdf"

# -------------------------------
# USER TOGGLES
# -------------------------------
APPLY_BLANK_CORRECTION = True    # subtract mean blanks per plate
APPLY_AIC = False                # if True use AIC to choose BEST models; if False use FORCED_MODEL
FORCED_MODEL = "3pl"             # only used when APPLY_AIC == False; one of "3pl","4pl","5pl"
SAMPLES_IN_DUPLICATE = True      # If True → average OD across duplicates per sample ID for summary analysis
ASSAY_TYPE = "sandwich"          # change to "sandwich" for sandwich ELISA data / "competitive"

#🚨save file before running

#source_python("name_of_file.py")  # 🚨update file name

# -------------------------------
# Models: 3PL, 4PL, 5PL (x = log concentration)
# -------------------------------

def five_pl(x, a, d, xmid, scal, g):
    return a + (d - a) / (1.0 + np.exp((xmid - x) / scal)) ** g


def four_pl(x, a, d, xmid, scal):
    return a + (d - a) / (1.0 + np.exp((xmid - x) / scal))


def three_pl(x, d, xmid, scal):
    a = 0.0
    return a + (d - a) / (1.0 + np.exp((xmid - x) / scal))


def calc_aic(k: int, ll: float) -> float:
    return 2 * k - 2 * ll


def log_likelihood(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_obs)
    residuals = y_obs - y_pred
    var = np.nanvar(residuals)
    if var <= 0 or not np.isfinite(var):
        return -1e6
    return -n / 2 * np.log(2 * math.pi) - n / 2 * np.log(var) - 1 / (2 * var) * np.nansum(residuals ** 2)
  

# -------------------------------
# NEW: Slope equation calculation and display
# -------------------------------

def calculate_slope_equation(params: np.ndarray, model_name: str, unit: str) -> str:
    """
    Calculate the slope equation parameters for the fitted curve.
    Returns a human-readable equation string.
    """
    if not np.isfinite(params).all():
        return "Unable to calculate slope equation - invalid parameters"
    
    a, d, xmid, scal, g = params
    
    if model_name.upper() == "3PL":
        # For 3PL: y = d / (1 + exp((xmid - x)/scal))
        return f"y = {d:.4f} / (1 + exp(({xmid:.4f} - x)/{scal:.4f}))"
    
    elif model_name.upper() == "4PL":
        # For 4PL: y = a + (d - a) / (1 + exp((xmid - x)/scal))
        return f"y = {a:.4f} + ({d:.4f} - {a:.4f}) / (1 + exp(({xmid:.4f} - x)/{scal:.4f}))"
    
    else:  # 5PL
        # For 5PL: y = a + (d - a) / (1 + exp((xmid - x)/scal))^g
        return f"y = {a:.4f} + ({d:.4f} - {a:.4f}) / (1 + exp(({xmid:.4f} - x)/{scal:.4f}))^{g:.4f}"

def calculate_slope_at_point(params: np.ndarray, conc: float) -> float:
    """
    Calculate the slope (dy/dx) at a specific concentration point.
    Uses numerical differentiation.
    """
    if not np.isfinite(params).all() or conc <= 0:
        return float('nan')
    
    # Small perturbation for numerical differentiation
    h = 1e-6
    log_conc = np.log(conc)
    
    # Calculate function values at log_conc ± h
    y1 = five_pl(log_conc - h, *params)
    y2 = five_pl(log_conc + h, *params)
    
    # Numerical derivative: dy/dx = (y2 - y1) / (2h)
    # Note: This is d(OD)/d(log(conc))
    slope = (y2 - y1) / (2 * h)
    
    return slope

def print_slope_analysis(params: np.ndarray, model_name: str, unit: str, conc_range: Tuple[float, float]):
    """
    Print comprehensive slope analysis for the fitted curve.
    """
    if not np.isfinite(params).all():
        print("  Cannot calculate slope analysis - invalid parameters")
        return
    
    print(f"\n📐 SLOPE ANALYSIS for {model_name}:")
    print(f"  Equation: {calculate_slope_equation(params, model_name, unit)}")
    print(f"  Where: x = log(concentration in {unit}), y = OD")
    
    # Calculate slopes at key concentration points
    min_conc, max_conc = conc_range
    mid_conc = np.exp((np.log(min_conc) + np.log(max_conc)) / 2)
    
    key_points = [
        ("Minimum", min_conc),
        ("25%", np.exp(np.log(min_conc) + 0.25 * (np.log(max_conc) - np.log(min_conc)))),
        ("Midpoint", mid_conc),
        ("75%", np.exp(np.log(min_conc) + 0.75 * (np.log(max_conc) - np.log(min_conc)))),
        ("Maximum", max_conc)
    ]
    
    print(f"  Slopes at key concentrations:")
    for label, conc in key_points:
        slope = calculate_slope_at_point(params, conc)
        if np.isfinite(slope):
            print(f"    {label} ({conc:.3f} {unit}): {slope:.4f} OD/log({unit})")
    
    # Find maximum slope (steepest point)
    log_min, log_max = np.log(min_conc), np.log(max_conc)
    test_points = np.linspace(log_min, log_max, 100)
    max_slope = -np.inf
    max_slope_conc = float('nan')
    
    for log_conc in test_points:
        conc = np.exp(log_conc)
        slope = calculate_slope_at_point(params, conc)
        if np.isfinite(slope) and abs(slope) > abs(max_slope):
            max_slope = slope
            max_slope_conc = conc
    
    if np.isfinite(max_slope):
        print(f"  Maximum slope: {max_slope:.4f} OD/log({unit}) at {max_slope_conc:.3f} {unit}")
    
    # Parameter interpretation
    a, d, xmid, scal, g = params
    print(f"\n  PARAMETER INTERPRETATION:")
    print(f"    Lower asymptote (a): {a:.4f} OD")
    print(f"    Upper asymptote (d): {d:.4f} OD") 
    print(f"    Inflection point (xmid): {np.exp(xmid):.4f} {unit}")
    print(f"    Slope factor (scal): {scal:.4f}")
    if model_name.upper() == "5PL":
        print(f"    Asymmetry factor (g): {g:.4f}")


# -------------------------------
# Confidence interval calculation
# -------------------------------

def calculate_confidence_intervals(concs: np.ndarray, ods: np.ndarray, params: np.ndarray, 
                                 model_func, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for the fitted curve using parametric bootstrap.
    Returns lower_ci, upper_ci for the curve predictions.
    """
    try:
        valid = np.isfinite(concs) & np.isfinite(ods) & (ods > 0)
        concs_v = np.clip(concs[valid].astype(float), 1e-12, None)
        ods_v = ods[valid].astype(float)
        x = np.log(concs_v)
        
        # Generate predictions on log scale
        pred = model_func(x, *params)
        residuals = ods_v - pred
        n = len(residuals)
        
        # Parametric bootstrap
        n_bootstrap = 1000
        bootstrap_predictions = []
        
        # Generate grid for predictions
        conc_grid = np.logspace(np.log10(concs_v.min()), np.log10(concs_v.max()), 100)
        log_grid = np.log(conc_grid)
        
        for _ in range(n_bootstrap):
            # Resample residuals with replacement
            bootstrap_residuals = np.random.choice(residuals, size=n, replace=True)
            bootstrap_ods = pred + bootstrap_residuals
            
            try:
                # Refit model to bootstrap data
                if len(params) == 5:  # 5PL
                    p0 = [np.nanmin(bootstrap_ods), np.nanmax(bootstrap_ods), 
                          np.median(x), 1.0, 1.0]
                    bounds = ([0, 0, x.min()-1, 0.01, 0.5], 
                             [np.nanmax(bootstrap_ods)*10, np.nanmax(bootstrap_ods)*10, 
                              x.max()+1, 50.0, 10.0])
                    bootstrap_params, _ = curve_fit(five_pl, x, bootstrap_ods, p0=p0, bounds=bounds, maxfev=5000)
                elif len(params) == 4:  # 4PL
                    p0 = [np.nanmin(bootstrap_ods), np.nanmax(bootstrap_ods), 
                          np.median(x), 1.0]
                    bounds = ([0, 0, x.min()-1, 0.01], 
                             [np.nanmax(bootstrap_ods)*10, np.nanmax(bootstrap_ods)*10, 
                              x.max()+1, 50.0])
                    bootstrap_params, _ = curve_fit(four_pl, x, bootstrap_ods, p0=p0, bounds=bounds, maxfev=5000)
                else:  # 3PL
                    p0 = [np.nanmax(bootstrap_ods), np.median(x), 1.0]
                    bounds = ([0, x.min()-1, 0.01], 
                             [np.nanmax(bootstrap_ods)*10, x.max()+1, 50.0])
                    bootstrap_params, _ = curve_fit(three_pl, x, bootstrap_ods, p0=p0, bounds=bounds, maxfev=5000)
                    bootstrap_params = np.array([0.0, float(bootstrap_params[0]), 
                                               float(bootstrap_params[1]), float(bootstrap_params[2]), 1.0])
                
                # Predict on grid
                bootstrap_pred = five_pl(log_grid, *bootstrap_params)
                bootstrap_predictions.append(bootstrap_pred)
                
            except Exception:
                continue
        
        if len(bootstrap_predictions) > 0:
            bootstrap_predictions = np.array(bootstrap_predictions)
            lower_ci = np.percentile(bootstrap_predictions, (alpha/2)*100, axis=0)
            upper_ci = np.percentile(bootstrap_predictions, (1-alpha/2)*100, axis=0)
            return lower_ci, upper_ci
        else:
            return np.full_like(log_grid, np.nan), np.full_like(log_grid, np.nan)
            
    except Exception as e:
        print(f"Warning: Could not calculate confidence intervals: {e}")
        conc_grid = np.logspace(np.log10(concs.min()), np.log10(concs.max()), 100)
        return np.full_like(conc_grid, np.nan), np.full_like(conc_grid, np.nan)

# -------------------------------
# Detection limit calculation
# -------------------------------

def calculate_detection_limits(params: np.ndarray, blank_ods: np.ndarray, 
                             confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate LOD (Limit of Detection) and LOQ (Limit of Quantification) based on blank measurements.
    LOD = mean_blank + 3*std_blank, LOQ = mean_blank + 10*std_blank
    Returns both concentration and OD values.
    """
    try:
        if len(blank_ods) == 0 or not np.isfinite(blank_ods).any():
            return {"LOD": float("nan"), "LOQ": float("nan"), 
                    "LOD_OD": float("nan"), "LOQ_OD": float("nan")}
        
        mean_blank = np.nanmean(blank_ods)
        std_blank = np.nanstd(blank_ods)
        
        if std_blank <= 0 or not np.isfinite(std_blank):
            return {"LOD": float("nan"), "LOQ": float("nan"),
                    "LOD_OD": float("nan"), "LOQ_OD": float("nan")}
        
        # Calculate LOD and LOQ in OD units (HOD and HOQ)
        lod_od = mean_blank + 3 * std_blank
        loq_od = mean_blank + 10 * std_blank
        
        # Convert to concentration units using the inverse function
        conc_min = 1e-12
        conc_max = 1e6  # Reasonable upper limit
        
        def od_to_conc_safe(od, params, conc_min, conc_max):
            try:
                return od_to_conc(od, params, conc_min, conc_max)
            except:
                return np.nan
        
        lod_conc = od_to_conc_safe(lod_od, params, conc_min, conc_max)
        loq_conc = od_to_conc_safe(loq_od, params, conc_min, conc_max)
        
        return {
            "LOD": float(lod_conc) if np.isfinite(lod_conc) else float("nan"),
            "LOQ": float(loq_conc) if np.isfinite(loq_conc) else float("nan"),
            "LOD_OD": float(lod_od),
            "LOQ_OD": float(loq_od)
        }
        
    except Exception as e:
        print(f"Warning: Could not calculate detection limits: {e}")
        return {"LOD": float("nan"), "LOQ": float("nan"), 
                "LOD_OD": float("nan"), "LOQ_OD": float("nan")}

# -------------------------------
# I/O helpers
# -------------------------------

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df


def ensure_columns(df: pd.DataFrame, required: List[str], fname: str):
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in file: {fname}")


def load_standards(file_path: str) -> Tuple[Dict[str, float], str]:
    df = pd.read_csv(file_path)
    df = clean_headers(df)
    conc_col = None
    for col in df.columns:
        if col.lower().startswith("concentration"):
            conc_col = col
            break
    if conc_col is None:
        raise ValueError("Standards file must contain a column starting with 'Concentration'")

    unit = conc_col.split("_",1)[1] if "_" in conc_col else "unknown"

    if "Standard" not in df.columns:
        # support fallback two-column style
        if df.shape[1] >= 2:
            first, second = df.columns[0], df.columns[1]
            try:
                pd.to_numeric(df[second])
                std_map = dict(zip(df[first].astype(str).str.strip(), df[second].astype(float)))
                return std_map, unit
            except Exception:
                pass
        raise ValueError("Standards file must contain a 'Standard' column or at least two columns (ID, conc)")

    std_map = dict(zip(df["Standard"].astype(str).str.strip(), df[conc_col].astype(float)))
    return std_map, unit


def parse_plate_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = clean_headers(df)
    ensure_columns(df, ["Well.Row", "Well.Col", "Content", "OD", "Dilution"], file_path)

    df["Content"] = df["Content"].astype(str).str.strip()
    df["Type"] = df["Content"].str.lower().apply(lambda x: "Standard" if "standard" in x else ("Blank" if "blank" in x else "Sample"))
    def _extract_id(x):
        parts = x.strip().split()
        if len(parts) >= 1:
            return parts[-1]
        return x
    df["ID"] = df["Content"].apply(lambda x: _extract_id(x) if isinstance(x, str) else x)
    df["Well"] = df["Well.Row"].astype(str) + df["Well.Col"].astype(str)
    df["OD"] = pd.to_numeric(df["OD"], errors="coerce")
    df["Dilution"] = pd.to_numeric(df["Dilution"], errors="coerce").fillna(1)
    return df[["Well","Well.Row","Well.Col","Content","Type","ID","OD","Dilution"]]
  

# -------------------------------
# NEW: Curve direction detection
# -------------------------------

def detect_curve_direction(concs: np.ndarray, ods: np.ndarray) -> str:
    """Detect if data follows standard (increasing) or competitive (decreasing) pattern"""
    valid = np.isfinite(concs) & np.isfinite(ods)
    if np.sum(valid) < 2:
        return "unknown"
    
    correlation = np.corrcoef(concs[valid], ods[valid])[0, 1]
    
    if correlation > 0.3:
        return "sandwich"
    elif correlation < -0.3:
        return "competitive"
    else:
        return "ambiguous"  
  

# -------------------------------
# Fit models (return 5-length params for consistency)
# -------------------------------

def fit_model(concs: np.ndarray, ods: np.ndarray, model: str, assay_type: str = "competitive", plate_idx=None) -> Tuple[np.ndarray, float]:
    valid = np.isfinite(concs) & np.isfinite(ods) & (ods > 0)
    concs_v = np.asarray(concs)[valid].astype(float)
    ods_v = np.asarray(ods)[valid].astype(float)
    if len(concs_v) < 4:
        return np.array([np.nan]*5), -1e6

    x = np.log(np.clip(concs_v, 1e-12, None))
    y = ods_v

    # NEW: DEBUG CODE 
    if len(concs_v) >= 2:
        correlation = np.corrcoef(concs_v, ods_v)[0, 1] if len(concs_v) > 1 else 0
        
        # DEBUG: Track where competitive ELISA warnings come from
        if assay_type.lower() == "competitive" and correlation > 0.2:
            print(f"\n🔍 DEBUG WARNING SOURCE:")
            import traceback
            traceback.print_stack(limit=3)  # Show last 3 calls in stack
            print(f"  Calling context: assay_type='{assay_type}', correlation={correlation:.3f}")
            print(f"  Model being fitted: {model}")
            
     # DIAGNOSTIC CODE - 29 JUNE 2026
            print("\n========== DIAGNOSTIC ==========")
            print(f"Dataset : {plate_idx}")
            print(f"Model   : {model}")
            print(f"Assay   : {assay_type}")
            print(f"Correlation : {correlation:.3f}")
            print("Concentrations:")
            print(concs_v)
            print("OD values:")
            print(ods_v)
            print("===============================\n")
            
            # ADDITIONAL CHECK - NOT DIAGNOSTIC, JUST QUALITY CHECK
            print(f"\n===== FITTING =====")
            print(f"Plate      : {plate_idx}")
            print(f"Model      : {model}")
            print(f"Assay type : {assay_type}")
            print(f"Correlation: {correlation:.3f}")
            print("===================")
    
    # THEN YOUR EXISTING WARNING LOGIC FOLLOWS:
    # FIXED LOGIC:
    if assay_type.lower() == "sandwich":
        # Sandwich ELISA should have POSITIVE correlation (increasing)
        if correlation < -0.2:  # Strong negative correlation
            print(f"  ⚠ WARNING: Data shows NEGATIVE correlation (r={correlation:.3f})")
            print(f"     Expected for SANDWICH ELISA: Positive correlation (OD increases with concentration)")
            print(f"     Your data shows: OD decreases with concentration")
            print(f"     This suggests COMPETITIVE ELISA pattern!")
        elif correlation > 0.2:  # Strong positive correlation - THIS IS CORRECT!
            print(f"  ✓ Data shows correct SANDWICH ELISA pattern (r={correlation:.3f})")
            print(f"     OD increases with concentration as expected")
            
    elif assay_type.lower() == "competitive":
        # Competitive ELISA should have NEGATIVE correlation (decreasing)
        if correlation > 0.2:  # Strong positive correlation
            print(f"  ⚠ WARNING: Data shows POSITIVE correlation (r={correlation:.3f})")
            print(f"     Expected for COMPETITIVE ELISA: Negative correlation (OD decreases with concentration)")
            print(f"     Your data shows: OD increases with concentration")
            print(f"     This suggests SANDWICH ELISA pattern!")
        elif correlation < -0.2:  # Strong negative correlation - THIS IS CORRECT!
            print(f"  ✓ Data shows correct COMPETITIVE ELISA pattern (r={correlation:.3f})")
            print(f"     OD decreases with concentration as expected")

    try:
        if model.lower() == "5pl":
            p0 = [np.nanmin(y), np.nanmax(y), np.median(x), 1.0, 1.0]
            bounds = ([0, 0, x.min()-1, 0.01, 0.5], [np.nanmax(y)*10, np.nanmax(y)*10, x.max()+1, 50.0, 10.0])
            params, _ = curve_fit(five_pl, x, y, p0=p0, bounds=bounds, maxfev=20000)
            return params, log_likelihood(y, five_pl(x, *params))

        if model.lower() == "4pl":
            p0 = [np.nanmin(y), np.nanmax(y), np.median(x), 1.0]
            bounds = ([0, 0, x.min()-1, 0.01], [np.nanmax(y)*10, np.nanmax(y)*10, x.max()+1, 50.0])
            params4, _ = curve_fit(four_pl, x, y, p0=p0, bounds=bounds, maxfev=20000)
            params5 = np.append(params4, 1.0)
            return params5, log_likelihood(y, four_pl(x, *params4))

        if model.lower() == "3pl":
            p0 = [np.nanmax(y), np.median(x), 1.0]
            bounds = ([0, x.min()-1, 0.01], [np.nanmax(y)*10, x.max()+1, 50.0])
            params3, _ = curve_fit(three_pl, x, y, p0=p0, bounds=bounds, maxfev=20000)
            params5 = np.array([0.0, float(params3[0]), float(params3[1]), float(params3[2]), 1.0])
            return params5, log_likelihood(y, three_pl(x, *params3))
    except Exception:
        return np.array([np.nan]*5), -1e6
      
      # NEW CODE - 29 JUNE
      
    print(params4)


# -------------------------------
# Select best model among 3PL/4PL/5PL (AIC)
# -------------------------------

#def select_best_model(concs: np.ndarray, ods: np.ndarray, assay_type: str = "competitive", plate_idx=plate_idx) -> Tuple[str, np.ndarray, Dict]:
def select_best_model(concs: np.ndarray, 
                      ods: np.ndarray,
                      assay_type: str = "competitive",
                      plate_idx=None) -> Tuple[str, np.ndarray, Dict]: 
                        results = {} 
                        for model, k in [("3pl",3), ("4pl",4), ("5pl",5)]:
                          params, ll = fit_model(concs, ods, model, assay_type=assay_type, plate_idx=plate_idx)
                          #params, ll = fit_model(concs, ods, model, )
                          if np.isfinite(ll) and not np.isnan(params).any():
                            results[model] = {"params": params, "ll": ll, "AIC": calc_aic(k,ll), "k": k}
                            if not results:
                              return "5PL", np.array([np.nan]*5), {}
                            best = min(results.keys(), key=lambda m: results[m]["AIC"])
                            return best.upper(), results[best]["params"], results


# -------------------------------
# QC & detection limits
# -------------------------------

def calculate_qc(concs: np.ndarray, ods: np.ndarray, params: np.ndarray) -> Dict:
    valid = np.isfinite(concs) & np.isfinite(ods) & (ods > 0)
    if np.sum(valid) == 0:
        return {"R_squared": float("nan"), "LLOQ": float("nan"), "ULOQ": float("nan"), 
                "Min_Conc": float("nan"), "Max_Conc": float("nan"), 
                "Min_OD": float("nan"), "Max_OD": float("nan")}
    
    concs_v = np.clip(concs[valid].astype(float), 1e-12, None)
    ods_v = ods[valid].astype(float)
    x = np.log(concs_v)
    
    # Calculate predictions
    pred = five_pl(x, *params)
    res = ods_v - pred
    
    # MORE ROBUST R² CALCULATION
    # Method 1: Direct formula (most reliable)
    ss_res = np.nansum(res ** 2)
    ss_tot = np.nansum((ods_v - np.nanmean(ods_v)) ** 2)
    
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = float("nan")
    
    # Alternative check: correlation-based R²
    # r2_corr = np.corrcoef(ods_v, pred)[0, 1] ** 2
    
    # Debug output for problematic cases
    if not np.isfinite(r2) or r2 < -0.5:
        print(f"  R² DEBUG: ss_res={ss_res:.4f}, ss_tot={ss_tot:.4f}, r2={r2:.4f}")
        print(f"  OD range: {ods_v.min():.4f} to {ods_v.max():.4f}")
        print(f"  Pred range: {pred.min():.4f} to {pred.max():.4f}")
    
    # Rest of your calculations...
    conc_sorted = np.sort(concs_v)
    lloq = float(np.percentile(conc_sorted, 5)) if len(conc_sorted)>0 else float("nan")
    uloq = float(np.percentile(conc_sorted, 95)) if len(conc_sorted)>0 else float("nan")
    
    min_conc = float(np.nanmin(concs_v)) if len(concs_v) > 0 else float("nan")
    max_conc = float(np.nanmax(concs_v)) if len(concs_v) > 0 else float("nan")
    min_od = float(np.nanmin(ods_v)) if len(ods_v) > 0 else float("nan")
    max_od = float(np.nanmax(ods_v)) if len(ods_v) > 0 else float("nan")
    
    return {"R_squared": r2, "LLOQ": lloq, "ULOQ": uloq,
            "Min_Conc": min_conc, "Max_Conc": max_conc,
            "Min_OD": min_od, "Max_OD": max_od}
# -------------------------------
# Inversion OD -> concentration
# -------------------------------

def invert_od_analytic(od: float, params: np.ndarray) -> float:
    try:
        if not np.isfinite(od):
            return np.nan
        a, d, xmid, scal, g = params
        denom = od - a
        if denom == 0:
            return np.nan
        inside = (d - a) / denom
        if inside <= 0 or not np.isfinite(inside):
            return np.nan
        val = inside ** (1.0 / g) - 1.0
        if val <= 0 or not np.isfinite(val):
            return np.nan
        logc = xmid - scal * np.log(val)
        return float(np.exp(logc))
    except Exception:
        return np.nan


def invert_od_numeric(od: float, params: np.ndarray, conc_min: float, conc_max: float) -> float:
    try:
        if not np.isfinite(od) or conc_min <= 0 or conc_max <= 0:
            return np.nan
        def f(logc):
            return five_pl(logc, *params) - od
        a = math.log(max(conc_min, 1e-12))
        b = math.log(max(conc_max, a + 1e-6))
        sol = root_scalar(f, bracket=[a, b], method='bisect', maxiter=200)
        if sol.converged:
            return float(np.exp(sol.root))
        return np.nan
    except Exception:
        return np.nan


def od_to_conc(od: float, params: np.ndarray, conc_min: float, conc_max: float) -> float:
    c = invert_od_analytic(od, params)
    if np.isfinite(c):
        return c
    return invert_od_numeric(od, params, conc_min, conc_max)

# -------------------------------
# Plot helpers (R² & sample range annotation) - UPDATED WITH CI
# -------------------------------

def generate_plot(df_plot: pd.DataFrame, params: np.ndarray, title: str, r2: float=None, unit: str="unknown", 
                 ci_lower: np.ndarray = None, ci_upper: np.ndarray = None, conc_grid: np.ndarray = None):
    df_plot = df_plot.copy()
    if "Type" not in df_plot.columns:
        df_plot["Type"] = "Standard"
    df_plot["Color"] = df_plot["Type"].map({"Standard":"black","Sample":"royalblue", "Blank":"grey"})
    if "Concentration" not in df_plot.columns or df_plot["Concentration"].dropna().empty:
        raise ValueError("No concentration values to plot.")
    concs = df_plot["Concentration"].dropna().astype(float).values
    cmin, cmax = concs.min(), concs.max()
    
    # Use provided grid or create default
    if conc_grid is None:
        grid = np.linspace(cmin, cmax, 300)
    else:
        grid = conc_grid
        
    log_grid = np.log(np.clip(grid, 1e-12, None))
    curve_y = five_pl(log_grid, *params)
    df_curve = pd.DataFrame({"Concentration": grid, "OD": curve_y})
    
    # Add confidence intervals if provided
    if ci_lower is not None and ci_upper is not None and len(ci_lower) == len(grid) and len(ci_upper) == len(grid):
        df_curve["CI_Lower"] = ci_lower
        df_curve["CI_Upper"] = ci_upper
    
    title_full = title if r2 is None else f"{title} | R\u00b2={r2:.3f}"
    
    # Build plot with or without confidence intervals
    plot = (ggplot(df_plot, aes(x="Concentration", y="OD", color="Color"))
            + geom_point(size=3))
    
    if ci_lower is not None and ci_upper is not None:
        plot = (plot 
                + geom_ribbon(data=df_curve, mapping=aes(x="Concentration", ymin="CI_Lower", ymax="CI_Upper"), 
                             alpha=0.2, fill="blue")
                + geom_line(data=df_curve, mapping=aes(x="Concentration", y="OD"), color="red", size=1))
    else:
        plot = plot + geom_line(data=df_curve, mapping=aes(x="Concentration", y="OD"), color="red")
    
    plot = (plot 
            + labs(title=title_full, x=f"Concentration ({unit})", y="OD")
            + theme_bw())
    
    return plot


def plot_unified_curve(pooled_std: pd.DataFrame, ref_params: np.ndarray, model_name: str,
                       plate_params: List[Tuple[int, np.ndarray]], output_dir: str, unit: str,
                       out_name_suffix: str = "", sample_range: Tuple[float,float]=None, r2: float=None,
                       ci_lower: np.ndarray = None, ci_upper: np.ndarray = None):
    os.makedirs(output_dir, exist_ok=True)
    concs = pooled_std["Concentration"].dropna().astype(float).values
    cmin, cmax = concs.min(), concs.max()
    grid = np.linspace(cmin, cmax, 400)
    log_grid = np.log(np.clip(grid, 1e-12, None))

    df_all = []
    df_all.append(pd.DataFrame({"Concentration": grid, "OD": five_pl(log_grid, *ref_params), "Curve": "Reference"}))
    
    # Add confidence intervals if provided
    if ci_lower is not None and ci_upper is not None:
        df_all.append(pd.DataFrame({"Concentration": grid, "OD": ci_lower, "Curve": "CI_Lower"}))
        df_all.append(pd.DataFrame({"Concentration": grid, "OD": ci_upper, "Curve": "CI_Upper"}))
    
    for plate_idx, params in plate_params:
        if np.isfinite(params).all():
            df_all.append(pd.DataFrame({"Concentration": grid, "OD": five_pl(log_grid, *params), "Curve": f"Plate {plate_idx}"}))
    curves = pd.concat(df_all, ignore_index=True)

    title = f"Unified Standard Curve ({model_name})"
    if r2 is not None and np.isfinite(r2):
        title = title + f" | R\u00b2={r2:.3f}"

    caption = None
    if sample_range is not None and not (np.isnan(sample_range[0]) or np.isnan(sample_range[1])):
        caption = f"Sample final concentration range: {sample_range[0]:.3g} - {sample_range[1]:.3g} {unit}"

    p = (ggplot()
         + geom_line(data=curves[curves["Curve"] == "Reference"], 
                    mapping=aes(x="Concentration", y="OD"), color="red", size=1.5)
         + geom_line(data=curves[curves["Curve"].str.startswith("Plate")], 
                    mapping=aes(x="Concentration", y="OD", color="Curve"), size=0.8, alpha=0.7))
    
    # Add confidence interval ribbon if available
    if ci_lower is not None and ci_upper is not None:
        ci_data = pd.DataFrame({
            "Concentration": grid,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
        p = (p + geom_ribbon(data=ci_data, mapping=aes(x="Concentration", ymin="CI_Lower", ymax="CI_Upper"), 
                           alpha=0.2, fill="red"))
    
    p = (p + geom_point(data=pooled_std, mapping=aes(x="Concentration", y="OD"), color="black", size=2)
         + labs(title=title, x=f"Concentration ({unit})", y="OD", caption=caption)
         + theme_bw())

    filename = f"unified_standard_curve{out_name_suffix}.png"
    p.save(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved: {filename}")

    plot_unified_curve_reference_only(pooled_std, ref_params, model_name, output_dir, unit, out_name_suffix, 
                                    r2=r2, sample_range=sample_range, ci_lower=ci_lower, ci_upper=ci_upper)
    return p


def plot_unified_curve_reference_only(pooled_std: pd.DataFrame, ref_params: np.ndarray,
                                      model_name: str, output_dir: str, unit: str,
                                      out_name_suffix: str = "", r2: float = None, 
                                      sample_range: Tuple[float,float]=None,
                                      ci_lower: np.ndarray = None, ci_upper: np.ndarray = None):
    os.makedirs(output_dir, exist_ok=True)
    concs = pooled_std["Concentration"].dropna().astype(float).values
    cmin, cmax = concs.min(), concs.max()
    grid = np.linspace(cmin, cmax, 400)
    log_grid = np.log(np.clip(grid, 1e-12, None))

    df_curve = pd.DataFrame({
        "Concentration": grid,
        "OD": five_pl(log_grid, *ref_params)
    })
    
    # Add confidence intervals if provided
    if ci_lower is not None and ci_upper is not None:
        df_curve["CI_Lower"] = ci_lower
        df_curve["CI_Upper"] = ci_upper

    title = f"Unified Standard Curve (Reference Only, {model_name})"
    if r2 is not None and np.isfinite(r2):
        title = title + f" | R\u00b2={r2:.3f}"

    caption = None
    if sample_range is not None and not (np.isnan(sample_range[0]) or np.isnan(sample_range[1])):
        caption = f"Sample final concentration range: {sample_range[0]:.3g} - {sample_range[1]:.3g} {unit}"

    p = (ggplot(df_curve, mapping=aes(x="Concentration", y="OD"))
         + geom_line(size=1.1))
    
    # Add confidence interval ribbon if available
    if ci_lower is not None and ci_upper is not None:
        p = (p + geom_ribbon(mapping=aes(ymin="CI_Lower", ymax="CI_Upper"), alpha=0.3, fill="blue"))
    
    p = (p + geom_point(data=pooled_std, mapping=aes(x="Concentration", y="OD"), color="black", size=2)
         + labs(title=title, x=f"Concentration ({unit})", y="OD", caption=caption)
         + theme_bw())

    file_path = os.path.join(output_dir, f"unified_standard_curve{out_name_suffix}_reference_only.png")
    p.save(file_path, dpi=300)
    print(f"Saved: {file_path}")
    return p

# -------------------------------
# Main analysis 
# -------------------------------
def run_analysis(standard_file: str, plate_files: List[str], output_dir: str):
    print(f"ASSAY_TYPE setting: {ASSAY_TYPE}")   # to see where warnings come from
    os.makedirs(output_dir, exist_ok=True)
    print(f"ELISAtools vignette path: {VIGNETTE_PATH}")

    standards_map, unit = load_standards(standard_file)
    print(f"Detected standard units: {unit}")
    print(f"Standards: {list(standards_map.keys())}")

    all_plate_dfs = []
    pooled_std_rows = []
    plate_blank_info = {}
    all_blanks = []

    # collect final sample concentrations (for unified caption)
    samples_final_list = []

    # Read plates and build pooled standards
    for i, pf in enumerate(plate_files):
        df = parse_plate_data(pf)
        df["PlateIndex"] = i+1
        df["OD_Raw"] = df["OD"].copy()

        blank_mean = np.nan
        blanks = df[df["Type"] == "Blank"]["OD"].astype(float)
        if APPLY_BLANK_CORRECTION:
            if len(blanks.dropna()) == 0:
                print(f"WARNING: Plate {i+1} ({pf}) has no blanks — skipping blank correction for this plate.")
            else:
                blank_mean = float(np.nanmean(blanks))
                df["OD"] = df["OD"].astype(float) - blank_mean
                print(f"Plate {i+1} blank mean = {blank_mean:.4f} (subtracted from all ODs on plate).")
        else:
            print(f"Blank correction disabled; using raw ODs for plate {i+1}.")

        plate_blank_info[i+1] = blank_mean
        all_plate_dfs.append((pf, df))
        
        # Collect blank ODs for detection limit calculation
        valid_blanks = blanks.dropna()
        if len(valid_blanks) > 0:
            all_blanks.extend(valid_blanks.tolist())

        std = df[df["Type"] == "Standard"].copy()
        std["ID"] = std["ID"].astype(str).str.strip()
        std["Concentration"] = std["ID"].map({k.strip(): v for k, v in standards_map.items()})
        std = std[std["Concentration"].notna()].copy()
        if not std.empty:
            avg = std.groupby("ID").agg(Concentration=("Concentration","mean"), OD=("OD","mean")).reset_index()
            avg["PlateIndex"] = i+1
            for _, row in avg.iterrows():
                pooled_std_rows.append({"PlateIndex": int(row["PlateIndex"]), "ID": row["ID"], "Concentration": float(row["Concentration"]), "OD": float(row["OD"])})

    pooled_std = pd.DataFrame(pooled_std_rows)
    if pooled_std.empty:
        raise ValueError("No standard points found across plates.")
    
    # ===========================================
    # NEW: DATA QUALITY CHECK
    # ===========================================
    print(f"\n📊 DATA QUALITY REPORT:")
    print(f"Number of standard points: {len(pooled_std)}")
    print(f"Concentration range: {pooled_std['Concentration'].min():.2f} - {pooled_std['Concentration'].max():.2f} {unit}")
    print(f"OD range: {pooled_std['OD'].min():.3f} - {pooled_std['OD'].max():.3f}")

    # Check for monotonic pattern expected for competitive ELISA
    sorted_std = pooled_std.sort_values('Concentration')
    od_trend = "decreasing" if sorted_std['OD'].iloc[0] > sorted_std['OD'].iloc[-1] else "increasing"
    print(f"OD trend with concentration: {od_trend}")
    print(f"Expected for {ASSAY_TYPE} ELISA: {'increasing' if ASSAY_TYPE.lower() == 'sandwich' else 'decreasing'}")
    
    # Show individual standard points
    print(f"\nStandard points detail:")
    for _, row in sorted_std.iterrows():
        print(f"  {row['ID']}: {row['Concentration']:.2f} {unit} -> OD {row['OD']:.3f}")
    
    # ===========================================
    # Reference curve: depending on APPLY_AIC
    # ===========================================
    ref_concs = pooled_std["Concentration"].values
    ref_ods = pooled_std["OD"].values
    
    # NEW: Detect and validate curve direction
    detected_direction = detect_curve_direction(ref_concs, ref_ods)
    print(f"Detected curve direction: {detected_direction}")
    
    if ASSAY_TYPE.lower() == "sandwich" and detected_direction == "competitive":
        print("⚠️  WARNING: Data appears COMPETITIVE but assay type is SANDWICH!")
        print("   Check your standard curve data or toggle setting")
    elif ASSAY_TYPE.lower() == "competitive" and detected_direction == "sandwich":
        print("⚠️  WARNING: Data appears SANDWICH but assay type is COMPETITIVE!")
        print("   Check your standard curve data or toggle setting")

    if APPLY_AIC:
        ref_name, ref_params, ref_models = select_best_model(ref_concs, ref_ods, ASSAY_TYPE, plate_idx="Unified")
        print("\n \U0001F4A1 Reference model selection (AIC):")
        for m, r in ref_models.items():
            print(f"  {m.upper()}: AIC={r['AIC']:.1f}")
        print(f"Selected reference model: {ref_name}")
        ref_qc = calculate_qc(ref_concs, ref_ods, ref_params)
        ref_r2 = ref_qc.get("R_squared", float("nan"))
        # NEW: Print slope analysis for reference curve
        conc_range = (pooled_std["Concentration"].min(), pooled_std["Concentration"].max())
        print_slope_analysis(ref_params, ref_name, unit, conc_range)
    else:
        # Force whatever user selected in FORCED_MODEL
        forced = FORCED_MODEL.lower()
        if forced not in ["3pl","4pl","5pl"]:
            raise ValueError("FORCED_MODEL must be one of '3pl','4pl','5pl'")
        ref_name = forced.upper()
        ref_params, ref_ll = fit_model(ref_concs, ref_ods, forced, ASSAY_TYPE)
        if not np.isfinite(ref_params).all():
            raise RuntimeError(f"Failed to fit pooled standards with forced model: {forced}")
        ref_qc = calculate_qc(ref_concs, ref_ods, ref_params)
        ref_r2 = ref_qc.get("R_squared", float("nan"))
        print(f"\n \U0001F4A1 POOLED reference (forced {ref_name}) fitted.")
        print(f"Unified reference ({ref_name}) R² = {ref_r2:.4f}")
        # NEW: Print slope analysis for forced model too
        conc_range = (pooled_std["Concentration"].min(), pooled_std["Concentration"].max())
        print_slope_analysis(ref_params, ref_name, unit, conc_range)

    # Calculate detection limits for reference curve
    if len(all_blanks) > 0:
        detection_limits = calculate_detection_limits(ref_params, np.array(all_blanks))
        ref_qc.update(detection_limits)
        print(f"  LOD (Limit of Detection): {ref_qc.get('LOD', 'N/A'):.4f} {unit}")
        print(f"  LOQ (Limit of Quantification): {ref_qc.get('LOQ', 'N/A'):.4f} {unit}")
    else:
        print("  No blank data available for detection limit calculation")

    # predicted OD range (for diagnostics)
    try:
        min_conc = max(pooled_std['Concentration'].min(), 1e-12)
        max_conc = pooled_std['Concentration'].max()
        grid = np.logspace(np.log10(min_conc), np.log10(max_conc), 200)
        pred_ods = five_pl(np.log(grid), *ref_params)
        pred_min, pred_max = float(np.nanmin(pred_ods)), float(np.nanmax(pred_ods))
        print(f"Unified reference predicted OD range: {pred_min:.4f} - {pred_max:.4f}")
    except Exception as e:
        pred_min, pred_max = np.nan, np.nan
        print(f"Could not compute unified predicted OD range: {e}")

    results_all = []

    # Determine per-plate params used for unified plots
    plate_params_for_unified = []
    for pf, plate_df in all_plate_dfs:
        plate_idx = int(plate_df["PlateIndex"].iloc[0])
        plate_std = pooled_std[pooled_std["PlateIndex"] == plate_idx].copy()
        if plate_std.empty:
            continue
        if APPLY_AIC:
            _, p_params, _ = select_best_model(plate_std["Concentration"].values, plate_std["OD"].values, ASSAY_TYPE, plate_idx=plate_idx)
        else:
            # per-plate forced to the selected FORCED_MODEL
            p_params, _ = fit_model(plate_std["Concentration"].values, plate_std["OD"].values, FORCED_MODEL.lower(), ASSAY_TYPE, plate_idx=plate_idx)
        if np.isfinite(p_params).all():
            plate_params_for_unified.append((plate_idx, p_params))
    
    # ---------------------------
    # Process each plate (fit and compute samples)
    # ---------------------------
    for pf, plate_df in all_plate_dfs:
        plate_idx = int(plate_df["PlateIndex"].iloc[0])
        print(f"\n \u23F3PROCESSING PLATE {plate_idx}: {pf}")

        plate_std = pooled_std[pooled_std["PlateIndex"] == plate_idx].copy()
        if plate_std.empty:
            print("WARNING: no standards on this plate")
            results_all.append(None)
            continue

        if APPLY_AIC:
            plate_name, plate_params, plate_models = select_best_model(plate_std["Concentration"].values, plate_std["OD"].values, ASSAY_TYPE)
            plate_model_label = plate_name
            print(f"  Plate model (AIC best): {plate_model_label}")
        else:
            plate_params, _ = fit_model(plate_std["Concentration"].values, plate_std["OD"].values, FORCED_MODEL.lower(), ASSAY_TYPE)
            plate_model_label = f"{FORCED_MODEL.upper()} (forced)"
            print(f"  Plate model (forced): {plate_model_label}")
            
        # NEW: Print slope analysis for each plate
        if not plate_std.empty:
            plate_conc_range = (plate_std["Concentration"].min(), plate_std["Concentration"].max())
            print_slope_analysis(plate_params, plate_model_label, unit, plate_conc_range)

        s_plate = float(plate_params[2] - ref_params[2]) if (np.isfinite(plate_params).all() and np.isfinite(ref_params).all()) else float('nan')
        print(f"  Shift factor s = {s_plate:.4f}")

        qc_plate = calculate_qc(plate_std["Concentration"].values, plate_std["OD"].values, plate_params)
        
        # Get plate-specific blanks for detection limit calculation
        plate_blanks = plate_df[plate_df["Type"] == "Blank"]["OD"].dropna().values
        if len(plate_blanks) > 0:
            plate_detection_limits = calculate_detection_limits(plate_params, plate_blanks)
            qc_plate.update(plate_detection_limits)
            print(f"  Plate LOD: {qc_plate.get('LOD', 'N/A'):.4f} {unit}")
            print(f"  Plate LOQ: {qc_plate.get('LOQ', 'N/A'):.4f} {unit}")

        out_df = plate_df.copy()
        out_df["Raw_Concentration"] = np.nan
        out_df["Adjusted_Concentration"] = np.nan
        out_df["Final_Concentration"] = np.nan

        conc_min = pooled_std["Concentration"].min()
        conc_max = pooled_std["Concentration"].max()

        samples = out_df[out_df["Type"] == "Sample"].copy()
        ids_in_range, ids_below, ids_above = [], [], []


        # =============================================================================
        # DUPLICATE SAMPLE ANALYSIS (BEFORE INDIVIDUAL PROCESSING)
        # =============================================================================
        print(f"  Analyzing sample duplicates...")

        # Prepare samples based on toggle
        if SAMPLES_IN_DUPLICATE:
            # Average duplicates for summary analysis
            df_samples_avg = samples.groupby("ID").agg({
                "OD": "mean",
                "Dilution": "first",  # Use first dilution value (should be same for duplicates)
                "Content": "first",
                "Well.Row": "first", 
                "Well.Col": "first"
            }).reset_index()
            samples_for_summary = df_samples_avg
            print(f"  Duplicate averaging ENABLED - {len(samples)} wells -> {len(df_samples_avg)} unique samples")
        else:
            # Use all individual wells for summary
            samples_for_summary = samples.copy()
            print(f"  Duplicate averaging DISABLED - analyzing {len(samples)} individual wells")

        sample_summary_rows = []

        for idx, row in samples_for_summary.iterrows():
            sample_id = row["ID"]
            
            # For duplicate mode: get original group to calculate stats
            if SAMPLES_IN_DUPLICATE:
                original_group = samples[samples["ID"] == sample_id]
                n_replicates = len(original_group)
                mean_od = float(row["OD"])  # Already averaged
                std_od = float(original_group["OD"].std()) if n_replicates > 1 else 0.0
                mean_dilution = float(row["Dilution"])
            else:
                # Individual mode: treat each well as separate "sample"
                n_replicates = 1
                mean_od = float(row["OD"])
                std_od = 0.0
                mean_dilution = float(row["Dilution"])
                # Create unique ID for each well to avoid grouping
                sample_id = f"{row['ID']}_{row['Well.Row']}{row['Well.Col']}"
            
            # Calculate CV% (handle division by zero and single replicates)
            cv_percent = 0.0
            if n_replicates > 1:
                cv_percent = float((std_od / mean_od) * 100) if mean_od > 0 else float('inf')
            
            # Compute concentration using MEAN OD (for summary)
            final_conc_from_mean = np.nan
            if np.isfinite(mean_od):
                conc_raw_mean = od_to_conc(mean_od, plate_params, conc_min, conc_max)
                if np.isfinite(conc_raw_mean):
                    final_conc_from_mean = conc_raw_mean / math.exp(s_plate) * mean_dilution if np.isfinite(s_plate) else conc_raw_mean * mean_dilution
            
            # Warning flag for high CV (only when duplicates exist)
            warning = ""
            if SAMPLES_IN_DUPLICATE and n_replicates > 1 and cv_percent > 10:  # 10% CV threshold
                warning = f"High CV: {cv_percent:.1f}%"
                print(f"    ⚠ WARNING: Sample '{sample_id}' has high CV: {cv_percent:.1f}%")
            
            sample_summary_rows.append({
                "Sample_ID": sample_id,
                "Mean_OD": mean_od,
                "Std_OD": std_od,
                "CV_percent": cv_percent,
                "Mean_Dilution": mean_dilution,
                "Final_Concentration": final_conc_from_mean,
                "Replicates": n_replicates,
                "Warning": warning
            })

        # Save sample summary for this plate
        if sample_summary_rows:
            sample_summary_df = pd.DataFrame(sample_summary_rows)
            summary_suffix = "_sample_summary_duplicates" if SAMPLES_IN_DUPLICATE else "_sample_summary_individual"
            sample_summary_path = os.path.join(output_dir, f"plate_{plate_idx}{summary_suffix}.csv")
            sample_summary_df.to_csv(sample_summary_path, index=False)
            
            mode = "duplicate-averaged" if SAMPLES_IN_DUPLICATE else "individual-well"
            print(f"  Saved {mode} sample summary: {len(sample_summary_rows)} entries -> {sample_summary_path}")
            
            # Print high-CV samples to console (only in duplicate mode)
            if SAMPLES_IN_DUPLICATE:
                high_cv_samples = sample_summary_df[sample_summary_df["Warning"].str.contains("High CV", na=False)]
                if not high_cv_samples.empty:
                    print(f"  ⚠ High-CV samples found: {len(high_cv_samples)}")
                    for _, row in high_cv_samples.iterrows():
                        print(f"      {row['Sample_ID']}: CV = {row['CV_percent']:.1f}%")
        else:
            print(f"  No samples found for summary.")

        # ======================================================================
        # CONTINUATION OF EXISTING INDIVIDUAL WELL PROCESSING
        # ======================================================================
        for idx, row in samples.iterrows():
            od = row["OD"]
            dil = row["Dilution"] if pd.notna(row["Dilution"]) else 1.0
            conc_raw = od_to_conc(od, plate_params, conc_min, conc_max)
            if np.isfinite(conc_raw):
                conc_adj = conc_raw / math.exp(s_plate) if np.isfinite(s_plate) else conc_raw
                final = conc_adj * dil
                out_df.at[idx, "Raw_Concentration"] = conc_raw
                out_df.at[idx, "Adjusted_Concentration"] = conc_adj
                out_df.at[idx, "Final_Concentration"] = final
                if np.isfinite(pred_min) and np.isfinite(pred_max):
                    if row["OD"] < pred_min:
                        ids_below.append(row.get("Content", idx))
                    elif row["OD"] > pred_max:
                        ids_above.append(row.get("Content", idx))
                    else:
                        ids_in_range.append(row.get("Content", idx))
                else:
                    ids_in_range.append(row.get("Content", idx))

        # collect final concentrations for unified-plot annotation
        final_vals = out_df["Final_Concentration"].dropna().astype(float).tolist()
        if final_vals:
            samples_final_list.extend(final_vals)

        print(f"  Samples computed: {int(out_df['Final_Concentration'].notna().sum())}/{len(samples)}")
        if ids_in_range:
            print(f"    Sample IDs within unified curve: {ids_in_range[:10]}{(' + more' if len(ids_in_range)>10 else '')}")
        if ids_below:
            print(f"    Sample IDs below unified curve: {ids_below[:10]}{(' + more' if len(ids_below)>10 else '')}")
        if ids_above:
            print(f"    Sample IDs above unified curve: {ids_above[:10]}{(' + more' if len(ids_above)>10 else '')}")

        out_df["Blank_Mean"] = plate_blank_info.get(plate_idx, np.nan)
        out_df.to_csv(os.path.join(output_dir, f"plate_{plate_idx}_results.csv"), index=False)

        # Calculate sample ranges for this plate
        sample_ods = samples["OD"].dropna()
        sample_final_concs = out_df["Final_Concentration"].dropna()
        
        model_summary = {
            "Plate": plate_idx,
            "Best_Model": plate_model_label,
            "Shift_s": s_plate,
            "R_squared": qc_plate.get("R_squared", float('nan')),
            "LLOQ": qc_plate.get("LLOQ", float('nan')),
            "ULOQ": qc_plate.get("ULOQ", float('nan')),
            "LOD": qc_plate.get("LOD", float('nan')),  # NEW: Detection limits
            "LOQ": qc_plate.get("LOQ", float('nan')),  # NEW: Detection limits
            "Std_Min_Conc": qc_plate.get("Min_Conc", float('nan')),
            "Std_Max_Conc": qc_plate.get("Max_Conc", float('nan')),
            "Std_Min_OD": qc_plate.get("Min_OD", float('nan')),
            "Std_Max_OD": qc_plate.get("Max_OD", float('nan')),
            "Sample_Min_Conc": float(sample_final_concs.min()) if len(sample_final_concs) > 0 else float('nan'),
            "Sample_Max_Conc": float(sample_final_concs.max()) if len(sample_final_concs) > 0 else float('nan'),
            "Sample_Min_OD": float(sample_ods.min()) if len(sample_ods) > 0 else float('nan'),
            "Sample_Max_OD": float(sample_ods.max()) if len(sample_ods) > 0 else float('nan'),
            "Blank_Mean": plate_blank_info.get(plate_idx, np.nan)
        }
        pd.DataFrame([model_summary]).to_csv(os.path.join(output_dir, f"plate_{plate_idx}_model_comprehensive.csv"), index=False)

        # Save per-plate plot (standards only) - SIMPLIFIED WITHOUT CI FOR NOW
        try:
            plot_df_std = plate_std.copy(); plot_df_std["Type"] = "Standard"
            r2_m = qc_plate.get("R_squared", float('nan'))
            p = generate_plot(plot_df_std, plate_params, f"Plate {plate_idx} | {plate_model_label}", r2=r2_m, unit=unit)
            p.save(os.path.join(output_dir, f"plate_{plate_idx}_curve.png"), dpi=300)
        except Exception as e:
            print(f"  Could not save plate {plate_idx} plot: {e}")

        results_all.append({
            "plate_index": plate_idx,
            "best_model": plate_model_label,
            "plate_params": plate_params,
            "s_plate": s_plate,
            "qc_plate": qc_plate,
            "n_found": int(out_df['Final_Concentration'].notna().sum()),
            "n_total": len(samples),
            "blank_mean": plate_blank_info.get(plate_idx, np.nan)
        })

    # ---------------------------
    # Unified curves generation
    # ---------------------------
    unified_summary_records = []
    unified_r2_records = []

    # compute sample final concentration range for annotations
    if len(samples_final_list) > 0:
        sample_range = (float(np.nanmin(samples_final_list)), float(np.nanmax(samples_final_list)))
    else:
        sample_range = (np.nan, np.nan)

    if APPLY_AIC:
        # produce BEST unified curve (pooled best + per-plate best)
        try:
            plate_params_for_best = []
            for pf, plate_df in all_plate_dfs:
                plate_idx = int(plate_df["PlateIndex"].iloc[0])
                plate_std = pooled_std[pooled_std["PlateIndex"] == plate_idx].copy()
                if plate_std.empty:
                    continue
                _, p_params, _ = select_best_model(plate_std["Concentration"].values, plate_std["OD"].values, ASSAY_TYPE, plate_idx)
                if np.isfinite(p_params).all():
                    plate_params_for_best.append((plate_idx, p_params))

            out_suffix = "_best"
            plot_unified_curve(pooled_std, ref_params, f"BEST:{ref_name}", plate_params_for_best, output_dir, unit, 
                             out_name_suffix=out_suffix, sample_range=sample_range, r2=ref_r2)
            curve_fn = f"unified_standard_curve{out_suffix}.png"
            unified_r2_records.append({"Model": "BEST", "Ref_Model": ref_name, "R_squared": ref_r2, "Filename": curve_fn})
            
            unified_summary_records.append({
                "Model": "BEST", 
                "Ref_Model": ref_name, 
                "R_squared": ref_r2, 
                "LLOQ": ref_qc.get("LLOQ", float('nan')),
                "ULOQ": ref_qc.get("ULOQ", float('nan')),
                "LOD": ref_qc.get("LOD", float('nan')),  # NEW: Detection limits
                "LOQ": ref_qc.get("LOQ", float('nan')),  # NEW: Detection limits
                "Min_Conc": ref_qc.get("Min_Conc", float('nan')),
                "Max_Conc": ref_qc.get("Max_Conc", float('nan')),
                "Min_OD": ref_qc.get("Min_OD", float('nan')),
                "Max_OD": ref_qc.get("Max_OD", float('nan')),
                "Curve_Filename": curve_fn, 
                "Reference_Only_Filename": f"unified_standard_curve{out_suffix}_reference_only.png", 
                "Warning": ""
            })
        except Exception as e:
            print(f"Could not generate BEST unified curve: {e}")
            unified_summary_records.append({"Model":"BEST","Ref_Model":ref_name,"R_squared":float('nan'),"LLOQ":float('nan'),"ULOQ":float('nan'),"LOD":float('nan'),"LOQ":float('nan'),"Curve_Filename":"","Reference_Only_Filename":"","Warning":str(e)})

        # also generate forced 3PL/4PL/5PL visual comparisons (as before)
        forced_list = ["3pl","4pl","5pl"]
    else:
        # APPLY_AIC == False -> Option A: only generate unified curves for FORCED_MODEL
        forced_list = [FORCED_MODEL.lower()]

    # generate unified curves for models listed in forced_list
    for forced in forced_list:
        try:
            params_forced_ref, _ = fit_model(ref_concs, ref_ods, forced, ASSAY_TYPE)
            if not np.isfinite(params_forced_ref).all():
                print(f"Could not fit pooled standards with {forced.upper()}; skipping")
                unified_summary_records.append({"Model":forced.upper(),"Ref_Model":forced.upper(),"R_squared":float('nan'),"LLOQ":float('nan'),"ULOQ":float('nan'),"LOD":float('nan'),"LOQ":float('nan'),"Curve_Filename":"","Reference_Only_Filename":"","Warning":"fit_failed"})
                unified_r2_records.append({"Model":forced.upper(),"Ref_Model":forced.upper(),"R_squared":float('nan'),"Filename":""})
                continue

            # Calculate QC for forced model
            qc_forced = calculate_qc(ref_concs, ref_ods, params_forced_ref)
            # Calculate detection limits for forced model
            if len(all_blanks) > 0:
                forced_detection_limits = calculate_detection_limits(params_forced_ref, np.array(all_blanks))
                qc_forced.update(forced_detection_limits)

            # per-plate forced fits for plotting
            plate_params_forced = []
            for pf, plate_df in all_plate_dfs:
                plate_idx = int(plate_df["PlateIndex"].iloc[0])
                plate_std = pooled_std[pooled_std["PlateIndex"] == plate_idx].copy()
                if plate_std.empty:
                    continue
                params_plate, _ = fit_model(plate_std["Concentration"].values, plate_std["OD"].values, forced, ASSAY_TYPE)
                if np.isfinite(params_plate).all():
                    plate_params_forced.append((plate_idx, params_plate))

            out_suffix = f"_{forced}"
            plot_unified_curve(pooled_std, params_forced_ref, forced.upper(), plate_params_forced, output_dir, unit, 
                             out_name_suffix=out_suffix, sample_range=sample_range, r2=qc_forced.get("R_squared"))
            unified_r2_records.append({"Model": forced.upper(), "Ref_Model": forced.upper(), "R_squared": qc_forced.get("R_squared", float("nan")), "Filename": f"unified_standard_curve{out_suffix}.png"})
            unified_summary_records.append({
                "Model": forced.upper(),
                "Ref_Model": forced.upper(),
                "R_squared": qc_forced.get("R_squared",float('nan')),
                "LLOQ": qc_forced.get("LLOQ"),
                "ULOQ": qc_forced.get("ULOQ"),
                "LOD": qc_forced.get("LOD", float('nan')),  # NEW: Detection limits
                "LOQ": qc_forced.get("LOQ", float('nan')),  # NEW: Detection limits
                "Min_Conc": qc_forced.get("Min_Conc", float('nan')),
                "Max_Conc": qc_forced.get("Max_Conc", float('nan')),
                "Min_OD": qc_forced.get("Min_OD", float('nan')),
                "Max_OD": qc_forced.get("Max_OD", float('nan')),
                "Curve_Filename": f"unified_standard_curve{out_suffix}.png",
                "Reference_Only_Filename": f"unified_standard_curve{out_suffix}_reference_only.png",
                "Warning": ""
            })
            print(f"Saved unified curve forced to {forced.upper()} (R²={qc_forced.get('R_squared',float('nan')):.4f})")
        except Exception as e:
            print(f"Error generating forced unified curve for {forced}: {e}")
            unified_summary_records.append({"Model":forced.upper(),"Ref_Model":forced.upper(),"R_squared":float('nan'),"LLOQ":float('nan'),"ULOQ":float('nan'),"LOD":float('nan'),"LOQ":float('nan'),"Curve_Filename":"","Reference_Only_Filename":"","Warning":str(e)})
            unified_r2_records.append({"Model": forced.upper(), "Ref_Model": forced.upper(), "R_squared": float('nan'), "Filename": ""})

    # Save unified R² table
    try:
        df_r2 = pd.DataFrame(unified_r2_records)
        df_r2.to_csv(os.path.join(output_dir, "unified_standard_curve_r2.csv"), index=False)
        print(f"Unified curves R² summary saved to {os.path.join(output_dir, 'unified_standard_curve_r2.csv')}")
    except Exception as e:
        print(f"Could not save unified R² summary: {e}")

    # Save comprehensive unified summary table
    try:
        df_usum = pd.DataFrame(unified_summary_records)
        df_usum.to_csv(os.path.join(output_dir, "unified_standard_curve_summary.csv"), index=False)
        print(f"Unified curves summary saved to {os.path.join(output_dir, 'unified_standard_curve_summary.csv')}")
    except Exception as e:
        print(f"Could not save unified summary CSV: {e}")

    # PRINTABLE unified batch summary
    try:
        unified_batch_summary = []
        for entry in unified_summary_records:
            unified_batch_summary.append({
                "Model": entry.get("Model", ""),
                "Ref_Model": entry.get("Ref_Model", ""),
                "Shift_Factor": 0.0,
                "R_squared": entry.get("R_squared", float("nan")),
                "LLOQ": entry.get("LLOQ", float("nan")),
                "ULOQ": entry.get("ULOQ", float("nan")),
                "LOD": entry.get("LOD", float("nan")),  # NEW: Detection limits
                "LOQ": entry.get("LOQ", float("nan")),  # NEW: Detection limits
                "Min_Conc": entry.get("Min_Conc", float("nan")),
                "Max_Conc": entry.get("Max_Conc", float("nan")),
                "Min_OD": entry.get("Min_OD", float("nan")),
                "Max_OD": entry.get("Max_OD", float("nan")),
                "Warning": entry.get("Warning", "")
            })

        df_ubatch = pd.DataFrame(unified_batch_summary)
        df_ubatch.to_csv(os.path.join(output_dir, "unified_standard_curve_batch_summary.csv"), index=False)

        print("\n \U0001F4C1 Unified Standard Curve Batch Summary:")
        print(df_ubatch.to_string(index=False))

    except Exception as e:
        print(f"Could not produce unified batch summary: {e}")

        # batch summary (per plate)
    batch_summary = []
    for r in results_all:
        if r is not None:
            # Calculate HOD and HOQ if LOD/LOQ are available
            lod = r["qc_plate"].get("LOD", float('nan'))
            loq = r["qc_plate"].get("LOQ", float('nan'))
            lod_od = r["qc_plate"].get("LOD_OD", float('nan'))
            loq_od = r["qc_plate"].get("LOQ_OD", float('nan'))
            
            batch_summary.append({
                "Plate": r["plate_index"],
                "Best_Model": r["best_model"],
                "Shift_Factor": r["s_plate"],
                "R_squared": r["qc_plate"].get("R_squared", float('nan')),
                "LOD": lod,
                "LOQ": loq,
                "HOD": lod_od,  # NEW: Highest OD for Detection
                "HOQ": loq_od,  # NEW: Highest OD for Quantification
                "Samples_Found": f"{r['n_found']}/{r['n_total']}",
                "Blank_Mean": r.get("blank_mean", np.nan)
            })
            
            # Print detection limits to console
            print(f"  Plate {r['plate_index']} Detection Limits:")
            if np.isfinite(lod):
                print(f"    LOD: {lod:.4f} {unit} (HOD: {lod_od:.4f} OD)")
            else:
                print(f"    LOD: Not calculable")
                
            if np.isfinite(loq):
                print(f"    LOQ: {loq:.4f} {unit} (HOQ: {loq_od:.4f} OD)")
            else:
                print(f"    LOQ: Not calculable")
    
    if batch_summary:
        pd.DataFrame(batch_summary).to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False)
        print("\n \U0001F4C1 Batch summary saved and printed below:")
        print(pd.DataFrame(batch_summary).to_string(index=False))

    return results_all
 
# -------------------------------
# Execution — edit filenames🚨🚨🚨
# -------------------------------
if __name__ == '__main__':
    STANDARD_FILE = "standard_file_name.csv" #🚨edit file name
    PLATE_FILES = ["plate1_readings.csv","plate2_readings.csv", "plate3_readings.csv"] #🚨edit file names
    OUTPUT_DIR = "analysis_results"          #🚨edit if necessary 
    print("🚀 STARTING ELISA BATCH ANALYSIS")
    results = run_analysis(STANDARD_FILE, PLATE_FILES, OUTPUT_DIR) 
    print("✅ DONE — results saved.")



