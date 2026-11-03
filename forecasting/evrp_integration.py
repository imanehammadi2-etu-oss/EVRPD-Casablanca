import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("EVRPIntegration")

def build_evrp_demand_input(
    selected_forecast: pd.DataFrame,
    zone_mapping: Dict[str, Any],
    timestamps: List[str],
    service_capacity: Any,
    optional_buffer: float = 0.0
) -> Dict[str, Any]:
    """
    Format predictions into a dictionary ingestible by the EVRP module (NSGA-II, MAPPO, etc.)
    """
    if "y_pred" not in selected_forecast.columns:
        raise ValueError("The forecast DataFrame must contain a 'y_pred' column.")
    
    # Ensure no negative demand
    selected_forecast["y_pred"] = selected_forecast["y_pred"].apply(lambda x: max(0.0, x))
    
    predicted_demand_by_zone = {}
    capacity_requirement_by_zone = {}
    safety_buffer = {}

    for zone in selected_forecast["zone_id"].unique():
        zone_data = selected_forecast[selected_forecast["zone_id"] == zone]
        
        # Rounding logic for items (parcels can't be fractional in EVRP)
        demand_values = np.ceil(zone_data["y_pred"].values).astype(int).tolist()
        predicted_demand_by_zone[zone] = demand_values
        capacity_requirement_by_zone[zone] = [d * service_capacity for d in demand_values] if isinstance(service_capacity, (int, float)) else demand_values
        safety_buffer[zone] = [int(np.ceil(d * optional_buffer)) for d in demand_values]

    result = {
      "forecast_horizon": len(timestamps),
      "timestamps": timestamps,
      "zones": list(predicted_demand_by_zone.keys()),
      "predicted_demand_by_zone": predicted_demand_by_zone,
      "capacity_requirement_by_zone": capacity_requirement_by_zone,
      "optional_safety_buffer": safety_buffer,
      "selected_model": "PatchTST", # Will be updated dynamically
      "forecast_metrics": {}
    }
    
    logger.info("Successfully built EVRP demand input structure without breaking existing optimization loops.")
    return result
