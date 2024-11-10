from typing import Dict


def get_ocean_plot(ocean: Dict[str, int]):
    return {
        "data": [{
            "type": "scatterpolar",
            "r": list(ocean.values()),
            "theta": list(ocean.keys()),
            "fill": "toself",
        }],
        "layout": {
            "polar": {"radialaxis": {"visible": True, "range": [0, 1]},
                      "angularaxis": {"rotation": -90}},
            "showlegend": False,
            "margin": {
                "l": 10,
                "r": 10,
                "b": 30,
                "t": 10,
                "pad": 4
            }
        },
    }
