STANDARD_MOISTURE = {
    "rice": 11.79,
    "maize": 13.5,
    "chickpea": 14.0,
    "kidneybeans": 9.77,
    "pigeonpeas": 12.0,
    "mothbeans": 10.46,
    "mungbean": 8.7,
    "blackgram": 15.75,
    "lentil": 12.7,
    "pomegranate": 81.0,
    "banana": 74.0,
    "mango": 81.05,
    "grapes": 78.0,
    "watermelon": 92.0,
    "muskmelon": 88.7,
    "apple": 82.0,
    "orange": 73.5,
    "papaya": 84.2,
    "coconut": 9.3,
    "cotton": 6.5,
    "jute": 13.2,
    "coffee": 12.0,
}

SQ_FT_PER_ACRE = 43_560
TONNES_PER_POUND = 0.00112085


def crop(grain_weight, grain_moisture, harvested_area, crop_name):
    """Estimate crop yield from grain weight, moisture, and harvested area."""
    if crop_name not in STANDARD_MOISTURE:
        raise ValueError(f"Unsupported crop: {crop_name}")

    if harvested_area <= 0:
        raise ValueError("Harvested area must be greater than zero.")

    standard_moisture = STANDARD_MOISTURE[crop_name]
    adjusted_weight = ((100 - grain_moisture) / (100 - standard_moisture)) * grain_weight
    acres = harvested_area / SQ_FT_PER_ACRE
    pounds_per_acre = adjusted_weight / acres

    return round(pounds_per_acre * TONNES_PER_POUND, 3)
