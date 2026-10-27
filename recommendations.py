"""Category-specific recycling advice."""

RECOMMENDATIONS = {
    "cardboard": "Flatten boxes and remove tape or heavy food contamination. Keep them dry before recycling.",
    "glass": "Rinse and follow your local glass collection rules. Wrap broken glass carefully before disposal.",
    "metal": "Empty and rinse containers, then place them in the metal or mixed-recycling stream.",
    "paper": "Keep paper dry and separate from food-contaminated waste. Greasy paper usually cannot be recycled.",
    "plastic": "Check the plastic type and your local recycling rules. Rinse containers and remove caps if required.",
    "trash": "This looks like residual waste. Use the general waste bin, and check whether any part can be recycled.",
}

DEFAULT = "Check your local waste-management guidelines for the correct disposal method."


def get_recommendation(category: str) -> str:
    return RECOMMENDATIONS.get(str(category).strip().lower(), DEFAULT)
