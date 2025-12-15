"""
Nutrition Data Module

This module contains nutritional information per 100g for all 20 grocery categories.
Data sourced from USDA FoodData Central and other reliable nutrition databases.
"""

# Nutritional values per 100g for each grocery category
# Values include: calories (kcal), protein (g), carbs (g), fat (g), fiber (g), sugar (g)
# Additional nutrients: vitamin_c (mg), potassium (mg), calcium (mg), iron (mg)

NUTRITION_DATA = {
    'bacon': {
        'name': 'Bacon (cured, pan-fried)',
        'calories': 541,
        'protein': 37.0,
        'carbs': 1.4,
        'fat': 42.0,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 565,
        'calcium': 11,
        'iron': 1.3,
        'emoji': '🥓',
        'health_note': 'High in protein and fat. Consume in moderation due to sodium content.'
    },
    'banana': {
        'name': 'Banana (raw)',
        'calories': 89,
        'protein': 1.1,
        'carbs': 22.8,
        'fat': 0.3,
        'fiber': 2.6,
        'sugar': 12.2,
        'vitamin_c': 8.7,
        'potassium': 358,
        'calcium': 5,
        'iron': 0.3,
        'emoji': '🍌',
        'health_note': 'Excellent source of potassium. Great for quick energy.'
    },
    'bread': {
        'name': 'Bread (white, commercially prepared)',
        'calories': 265,
        'protein': 9.4,
        'carbs': 49.0,
        'fat': 3.2,
        'fiber': 2.7,
        'sugar': 5.0,
        'vitamin_c': 0,
        'potassium': 115,
        'calcium': 151,
        'iron': 3.6,
        'emoji': '🍞',
        'health_note': 'Good source of carbohydrates. Choose whole grain for more fiber.'
    },
    'broccoli': {
        'name': 'Broccoli (raw)',
        'calories': 34,
        'protein': 2.8,
        'carbs': 7.0,
        'fat': 0.4,
        'fiber': 2.6,
        'sugar': 1.7,
        'vitamin_c': 89.2,
        'potassium': 316,
        'calcium': 47,
        'iron': 0.7,
        'emoji': '🥦',
        'health_note': 'Superfood! High in vitamin C and antioxidants.'
    },
    'butter': {
        'name': 'Butter (salted)',
        'calories': 717,
        'protein': 0.9,
        'carbs': 0.1,
        'fat': 81.0,
        'fiber': 0,
        'sugar': 0.1,
        'vitamin_c': 0,
        'potassium': 24,
        'calcium': 24,
        'iron': 0,
        'emoji': '🧈',
        'health_note': 'High in saturated fat. Use sparingly for flavoring.'
    },
    'carrots': {
        'name': 'Carrots (raw)',
        'calories': 41,
        'protein': 0.9,
        'carbs': 9.6,
        'fat': 0.2,
        'fiber': 2.8,
        'sugar': 4.7,
        'vitamin_c': 5.9,
        'potassium': 320,
        'calcium': 33,
        'iron': 0.3,
        'emoji': '🥕',
        'health_note': 'Rich in beta-carotene (Vitamin A). Great for eye health.'
    },
    'cheese': {
        'name': 'Cheese (cheddar)',
        'calories': 403,
        'protein': 25.0,
        'carbs': 1.3,
        'fat': 33.0,
        'fiber': 0,
        'sugar': 0.5,
        'vitamin_c': 0,
        'potassium': 98,
        'calcium': 721,
        'iron': 0.7,
        'emoji': '🧀',
        'health_note': 'Excellent source of calcium and protein.'
    },
    'chicken': {
        'name': 'Chicken (breast, roasted)',
        'calories': 165,
        'protein': 31.0,
        'carbs': 0,
        'fat': 3.6,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 256,
        'calcium': 15,
        'iron': 1.0,
        'emoji': '🍗',
        'health_note': 'Lean protein source. Low in fat, high in protein.'
    },
    'cucumber': {
        'name': 'Cucumber (with peel, raw)',
        'calories': 15,
        'protein': 0.7,
        'carbs': 3.6,
        'fat': 0.1,
        'fiber': 0.5,
        'sugar': 1.7,
        'vitamin_c': 2.8,
        'potassium': 147,
        'calcium': 16,
        'iron': 0.3,
        'emoji': '🥒',
        'health_note': 'Very low calorie, hydrating. Great for weight management.'
    },
    'eggs': {
        'name': 'Eggs (whole, hard-boiled)',
        'calories': 155,
        'protein': 13.0,
        'carbs': 1.1,
        'fat': 11.0,
        'fiber': 0,
        'sugar': 1.1,
        'vitamin_c': 0,
        'potassium': 126,
        'calcium': 50,
        'iron': 1.2,
        'emoji': '🥚',
        'health_note': 'Complete protein source with all essential amino acids.'
    },
    'fish': {
        'name': 'Fish (salmon, cooked)',
        'calories': 208,
        'protein': 20.0,
        'carbs': 0,
        'fat': 13.0,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 363,
        'calcium': 12,
        'iron': 0.8,
        'emoji': '🐟',
        'health_note': 'Rich in Omega-3 fatty acids. Heart-healthy protein.'
    },
    'lettuce': {
        'name': 'Lettuce (romaine, raw)',
        'calories': 17,
        'protein': 1.2,
        'carbs': 3.3,
        'fat': 0.3,
        'fiber': 2.1,
        'sugar': 1.2,
        'vitamin_c': 4.0,
        'potassium': 247,
        'calcium': 33,
        'iron': 1.0,
        'emoji': '🥬',
        'health_note': 'Very low calorie, high in water content. Good for salads.'
    },
    'milk': {
        'name': 'Milk (whole, 3.25% fat)',
        'calories': 61,
        'protein': 3.2,
        'carbs': 4.8,
        'fat': 3.3,
        'fiber': 0,
        'sugar': 5.0,
        'vitamin_c': 0,
        'potassium': 132,
        'calcium': 113,
        'iron': 0,
        'emoji': '🥛',
        'health_note': 'Excellent source of calcium and vitamin D.'
    },
    'onions': {
        'name': 'Onions (raw)',
        'calories': 40,
        'protein': 1.1,
        'carbs': 9.3,
        'fat': 0.1,
        'fiber': 1.7,
        'sugar': 4.2,
        'vitamin_c': 7.4,
        'potassium': 146,
        'calcium': 23,
        'iron': 0.2,
        'emoji': '🧅',
        'health_note': 'Contains antioxidants. May help reduce inflammation.'
    },
    'peppers': {
        'name': 'Bell Peppers (red, raw)',
        'calories': 31,
        'protein': 1.0,
        'carbs': 6.0,
        'fat': 0.3,
        'fiber': 2.1,
        'sugar': 4.2,
        'vitamin_c': 127.7,
        'potassium': 211,
        'calcium': 7,
        'iron': 0.4,
        'emoji': '🫑',
        'health_note': 'Extremely high in Vitamin C - more than oranges!'
    },
    'potatoes': {
        'name': 'Potatoes (flesh and skin, baked)',
        'calories': 93,
        'protein': 2.5,
        'carbs': 21.0,
        'fat': 0.1,
        'fiber': 2.2,
        'sugar': 1.7,
        'vitamin_c': 9.6,
        'potassium': 535,
        'calcium': 15,
        'iron': 1.1,
        'emoji': '🥔',
        'health_note': 'Good source of potassium. Leave skin on for more fiber.'
    },
    'sausages': {
        'name': 'Sausages (pork, cooked)',
        'calories': 301,
        'protein': 19.0,
        'carbs': 2.4,
        'fat': 24.0,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 1.1,
        'potassium': 294,
        'calcium': 13,
        'iron': 1.5,
        'emoji': '🌭',
        'health_note': 'High in protein and fat. Watch sodium content.'
    },
    'spinach': {
        'name': 'Spinach (raw)',
        'calories': 23,
        'protein': 2.9,
        'carbs': 3.6,
        'fat': 0.4,
        'fiber': 2.2,
        'sugar': 0.4,
        'vitamin_c': 28.1,
        'potassium': 558,
        'calcium': 99,
        'iron': 2.7,
        'emoji': '🥬',
        'health_note': 'Iron-rich superfood. Great for vegetarians.'
    },
    'tomato': {
        'name': 'Tomatoes (red, ripe, raw)',
        'calories': 18,
        'protein': 0.9,
        'carbs': 3.9,
        'fat': 0.2,
        'fiber': 1.2,
        'sugar': 2.6,
        'vitamin_c': 13.7,
        'potassium': 237,
        'calcium': 10,
        'iron': 0.3,
        'emoji': '🍅',
        'health_note': 'Rich in lycopene, a powerful antioxidant.'
    },
    'yogurt': {
        'name': 'Yogurt (plain, whole milk)',
        'calories': 61,
        'protein': 3.5,
        'carbs': 4.7,
        'fat': 3.3,
        'fiber': 0,
        'sugar': 4.7,
        'vitamin_c': 0.5,
        'potassium': 155,
        'calcium': 121,
        'iron': 0.1,
        'emoji': '🥛',
        'health_note': 'Contains probiotics for gut health. Good calcium source.'
    }
}


def get_nutrition(class_name: str) -> dict:
    """
    Get nutritional information for a grocery class.
    
    Args:
        class_name: The predicted grocery class (e.g., 'banana')
        
    Returns:
        Dictionary containing nutritional data, or None if not found
    """
    return NUTRITION_DATA.get(class_name.lower())


def get_nutrition_html_card(class_name: str) -> str:
    """
    Generate a beautiful HTML card displaying nutrition facts.
    
    Args:
        class_name: The predicted grocery class
        
    Returns:
        HTML string for the nutrition card
    """
    nutrition = get_nutrition(class_name)
    
    if not nutrition:
        return f"""
<div style="background: rgba(255,0,0,0.1); padding: 1rem; border-radius: 10px; text-align: center;">
<p>Nutrition data not available for {class_name}</p>
</div>
"""
    
    return f"""
<div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%); 
border-radius: 15px; padding: 1.5rem; border: 1px solid rgba(34, 197, 94, 0.3); margin-top: 1rem;">
<h3 style="color: #22c55e; margin-bottom: 0.5rem;">
{nutrition['emoji']} Nutrition Facts
</h3>
<p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
{nutrition['name']} - per 100g
</p>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;">
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #f97316;">{nutrition['calories']}</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Calories</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">{nutrition['protein']}g</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Protein</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #a855f7;">{nutrition['carbs']}g</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Carbs</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #eab308;">{nutrition['fat']}g</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Fat</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #22c55e;">{nutrition['fiber']}g</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Fiber</div>
</div>
<div style="background: rgba(0,0,0,0.2); padding: 0.75rem; border-radius: 8px; text-align: center;">
<div style="font-size: 1.5rem; font-weight: bold; color: #ec4899;">{nutrition['sugar']}g</div>
<div style="font-size: 0.75rem; color: #94a3b8;">Sugar</div>
</div>
</div>

<div style="margin-top: 1rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
<div style="background: rgba(0,0,0,0.15); padding: 0.5rem; border-radius: 6px; display: flex; justify-content: space-between;">
<span style="color: #94a3b8; font-size: 0.85rem;">Vitamin C</span>
<span style="color: #ffffff; font-weight: 500;">{nutrition['vitamin_c']}mg</span>
</div>
<div style="background: rgba(0,0,0,0.15); padding: 0.5rem; border-radius: 6px; display: flex; justify-content: space-between;">
<span style="color: #94a3b8; font-size: 0.85rem;">Potassium</span>
<span style="color: #ffffff; font-weight: 500;">{nutrition['potassium']}mg</span>
</div>
<div style="background: rgba(0,0,0,0.15); padding: 0.5rem; border-radius: 6px; display: flex; justify-content: space-between;">
<span style="color: #94a3b8; font-size: 0.85rem;">Calcium</span>
<span style="color: #ffffff; font-weight: 500;">{nutrition['calcium']}mg</span>
</div>
<div style="background: rgba(0,0,0,0.15); padding: 0.5rem; border-radius: 6px; display: flex; justify-content: space-between;">
<span style="color: #94a3b8; font-size: 0.85rem;">Iron</span>
<span style="color: #ffffff; font-weight: 500;">{nutrition['iron']}mg</span>
</div>
</div>

<div style="margin-top: 1rem; background: rgba(34, 197, 94, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #22c55e;">
<p style="margin: 0; font-size: 0.85rem; color: #94a3b8;">
💡 <strong style="color: #22c55e;">Health Tip:</strong> {nutrition['health_note']}
</p>
</div>
</div>
"""


if __name__ == "__main__":
    # Test the module
    print("Testing Nutrition Data Module")
    print("=" * 50)
    
    for class_name in ['banana', 'chicken', 'broccoli']:
        nutrition = get_nutrition(class_name)
        if nutrition:
            print(f"\\n{nutrition['emoji']} {nutrition['name']}")
            print(f"  Calories: {nutrition['calories']} kcal")
            print(f"  Protein: {nutrition['protein']}g")
            print(f"  Carbs: {nutrition['carbs']}g")
            print(f"  Fat: {nutrition['fat']}g")
