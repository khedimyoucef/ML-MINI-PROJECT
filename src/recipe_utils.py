"""
Recipe Recommendation Utilities

This module provides functionality to load recipe data and recommend recipes
based on available ingredients.
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path


class RecipeRecommender:
    def __init__(self, data_path: str):
        """
        Initialize the RecipeRecommender.
        
        Args:
            data_path: Path to the recipes CSV file
        """
        self.data_path = data_path
        self.df = None
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the recipes dataset."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Recipe dataset not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Parse the Cleaned_Ingredients column from string to list
        # We use a safe evaluation since the data is a string representation of a list
        self.df['parsed_ingredients'] = self.df['Cleaned_Ingredients'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        
        # We also want a lowercased simplified version for easier matching
        # This is a basic extraction of words
        self.df['search_ingredients'] = self.df['parsed_ingredients'].apply(
            lambda ig_list: [self._simplify_ingredient(ig) for ig in ig_list]
        )
        
    def _simplify_ingredient(self, ingredient: str) -> str:
        """
        Simplify ingredient string for matching.
        Converts to lowercase and removes special chars.
        """
        return ingredient.lower()
        
    def search_recipes(
        self, 
        user_ingredients: List[str], 
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for recipes that can be made with user ingredients.
        
        Args:
            user_ingredients: List of ingredients the user has
            top_k: Number of recipes to return
            
        Returns:
            List of recipe dictionaries with match details
        """
        if self.df is None:
            self._load_data()
            
        user_ingredients_lower = [i.lower().strip() for i in user_ingredients]
        
        results = []
        
        # Iterate through recipes and calculate match score
        # Note: For a very large dataset, this linear scan could be optimized 
        # (e.g., using an inverted index), but for this mini-project it's sufficient.
        
        for idx, row in self.df.iterrows():
            recipe_ingredients = row['search_ingredients']
            
            # Find which user ingredients are in this recipe
            # logic: check if any user ingredient is a substring of the recipe ingredient
            # or vice versa to be flexible
            
            matching_indices = set()
            matches = []
            
            for u_ing in user_ingredients_lower:
                for r_idx, r_ing in enumerate(recipe_ingredients):
                    # Check for partial match (word overlap)
                    if u_ing in r_ing or any(word in r_ing.split() for word in u_ing.split() if len(word) > 2):
                        matching_indices.add(r_idx)
                        # We store the user ingredient that matched
                        matches.append(r_ing)
                        break 
            
            # Identify missing ingredients
            all_indices = set(range(len(recipe_ingredients)))
            missing_indices = all_indices - matching_indices
            
            missing_ingredients = [row['parsed_ingredients'][i] for i in missing_indices]
            matching_ingredients_display = [row['parsed_ingredients'][i] for i in matching_indices]
            
            match_count = len(matching_indices)
            total_ingredients = len(recipe_ingredients)
            
            if total_ingredients == 0:
                continue
                
            match_percentage = match_count / total_ingredients
            
            if match_count > 0:
                results.append({
                    'Title': row['Title'],
                    'Instructions': row['Instructions'],
                    'Image_Name': row['Image_Name'],
                    'Ingredients': row['parsed_ingredients'],
                    'Matching_Ingredients': matching_ingredients_display,
                    'Missing_Ingredients': missing_ingredients,
                    'Match_Count': match_count,
                    'Match_Percentage': match_percentage,
                    'Total_Ingredients': total_ingredients
                })
        
        # Sort by match percentage (descending) and then match count
        results.sort(key=lambda x: (x['Match_Percentage'], x['Match_Count']), reverse=True)
        
        return results[:top_k]

if __name__ == "__main__":
    # Test
    path = "DS3RECIPES/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    if Path(path).exists():
        recommender = RecipeRecommender(path)
        print("Data loaded successfully")
        
        test_ingredients = ["chicken", "salt", "pepper", "butter"]
        recommendations = recommender.search_recipes(test_ingredients, top_k=2)
        
        for rec in recommendations:
            print(f"\nRecipe: {rec['Title']}")
            print(f"Match: {rec['Match_Percentage']:.1%}")
            print(f"Missing: {len(rec['Missing_Ingredients'])} ingredients")
