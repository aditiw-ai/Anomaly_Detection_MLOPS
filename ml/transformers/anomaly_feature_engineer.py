"""
Anomaly Feature Engineering Pipeline
Creates anomaly detection features for product comparison.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class AnomalyFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Price Difference
        df["Price_Difference"] = (
            df["Price In Dollar"] - df["Original Price In Dollar"]
        )

        # Weight Difference
        df["Weight_Difference"] = (
            df["Final Weights in Grams"] - df["Original Final Weights in Grams"]
        )

        # Length Difference
        df["Length_Difference"] = (
            df["Length"] - df["Original Length"]
        )

        # Width Difference
        df["Width_Difference"] = (
            df["Width"] - df["Original Width"]
        )

        # Height Difference
        df["Height_Difference"] = (
            df["Height"] - df["Original Height"]
        )
                # Price Percentage Change
        df["Price_Percentage_Change"] = (
            (df["Price_Difference"] / df["Original Price In Dollar"]) * 100
        )

        # Weight Percentage Change
        df["Weight_Percentage_Change"] = (
            (df["Weight_Difference"] / df["Original Final Weights in Grams"]) * 100
        )


        # Price Percentage Change
        df["Price_Percentage_Change"] = (
                    (df["Price In Dollar"] - df["Original Price In Dollar"])
                    / df["Original Price In Dollar"]
        ) * 100
        
        
                # Weight Percentage Change
        df["Weight_Percentage_Change"] = (
                   (df["Final Weights in Grams"] - df["Original Final Weights in Grams"])
                   / df["Original Final Weights in Grams"]
        ) * 100

        # Volume
        df["Volume"] = (
            df["Length"]
            * df["Width"]
            * df["Height"]
        )

        # Original Volume
        df["Original_Volume"] = (
           df["Original Length"]
           * df["Original Width"]
           * df["Original Height"]
        )

        # Volume Difference
        df["Volume_Difference"] = (
           df["Volume"]
           - df["Original_Volume"]
        )

        # Density
        df["Density"] = (
           df["Final Weights in Grams"]
           / df["Volume"]
        )

        # Original Density
        df["Original_Density"] = (
           df["Original Final Weights in Grams"]
           / df["Original_Volume"]
        )

        # Density Difference
        df["Density_Difference"] = (
           df["Density"]
           - df["Original_Density"]
        )
        # Price per Gram
        df["Price_per_Gram"] = (
           df["Price In Dollar"]
           / (df["Final Weights in Grams"] + 1)
        )

        # Original Price per Gram
        df["Original_Price_per_Gram"] = (
           df["Original Price In Dollar"]
           / (df["Original Final Weights in Grams"] + 1)
        )

        # Price per Gram Difference
        df["Price_per_Gram_Difference"] = (
           df["Price_per_Gram"]
           - df["Original_Price_per_Gram"]
        )

        # Volume Percentage Change
        df["Volume_Percentage_Change"] = (
           (df["Volume"] - df["Original_Volume"])
           / (df["Original_Volume"] + 1)
        ) * 100



        return df