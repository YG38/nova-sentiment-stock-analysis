"""
Prediction models for stock price movement based on sentiment and technical indicators.
"""

__all__ = ["SentimentModel"]

class SentimentModel:
    """
    A model for predicting stock price movements based on sentiment analysis
    and technical indicators.
    """
    
    def __init__(self):
        """Initialize the sentiment model."""
        self.model = None
        self.feature_columns = None
    
    def train(self, X, y):
        """
        Train the sentiment model.
        
        Args:
            X (pd.DataFrame): Features for training
            y (pd.Series): Target variable
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted classes
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            pd.DataFrame: Feature importances
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        import pandas as pd
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
