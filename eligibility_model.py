import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Dict, Any, Tuple
import joblib
import os
from models.applicant import Applicant, EligibilityResult


class EligibilityModel:
    """ML model for determining social support eligibility."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'monthly_income', 'family_size', 'dependents', 'monthly_expenses',
            'income_per_member', 'debt_to_income_ratio', 'savings_rate',
            'age', 'has_emirates_id', 'has_bank_statement', 'has_income_certificate'
        ]
        self.model_path = "models/eligibility_model.pkl"
        self.scaler_path = "models/scaler.pkl"
    
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data for the model."""
        np.random.seed(42)
        
        data = []
        labels = []
        
        for _ in range(n_samples):
            # Generate realistic data
            family_size = np.random.randint(1, 8)
            dependents = np.random.randint(0, family_size)
            age = np.random.randint(18, 65)
            
            # Income based on family size and age
            base_income = np.random.normal(3000, 1500)
            income_multiplier = 1 + (family_size - 1) * 0.3
            monthly_income = max(500, base_income * income_multiplier)
            
            # Expenses based on income and family size
            expense_ratio = np.random.uniform(0.6, 1.2)
            monthly_expenses = monthly_income * expense_ratio
            
            # Calculate derived features
            income_per_member = monthly_income / family_size
            debt_to_income_ratio = monthly_expenses / monthly_income
            savings_rate = max(0, (monthly_income - monthly_expenses) / monthly_income)
            
            # Document availability (higher chance for lower income)
            has_emirates_id = np.random.random() > 0.05
            has_bank_statement = np.random.random() > 0.1
            has_income_certificate = np.random.random() > 0.15
            
            # Determine eligibility based on rules
            is_eligible = self._determine_eligibility_rules(
                monthly_income, family_size, dependents, monthly_expenses,
                income_per_member, debt_to_income_ratio, age
            )
            
            data.append([
                monthly_income, family_size, dependents, monthly_expenses,
                income_per_member, debt_to_income_ratio, savings_rate,
                age, has_emirates_id, has_bank_statement, has_income_certificate
            ])
            labels.append(1 if is_eligible else 0)
        
        df = pd.DataFrame(data, columns=self.feature_columns)
        return df, np.array(labels)
    
    def _determine_eligibility_rules(self, monthly_income: float, family_size: int, 
                                   dependents: int, monthly_expenses: float,
                                   income_per_member: float, debt_to_income_ratio: float,
                                   age: int) -> bool:
        """Determine eligibility based on business rules."""
        # High priority: Very low income
        if income_per_member < 1500:
            return True
        
        # High priority: Large family with low income
        if family_size >= 4 and income_per_member < 2500:
            return True
        
        # High priority: High debt ratio with dependents
        if debt_to_income_ratio > 0.9 and dependents > 0:
            return True
        
        # Medium priority: Moderate income with many dependents
        if income_per_member < 3000 and dependents >= 3:
            return True
        
        # Low priority: Some savings but still struggling
        if income_per_member < 4000 and debt_to_income_ratio > 0.8:
            return True
        
        # Not eligible: High income
        if income_per_member > 5000:
            return False
        
        # Not eligible: Good financial situation
        if debt_to_income_ratio < 0.6 and income_per_member > 3000:
            return False
        
        return False
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train the eligibility model."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            return {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred),
                "feature_importance": dict(zip(self.feature_columns, self.model.feature_importances_))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def predict_eligibility(self, applicant: Applicant, documents: List[Any]) -> EligibilityResult:
        """Predict eligibility for a single applicant."""
        if not self.is_trained:
            # Train with generated data if not already trained
            X, y = self.generate_training_data()
            self.train_model(X, y)
        
        try:
            # Prepare features
            features = self._prepare_features(applicant, documents)
            features_df = pd.DataFrame([features], columns=self.feature_columns)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            confidence = max(probability)
            
            # Generate reasons and recommendations
            reasons, recommendations = self._generate_explanation(applicant, prediction, confidence)
            
            # Determine priority level
            priority_level = self._determine_priority_level(applicant, prediction, confidence)
            
            return EligibilityResult(
                is_eligible=bool(prediction),
                confidence_score=confidence,
                reasons=reasons,
                recommendations=recommendations,
                priority_level=priority_level
            )
            
        except Exception as e:
            return EligibilityResult(
                is_eligible=False,
                confidence_score=0.0,
                reasons=[f"Error in prediction: {str(e)}"],
                recommendations=["Please contact support"],
                priority_level="low"
            )
    
    def _prepare_features(self, applicant: Applicant, documents: List[Any]) -> List[float]:
        """Prepare features for prediction."""
        # Calculate derived features
        income_per_member = applicant.monthly_income / applicant.family_size
        debt_to_income_ratio = applicant.monthly_expenses / applicant.monthly_income if applicant.monthly_income > 0 else 0
        savings_rate = max(0, (applicant.monthly_income - applicant.monthly_expenses) / applicant.monthly_income) if applicant.monthly_income > 0 else 0
        
        # Calculate age
        try:
            from datetime import datetime
            birth_date = datetime.strptime(applicant.date_of_birth, '%Y-%m-%d')
            age = (datetime.now() - birth_date).days // 365
        except:
            age = 30  # Default age
        
        # Check document availability
        doc_types = [doc.document_type for doc in documents] if documents else []
        has_emirates_id = 1.0 if any(doc_type.value == 'emirates_id' for doc_type in doc_types) else 0.0
        has_bank_statement = 1.0 if any(doc_type.value == 'bank_statement' for doc_type in doc_types) else 0.0
        has_income_certificate = 1.0 if any(doc_type.value == 'income_certificate' for doc_type in doc_types) else 0.0
        
        return [
            applicant.monthly_income, applicant.family_size, applicant.dependents, applicant.monthly_expenses,
            income_per_member, debt_to_income_ratio, savings_rate,
            age, has_emirates_id, has_bank_statement, has_income_certificate
        ]
    
    def _generate_explanation(self, applicant: Applicant, prediction: int, confidence: float) -> Tuple[List[str], List[str]]:
        """Generate explanation for the prediction."""
        reasons = []
        recommendations = []
        
        income_per_member = applicant.monthly_income / applicant.family_size
        
        if prediction == 1:  # Eligible
            if income_per_member < 1500:
                reasons.append("Very low income per family member")
                recommendations.append("Consider additional financial support programs")
            elif applicant.family_size >= 4 and income_per_member < 2500:
                reasons.append("Large family with limited income")
                recommendations.append("Family support programs may be beneficial")
            elif applicant.monthly_expenses / applicant.monthly_income > 0.9:
                reasons.append("High debt-to-income ratio")
                recommendations.append("Financial counseling recommended")
            else:
                reasons.append("Meets eligibility criteria for social support")
                recommendations.append("Application approved for review")
        else:  # Not eligible
            if income_per_member > 5000:
                reasons.append("Income exceeds eligibility threshold")
                recommendations.append("Consider other government programs")
            elif applicant.monthly_expenses / applicant.monthly_income < 0.6:
                reasons.append("Good financial stability")
                recommendations.append("No immediate support needed")
            else:
                reasons.append("Does not meet current eligibility criteria")
                recommendations.append("Reapply if circumstances change")
        
        if confidence < 0.7:
            reasons.append("Low confidence in assessment")
            recommendations.append("Manual review recommended")
        
        return reasons, recommendations
    
    def _determine_priority_level(self, applicant: Applicant, prediction: int, confidence: float) -> str:
        """Determine priority level for processing."""
        if prediction == 0:  # Not eligible
            return "low"
        
        income_per_member = applicant.monthly_income / applicant.family_size
        
        if income_per_member < 1000 or applicant.family_size >= 6:
            return "high"
        elif income_per_member < 2000 or applicant.dependents >= 3:
            return "medium"
        else:
            return "low"
    
    def _save_model(self):
        """Save trained model and scaler."""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def load_model(self) -> bool:
        """Load pre-trained model and scaler."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_columns, self.model.feature_importances_))
