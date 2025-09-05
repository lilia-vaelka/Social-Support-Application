import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
from models.applicant import Applicant, Document, DocumentType, ApplicationStatus


class DataIngestionService:
    """Handles data ingestion and extraction from various sources."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.csv', '.xlsx']
    
    def load_mock_applications(self, file_path: str) -> List[Applicant]:
        """Load sample applications from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            applications = []
            for app_data in data:
                applicant = Applicant(
                    full_name=app_data['full_name'],
                    emirates_id=app_data['emirates_id'],
                    phone_number=app_data['phone_number'],
                    email=app_data['email'],
                    date_of_birth=app_data['date_of_birth'],
                    monthly_income=app_data['monthly_income'],
                    family_size=app_data['family_size'],
                    dependents=app_data['dependents'],
                    monthly_expenses=app_data['monthly_expenses'],
                    application_date=app_data['application_date'],
                    documents=[DocumentType(doc) for doc in app_data['documents']]
                )
                applications.append(applicant)
            
            return applications
        except Exception as e:
            st.error(f"Error loading mock applications: {str(e)}")
            return []
    
    def extract_data_from_form(self, form_data: Dict[str, Any]) -> Applicant:
        """Extract data from uploaded form."""
        try:
            applicant = Applicant(
                full_name=form_data.get('full_name', ''),
                emirates_id=form_data.get('emirates_id', ''),
                phone_number=form_data.get('phone_number', ''),
                email=form_data.get('email', ''),
                date_of_birth=form_data.get('date_of_birth', ''),
                monthly_income=float(form_data.get('monthly_income', 0)),
                family_size=int(form_data.get('family_size', 1)),
                dependents=int(form_data.get('dependents', 0)),
                monthly_expenses=float(form_data.get('monthly_expenses', 0)),
                application_date=datetime.now().strftime('%Y-%m-%d'),
                documents=[]
            )
            return applicant
        except Exception as e:
            st.error(f"Error extracting form data: {str(e)}")
            return None
    
    def process_uploaded_documents(self, uploaded_files: List[Any]) -> List[Document]:
        """Process uploaded documents and extract metadata."""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Determine document type based on filename
                doc_type = self._determine_document_type(uploaded_file.name)
                
                document = Document(
                    document_type=doc_type,
                    file_path=uploaded_file.name,
                    file_name=uploaded_file.name,
                    file_size=uploaded_file.size,
                    upload_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    extracted_data=self._extract_document_data(uploaded_file, doc_type)
                )
                documents.append(document)
                
            except Exception as e:
                st.warning(f"Error processing document {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def _determine_document_type(self, filename: str) -> DocumentType:
        """Determine document type based on filename."""
        filename_lower = filename.lower()
        
        if 'emirates' in filename_lower or 'id' in filename_lower:
            return DocumentType.EMIRATES_ID
        elif 'bank' in filename_lower or 'statement' in filename_lower:
            return DocumentType.BANK_STATEMENT
        elif 'income' in filename_lower or 'salary' in filename_lower:
            return DocumentType.INCOME_CERTIFICATE
        elif 'asset' in filename_lower or 'property' in filename_lower:
            return DocumentType.ASSET_LIST
        else:
            return DocumentType.EMIRATES_ID  # Default
    
    def _extract_document_data(self, uploaded_file: Any, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract data from uploaded document (mock implementation)."""
        # In a real implementation, this would use OCR, PDF parsing, etc.
        mock_data = {
            DocumentType.EMIRATES_ID: {
                "id_number": "784-1985-1234567-8",
                "name": "Ahmed Al-Rashid",
                "nationality": "UAE",
                "expiry_date": "2029-03-15"
            },
            DocumentType.BANK_STATEMENT: {
                "account_balance": 5000.0,
                "monthly_transactions": 15,
                "average_balance": 4500.0,
                "statement_period": "2024-01-01 to 2024-01-31"
            },
            DocumentType.INCOME_CERTIFICATE: {
                "employer": "ABC Company LLC",
                "position": "Software Developer",
                "monthly_salary": 2500.0,
                "employment_start_date": "2022-01-01"
            },
            DocumentType.ASSET_LIST: {
                "total_assets": 15000.0,
                "property_value": 10000.0,
                "vehicle_value": 5000.0,
                "other_assets": 0.0
            }
        }
        
        return mock_data.get(doc_type, {})
    
    def validate_emirates_id(self, emirates_id: str) -> bool:
        """Validate Emirates ID format."""
        if not emirates_id:
            return False
        
        # Basic format validation: 784-YYYY-NNNNNNN-N
        parts = emirates_id.split('-')
        if len(parts) != 4:
            return False
        
        try:
            # Check if it's a valid format
            if parts[0] == '784' and len(parts[1]) == 4 and len(parts[2]) == 7 and len(parts[3]) == 1:
                return True
        except:
            pass
        
        return False
    
    def calculate_financial_ratios(self, applicant: Applicant) -> Dict[str, float]:
        """Calculate financial ratios for assessment."""
        if applicant.monthly_income == 0:
            return {}
        
        return {
            "debt_to_income_ratio": applicant.monthly_expenses / applicant.monthly_income,
            "savings_rate": (applicant.monthly_income - applicant.monthly_expenses) / applicant.monthly_income,
            "income_per_family_member": applicant.monthly_income / applicant.family_size,
            "expense_per_family_member": applicant.monthly_expenses / applicant.family_size
        }
