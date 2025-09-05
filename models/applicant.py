from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ApplicationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    NEEDS_REVIEW = "needs_review"


class DocumentType(Enum):
    EMIRATES_ID = "emirates_id"
    BANK_STATEMENT = "bank_statement"
    INCOME_CERTIFICATE = "income_certificate"
    ASSET_LIST = "asset_list"


@dataclass
class Applicant:
    """Represents a social support application applicant."""
    
    # Personal Information
    full_name: str
    emirates_id: str
    phone_number: str
    email: str
    date_of_birth: str
    
    # Financial Information
    monthly_income: float
    family_size: int
    dependents: int
    monthly_expenses: float
    
    # Application Details
    application_date: str
    status: ApplicationStatus = ApplicationStatus.PENDING
    eligibility_score: Optional[float] = None
    recommendation: Optional[str] = None
    
    # Documents
    documents: List[DocumentType] = None
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = []


@dataclass
class Document:
    """Represents an uploaded document."""
    
    document_type: DocumentType
    file_path: str
    file_name: str
    file_size: int
    upload_date: str
    extracted_data: Optional[dict] = None


@dataclass
class EligibilityResult:
    """Represents the result of eligibility assessment."""
    
    is_eligible: bool
    confidence_score: float
    reasons: List[str]
    recommendations: List[str]
    priority_level: str  # "high", "medium", "low"
