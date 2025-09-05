import streamlit as st
from typing import List, Dict, Any
import random
from datetime import datetime


class SocialSupportChatbot:
    """Mock chatbot for social support application guidance."""
    
    def __init__(self):
        self.conversation_history = []
        self.responses = self._initialize_responses()
    
    def _initialize_responses(self) -> Dict[str, List[str]]:
        """Initialize chatbot responses for different scenarios."""
        return {
            "greeting": [
                "Hello! I'm here to help you with your social support application. How can I assist you today?",
                "Welcome! I can guide you through the social support application process. What would you like to know?",
                "Hi there! I'm your virtual assistant for social support applications. How may I help you?"
            ],
            "eligibility": [
                "To be eligible for social support, your monthly income per family member should generally be below 3,000 AED.",
                "Eligibility depends on several factors including income, family size, and financial situation.",
                "The system considers your income, expenses, family size, and number of dependents when determining eligibility."
            ],
            "documents": [
                "You'll need your Emirates ID, bank statements, income certificate, and any asset documentation.",
                "Required documents include: Emirates ID, recent bank statements, income certificate, and asset list if applicable.",
                "Make sure to have your Emirates ID, bank statements for the last 3 months, and income certificate ready."
            ],
            "process": [
                "The application process involves: 1) Filling out the form, 2) Uploading documents, 3) Automated validation, 4) Eligibility assessment, and 5) Decision notification.",
                "Our AI system will automatically process your application and provide a recommendation within minutes.",
                "The process is fully automated - just upload your documents and our system will handle the rest!"
            ],
            "timeline": [
                "Most applications are processed within 24-48 hours.",
                "You'll receive an email notification once your application is reviewed.",
                "The automated system typically provides immediate feedback on your eligibility status."
            ],
            "appeal": [
                "If you disagree with the decision, you can submit an appeal within 30 days.",
                "Appeals can be submitted through the same portal with additional supporting documentation.",
                "The appeal process involves a manual review by our case workers."
            ],
            "support": [
                "You can contact our support team at support@socialcare.ae or call +971-4-123-4567.",
                "Our support team is available Sunday to Thursday, 8 AM to 6 PM.",
                "For urgent matters, you can visit our office at Dubai Government Center."
            ],
            "default": [
                "I understand you're asking about that. Let me help you find the right information.",
                "That's a great question! Let me provide you with some guidance on that topic.",
                "I can help you with that. Let me give you some information that might be useful."
            ]
        }
    
    def get_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate a response based on user input and context."""
        user_input_lower = user_input.lower()
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "bot": ""
        })
        
        # Determine response category
        response_category = self._categorize_input(user_input_lower, context)
        
        # Get appropriate response
        responses = self.responses.get(response_category, self.responses["default"])
        response = random.choice(responses)
        
        # Add context-specific information if available
        if context:
            response = self._enhance_response_with_context(response, context)
        
        # Update conversation history
        self.conversation_history[-1]["bot"] = response
        
        return response
    
    def _categorize_input(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Categorize user input to determine appropriate response."""
        # Greeting patterns
        if any(word in user_input for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "greeting"
        
        # Eligibility questions
        if any(word in user_input for word in ["eligible", "qualify", "criteria", "requirements", "income"]):
            return "eligibility"
        
        # Document questions
        if any(word in user_input for word in ["document", "paper", "certificate", "statement", "upload"]):
            return "documents"
        
        # Process questions
        if any(word in user_input for word in ["process", "step", "how", "procedure", "workflow"]):
            return "process"
        
        # Timeline questions
        if any(word in user_input for word in ["time", "when", "how long", "duration", "timeline"]):
            return "timeline"
        
        # Appeal questions
        if any(word in user_input for word in ["appeal", "disagree", "reject", "denied", "challenge"]):
            return "appeal"
        
        # Support questions
        if any(word in user_input for word in ["contact", "help", "support", "phone", "email", "office"]):
            return "support"
        
        return "default"
    
    def _enhance_response_with_context(self, response: str, context: Dict[str, Any]) -> str:
        """Enhance response with context-specific information."""
        if not context:
            return response
        
        # Add application status if available
        if "application_status" in context:
            status = context["application_status"]
            if status == "approved":
                response += "\n\nGreat news! Your application has been approved. You should receive further instructions soon."
            elif status == "declined":
                response += "\n\nI see your application was declined. Would you like to know about the appeal process?"
            elif status == "pending":
                response += "\n\nYour application is currently being processed. You'll be notified once it's complete."
        
        # Add eligibility information if available
        if "eligibility_result" in context:
            eligibility = context["eligibility_result"]
            if eligibility.get("is_eligible", False):
                response += f"\n\nBased on your application, you appear to be eligible with a confidence score of {eligibility.get('confidence_score', 0):.1%}."
            else:
                response += f"\n\nBased on your application, you may not meet the current eligibility criteria. However, you can still apply for manual review."
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for the user."""
        return [
            "What documents do I need to upload?",
            "What are the eligibility criteria?",
            "How long does the application process take?",
            "What if my application is declined?",
            "How can I contact support?",
            "What is the income threshold for eligibility?",
            "Can I appeal a decision?",
            "What happens after I submit my application?"
        ]
    
    def get_quick_actions(self) -> List[Dict[str, str]]:
        """Get quick action buttons for common tasks."""
        return [
            {"label": "Check Eligibility", "action": "check_eligibility"},
            {"label": "Required Documents", "action": "show_documents"},
            {"label": "Application Status", "action": "check_status"},
            {"label": "Contact Support", "action": "contact_support"},
            {"label": "FAQ", "action": "show_faq"},
            {"label": "Start New Application", "action": "new_application"}
        ]
    
    def handle_quick_action(self, action: str) -> str:
        """Handle quick action requests."""
        action_responses = {
            "check_eligibility": "To check your eligibility, please fill out the application form with your personal and financial information. Our AI system will automatically assess your eligibility based on the criteria.",
            "show_documents": "Required documents include:\n• Emirates ID (front and back)\n• Bank statements (last 3 months)\n• Income certificate\n• Asset list (if applicable)\n• Any other supporting documents",
            "check_status": "To check your application status, please provide your application reference number. You can also check the status in your dashboard.",
            "contact_support": "You can contact our support team through:\n• Email: support@socialcare.ae\n• Phone: +971-4-123-4567\n• Office: Dubai Government Center\n• Hours: Sunday-Thursday, 8 AM-6 PM",
            "show_faq": "Here are some frequently asked questions:\n• Q: What is the income threshold? A: Generally below 3,000 AED per family member\n• Q: How long does it take? A: Usually 24-48 hours\n• Q: Can I appeal? A: Yes, within 30 days of decision",
            "new_application": "To start a new application, please click on the 'New Application' button in the main menu. Make sure you have all required documents ready before starting."
        }
        
        return action_responses.get(action, "I'm not sure how to help with that action. Please ask me a specific question.")
