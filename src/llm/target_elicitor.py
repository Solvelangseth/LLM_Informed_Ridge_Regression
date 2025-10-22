import os
from dotenv import load_dotenv 
from typing import List, Dict, Optional, Tuple
import openai
import pandas as pd 
import json
import re

class LLMTargetElicitor:
    """
    Handles LLM communication for regularization target elicitation.
    Focuses on processing user-crafted prompts and extracting coefficient targets.
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize LLM Target Elicitor 

        Parameters: 
        -----------
        model_name : str
            LLM model to use (e.g., 'gpt-4', 'claude-3-sonnet')
        api_key : str, optional
            API-key for LLM service (if None, will try to get from .env)
        """
        
        self.model_name = model_name 
        self.api_key = api_key or self._get_api_key_from_env()

        # Supported models and their API endpoints
        self.supported_models = {
            'gpt-4': 'openai',
            'gpt-3.5-turbo': 'openai',
            'claude-3-sonnet': 'anthropic',
            'claude-3-haiku': 'anthropic'
        }

        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model {model_name}. Supported list {list(self.supported_models.keys())}")
        
        self.api_provider = self.supported_models[model_name]

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        load_dotenv()  # loads variables from .env file
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        
        for key_name in api_keys:
            api_key = os.getenv(key_name)
            if api_key:
                return api_key
        
        return None

    def validate_prompt_structure(self, prompt: str, feature_names: List[str]) -> Dict[str, bool]:
        """
        Validate that the prompt has the correct structure for target extraction.
        
        Parameters:
        -----------
        prompt : str
            User-crafted prompt to validate
        feature_names : List[str]
            Expected feature names that should be in the JSON response
            
        Returns:
        --------
        Dict with validation results
        """
        validation = {
            'has_json_format': False,
            'mentions_targets_key': False,
            'includes_all_features': False,
            'has_domain_key': False,
            'overall_valid': False,
            'issues': []
        }
        
        # Check for JSON format instruction
        json_indicators = ['```json', '```JSON', 'JSON', 'json', '"targets"', '"domain"']
        if any(indicator in prompt for indicator in json_indicators):
            validation['has_json_format'] = True
        else:
            validation['issues'].append("Prompt should instruct LLM to return JSON format")
        
        # Check for targets key instruction
        if '"targets"' in prompt or 'targets' in prompt.lower():
            validation['mentions_targets_key'] = True
        else:
            validation['issues'].append("Prompt should instruct LLM to use 'targets' key in JSON")
        
        # Check if prompt mentions all features
        features_mentioned = sum(1 for feature in feature_names if feature in prompt)
        if features_mentioned >= len(feature_names) * 0.8:  # Allow some flexibility
            validation['includes_all_features'] = True
        else:
            validation['issues'].append(f"Prompt mentions only {features_mentioned}/{len(feature_names)} features")
        
        # Check for domain analysis instruction
        domain_indicators = ['domain', 'Domain', 'DOMAIN']
        if any(indicator in prompt for indicator in domain_indicators):
            validation['has_domain_key'] = True
        else:
            validation['issues'].append("Consider asking LLM to identify the domain")
        
        # Overall validation
        required_checks = ['has_json_format', 'mentions_targets_key']
        validation['overall_valid'] = all(validation[check] for check in required_checks)
        
        return validation
    
    def call_llm_api(self, prompt: str) -> Optional[str]:
        """Make API call to LLM"""
        try:
            if self.api_provider == 'openai':
                # Use new OpenAI API v1.0+ syntax
                client = openai.OpenAI(api_key=self.api_key)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                return response.choices[0].message.content
                
            elif self.api_provider == 'anthropic':
                import anthropic
                
                client = anthropic.Anthropic(api_key=self.api_key)
                
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.content[0].text
                
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def parse_response(self, response_text: str, feature_names: List[str]) -> Optional[Dict]:
        """Parse LLM response and extract coefficient targets"""
        if not response_text:
            print("No response text to parse")
            return None
        
        try:
            # Try to find JSON block in the response
            # Look for ```json blocks first, then fallback to any JSON-like structure
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # ```json blocks
                r'```JSON\s*(\{.*?\})\s*```',  # ```JSON blocks  
                r'```\s*(\{.*?\})\s*```',      # Generic code blocks with JSON
                r'(\{[^}]*"targets"[^}]*\{[^}]*\}[^}]*\})',  # JSON with targets key
                r'(\{.*?"domain".*?"targets".*?\})',  # JSON with domain and targets
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    break
            
            if not json_str:
                # Final fallback: find any JSON-like structure
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                else:
                    print("Could not find JSON structure in response")
                    return None
            
            # Parse the JSON
            parsed_data = json.loads(json_str)
            
            # Extract coefficient targets
            if 'targets' not in parsed_data:
                print("No 'targets' key found in JSON response")
                print(f"Available keys: {list(parsed_data.keys())}")
                return None
            
            target_coeffs = parsed_data['targets']
            
            # Validate that we have coefficients for all features
            coeff_values = []
            missing_features = []
            
            for feature in feature_names:
                if feature in target_coeffs:
                    value = target_coeffs[feature]
                    # Convert to float if it's a string number
                    try:
                        coeff_values.append(float(value))
                    except (ValueError, TypeError):
                        print(f"Invalid coefficient target value for {feature}: {value}")
                        return None
                else:
                    missing_features.append(feature)
            
            if missing_features:
                print(f"Missing coefficient targets for features: {missing_features}")
                print(f"Available targets: {list(target_coeffs.keys())}")
                return None
            
            # Return structured response
            return {
                'targets': coeff_values,
                'domain': parsed_data.get('domain', 'unknown'),
                'raw_response': response_text,
                'parsed_json': parsed_data
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str[:200] if json_str else 'None'}...")
            return None
        except Exception as e:
            print(f"Unexpected error in parsing: {e}")
            return None

    def get_targets_with_prompt(self, custom_prompt: str, feature_names: List[str], 
                               validate_prompt: bool = True) -> Optional[Dict]:
        """
        Get regularization targets using a user-provided prompt.
        
        Parameters:
        -----------
        custom_prompt : str
            User's custom prompt for the LLM
        feature_names : List[str]
            Names of features (for parsing validation)
        validate_prompt : bool, default=True
            Whether to validate prompt structure before sending
            
        Returns:
        --------
        Dict with targets and metadata, or None if failed
        """
        try:
            print("Using user-crafted prompt")
            print(f"Feature names expected: {feature_names}")
            print(f"Prompt length: {len(custom_prompt)} characters")
            
            # Validate prompt structure if requested
            if validate_prompt:
                validation = self.validate_prompt_structure(custom_prompt, feature_names)
                print(f"Prompt validation: {'PASSED' if validation['overall_valid'] else 'ISSUES FOUND'}")
                
                if validation['issues']:
                    print("Validation issues:")
                    for issue in validation['issues']:
                        print(f"  - {issue}")
                
                if not validation['overall_valid']:
                    print("Warning: Prompt may not work as expected. Continue anyway.")
            
            # Call LLM API with custom prompt
            print("Calling LLM API...")
            response_text = self.call_llm_api(custom_prompt)
            
            if not response_text:
                print("LLM API call failed")
                return None
            
            print(f"Got LLM response ({len(response_text)} characters)")
            print(f"Response preview: {response_text[:200]}...")
            
            # Parse response
            print("Parsing LLM response...")
            parsed_result = self.parse_response(response_text, feature_names)
            
            if not parsed_result:
                print("Response parsing failed")
                return None
            
            print(f"Successfully extracted targets: {parsed_result['targets']}")
            
            # Add metadata
            parsed_result['prompt_type'] = 'user_crafted'
            parsed_result['custom_prompt'] = custom_prompt
            parsed_result['validation'] = validation if validate_prompt else None
            
            return parsed_result
            
        except Exception as e:
            print(f"Error in get_targets_with_prompt: {e}")
            return None

    def get_fallback_targets(self, feature_names: List[str]) -> Dict:
        """
        Fallback targets when LLM fails (all zeros = traditional Ridge)
        """
        print(f"Using fallback targets (zeros) for {len(feature_names)} features")
        
        return {
            'targets': [0.0] * len(feature_names),
            'domain': 'unknown',
            'raw_response': 'fallback_used_zero_targets',
            'parsed_json': {'targets': {name: 0.0 for name in feature_names}},
            'prompt_type': 'fallback'
        }