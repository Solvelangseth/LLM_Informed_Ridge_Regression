import os
from dotenv import load_dotenv 
from typing import List, Dict, Optional, Tuple
import openai
import pandas as pd 
import json
import re

class LLMPriorElicitor:
    """
    Handles LLM communication for prior elicitation
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize LLM Prior Elicitor 

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

        # TODO: Add later for advanced features
        # self.domain_templates = self._load_domain_templates()
        # self.confidence_mapping = {...}

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        load_dotenv()  # loads variables from .env file
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        
        for key_name in api_keys:
            api_key = os.getenv(key_name)
            if api_key:
                return api_key
        
        return None

    def _load_domain_templates(self) -> Dict:
        """Load domain-specific templates - TODO: implement later"""
        return {}
    
    def create_domain_context(self, domain, feature_names):
        """Create domain context - TODO: implement later"""
        return {}

    def create_simple_prompt(self, feature_names: List[str], target_name: str) -> str:
        """Create hardcoded prompt for MVP"""
        prompt = f"""You are a world-class statistics and domain expert. I need your expertise to set informative prior beliefs for a ridge regression model.

TASK: Analyze the relationship between these features and target variable, then provide prior coefficient estimates.

FEATURES: {feature_names}
TARGET: {target_name}

Please follow this process:

1. DOMAIN ANALYSIS: First, reason about what domain this appears to be (real estate, marketing, healthcare, etc.) based on the variable names.

2. LITERATURE REASONING: Discuss what existing research and domain knowledge suggests about how each feature typically relates to the target variable. Consider:
   - Expected direction of relationship (positive/negative)
   - Typical magnitude of effects in this domain
   - Any known benchmarks or standard coefficients

3. COEFFICIENT ESTIMATES: Based on your analysis, provide your best estimate for each coefficient.

RESPONSE FORMAT:
Please structure your response with your reasoning first, then end with a JSON block like this:

```json
{{
  "domain": "your_domain_assessment",
  "priors": {{"""

        # Add each feature dynamically
        for i, feature in enumerate(feature_names):
            prompt += f'\n    "{feature}": your_coefficient_estimate'
            if i < len(feature_names) - 1:
                prompt += ","

        prompt += """
  }}
}}
```

Be thorough in your reasoning but concise in your final estimates."""

        return prompt 
    
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

    def parse_simple_response(self, response_text: str, feature_names: List[str]) -> Optional[Dict]:
        """Parse LLM response and extract target coefficients"""
        if not response_text:
            print("No response text to parse")
            return None
        
        try:
            # Try to find JSON block in the response
            # Look for ```json or just { ... } with target_coefficients
            json_pattern = r'```json\s*(\{.*?\})\s*```|(\{[^}]*"target_coefficients"[^}]*\{[^}]*\}[^}]*\})'
            
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                # Get the matched JSON string (either from group 1 or 2)
                json_str = match.group(1) or match.group(2)
            else:
                # Fallback: try to find any JSON-like structure
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                else:
                    print("Could not find JSON in response")
                    return None
            
            # Parse the JSON
            parsed_data = json.loads(json_str)
            
            # Extract target coefficients
            if 'target_coefficients' not in parsed_data:
                print("No 'target_coefficients' key found in JSON")
                return None
            
            target_coeffs = parsed_data['target_coefficients']
            
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
                        print(f"Invalid coefficient value for {feature}: {value}")
                        return None
                else:
                    missing_features.append(feature)
            
            if missing_features:
                print(f"Missing coefficients for features: {missing_features}")
                return None
            
            # Return structured response (keeping 'priors' key for backward compatibility)
            return {
                'priors': coeff_values,  # Keep this key name for existing code
                'domain': parsed_data.get('domain', 'unknown'),
                'raw_response': response_text,
                'parsed_json': parsed_data
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str[:200]}...")
            return None
        except Exception as e:
            print(f"Unexpected error in parsing: {e}")
            return None

    def get_priors(self, X: pd.DataFrame, y: pd.Series, target_name: str, domain: Optional[str] = None) -> Optional[Dict]:
        """
        Main method to get priors from LLM
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series  
            Target data
        target_name : str
            Name of target variable
        domain : str, optional
            Domain context (not used in MVP)
            
        Returns:
        --------
        Dict with priors and metadata, or None if failed
        """
        try:
            # Step 1: Get feature names
            feature_names = list(X.columns)
            print(f"Getting priors for features: {feature_names}")
            print(f"Target variable: {target_name}")
            
            # Step 2: Create prompt
            prompt = self.create_simple_prompt(feature_names, target_name)
            print(f"Created prompt (first 200 chars): {prompt[:200]}...")
            
            # Step 3: Call LLM API
            print("Calling LLM API...")
            response_text = self.call_llm_api(prompt)
            
            if not response_text:
                print("LLM API call failed, using fallback")
                return self.get_fallback_priors(feature_names)
            
            print(f"Got LLM response (first 300 chars): {response_text[:300]}...")
            
            # Step 4: Parse response
            print("Parsing LLM response...")
            parsed_result = self.parse_simple_response(response_text, feature_names)
            
            if not parsed_result:
                print("Response parsing failed, using fallback")
                return self.get_fallback_priors(feature_names)
            
            print(f"Successfully extracted priors: {parsed_result['priors']}")
            return parsed_result
            
        except Exception as e:
            print(f"Error in get_priors: {e}")
            return self.get_fallback_priors(feature_names)

    def get_fallback_priors(self, feature_names: List[str]) -> Dict:
        """
        Fallback priors when LLM fails
        """
        print(f"Using fallback priors (zeros) for {len(feature_names)} features")
        
        return {
            'priors': [0.0] * len(feature_names),
            'domain': 'unknown',
            'raw_response': 'fallback_used',
            'parsed_json': {'priors': {name: 0.0 for name in feature_names}}
        }

    def get_priors_with_custom_prompt(self, custom_prompt: str, feature_names: List[str]) -> Optional[Dict]:
        """
        Get priors using a custom user-provided prompt
        
        Parameters:
        -----------
        custom_prompt : str
            User's custom prompt for the LLM
        feature_names : List[str]
            Names of features (for parsing validation)
            
        Returns:
        --------
        Dict with priors and metadata, or None if failed
        """
        try:
            print("Using custom prompt from user")
            print(f"Feature names for parsing: {feature_names}")
            print(f"Custom prompt (first 200 chars): {custom_prompt[:200]}...")
            
            # Call LLM API with custom prompt
            print("Calling LLM API with custom prompt...")
            response_text = self.call_llm_api(custom_prompt)
            
            if not response_text:
                print("LLM API call failed, using fallback")
                return self.get_fallback_priors(feature_names)
            
            print(f"Got LLM response (first 300 chars): {response_text[:300]}...")
            
            # Parse response
            print("Parsing LLM response...")
            parsed_result = self.parse_simple_response(response_text, feature_names)
            
            if not parsed_result:
                print("Response parsing failed, using fallback")
                return self.get_fallback_priors(feature_names)
            
            print(f"Successfully extracted priors: {parsed_result['priors']}")
            
            # Add marker that this was a custom prompt
            parsed_result['prompt_type'] = 'custom'
            parsed_result['custom_prompt'] = custom_prompt
            
            return parsed_result
            
        except Exception as e:
            print(f"Error in get_priors_with_custom_prompt: {e}")
            return self.get_fallback_priors(feature_names)

    def get_priors_interactive(self, X: pd.DataFrame, y: pd.Series, target_name: str, 
                              use_custom_prompt: bool = False, custom_prompt: str = "") -> Optional[Dict]:
        """
        Interactive method for notebook use - choose between auto and custom prompt
        
        Parameters:
        -----------
        X, y, target_name : standard inputs
        use_custom_prompt : bool
            If True, use custom_prompt instead of auto-generated
        custom_prompt : str
            Custom prompt to use if use_custom_prompt=True
        """
        feature_names = list(X.columns)
        
        if use_custom_prompt and custom_prompt:
            print("=== USING CUSTOM PROMPT ===")
            return self.get_priors_with_custom_prompt(custom_prompt, feature_names)
        else:
            print("=== USING AUTO-GENERATED PROMPT ===")
            return self.get_priors(X, y, target_name)