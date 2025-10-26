"""
Advanced HYPE and Risk Detection System
Enhanced algorithms for detecting market hype and investment risks
"""

import re
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    score: float
    confidence: float
    indicators: List[str]
    patterns: List[str]
    explanation: str

class AdvancedHypeDetector:
    """Advanced HYPE detection using multiple algorithms"""
    
    def __init__(self):
        # Comprehensive hype patterns
        self.hype_patterns = {
            # Exaggerated language patterns
            'exaggeration': [
                r'\b(moon|rocket|skyrocket|explosive|breakthrough|revolutionary)\b',
                r'\b(game.?changer|disrupt|massive|huge|enormous|incredible)\b',
                r'\b(amazing|unbelievable|stunning|shocking|spectacular)\b',
                r'\b(guaranteed|sure thing|can\'t lose|easy money|quick profit)\b',
                r'\b(get rich quick|life changing|once in a lifetime)\b',
                r'\b(never seen before|unprecedented|historic|legendary)\b'
            ],
            
            # Pump and dump language
            'pump_language': [
                r'\b(to the moon|diamond hands|hodl|yolo|apes together)\b',
                r'\b(this is the way|tendies|stonks|buy the dip)\b',
                r'\b(hold the line|paper hands|diamond hands)\b',
                r'\b(rocket ship|lambo|mooning|pumping)\b'
            ],
            
            # Urgency and FOMO tactics
            'urgency': [
                r'\b(act now|limited time|don\'t miss out|last chance)\b',
                r'\b(urgent|breaking|exclusive|insider|secret)\b',
                r'\b(hidden gem|undervalued|under the radar)\b',
                r'\b(only for today|expires soon|while supplies last)\b'
            ],
            
            # Price target exaggeration
            'price_hype': [
                r'\$?\d+(?:\.\d+)?\s*(?:to|->|→)\s*\$?\d+(?:\.\d+)?',
                r'\b(10x|100x|1000x|millionaire|billionaire)\b',
                r'\b(price target|price prediction|will hit)\b',
                r'\b(guaranteed return|sure profit|can\'t go wrong)\b'
            ],
            
            # Social media hype patterns
            'social_hype': [
                r'\b(viral|trending|everyone is talking)\b',
                r'\b(influencer|celebrity|endorsement)\b',
                r'\b(community|group|following|fans)\b',
                r'\b(hashtag|#|@|follow|share|like)\b'
            ]
        }
        
        # Risk patterns
        self.risk_patterns = {
            # Financial risks
            'financial_risk': [
                r'\b(bankruptcy|delisting|penny stock|pump and dump)\b',
                r'\b(scam|fraud|insider trading|sec investigation)\b',
                r'\b(regulatory action|audit concerns|accounting issues)\b',
                r'\b(financial irregularities|debt problems|cash flow)\b'
            ],
            
            # Market risks
            'market_risk': [
                r'\b(highly volatile|speculative|risky investment)\b',
                r'\b(no guarantee|past performance|future uncertain)\b',
                r'\b(market crash|bubble|overvalued|correction)\b',
                r'\b(bear market|recession risk|economic downturn)\b'
            ],
            
            # Company risks
            'company_risk': [
                r'\b(ceo resignation|management changes|layoffs)\b',
                r'\b(restructuring|liquidity concerns|competition threat)\b',
                r'\b(market share loss|product recalls|legal issues)\b',
                r'\b(earnings miss|revenue decline|profit warning)\b'
            ],
            
            # Warning phrases
            'warning_phrases': [
                r'\b(investor beware|buyer beware|caveat emptor)\b',
                r'\b(do your own research|not financial advice)\b',
                r'\b(high risk|proceed with caution|due diligence)\b',
                r'\b(consult financial advisor|seek professional help)\b'
            ]
        }
        
        # Sentiment intensity modifiers
        self.intensity_modifiers = {
            'high': ['extremely', 'incredibly', 'absolutely', 'completely', 'totally'],
            'medium': ['very', 'quite', 'rather', 'pretty', 'fairly'],
            'low': ['somewhat', 'slightly', 'a bit', 'kind of', 'sort of']
        }
    
    def detect_hype(self, text: str) -> DetectionResult:
        """Detect hype using advanced pattern matching and scoring"""
        text_lower = text.lower()
        
        # Initialize scoring
        total_score = 0
        detected_indicators = []
        detected_patterns = []
        confidence_factors = []
        
        # Pattern detection
        for category, patterns in self.hype_patterns.items():
            category_score = 0
            category_indicators = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    category_score += len(matches) * self._get_pattern_weight(category)
                    category_indicators.extend(matches)
                    detected_patterns.append(f"{category}: {pattern}")
            
            if category_score > 0:
                total_score += category_score
                detected_indicators.extend(category_indicators)
                confidence_factors.append(category_score)
        
        # Additional scoring factors
        caps_score = self._analyze_caps_usage(text)
        exclamation_score = self._analyze_exclamation_usage(text)
        repetition_score = self._analyze_repetition(text_lower)
        
        total_score += caps_score + exclamation_score + repetition_score
        
        # Calculate confidence
        confidence = min(1.0, len(confidence_factors) * 0.2 + total_score * 0.1)
        
        # Generate explanation
        explanation = self._generate_hype_explanation(total_score, detected_indicators, caps_score, exclamation_score)
        
        return DetectionResult(
            score=min(total_score, 10.0),  # Cap at 10
            confidence=confidence,
            indicators=detected_indicators[:10],  # Limit to top 10
            patterns=detected_patterns[:5],  # Limit to top 5
            explanation=explanation
        )
    
    def detect_risk(self, text: str) -> DetectionResult:
        """Detect investment risks using advanced pattern matching"""
        text_lower = text.lower()
        
        total_score = 0
        detected_indicators = []
        detected_patterns = []
        confidence_factors = []
        
        # Pattern detection
        for category, patterns in self.risk_patterns.items():
            category_score = 0
            category_indicators = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    category_score += len(matches) * self._get_risk_pattern_weight(category)
                    category_indicators.extend(matches)
                    detected_patterns.append(f"{category}: {pattern}")
            
            if category_score > 0:
                total_score += category_score
                detected_indicators.extend(category_indicators)
                confidence_factors.append(category_score)
        
        # Additional risk factors
        negative_sentiment_score = self._analyze_negative_sentiment(text_lower)
        disclaimer_score = self._analyze_disclaimers(text_lower)
        uncertainty_score = self._analyze_uncertainty_language(text_lower)
        
        total_score += negative_sentiment_score + disclaimer_score + uncertainty_score
        
        # Calculate confidence
        confidence = min(1.0, len(confidence_factors) * 0.25 + total_score * 0.15)
        
        # Generate explanation
        explanation = self._generate_risk_explanation(total_score, detected_indicators, negative_sentiment_score)
        
        return DetectionResult(
            score=min(total_score, 10.0),  # Cap at 10
            confidence=confidence,
            indicators=detected_indicators[:10],
            patterns=detected_patterns[:5],
            explanation=explanation
        )
    
    def _get_pattern_weight(self, category: str) -> float:
        """Get weight for different hype pattern categories"""
        weights = {
            'exaggeration': 2.0,
            'pump_language': 3.0,
            'urgency': 2.5,
            'price_hype': 3.5,
            'social_hype': 1.5
        }
        return weights.get(category, 1.0)
    
    def _get_risk_pattern_weight(self, category: str) -> float:
        """Get weight for different risk pattern categories"""
        weights = {
            'financial_risk': 3.0,
            'market_risk': 2.5,
            'company_risk': 2.0,
            'warning_phrases': 1.5
        }
        return weights.get(category, 1.0)
    
    def _analyze_caps_usage(self, text: str) -> float:
        """Analyze excessive use of capital letters"""
        if not text:
            return 0
        
        caps_count = sum(1 for c in text if c.isupper())
        total_chars = len(text)
        caps_ratio = caps_count / total_chars
        
        if caps_ratio > 0.5:
            return 3.0
        elif caps_ratio > 0.3:
            return 2.0
        elif caps_ratio > 0.2:
            return 1.0
        return 0
    
    def _analyze_exclamation_usage(self, text: str) -> float:
        """Analyze excessive use of exclamation marks"""
        exclamation_count = text.count('!')
        
        if exclamation_count > 5:
            return 2.0
        elif exclamation_count > 3:
            return 1.5
        elif exclamation_count > 1:
            return 1.0
        return 0
    
    def _analyze_repetition(self, text: str) -> float:
        """Analyze repetitive words or phrases"""
        words = text.split()
        if len(words) < 5:
            return 0
        
        word_counts = Counter(words)
        repeated_words = [word for word, count in word_counts.items() if count > 2]
        
        return min(len(repeated_words) * 0.5, 2.0)
    
    def _analyze_negative_sentiment(self, text: str) -> float:
        """Analyze negative sentiment patterns"""
        negative_words = [
            'but', 'however', 'despite', 'although', 'warning', 'concern',
            'risk', 'uncertainty', 'volatile', 'speculative', 'dangerous',
            'problem', 'issue', 'trouble', 'difficulty', 'challenge'
        ]
        
        negative_count = sum(1 for word in negative_words if word in text)
        return min(negative_count * 0.3, 2.0)
    
    def _analyze_disclaimers(self, text: str) -> float:
        """Analyze presence of disclaimers"""
        disclaimer_patterns = [
            r'not financial advice',
            r'do your own research',
            r'invest at your own risk',
            r'consult.*advisor',
            r'past performance.*not.*guarantee'
        ]
        
        disclaimer_count = sum(1 for pattern in disclaimer_patterns if re.search(pattern, text))
        return min(disclaimer_count * 0.5, 1.5)
    
    def _analyze_uncertainty_language(self, text: str) -> float:
        """Analyze uncertainty and hedging language"""
        uncertainty_words = [
            'might', 'could', 'possibly', 'perhaps', 'maybe', 'potentially',
            'uncertain', 'unclear', 'unknown', 'speculative', 'volatile'
        ]
        
        uncertainty_count = sum(1 for word in uncertainty_words if word in text)
        return min(uncertainty_count * 0.2, 1.0)
    
    def _generate_hype_explanation(self, score: float, indicators: List[str], caps_score: float, exclamation_score: float) -> str:
        """Generate explanation for hype detection"""
        if score >= 7:
            level = "HIGH HYPE"
            description = "Strong indicators of market hype and potential pump tactics"
        elif score >= 4:
            level = "MODERATE HYPE"
            description = "Some indicators of hype and promotional language"
        elif score >= 2:
            level = "LOW HYPE"
            description = "Minimal hype indicators detected"
        else:
            level = "NO HYPE"
            description = "No significant hype indicators found"
        
        factors = []
        if caps_score > 0:
            factors.append("excessive capitalization")
        if exclamation_score > 0:
            factors.append("excessive exclamation marks")
        if indicators:
            factors.append(f"{len(indicators)} hype keywords")
        
        factor_text = f" ({', '.join(factors)})" if factors else ""
        
        return f"{level}: {description}{factor_text}"
    
    def _generate_risk_explanation(self, score: float, indicators: List[str], negative_score: float) -> str:
        """Generate explanation for risk detection"""
        if score >= 7:
            level = "HIGH RISK"
            description = "Multiple risk indicators suggest caution"
        elif score >= 4:
            level = "MODERATE RISK"
            description = "Some risk indicators present"
        elif score >= 2:
            level = "LOW RISK"
            description = "Minimal risk indicators detected"
        else:
            level = "LOW RISK"
            description = "No significant risk indicators found"
        
        factors = []
        if negative_score > 0:
            factors.append("negative sentiment")
        if indicators:
            factors.append(f"{len(indicators)} risk keywords")
        
        factor_text = f" ({', '.join(factors)})" if factors else ""
        
        return f"{level}: {description}{factor_text}"

class EnhancedNewsAnalyzer:
    """Enhanced news analyzer using advanced detection algorithms"""
    
    def __init__(self):
        self.hype_detector = AdvancedHypeDetector()
    
    def analyze_article(self, title: str, content: str = "") -> Dict:
        """Analyze a single article for hype and risk"""
        full_text = f"{title} {content}".strip()
        
        # Detect hype
        hype_result = self.hype_detector.detect_hype(full_text)
        
        # Detect risk
        risk_result = self.hype_detector.detect_risk(full_text)
        
        return {
            'hype': {
                'score': round(hype_result.score, 2),
                'confidence': round(hype_result.confidence, 2),
                'indicators': hype_result.indicators,
                'patterns': hype_result.patterns,
                'explanation': hype_result.explanation,
                'level': self._get_hype_level(hype_result.score)
            },
            'risk': {
                'score': round(risk_result.score, 2),
                'confidence': round(risk_result.confidence, 2),
                'indicators': risk_result.indicators,
                'patterns': risk_result.patterns,
                'explanation': risk_result.explanation,
                'level': self._get_risk_level(risk_result.score)
            }
        }
    
    def _get_hype_level(self, score: float) -> str:
        """Convert hype score to level"""
        if score >= 7:
            return "HIGH HYPE"
        elif score >= 4:
            return "MODERATE HYPE"
        elif score >= 2:
            return "LOW HYPE"
        else:
            return "NO HYPE"
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to level"""
        if score >= 7:
            return "HIGH RISK"
        elif score >= 4:
            return "MODERATE RISK"
        elif score >= 2:
            return "LOW RISK"
        else:
            return "LOW RISK"
