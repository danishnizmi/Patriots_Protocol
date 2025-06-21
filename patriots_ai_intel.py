#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Tactical AI-Powered Cyber Intelligence Engine v4.2
Fixed for GitHub Models API Compatibility & Enhanced Logging

Repository: https://github.com/danishnizmi/Patriots_Protocol
"""

import os
import json
import asyncio
import aiohttp
import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import feedparser

# Enhanced tactical logging configuration with file handlers
def setup_tactical_logging():
    """Setup enhanced tactical logging with file outputs"""
    # Create log directories
    log_dirs = ['logs/tactical', 'logs/performance', 'logs/errors', 'logs/audit']
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S UTC')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for tactical logs
    file_handler = logging.FileHandler(f'logs/tactical/tactical_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S UTC')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error handler
    error_handler = logging.FileHandler(f'logs/errors/errors_{datetime.now().strftime("%Y%m%d")}.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_tactical_logging()

@dataclass
class TacticalThreatReport:
    """Enhanced Tactical Threat Intelligence Report"""
    title: str
    summary: str
    source: str
    source_url: str
    timestamp: str
    threat_level: str
    confidence_score: float
    severity_rating: int
    attack_vectors: List[str]
    affected_sectors: List[str]
    geographic_scope: str
    country_code: str
    threat_actors: List[str]
    mitigation_priority: str
    cve_references: List[str]
    threat_family: str
    attack_sophistication: str
    risk_score: int
    correlation_id: str
    key_insights: List[str]
    business_impact: str
    tactical_impact: str
    operational_urgency: str
    risk_factors: Dict[str, str]
    recommended_actions: List[str]

@dataclass
class TacticalSituationReport:
    """Daily Tactical Situation Report"""
    date: str
    executive_summary: str
    key_developments: List[str]
    critical_threats_overview: str
    trending_attack_vectors: List[str]
    geographic_hotspots: List[str]
    sector_impact_analysis: str
    recommended_actions: List[str]
    threat_landscape_assessment: str
    zero_day_activity: str
    attribution_insights: str
    defensive_priorities: List[str]
    tactical_recommendations: List[str]
    force_protection_level: str

@dataclass
class TacticalMetrics:
    """Enhanced Tactical Intelligence Metrics"""
    total_threats: int
    critical_threats: int
    high_threats: int
    medium_threats: int
    low_threats: int
    global_threat_level: str
    intelligence_confidence: int
    recent_threats_24h: int
    top_threat_families: List[Dict[str, Any]]
    geographic_distribution: Dict[str, int]
    zero_day_count: int
    trending_threats: List[Dict[str, Any]]
    threat_velocity: str
    fresh_intel_24h: int
    source_credibility: float
    emerging_trends: List[str]
    threat_evolution: str
    daily_summary_confidence: int
    ai_insights_quality: int
    tactical_readiness: str
    force_protection_status: str

class TacticalPatriotsIntelligence:
    """Enhanced Tactical AI-Powered Cyber Threat Intelligence with Fixed API Support"""
    
    def __init__(self):
        # API configuration - try multiple endpoints
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN') or os.getenv('OPENAI_API_KEY')
        
        # Try GitHub Models first, then fallback to OpenAI
        self.github_base_url = "https://models.inference.ai.azure.com"
        self.openai_base_url = "https://api.openai.com/v1"
        
        # Model configurations
        self.github_model = "gpt-4o-mini"  # Fixed model name for GitHub Models
        self.openai_model = "gpt-4o-mini"
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Tactical optimization
        self.ai_summary_generated = False
        self.ai_calls_made = 0
        self.max_ai_calls = 2  # Cost optimization
        self.api_endpoint = None  # Will be determined during testing
        self.current_model = None
        
        # Enhanced intelligence sources with tactical priorities
        self.intelligence_sources = [
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.92,
                'priority': 1,
                'tactical_value': 'HIGH'
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'priority': 1,
                'tactical_value': 'MAXIMUM'
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'priority': 1,
                'tactical_value': 'HIGH'
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.88,
                'priority': 2,
                'tactical_value': 'MEDIUM'
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.85,
                'priority': 2,
                'tactical_value': 'MEDIUM'
            },
            {
                'name': 'SECURITY_WEEK',
                'url': 'https://www.securityweek.com/feed/',
                'reliability': 0.87,
                'priority': 2,
                'tactical_value': 'MEDIUM'
            }
        ]
        
        self.geographic_mapping = {
            'united states': 'US', 'usa': 'US', 'america': 'US', 'u.s.': 'US',
            'canada': 'CA', 'mexico': 'MX', 'china': 'CN', 'japan': 'JP',
            'south korea': 'KR', 'korea': 'KR', 'australia': 'AU', 'new zealand': 'NZ',
            'singapore': 'SG', 'india': 'IN', 'thailand': 'TH', 'vietnam': 'VN',
            'philippines': 'PH', 'malaysia': 'MY', 'indonesia': 'ID', 'taiwan': 'TW',
            'hong kong': 'HK', 'germany': 'DE', 'france': 'FR', 'united kingdom': 'GB',
            'uk': 'GB', 'italy': 'IT', 'spain': 'ES', 'netherlands': 'NL',
            'russia': 'RU', 'brazil': 'BR', 'europe': 'EU', 'global': 'GLOBAL'
        }
        
        self.data_directory = Path('./data')
        self.data_directory.mkdir(exist_ok=True)
        
        logger.info("🎖️ Tactical Patriots Protocol Intelligence Engine v4.2 - Fixed API Integration")
        if self.api_token:
            logger.info("🤖 API Token found - testing multiple endpoints...")
        else:
            logger.warning("⚠️ No API token found - will use enhanced basic analysis")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={
                'User-Agent': 'Patriots-Protocol-Tactical/4.2 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_api_endpoints(self) -> bool:
        """Test multiple API endpoints to find working one"""
        if not self.api_token:
            logger.warning("⚠️ No API token available for testing")
            return False
        
        # Test configurations
        test_configs = [
            {
                'name': 'GitHub Models',
                'url': f"{self.github_base_url}/chat/completions",
                'model': self.github_model,
                'headers': {
                    'Authorization': f'Bearer {self.api_token}',
                    'Content-Type': 'application/json'
                }
            },
            {
                'name': 'OpenAI',
                'url': f"{self.openai_base_url}/chat/completions", 
                'model': self.openai_model,
                'headers': {
                    'Authorization': f'Bearer {self.api_token}',
                    'Content-Type': 'application/json'
                }
            }
        ]
        
        for config in test_configs:
            try:
                test_payload = {
                    "model": config['model'],
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Respond with 'API_TEST_SUCCESS' only."}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 10
                }

                logger.info(f"🔍 Testing {config['name']} API...")
                
                async with self.session.post(config['url'], 
                                           headers=config['headers'], 
                                           json=test_payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        test_response = result['choices'][0]['message']['content']
                        if 'API_TEST_SUCCESS' in test_response or 'SUCCESS' in test_response.upper():
                            logger.info(f"✅ {config['name']} API test successful")
                            self.api_endpoint = config['url']
                            self.current_model = config['model']
                            self.api_headers = config['headers']
                            return True
                        else:
                            logger.info(f"✅ {config['name']} API responding (got: {test_response})")
                            self.api_endpoint = config['url']
                            self.current_model = config['model']
                            self.api_headers = config['headers']
                            return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"⚠️ {config['name']} API test failed: {response.status}")
                        
            except Exception as e:
                logger.warning(f"⚠️ {config['name']} API test error: {str(e)[:100]}")
                continue
                
        logger.error("❌ All API endpoints failed - using enhanced fallback analysis")
        return False

    async def generate_tactical_sitrep(self, all_threats: List[TacticalThreatReport]) -> TacticalSituationReport:
        """Generate comprehensive tactical situation report using available AI"""
        if not self.api_token or not all_threats:
            logger.info("🔄 Using tactical fallback analysis (no API token or no threats)")
            return self.create_tactical_fallback(all_threats)

        # Test API endpoints
        api_working = await self.test_api_endpoints()
        if not api_working:
            logger.warning("⚠️ No working API endpoint found - using tactical fallback")
            return self.create_tactical_fallback(all_threats)

        try:
            threat_data = self.prepare_tactical_summary_data(all_threats)
            
            # Optimized prompt for cost efficiency
            tactical_prompt = f"""Analyze these cyber threats for {datetime.now().strftime('%Y-%m-%d')} and provide a JSON tactical report:

THREATS ({len(all_threats)} total):
{threat_data}

Return ONLY valid JSON in this exact format:
{{
    "executive_summary": "Brief tactical overview of current threat landscape",
    "key_developments": [
        "Most critical cyber incident",
        "Important vulnerability or attack"
    ],
    "critical_threats_overview": "Analysis of critical threats requiring immediate response",
    "trending_attack_vectors": ["primary_attack_method", "secondary_method"],
    "geographic_hotspots": ["region_with_most_activity"],
    "sector_impact_analysis": "Which sectors are being targeted",
    "recommended_actions": [
        "Most urgent action required",
        "Critical security measure"
    ],
    "threat_landscape_assessment": "Overall security posture assessment",
    "zero_day_activity": "Zero-day vulnerability status",
    "attribution_insights": "Threat actor intelligence summary",
    "defensive_priorities": [
        "Top defensive priority",
        "Secondary priority"
    ],
    "tactical_recommendations": [
        "Key operational recommendation"
    ],
    "force_protection_level": "CRITICAL/HIGH/MEDIUM/LOW"
}}

Focus on actionable intelligence. Respond with ONLY the JSON object."""

            payload = {
                "model": self.current_model,
                "messages": [
                    {"role": "system", "content": "You are a cyber intelligence analyst. Provide tactical threat assessments in JSON format only."},
                    {"role": "user", "content": tactical_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }

            self.ai_calls_made += 1
            logger.info(f"🤖 Generating Tactical SITREP for {len(all_threats)} threats...")

            async with self.session.post(self.api_endpoint, 
                                       headers=self.api_headers, 
                                       json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result['choices'][0]['message']['content']
                    
                    # Enhanced JSON extraction
                    json_content = self.extract_json_from_response(ai_response)
                    
                    if json_content:
                        try:
                            sitrep_data = json.loads(json_content)
                            self.ai_summary_generated = True
                            logger.info("✅ Tactical SITREP generated successfully using AI")
                            return self.format_tactical_sitrep(sitrep_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ JSON parsing failed: {str(e)}")
                    else:
                        logger.warning("⚠️ No valid JSON found in AI response")
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ AI API error: {response.status} - {error_text[:200]}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Tactical SITREP generation failed: {str(e)}")
            
        return self.create_tactical_fallback(all_threats)

    def extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from AI response with multiple strategies"""
        # Strategy 1: Find JSON object boundaries
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_content = response[json_start:json_end]
            # Basic validation
            if json_content.count('{') == json_content.count('}'):
                return json_content
        
        # Strategy 2: Extract between code blocks
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Strategy 3: Extract first complete JSON object
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(response):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return response[start_idx:i+1]
        
        return None

    def prepare_tactical_summary_data(self, threats: List[TacticalThreatReport]) -> str:
        """Prepare concise tactical threat data for AI analysis"""
        if not threats:
            return "No threats detected"
        
        summary_lines = []
        
        # Overall statistics
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        zero_day_threats = [t for t in threats if 'zero' in t.threat_family.lower()]
        
        summary_lines.append(f"TOTAL: {len(threats)} threats")
        summary_lines.append(f"LEVELS: Critical({len(critical_threats)}) High({len(high_threats)}) Zero-day({len(zero_day_threats)})")
        
        # Top critical threats
        if critical_threats:
            summary_lines.append("\nCRITICAL THREATS:")
            for threat in critical_threats[:3]:
                summary_lines.append(f"- {threat.title[:60]} | Risk:{threat.risk_score}/10 | {threat.threat_family}")
        
        # Top high threats
        if high_threats and not critical_threats:
            summary_lines.append("\nHIGH THREATS:")
            for threat in high_threats[:3]:
                summary_lines.append(f"- {threat.title[:60]} | Risk:{threat.risk_score}/10 | {threat.threat_family}")
        
        # Threat analysis
        families = {}
        regions = {}
        vectors = set()
        
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
            regions[threat.country_code] = regions.get(threat.country_code, 0) + 1
            vectors.update(threat.attack_vectors[:2])  # Limit vectors
        
        # Top categories
        top_families = sorted(families.items(), key=lambda x: x[1], reverse=True)[:3]
        top_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary_lines.append(f"\nTOP THREAT TYPES: {', '.join([f'{k}({v})' for k, v in top_families])}")
        summary_lines.append(f"TOP REGIONS: {', '.join([f'{k}({v})' for k, v in top_regions])}")
        if vectors:
            summary_lines.append(f"ATTACK VECTORS: {', '.join(list(vectors)[:3])}")
        
        return '\n'.join(summary_lines)

    def format_tactical_sitrep(self, ai_data: Dict) -> TacticalSituationReport:
        """Format AI response into tactical SITREP structure"""
        return TacticalSituationReport(
            date=datetime.now().strftime('%Y-%m-%d'),
            executive_summary=ai_data.get('executive_summary', 'Tactical analysis complete - threat landscape assessed'),
            key_developments=ai_data.get('key_developments', ["Intelligence network operational", "Threat monitoring active"]),
            critical_threats_overview=ai_data.get('critical_threats_overview', 'No critical threats requiring immediate response'),
            trending_attack_vectors=ai_data.get('trending_attack_vectors', ["monitoring"]),
            geographic_hotspots=ai_data.get('geographic_hotspots', ["Global"]),
            sector_impact_analysis=ai_data.get('sector_impact_analysis', 'Multi-sector monitoring active'),
            recommended_actions=ai_data.get('recommended_actions', ["Continue standard monitoring", "Maintain security posture"]),
            threat_landscape_assessment=ai_data.get('threat_landscape_assessment', 'Threat landscape stable'),
            zero_day_activity=ai_data.get('zero_day_activity', 'No zero-day activity detected'),
            attribution_insights=ai_data.get('attribution_insights', 'Attribution analysis ongoing'),
            defensive_priorities=ai_data.get('defensive_priorities', ["Maintain vigilance", "Standard monitoring"]),
            tactical_recommendations=ai_data.get('tactical_recommendations', ["Continue operations", "Monitor threat feeds"]),
            force_protection_level=ai_data.get('force_protection_level', 'MEDIUM')
        )

    def create_tactical_fallback(self, threats: List[TacticalThreatReport]) -> TacticalSituationReport:
        """Create enhanced tactical fallback SITREP when AI unavailable"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        if not threats:
            return TacticalSituationReport(
                date=current_date,
                executive_summary="Tactical intelligence network operational with baseline monitoring. No significant threats detected in current intelligence cycle. All systems maintaining standard security posture.",
                key_developments=["Intelligence network operational", "All monitoring systems active", "Standard threat assessment complete"],
                critical_threats_overview="No critical threats requiring immediate response detected. Routine monitoring protocols active.",
                trending_attack_vectors=["monitoring", "baseline_assessment"],
                geographic_hotspots=["Global"],
                sector_impact_analysis="No specific sector targeting identified. Multi-sector monitoring maintained.",
                recommended_actions=["Continue standard monitoring procedures", "Maintain current security posture", "Verify intelligence feed operational status"],
                threat_landscape_assessment="Stable baseline monitoring established. No elevated threat activity detected.",
                zero_day_activity="No zero-day activity detected in current intelligence cycle",
                attribution_insights="No specific threat actor activity identified in current timeframe",
                defensive_priorities=["Maintain operational vigilance", "Continue standard monitoring", "Verify system integrity"],
                tactical_recommendations=["Continue routine operations", "Monitor threat intelligence feeds", "Maintain readiness posture"],
                force_protection_level="LOW"
            )

        # Enhanced tactical fallback analysis
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        zero_days = sum(1 for t in threats if 'zero' in t.threat_family.lower())
        
        # Analyze patterns
        families = {}
        geography = {}
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
            geography[threat.country_code] = geography.get(threat.country_code, 0) + 1
        
        top_family = max(families.items(), key=lambda x: x[1])[0] if families else 'Mixed Threats'
        top_regions = sorted(geography.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate executive summary
        exec_summary = f"Tactical intelligence analysis of {len(threats)} security incidents shows "
        if critical_threats:
            exec_summary += f"{len(critical_threats)} critical threats requiring immediate tactical response. "
        if zero_days:
            exec_summary += f"{zero_days} zero-day vulnerabilities detected requiring emergency action. "
        exec_summary += f"Primary threat category: {top_family}. "
        
        # Determine force protection level
        if len(critical_threats) >= 3 or zero_days >= 2:
            force_protection = "CRITICAL"
            exec_summary += "Force protection level elevated to CRITICAL."
            assessment = "Critical threat activity detected - enhanced security posture required"
        elif len(critical_threats) >= 1 or zero_days >= 1:
            force_protection = "HIGH"
            exec_summary += "Force protection level elevated to HIGH."
            assessment = "Elevated threat activity - increased vigilance required"
        elif len(high_threats) >= 3:
            force_protection = "MEDIUM"
            exec_summary += "Force protection level: MEDIUM."
            assessment = "Moderate threat activity - standard enhanced monitoring"
        else:
            force_protection = "LOW"
            exec_summary += "Force protection level: LOW."
            assessment = "Standard threat monitoring - routine security posture maintained"
        
        # Generate key developments
        key_developments = []
        if critical_threats:
            key_developments.append(f"Critical security incident: {critical_threats[0].title[:70]}")
        if zero_days:
            key_developments.append(f"Zero-day vulnerability activity detected ({zero_days} incidents)")
        if high_threats and len(key_developments) < 2:
            key_developments.append(f"High-priority threat: {high_threats[0].title[:70]}")
        
        if len(key_developments) < 2:
            key_developments.extend(["Standard threat monitoring active", "Security intelligence collection operational"])
        
        # Generate recommendations
        recommendations = []
        if critical_threats:
            recommendations.extend([
                "Immediate escalation to security operations center required",
                "Implement enhanced monitoring and incident response protocols"
            ])
        if zero_days:
            recommendations.append("Activate emergency patch management procedures immediately")
        if not recommendations:
            recommendations.extend([
                "Maintain current security monitoring procedures",
                "Continue routine threat intelligence assessment"
            ])
        
        # Generate tactical recommendations
        tactical_recommendations = []
        if force_protection in ["CRITICAL", "HIGH"]:
            tactical_recommendations.extend([
                "Escalate to command for tactical response authorization",
                "Implement enhanced force protection measures"
            ])
        else:
            tactical_recommendations.extend([
                "Maintain standard operational security posture",
                "Continue routine monitoring and assessment"
            ])
        
        return TacticalSituationReport(
            date=current_date,
            executive_summary=exec_summary,
            key_developments=key_developments[:3],
            critical_threats_overview=f"Analysis identified {len(critical_threats)} critical and {len(high_threats)} high-priority threats. {f'Zero-day activity: {zero_days} incidents. ' if zero_days else ''}Immediate security team attention {'required' if critical_threats else 'recommended'}.",
            trending_attack_vectors=list(set([v for t in threats for v in t.attack_vectors]))[:3] or ["standard_monitoring"],
            geographic_hotspots=[region for region, count in top_regions] or ["Global"],
            sector_impact_analysis=f"Multi-sector security impact analysis shows {top_family} as primary threat category affecting monitored infrastructure",
            recommended_actions=recommendations[:3],
            threat_landscape_assessment=assessment,
            zero_day_activity=f"{zero_days} zero-day incidents detected requiring immediate patch management response" if zero_days else "No zero-day activity detected in current intelligence cycle",
            attribution_insights="Advanced threat actor analysis pending" if len(threats) > 8 else "No specific threat actor attribution identified",
            defensive_priorities=[
                "Critical security patch management" if critical_threats else "Standard security monitoring",
                "Enhanced endpoint and network protection verification",
                "Security awareness and incident response readiness"
            ],
            tactical_recommendations=tactical_recommendations,
            force_protection_level=force_protection
        )

    # [Rest of the methods remain the same as in original code]
    def enhanced_tactical_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Enhanced tactical analysis with sophisticated risk rating"""
        full_text = (title + ' ' + content).lower()
        
        # Enhanced threat family detection with tactical categories
        threat_families = {
            'Zero-Day Exploit': {
                'keywords': ['zero-day', 'zero day', '0-day', 'unknown vulnerability', 'cve-2024', 'cve-2025'],
                'base_risk': 9,
                'tactical_priority': 'CRITICAL'
            },
            'Ransomware': {
                'keywords': ['ransomware', 'encryption', 'ransom', 'lockbit', 'conti', 'revil', 'blackcat'],
                'base_risk': 8,
                'tactical_priority': 'HIGH'
            },
            'APT Campaign': {
                'keywords': ['apt', 'nation-state', 'state-sponsored', 'advanced persistent', 'sophisticated attack'],
                'base_risk': 8,
                'tactical_priority': 'HIGH'
            },
            'Critical Infrastructure': {
                'keywords': ['infrastructure', 'utility', 'scada', 'industrial', 'power grid', 'water system'],
                'base_risk': 9,
                'tactical_priority': 'CRITICAL'
            },
            'Supply Chain Attack': {
                'keywords': ['supply chain', 'software supply', 'third-party', 'vendor compromise'],
                'base_risk': 7,
                'tactical_priority': 'HIGH'
            },
            'Data Breach': {
                'keywords': ['data breach', 'breach', 'stolen data', 'exposed records', 'leaked database'],
                'base_risk': 6,
                'tactical_priority': 'MEDIUM'
            },
            'Vulnerability Disclosure': {
                'keywords': ['patch tuesday', 'security update', 'patches', 'vulnerability disclosure'],
                'base_risk': 5,
                'tactical_priority': 'MEDIUM'
            },
            'Malware Campaign': {
                'keywords': ['malware', 'trojan', 'backdoor', 'spyware', 'botnet', 'stealer'],
                'base_risk': 5,
                'tactical_priority': 'MEDIUM'
            },
            'Phishing Campaign': {
                'keywords': ['phishing', 'spear phishing', 'business email', 'email attack'],
                'base_risk': 4,
                'tactical_priority': 'MEDIUM'
            },
            'Security Incident': {
                'keywords': ['systems restored', 'security incident', 'cyber attack', 'investigation'],
                'base_risk': 4,
                'tactical_priority': 'LOW'
            }
        }
        
        # Detect threat family and base characteristics
        detected_family = 'Security Incident'
        base_risk = 4
        tactical_priority = 'LOW'
        confidence = 0.6
        
        for family, data in threat_families.items():
            matches = [keyword for keyword in data['keywords'] if keyword in full_text]
            if matches:
                detected_family = family
                base_risk = data['base_risk']
                tactical_priority = data['tactical_priority']
                confidence = min(0.95, 0.7 + len(matches) * 0.05)
                break
        
        # Enhanced risk calculation with multiple factors
        risk_score = base_risk
        
        # Risk modifiers
        risk_modifiers = {
            'active_exploitation': ['under attack', 'actively exploited', 'in the wild', 'being exploited'],
            'critical_systems': ['critical', 'infrastructure', 'essential', 'mission critical'],
            'widespread_impact': ['widespread', 'global', 'mass', 'large scale'],
            'no_patch': ['no patch', 'unpatched', 'zero-day', 'no fix available'],
            'high_sophistication': ['sophisticated', 'advanced', 'nation-state', 'apt'],
            'confirmed_attribution': ['confirmed', 'attribution', 'tracked', 'identified group']
        }
        
        applied_modifiers = []
        for modifier, keywords in risk_modifiers.items():
            if any(keyword in full_text for keyword in keywords):
                applied_modifiers.append(modifier)
                if modifier == 'active_exploitation':
                    risk_score = min(10, risk_score + 2)
                elif modifier == 'critical_systems':
                    risk_score = min(10, risk_score + 1)
                elif modifier == 'no_patch':
                    risk_score = min(10, risk_score + 2)
                elif modifier == 'high_sophistication':
                    risk_score = min(10, risk_score + 1)
        
        # Calculate sophisticated risk factors
        impact_level = self.calculate_impact_level(full_text, risk_score)
        probability_level = self.calculate_probability_level(full_text, applied_modifiers)
        sophistication_level = self.calculate_sophistication_level(full_text, detected_family)
        
        # Generate enhanced insights
        key_insights = self.generate_tactical_insights(full_text, detected_family, applied_modifiers)
        
        # Generate tactical recommendations
        recommended_actions = self.generate_tactical_actions(full_text, detected_family, risk_score)
        
        # Determine threat level based on sophisticated calculation
        if risk_score >= 9 or 'active_exploitation' in applied_modifiers:
            threat_level, urgency = 'CRITICAL', 'IMMEDIATE'
        elif risk_score >= 7 or tactical_priority == 'HIGH':
            threat_level, urgency = 'HIGH', 'URGENT'
        elif risk_score >= 5:
            threat_level, urgency = 'MEDIUM', 'ROUTINE'
        else:
            threat_level, urgency = 'LOW', 'ROUTINE'
        
        # Generate tactical impact assessment
        tactical_impact = self.assess_tactical_impact(full_text, risk_score, detected_family)
        
        return {
            'threat_family': detected_family,
            'risk_score': risk_score,
            'threat_level': threat_level,
            'confidence': confidence,
            'key_insights': key_insights,
            'recommended_actions': recommended_actions,
            'attack_vectors': self.extract_attack_vectors(full_text),
            'affected_sectors': self.extract_sectors(full_text),
            'geographic_scope': self.extract_geography(full_text),
            'cve_references': re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()),
            'threat_actors': self.extract_threat_actors(full_text),
            'business_impact': self.assess_business_impact(full_text, risk_score),
            'tactical_impact': tactical_impact,
            'operational_urgency': urgency,
            'risk_factors': {
                'impact': impact_level,
                'probability': probability_level,
                'sophistication': sophistication_level
            },
            'applied_modifiers': applied_modifiers
        }

    def calculate_impact_level(self, content: str, risk_score: int) -> str:
        """Calculate tactical impact level"""
        if risk_score >= 9 or any(term in content for term in ['critical', 'infrastructure', 'widespread']):
            return 'HIGH'
        elif risk_score >= 7 or any(term in content for term in ['significant', 'major', 'serious']):
            return 'MED'
        elif risk_score >= 5:
            return 'MED'
        else:
            return 'LOW'

    def calculate_probability_level(self, content: str, modifiers: List[str]) -> str:
        """Calculate probability of exploitation"""
        if 'active_exploitation' in modifiers or any(term in content for term in ['under attack', 'exploited']):
            return 'HIGH'
        elif any(term in content for term in ['proof of concept', 'demonstrated', 'likely']):
            return 'MED'
        else:
            return 'LOW'

    def calculate_sophistication_level(self, content: str, family: str) -> str:
        """Calculate attack sophistication level"""
        if family in ['APT Campaign', 'Zero-Day Exploit'] or any(term in content for term in ['nation-state', 'advanced']):
            return 'HIGH'
        elif family in ['Ransomware', 'Supply Chain Attack'] or any(term in content for term in ['sophisticated', 'coordinated']):
            return 'MED'
        else:
            return 'LOW'

    def generate_tactical_insights(self, content: str, family: str, modifiers: List[str]) -> List[str]:
        """Generate tactical insights based on content analysis"""
        insights = []
        
        if 'zero-day' in content:
            insights.append("Zero-day exploitation requires immediate tactical response and emergency patching")
        if 'ransomware' in content:
            insights.append("Ransomware threat - verify backup integrity and test recovery procedures")
        if 'supply chain' in content:
            insights.append("Supply chain compromise affects downstream security dependencies")
        if 'active_exploitation' in modifiers:
            insights.append("Active exploitation detected - implement immediate containment measures")
        if 'critical' in content and 'patch' in content:
            insights.append("Critical patches available - prioritize emergency deployment")
        if 'apt' in content or 'nation-state' in content:
            insights.append("Advanced persistent threat - enhance threat hunting and monitoring")
        if family == 'Critical Infrastructure':
            insights.append("Critical infrastructure targeting - coordinate with relevant authorities")
        
        if not insights:
            if family in ['Zero-Day Exploit', 'Ransomware', 'APT Campaign']:
                insights.append("High-priority security incident requiring immediate tactical attention")
            else:
                insights.append("Security development requiring assessment and monitoring")
        
        return insights[:3]

    def generate_tactical_actions(self, content: str, family: str, risk_score: int) -> List[str]:
        """Generate specific tactical actions"""
        actions = []
        
        if any(term in content for term in ['patch', 'update', 'fix']):
            actions.append('Deploy security patches immediately on all affected systems')
        if 'zero-day' in content:
            actions.append('Implement network segmentation and enhanced monitoring')
        if 'ransomware' in content:
            actions.append('Verify backup systems and test offline recovery procedures')
        if 'phishing' in content:
            actions.append('Configure advanced email filtering and user awareness alerts')
        if 'infrastructure' in content:
            actions.append('Coordinate with critical infrastructure protection teams')
        if family == 'APT Campaign':
            actions.append('Activate advanced threat hunting procedures')
        
        # Add risk-based actions
        if risk_score >= 8:
            actions.append('Escalate to security operations center for immediate response')
        if risk_score >= 6:
            actions.append('Enhance monitoring for related threat indicators')
        
        if not actions:
            actions.append('Monitor threat developments and assess security posture')
        
        return actions[:4]

    def assess_tactical_impact(self, content: str, risk_score: int, family: str) -> str:
        """Assess tactical impact for operations"""
        if 'systems restored' in content:
            return "Operations restored - conduct lessons learned analysis"
        elif risk_score >= 9 or family in ['Zero-Day Exploit', 'Critical Infrastructure']:
            return "High business impact - immediate command attention required"
        elif risk_score >= 7 or family in ['Ransomware', 'APT Campaign']:
            return "Moderate business impact - enhanced security procedures required"
        elif risk_score >= 5:
            return "Moderate business impact - standard security procedures apply"
        else:
            return "Limited business impact - awareness and routine monitoring sufficient"

    def extract_attack_vectors(self, content: str) -> List[str]:
        """Extract attack vectors with tactical categories"""
        vectors = []
        vector_mapping = {
            'email': ['phishing', 'email', 'attachment', 'spam'],
            'web': ['website', 'web', 'browser', 'online'],
            'network': ['network', 'remote', 'lateral', 'privilege'],
            'supply_chain': ['supply chain', 'third-party', 'vendor'],
            'social_engineering': ['social', 'human', 'employee', 'insider'],
            'physical': ['physical', 'usb', 'device', 'hardware'],
            'credential': ['password', 'credential', 'authentication', 'login']
        }
        
        for vector, keywords in vector_mapping.items():
            if any(keyword in content for keyword in keywords):
                vectors.append(vector)
        
        return vectors[:3]

    def extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors with tactical relevance"""
        sectors = []
        sector_keywords = {
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power', 'water'],
            'government': ['government', 'federal', 'agency', 'military', 'defense'],
            'financial': ['bank', 'finance', 'payment', 'financial', 'economic'],
            'healthcare': ['hospital', 'medical', 'patient', 'health', 'healthcare'],
            'technology': ['tech', 'software', 'cloud', 'microsoft', 'google', 'apple'],
            'telecommunications': ['telecom', 'communication', 'network', 'internet'],
            'transportation': ['transport', 'airline', 'shipping', 'logistics'],
            'manufacturing': ['manufacturing', 'industrial', 'factory', 'production'],
            'education': ['university', 'school', 'education', 'academic', 'research']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        
        return sectors[:2]

    def extract_geography(self, content: str) -> List[str]:
        """Extract geographic scope with tactical relevance"""
        geography = []
        for region, code in self.geographic_mapping.items():
            if region in content:
                geography.append(region.title())
                if len(geography) >= 2:
                    break
        
        return geography or ['Global']

    def extract_threat_actors(self, content: str) -> List[str]:
        """Extract threat actor references"""
        actors = []
        actor_patterns = [
            r'apt[\s-]?\d+', r'lazarus', r'fancy bear', r'cozy bear', r'carbanak',
            r'fin\d+', r'ta\d+', r'conti', r'lockbit', r'scattered spider'
        ]
        
        for pattern in actor_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            actors.extend(matches)
        
        return list(set(actors))[:3]

    def assess_business_impact(self, content: str, risk_score: int) -> str:
        """Assess business impact with tactical perspective"""
        if 'systems restored' in content:
            return "Operations restored - monitor for residual impact and conduct post-incident analysis"
        elif risk_score >= 9:
            return "High business impact - immediate executive and operational leadership attention required"
        elif risk_score >= 7:
            return "Moderate business impact - security team escalation and enhanced response required"
        elif risk_score >= 5:
            return "Moderate business impact - standard security procedures and monitoring apply"
        else:
            return "Limited business impact - awareness and routine security review sufficient"

    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced tactical intelligence collection with improved logging"""
        collected_intel = []
        
        logger.info(f"🔍 Starting intelligence collection from {len(self.intelligence_sources)} sources...")
        
        for source in sorted(self.intelligence_sources, key=lambda x: x['priority']):
            try:
                logger.info(f"📡 Collecting from {source['name']} (Priority: {source['priority']}, Value: {source['tactical_value']})...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:15]:  # Increased from 12
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Enhanced tactical relevance filtering
                            full_content = (title + ' ' + summary).lower()
                            tactical_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'zero-day',
                                'patch', 'cve', 'incident', 'compromise', 'backdoor', 'trojan',
                                'apt', 'nation-state', 'critical', 'infrastructure', 'data breach'
                            ]
                            
                            relevance_score = sum(1 for indicator in tactical_indicators if indicator in full_content)
                            
                            # Enhanced filtering criteria
                            if relevance_score < 1:
                                continue
                            
                            if len(summary) < 30 or len(title) < 10:
                                continue
                            
                            # Tactical value multiplier
                            tactical_multiplier = {
                                'MAXIMUM': 1.5,
                                'HIGH': 1.2,
                                'MEDIUM': 1.0
                            }.get(source.get('tactical_value', 'MEDIUM'), 1.0)
                            
                            final_relevance = relevance_score * tactical_multiplier
                            
                            intel_item = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'relevance_score': final_relevance,
                                'tactical_value': source.get('tactical_value', 'MEDIUM')
                            }
                            
                            source_intel.append(intel_item)
                        
                        source_intel.sort(key=lambda x: x['relevance_score'], reverse=True)
                        collected_intel.extend(source_intel[:12])  # Top 12 per source
                        
                        logger.info(f"✅ {source['name']}: {len(source_intel)} tactical reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(0.5)  # Rate limiting

        logger.info(f"🎯 Total Tactical Intelligence Collected: {len(collected_intel)} reports")
        return self.tactical_deduplication(collected_intel)

    def tactical_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Enhanced tactical intelligence deduplication"""
        unique_intel = []
        seen_signatures = set()
        
        for intel in raw_intel:
            # Create signature from title words
            title_words = set(re.findall(r'\w+', intel['title'].lower()))
            # Remove common words
            title_words = title_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            content_signature = hashlib.sha256(''.join(sorted(title_words)).encode()).hexdigest()[:12]
            
            if content_signature not in seen_signatures:
                seen_signatures.add(content_signature)
                unique_intel.append(intel)
        
        # Sort by tactical relevance
        unique_intel.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"🔄 Tactical Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel[:25]  # Top 25 most tactically relevant

    async def process_intelligence(self, raw_intel: List[Dict]) -> List[TacticalThreatReport]:
        """Process intelligence with enhanced tactical analysis and logging"""
        threat_reports = []
        
        logger.info(f"🔄 Processing {len(raw_intel)} intelligence reports...")
        
        for i, intel_item in enumerate(raw_intel, 1):
            try:
                # Enhanced tactical analysis
                analysis = self.enhanced_tactical_analysis(
                    intel_item['title'], 
                    intel_item['summary'],
                    intel_item['source']
                )
                
                # Geographic processing
                geographic_scope = 'Global'
                country_code = 'GLOBAL'
                
                if analysis.get('geographic_scope'):
                    geographic_scope = ', '.join(analysis['geographic_scope'][:2])
                    country_code = self.geographic_mapping.get(
                        analysis['geographic_scope'][0].lower(), 'GLOBAL'
                    )
                
                # Create enhanced tactical threat report
                threat_report = TacticalThreatReport(
                    title=intel_item['title'],
                    summary=intel_item['summary'],
                    source=intel_item['source'],
                    source_url=intel_item['source_url'],
                    timestamp=intel_item['timestamp'],
                    threat_level=analysis['threat_level'],
                    confidence_score=analysis['confidence'],
                    severity_rating=analysis['risk_score'],
                    attack_vectors=analysis.get('attack_vectors', []),
                    affected_sectors=analysis.get('affected_sectors', []),
                    geographic_scope=geographic_scope,
                    country_code=country_code,
                    threat_actors=analysis.get('threat_actors', []),
                    mitigation_priority=analysis['operational_urgency'],
                    cve_references=analysis.get('cve_references', []),
                    threat_family=analysis['threat_family'],
                    attack_sophistication=analysis['risk_factors']['sophistication'],
                    risk_score=analysis['risk_score'],
                    correlation_id=hashlib.md5(intel_item['title'].encode()).hexdigest()[:8],
                    key_insights=analysis.get('key_insights', []),
                    business_impact=analysis['business_impact'],
                    tactical_impact=analysis['tactical_impact'],
                    operational_urgency=analysis['operational_urgency'],
                    risk_factors=analysis['risk_factors'],
                    recommended_actions=analysis.get('recommended_actions', [])
                )
                
                threat_reports.append(threat_report)
                
                # Log processing details
                if threat_report.threat_level in ['CRITICAL', 'HIGH']:
                    logger.info(f"⚠️ {threat_report.threat_level} threat processed ({i}/{len(raw_intel)}): {threat_report.title[:60]}...")
                else:
                    logger.info(f"✅ Processed ({i}/{len(raw_intel)}): {threat_report.threat_level} - {threat_report.title[:50]}...")
                
            except Exception as e:
                logger.error(f"❌ Processing error for item {i}: {str(e)}")
                continue

        logger.info(f"📊 Processing complete: {len(threat_reports)} threat reports generated")
        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_tactical_metrics(self, reports: List[TacticalThreatReport]) -> TacticalMetrics:
        """Calculate tactical metrics with enhanced assessment and logging"""
        logger.info("📊 Calculating tactical metrics...")
        
        if not reports:
            logger.warning("⚠️ No threat reports available for metrics calculation")
            return TacticalMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], threat_velocity="stable", fresh_intel_24h=0, 
                source_credibility=0.0, emerging_trends=["Baseline Monitoring"], threat_evolution="stable",
                daily_summary_confidence=85 if self.ai_summary_generated else 70,
                ai_insights_quality=95 if self.ai_summary_generated else 75,
                tactical_readiness="GREEN", force_protection_status="NORMAL"
            )

        # Tactical threat level counting
        threat_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            level = report.threat_level.upper()
            if level in threat_counts:
                threat_counts[level] += 1

        # Enhanced tactical threat level assessment
        critical_count = threat_counts['CRITICAL']
        high_count = threat_counts['HIGH']
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower())
        
        logger.info(f"🎯 Threat Distribution: Critical={critical_count}, High={high_count}, Zero-days={zero_day_count}")
        
        if critical_count >= 5 or zero_day_count >= 3:
            global_level = "CRITICAL"
            tactical_readiness = "RED"
            force_protection = "MAXIMUM"
        elif critical_count >= 3 or zero_day_count >= 2:
            global_level = "HIGH"
            tactical_readiness = "ORANGE"
            force_protection = "HIGH"
        elif critical_count >= 1 or high_count >= 5:
            global_level = "ELEVATED"
            tactical_readiness = "YELLOW"
            force_protection = "ELEVATED"
        elif high_count >= 2:
            global_level = "MEDIUM"
            tactical_readiness = "YELLOW"
            force_protection = "NORMAL"
        else:
            global_level = "LOW"
            tactical_readiness = "GREEN"
            force_protection = "NORMAL"

        logger.info(f"🔥 THREATCON Level: {global_level} | Readiness: {tactical_readiness} | Force Protection: {force_protection}")

        # Geographic distribution
        geo_dist = {}
        for report in reports:
            country = report.country_code
            geo_dist[country] = geo_dist.get(country, 0) + 1

        # Enhanced threat family analysis
        family_analysis = {}
        for report in reports:
            family = report.threat_family
            if family not in family_analysis:
                family_analysis[family] = {'count': 0, 'total_risk': 0, 'max_risk': 0}
            family_analysis[family]['count'] += 1
            family_analysis[family]['total_risk'] += report.risk_score
            family_analysis[family]['max_risk'] = max(family_analysis[family]['max_risk'], report.risk_score)

        top_families = []
        for family, data in sorted(family_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:6]:
            avg_risk = data['total_risk'] / data['count']
            risk_level = "CRITICAL" if avg_risk >= 8 else "HIGH" if avg_risk >= 6 else "MEDIUM"
            top_families.append({
                "name": family,
                "count": data['count'],
                "risk_level": risk_level,
                "avg_risk": round(avg_risk, 1),
                "max_risk": data['max_risk']
            })

        # Recent threat calculations
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_24h = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff:
                    recent_24h += 1
            except:
                pass

        # Trending threats (high risk + recent)
        trending = sorted([r for r in reports if r.risk_score >= 6], 
                         key=lambda x: (x.risk_score, x.confidence_score), reverse=True)[:5]

        trending_threats = [{
            'title': r.title,
            'risk_score': r.risk_score,
            'threat_family': r.threat_family,
            'threat_level': r.threat_level,
            'tactical_impact': r.tactical_impact
        } for r in trending]

        # Enhanced tactical metrics
        avg_confidence = sum(r.confidence_score for r in reports) / len(reports)
        intelligence_confidence = int(avg_confidence * 100)

        # Emerging trends analysis
        emerging_trends = []
        if critical_count > 0:
            emerging_trends.append("Critical Vulnerability Activity")
        if zero_day_count > 0:
            emerging_trends.append("Zero-Day Exploitation")
        if any('ransomware' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Ransomware Campaign Activity")
        if any('apt' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Advanced Persistent Threat Activity")
        if any('supply chain' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Supply Chain Targeting")

        if not emerging_trends:
            emerging_trends = ["Standard Threat Monitoring"]

        # Threat evolution assessment
        if recent_24h > len(reports) * 0.5:
            threat_evolution = "escalating"
        elif critical_count >= 2:
            threat_evolution = "elevated"
        else:
            threat_evolution = "stable"

        metrics = TacticalMetrics(
            total_threats=len(reports),
            critical_threats=threat_counts['CRITICAL'],
            high_threats=threat_counts['HIGH'],
            medium_threats=threat_counts['MEDIUM'],
            low_threats=threat_counts['LOW'],
            global_threat_level=global_level,
            intelligence_confidence=intelligence_confidence,
            recent_threats_24h=recent_24h,
            top_threat_families=top_families,
            geographic_distribution=geo_dist,
            zero_day_count=zero_day_count,
            trending_threats=trending_threats,
            threat_velocity="accelerating" if recent_24h > len(reports) * 0.4 else "stable",
            fresh_intel_24h=recent_24h,
            source_credibility=round(avg_confidence, 2),
            emerging_trends=emerging_trends,
            threat_evolution=threat_evolution,
            daily_summary_confidence=95 if self.ai_summary_generated else 80,
            ai_insights_quality=90 if self.ai_summary_generated else 75,
            tactical_readiness=tactical_readiness,
            force_protection_status=force_protection
        )

        logger.info(f"📊 Tactical metrics calculated: Confidence={intelligence_confidence}%, AI Quality={metrics.ai_insights_quality}%")
        return metrics

    def save_tactical_data(self, reports: List[TacticalThreatReport], metrics: TacticalMetrics, sitrep: TacticalSituationReport) -> None:
        """Save tactical intelligence with enhanced structure and logging"""
        logger.info("💾 Saving tactical intelligence data...")
        
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "daily_summary": asdict(sitrep),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "ai_usage": {
                "tactical_sitrep_generated": self.ai_summary_generated,
                "approach": "Enhanced Tactical Intelligence" + (" with AI" if self.ai_summary_generated else " (Fallback)"),
                "model_used": self.current_model if self.ai_summary_generated else "Enhanced Analytics",
                "api_calls_made": self.ai_calls_made,
                "efficiency_score": 95 if self.ai_summary_generated else 80,
                "cost_optimization": "TACTICAL_VALUE"
            },
            "intelligence_summary": {
                "mission_status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "intelligence_sources": len(self.intelligence_sources),
                "confidence_level": metrics.intelligence_confidence,
                "threat_landscape": metrics.global_threat_level,
                "tactical_readiness": metrics.tactical_readiness,
                "force_protection": metrics.force_protection_status,
                "next_update": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        try:
            output_file = self.data_directory / 'news-analysis.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Tactical Intelligence saved successfully")
            logger.info(f"📊 Reports: {len(reports)} | THREATCON: {metrics.global_threat_level} | AI: {'Generated' if self.ai_summary_generated else 'Fallback'}")
            logger.info(f"🎯 Critical: {metrics.critical_threats} | High: {metrics.high_threats} | Zero-days: {metrics.zero_day_count}")
            logger.info(f"🤖 AI Quality: {metrics.ai_insights_quality}% | Confidence: {metrics.intelligence_confidence}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to save tactical data: {str(e)}")
            raise

async def execute_tactical_intelligence_mission():
    """Execute tactical cyber threat intelligence mission with enhanced logging"""
    logger.info("🎖️ ═══════════════════════════════════════════════════════════════════")
    logger.info("     PATRIOTS PROTOCOL v4.2 - TACTICAL INTELLIGENCE MISSION")
    logger.info("  ═══════════════════════════════════════════════════════════════════")
    
    try:
        async with TacticalPatriotsIntelligence() as intel_engine:
            # Collect tactical intelligence
            logger.info("📡 Initiating tactical intelligence collection...")
            raw_intelligence = await intel_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No tactical intelligence collected - creating baseline report")
                intel_engine.save_tactical_data([], 
                    intel_engine.calculate_tactical_metrics([]), 
                    intel_engine.create_tactical_fallback([]))
                return
            
            # Process with enhanced tactical analysis
            logger.info("🔄 Processing intelligence with tactical analysis...")
            threat_reports = await intel_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed - creating baseline report")
                intel_engine.save_tactical_data([], 
                    intel_engine.calculate_tactical_metrics([]), 
                    intel_engine.create_tactical_fallback([]))
                return
            
            # Generate tactical SITREP
            logger.info("📋 Generating tactical situation report...")
            tactical_sitrep = await intel_engine.generate_tactical_sitrep(threat_reports)
            
            # Calculate tactical metrics
            logger.info("📊 Calculating tactical metrics...")
            metrics = intel_engine.calculate_tactical_metrics(threat_reports)
            
            # Save tactical data
            intel_engine.save_tactical_data(threat_reports, metrics, tactical_sitrep)
            
            # Tactical mission summary
            logger.info("🎖️ ═══════════════════════════════════════════════════════════════════")
            logger.info("     TACTICAL INTELLIGENCE MISSION COMPLETE")
            logger.info("  ═══════════════════════════════════════════════════════════════════")
            logger.info(f"  🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"  🔥 THREATCON Level: {metrics.global_threat_level}")
            logger.info(f"  ⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"  📈 High Priority: {metrics.high_threats}")
            logger.info(f"  💥 Zero-Day Activity: {metrics.zero_day_count}")
            logger.info(f"  🤖 AI Quality: {metrics.ai_insights_quality}%")
            logger.info(f"  📊 Intelligence Confidence: {metrics.intelligence_confidence}%")
            logger.info(f"  🛡️ Tactical Readiness: {metrics.tactical_readiness}")
            logger.info(f"  🛡️ Force Protection: {metrics.force_protection_status}")
            logger.info(f"  🔄 Threat Evolution: {metrics.threat_evolution}")
            logger.info("  ═══════════════════════════════════════════════════════════════════")
            logger.info("  🎖️ Patriots Protocol Tactical v4.2: MISSION SUCCESS")
            
    except Exception as e:
        logger.error(f"❌ TACTICAL INTELLIGENCE MISSION FAILED: {str(e)}")
        logger.error("🔄 Creating emergency fallback report...")
        
        # Create tactical error state with enhanced logging
        try:
            error_data = {
                "articles": [],
                "metrics": {
                    "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                    "medium_threats": 0, "low_threats": 0, "global_threat_level": "OFFLINE",
                    "intelligence_confidence": 0, "recent_threats_24h": 0,
                    "top_threat_families": [], "geographic_distribution": {},
                    "zero_day_count": 0, "trending_threats": [], 
                    "threat_velocity": "unknown", "fresh_intel_24h": 0, "source_credibility": 0.0,
                    "emerging_trends": ["System Recovery Mode"], "threat_evolution": "offline",
                    "daily_summary_confidence": 0, "ai_insights_quality": 0,
                    "tactical_readiness": "OFFLINE", "force_protection_status": "UNKNOWN"
                },
                "daily_summary": {
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "executive_summary": "Tactical intelligence network temporarily offline due to system error. Recovery operations initiated.",
                    "key_developments": ["System error encountered", "Recovery protocols activated", "Standby monitoring active"],
                    "critical_threats_overview": "System offline - no threat analysis available",
                    "trending_attack_vectors": [],
                    "geographic_hotspots": [],
                    "sector_impact_analysis": "Analysis unavailable during system recovery",
                    "recommended_actions": ["Monitor system recovery status", "Await system restoration", "Standby monitoring maintained"],
                    "threat_landscape_assessment": "System offline - assessment unavailable",
                    "zero_day_activity": "Analysis unavailable during recovery",
                    "attribution_insights": "Analysis unavailable during recovery", 
                    "defensive_priorities": ["System recovery", "Standby monitoring"],
                    "tactical_recommendations": ["Await system recovery", "Monitor for updates"],
                    "force_protection_level": "UNKNOWN"
                },
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "version": "4.2",
                "intelligence_summary": {
                    "mission_status": "ERROR",
                    "threats_analyzed": 0,
                    "intelligence_confidence": 0,
                    "threat_landscape": "OFFLINE",
                    "tactical_readiness": "OFFLINE"
                }
            }
            
            os.makedirs('./data', exist_ok=True)
            with open('./data/news-analysis.json', 'w') as f:
                json.dump(error_data, f, indent=2)
                
            logger.info("✅ Emergency fallback report created")
            
        except Exception as fallback_error:
            logger.error(f"❌ Failed to create fallback report: {str(fallback_error)}")

if __name__ == "__main__":
    asyncio.run(execute_tactical_intelligence_mission())
