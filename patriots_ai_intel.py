#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Tactical AI-Powered Cyber Intelligence Engine v4.2
Fixed for GitHub Models API Compatibility

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

# Tactical logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

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
    """Enhanced Tactical AI-Powered Cyber Threat Intelligence with GitHub Models Support"""
    
    def __init__(self):
        # GitHub Models configuration
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        
        # Updated model for GitHub Models compatibility
        self.model = "openai/gpt-4.1"  # Changed from gpt-4o-mini to gpt-4.1
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Tactical optimization
        self.ai_summary_generated = False
        self.ai_calls_made = 0
        self.max_ai_calls = 3  # Conservative limit for GitHub Models
        
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
        
        logger.info("🎖️ Tactical Patriots Protocol Intelligence Engine v4.2 - GitHub Models Compatible")
        if self.api_token:
            logger.info(f"🤖 GitHub Models API configured - Model: {self.model}")
        else:
            logger.warning("⚠️ No GitHub Models API token found - will use enhanced basic analysis")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=90),  # Increased timeout for GitHub Models
            headers={
                'User-Agent': 'Patriots-Protocol-Tactical/4.2 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_github_models_api(self) -> bool:
        """Test GitHub Models API connectivity"""
        if not self.api_token:
            logger.warning("⚠️ No API token available for testing")
            return False
            
        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            test_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test message - respond with 'API_TEST_SUCCESS'"}
                ],
                "temperature": 0.1,
                "max_tokens": 50
            }

            logger.info(f"🔍 Testing GitHub Models API connectivity...")
            
            async with self.session.post(self.base_url + "/chat/completions", 
                                       headers=headers, 
                                       json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    test_response = result['choices'][0]['message']['content']
                    if 'API_TEST_SUCCESS' in test_response:
                        logger.info("✅ GitHub Models API test successful")
                        return True
                    else:
                        logger.info(f"✅ GitHub Models API responding (got: {test_response[:50]}...)")
                        return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ GitHub Models API test failed: {response.status} - {error_text[:200]}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ GitHub Models API test error: {str(e)}")
            return False

    async def generate_tactical_sitrep(self, all_threats: List[TacticalThreatReport]) -> TacticalSituationReport:
        """Generate comprehensive tactical situation report using GitHub Models"""
        if not self.api_token or not all_threats:
            logger.info("🔄 Using tactical fallback analysis (no API token or no threats)")
            return self.create_tactical_fallback(all_threats)

        # Test API connectivity first
        api_working = await self.test_github_models_api()
        if not api_working:
            logger.warning("⚠️ GitHub Models API not accessible - using tactical fallback")
            return self.create_tactical_fallback(all_threats)

        try:
            threat_data = self.prepare_tactical_summary_data(all_threats)
            
            # Optimized prompt for GitHub Models
            tactical_prompt = f"""Generate a TACTICAL SITUATION REPORT for {datetime.now().strftime('%Y-%m-%d')}.

THREAT DATA:
{threat_data}

Provide tactical analysis in JSON format:
{{
    "executive_summary": "Brief tactical overview of threat landscape",
    "key_developments": [
        "Most significant cyber incident",
        "Critical vulnerability or exploit",
        "Notable threat actor activity"
    ],
    "critical_threats_overview": "Analysis of critical threats and required response",
    "trending_attack_vectors": ["primary_method", "secondary_vector"],
    "geographic_hotspots": ["region_with_activity"],
    "sector_impact_analysis": "Assessment of targeted sectors",
    "recommended_actions": [
        "Immediate action required",
        "Critical patches to deploy",
        "Enhanced monitoring needed"
    ],
    "threat_landscape_assessment": "Overall threat posture assessment",
    "zero_day_activity": "Zero-day vulnerability assessment",
    "attribution_insights": "Threat actor intelligence",
    "defensive_priorities": [
        "Primary defensive priority",
        "Secondary measure",
        "Third priority"
    ],
    "tactical_recommendations": [
        "Operational security recommendation",
        "Force protection measure"
    ],
    "force_protection_level": "CRITICAL/HIGH/MEDIUM/LOW"
}}

Focus on actionable tactical intelligence."""

            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a senior military cyber intelligence analyst. Provide tactical situation reports with actionable intelligence."},
                    {"role": "user", "content": tactical_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }

            self.ai_calls_made += 1
            logger.info(f"🤖 Generating Tactical SITREP for {len(all_threats)} threats using GitHub Models...")

            async with self.session.post(self.base_url + "/chat/completions", 
                                       headers=headers, 
                                       json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result['choices'][0]['message']['content']
                    
                    # Extract JSON from response
                    json_start = ai_response.find('{')
                    json_end = ai_response.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = ai_response[json_start:json_end]
                        sitrep_data = json.loads(json_content)
                        self.ai_summary_generated = True
                        logger.info("✅ Tactical SITREP generated successfully using GitHub Models")
                        return self.format_tactical_sitrep(sitrep_data)
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ GitHub Models API error: {response.status} - {error_text[:200]}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Tactical SITREP generation failed: {str(e)[:100]}... - using tactical fallback")
            
        return self.create_tactical_fallback(all_threats)

    def prepare_tactical_summary_data(self, threats: List[TacticalThreatReport]) -> str:
        """Prepare tactical threat data for AI analysis"""
        summary_data = []
        
        # Categorize threats by tactical priority
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        zero_day_threats = [t for t in threats if 'zero' in t.threat_family.lower()]
        
        summary_data.append(f"TOTAL THREATS: {len(threats)}")
        summary_data.append(f"CRITICAL: {len(critical_threats)}, HIGH: {len(high_threats)}, ZERO-DAY: {len(zero_day_threats)}")
        
        # Critical threats intelligence
        if critical_threats:
            summary_data.append("\nCRITICAL THREATS:")
            for threat in critical_threats[:3]:
                summary_data.append(f"- {threat.title[:80]} ({threat.threat_family}) - Risk: {threat.risk_score}/10")
        
        # High priority threats
        if high_threats:
            summary_data.append("\nHIGH THREATS:")
            for threat in high_threats[:3]:
                summary_data.append(f"- {threat.title[:80]} ({threat.threat_family}) - Risk: {threat.risk_score}/10")
        
        # Tactical analysis data
        families = {}
        geography = {}
        attack_vectors = {}
        
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
            geography[threat.country_code] = geography.get(threat.country_code, 0) + 1
            for vector in threat.attack_vectors:
                attack_vectors[vector] = attack_vectors.get(vector, 0) + 1
        
        top_families = sorted(families.items(), key=lambda x: x[1], reverse=True)[:3]
        top_regions = sorted(geography.items(), key=lambda x: x[1], reverse=True)[:3]
        top_vectors = sorted(attack_vectors.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary_data.append(f"\nTOP FAMILIES: {', '.join([f'{k}({v})' for k, v in top_families])}")
        summary_data.append(f"REGIONS: {', '.join([f'{k}({v})' for k, v in top_regions])}")
        summary_data.append(f"VECTORS: {', '.join([f'{k}({v})' for k, v in top_vectors])}")
        
        return '\n'.join(summary_data)

    def format_tactical_sitrep(self, ai_data: Dict) -> TacticalSituationReport:
        """Format AI response into tactical SITREP structure"""
        return TacticalSituationReport(
            date=datetime.now().strftime('%Y-%m-%d'),
            executive_summary=ai_data.get('executive_summary', 'Tactical analysis complete - threat landscape assessed'),
            key_developments=ai_data.get('key_developments', []),
            critical_threats_overview=ai_data.get('critical_threats_overview', 'No critical threats requiring immediate response'),
            trending_attack_vectors=ai_data.get('trending_attack_vectors', []),
            geographic_hotspots=ai_data.get('geographic_hotspots', []),
            sector_impact_analysis=ai_data.get('sector_impact_analysis', 'Multi-sector monitoring active'),
            recommended_actions=ai_data.get('recommended_actions', []),
            threat_landscape_assessment=ai_data.get('threat_landscape_assessment', 'Threat landscape stable'),
            zero_day_activity=ai_data.get('zero_day_activity', 'No zero-day activity detected'),
            attribution_insights=ai_data.get('attribution_insights', 'Attribution analysis ongoing'),
            defensive_priorities=ai_data.get('defensive_priorities', []),
            tactical_recommendations=ai_data.get('tactical_recommendations', []),
            force_protection_level=ai_data.get('force_protection_level', 'MEDIUM')
        )

    def create_tactical_fallback(self, threats: List[TacticalThreatReport]) -> TacticalSituationReport:
        """Create tactical fallback SITREP when AI unavailable"""
        if not threats:
            return TacticalSituationReport(
                date=datetime.now().strftime('%Y-%m-%d'),
                executive_summary="No significant threats detected in current intelligence cycle",
                key_developments=["Intelligence network operational", "All systems monitoring"],
                critical_threats_overview="No critical threats requiring immediate response",
                trending_attack_vectors=["monitoring"],
                geographic_hotspots=["Global"],
                sector_impact_analysis="No specific sector targeting identified",
                recommended_actions=["Continue standard monitoring", "Maintain security posture"],
                threat_landscape_assessment="Stable monitoring baseline",
                zero_day_activity="No zero-day activity detected",
                attribution_insights="No specific attribution intelligence",
                defensive_priorities=["Maintain vigilance", "Standard monitoring"],
                tactical_recommendations=["Continue operations", "Monitor threat feeds"],
                force_protection_level="LOW"
            )

        # Enhanced tactical fallback analysis
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        zero_days = sum(1 for t in threats if 'zero' in t.threat_family.lower())
        
        # Analyze threat families and geography
        families = {}
        geography = {}
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
            geography[threat.country_code] = geography.get(threat.country_code, 0) + 1
        
        top_family = max(families.items(), key=lambda x: x[1])[0] if families else 'Mixed'
        top_regions = sorted(geography.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate tactical executive summary
        exec_summary = f"Tactical analysis of {len(threats)} contacts shows "
        if critical_threats:
            exec_summary += f"{len(critical_threats)} critical incidents requiring immediate response. "
        exec_summary += f"Primary threat vector: {top_family}. "
        
        # Determine force protection level
        if len(critical_threats) >= 3 or zero_days >= 2:
            force_protection = "CRITICAL"
            exec_summary += "Force protection level: CRITICAL."
        elif len(critical_threats) >= 1 or zero_days >= 1:
            force_protection = "HIGH"
            exec_summary += "Force protection level: HIGH."
        else:
            force_protection = "MEDIUM"
            exec_summary += "Force protection level: MEDIUM."
        
        # Generate key developments
        key_developments = []
        if critical_threats:
            key_developments.append(f"Critical threat detected: {critical_threats[0].title[:60]}...")
        if high_threats:
            key_developments.append(f"High-priority incident: {high_threats[0].title[:60]}...")
        if zero_days:
            key_developments.append(f"Zero-day vulnerability activity detected ({zero_days} incidents)")
        
        if not key_developments:
            key_developments = ["Standard threat monitoring active", "No critical incidents detected"]
        
        # Generate tactical recommendations
        tactical_recommendations = []
        if critical_threats:
            tactical_recommendations.append("Escalate to SOC for immediate response")
            tactical_recommendations.append("Implement enhanced monitoring protocols")
        if zero_days:
            tactical_recommendations.append("Activate emergency patch management procedures")
        if not tactical_recommendations:
            tactical_recommendations = ["Maintain current security posture", "Continue routine monitoring"]
        
        return TacticalSituationReport(
            date=datetime.now().strftime('%Y-%m-%d'),
            executive_summary=exec_summary,
            key_developments=key_developments[:3],
            critical_threats_overview=f"Analysis identified {len(critical_threats)} critical and {len(high_threats)} high-priority threats requiring security team attention",
            trending_attack_vectors=list(set([v for t in threats for v in t.attack_vectors]))[:3],
            geographic_hotspots=[region for region, count in top_regions],
            sector_impact_analysis=f"Multi-sector impact detected with {top_family} being the primary threat category",
            recommended_actions=[
                "Review and apply security patches immediately" if any('patch' in t.title.lower() for t in threats) else "Maintain security monitoring",
                "Enhance monitoring for trending attack vectors",
                "Review incident response procedures"
            ],
            threat_landscape_assessment="Elevated" if critical_threats else "Stable",
            zero_day_activity=f"{zero_days} zero-day incidents detected" if zero_days else "No zero-day activity detected",
            attribution_insights="Threat actor analysis pending" if len(threats) > 5 else "No specific attribution identified",
            defensive_priorities=[
                "Critical patch management" if critical_threats else "Standard monitoring",
                "Endpoint protection verification",
                "Security awareness maintenance"
            ],
            tactical_recommendations=tactical_recommendations,
            force_protection_level=force_protection
        )

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
            'Critical Infrastructure Attack': {
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
            return 'CRITICAL'
        elif risk_score >= 7 or any(term in content for term in ['significant', 'major', 'serious']):
            return 'HIGH'
        elif risk_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def calculate_probability_level(self, content: str, modifiers: List[str]) -> str:
        """Calculate probability of exploitation"""
        if 'active_exploitation' in modifiers or any(term in content for term in ['under attack', 'exploited']):
            return 'HIGH'
        elif any(term in content for term in ['proof of concept', 'demonstrated', 'likely']):
            return 'MEDIUM'
        else:
            return 'LOW'

    def calculate_sophistication_level(self, content: str, family: str) -> str:
        """Calculate attack sophistication level"""
        if family in ['APT Campaign', 'Zero-Day Exploit'] or any(term in content for term in ['nation-state', 'advanced']):
            return 'ADVANCED'
        elif family in ['Ransomware', 'Supply Chain Attack'] or any(term in content for term in ['sophisticated', 'coordinated']):
            return 'HIGH'
        elif any(term in content for term in ['automated', 'tool', 'framework']):
            return 'MEDIUM'
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
        if family == 'Critical Infrastructure Attack':
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
        elif risk_score >= 9 or family in ['Zero-Day Exploit', 'Critical Infrastructure Attack']:
            return "Critical tactical impact - immediate command attention required"
        elif risk_score >= 7 or family in ['Ransomware', 'APT Campaign']:
            return "Significant tactical impact - enhanced security posture required"
        elif risk_score >= 5:
            return "Moderate tactical impact - standard security procedures apply"
        else:
            return "Minimal tactical impact - awareness and routine monitoring sufficient"

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
            return "Critical business impact - immediate executive and operational leadership attention required"
        elif risk_score >= 7:
            return "Significant business impact - security team escalation and enhanced response required"
        elif risk_score >= 5:
            return "Moderate business impact - standard security procedures and monitoring apply"
        else:
            return "Limited business impact - awareness and routine security review sufficient"

    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced tactical intelligence collection"""
        collected_intel = []
        
        for source in sorted(self.intelligence_sources, key=lambda x: x['priority']):
            try:
                logger.info(f"🔍 Collecting from {source['name']} (Priority: {source['priority']})...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:12]:
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Enhanced tactical relevance filtering
                            full_content = (title + ' ' + summary).lower()
                            tactical_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'zero-day',
                                'patch', 'cve', 'incident', 'compromise', 'backdoor', 'trojan',
                                'apt', 'nation-state', 'critical', 'infrastructure'
                            ]
                            
                            relevance_score = sum(1 for indicator in tactical_indicators if indicator in full_content)
                            
                            # Enhanced filtering criteria
                            if relevance_score < 1:
                                continue
                            
                            if len(summary) < 50 or len(title) < 15:
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
                        collected_intel.extend(source_intel[:10])  # Top 10 per source
                        
                        logger.info(f"📊 {source['name']}: {len(source_intel)} tactical reports")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(0.3)

        logger.info(f"🎯 Total Tactical Intelligence: {len(collected_intel)} reports")
        return self.tactical_deduplication(collected_intel)

    def tactical_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Tactical intelligence deduplication"""
        unique_intel = []
        seen_signatures = set()
        
        for intel in raw_intel:
            title_words = set(re.findall(r'\w+', intel['title'].lower()))
            content_signature = hashlib.sha256(''.join(sorted(title_words)).encode()).hexdigest()[:12]
            
            if content_signature not in seen_signatures:
                seen_signatures.add(content_signature)
                unique_intel.append(intel)
        
        unique_intel.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"🔄 Tactical Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel[:30]  # Top 30 most tactically relevant

    async def process_intelligence(self, raw_intel: List[Dict]) -> List[TacticalThreatReport]:
        """Process intelligence with enhanced tactical analysis"""
        threat_reports = []
        
        for intel_item in raw_intel:
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
                logger.info(f"✅ Processed: {threat_report.title[:50]}... ({threat_report.threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_tactical_metrics(self, reports: List[TacticalThreatReport]) -> TacticalMetrics:
        """Calculate tactical metrics with enhanced assessment"""
        if not reports:
            return TacticalMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], threat_velocity="stable", fresh_intel_24h=0, 
                source_credibility=0.0, emerging_trends=[], threat_evolution="stable",
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
        trending = sorted([r for r in reports if r.risk_score >= 7], 
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
            emerging_trends.append("Critical Vulnerability Surge")
        if zero_day_count > 0:
            emerging_trends.append("Zero-Day Activity")
        if any('ransomware' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Ransomware Campaign")
        if any('apt' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Advanced Persistent Threat")

        # Threat evolution assessment
        if recent_24h > len(reports) * 0.5:
            threat_evolution = "escalating"
        elif critical_count >= 2:
            threat_evolution = "elevated"
        else:
            threat_evolution = "stable"

        return TacticalMetrics(
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
            emerging_trends=emerging_trends or ["Intelligence Network Monitoring"],
            threat_evolution=threat_evolution,
            daily_summary_confidence=95 if self.ai_summary_generated else 75,
            ai_insights_quality=90 if self.ai_summary_generated else 70,
            tactical_readiness=tactical_readiness,
            force_protection_status=force_protection
        )

    def save_tactical_data(self, reports: List[TacticalThreatReport], metrics: TacticalMetrics, sitrep: TacticalSituationReport) -> None:
        """Save tactical intelligence with enhanced structure"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "daily_summary": asdict(sitrep),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "ai_usage": {
                "tactical_sitrep_generated": self.ai_summary_generated,
                "approach": "GitHub Models Tactical Intelligence",
                "model_used": self.model,
                "api_calls_made": self.ai_calls_made,
                "efficiency_score": 95 if self.ai_summary_generated else 75,
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

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Tactical Intelligence saved: {len(reports)} reports")
        logger.info(f"🎯 THREATCON Level: {metrics.global_threat_level}")
        logger.info(f"🤖 GitHub Models SITREP: {'Generated' if self.ai_summary_generated else 'Fallback Used'}")
        logger.info(f"📊 AI Analysis Quality: {metrics.ai_insights_quality}%")

async def execute_tactical_intelligence_mission():
    """Execute tactical cyber threat intelligence mission with GitHub Models"""
    logger.info("🎖️ PATRIOTS PROTOCOL v4.2 - Tactical Intelligence Mission (GitHub Models)")
    
    try:
        async with TacticalPatriotsIntelligence() as intel_engine:
            # Test GitHub Models API connectivity
            logger.info("🔍 Testing GitHub Models API connectivity...")
            
            # Collect tactical intelligence
            raw_intelligence = await intel_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No tactical intelligence collected")
                return
            
            # Process with enhanced tactical analysis
            threat_reports = await intel_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed")
                return
            
            # Generate tactical SITREP using GitHub Models
            tactical_sitrep = await intel_engine.generate_tactical_sitrep(threat_reports)
            
            # Calculate tactical metrics
            metrics = intel_engine.calculate_tactical_metrics(threat_reports)
            
            # Save tactical data
            intel_engine.save_tactical_data(threat_reports, metrics, tactical_sitrep)
            
            # Tactical mission summary
            logger.info("✅ Tactical Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 THREATCON Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"💥 Zero-Day Activity: {metrics.zero_day_count}")
            logger.info(f"🤖 AI Quality: {metrics.ai_insights_quality}%")
            logger.info(f"📈 Intelligence Confidence: {metrics.intelligence_confidence}%")
            logger.info(f"🛡️ Tactical Readiness: {metrics.tactical_readiness}")
            logger.info(f"🎖️ Patriots Protocol Tactical v4.2: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Tactical intelligence mission failed: {str(e)}")
        
        # Create tactical error state
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                "medium_threats": 0, "low_threats": 0, "global_threat_level": "OFFLINE",
                "intelligence_confidence": 0, "recent_threats_24h": 0,
                "top_threat_families": [], "geographic_distribution": {},
                "zero_day_count": 0, "trending_threats": [], 
                "threat_velocity": "unknown", "fresh_intel_24h": 0, "source_credibility": 0.0,
                "emerging_trends": ["System Recovery"], "threat_evolution": "offline",
                "daily_summary_confidence": 0, "ai_insights_quality": 0,
                "tactical_readiness": "OFFLINE", "force_protection_status": "UNKNOWN"
            },
            "daily_summary": {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "executive_summary": "Tactical intelligence network temporarily offline - recovery operations in progress",
                "key_developments": ["System maintenance", "Recovery protocols active"],
                "critical_threats_overview": "System offline - no threat analysis available",
                "trending_attack_vectors": [],
                "geographic_hotspots": [],
                "sector_impact_analysis": "Analysis unavailable during maintenance",
                "recommended_actions": ["Monitor system status", "Await system recovery"],
                "threat_landscape_assessment": "System offline",
                "zero_day_activity": "Analysis unavailable",
                "attribution_insights": "Analysis unavailable", 
                "defensive_priorities": ["Await system recovery"],
                "tactical_recommendations": ["Monitor system recovery"],
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

if __name__ == "__main__":
    asyncio.run(execute_tactical_intelligence_mission())
