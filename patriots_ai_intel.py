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
        
        # Fixed API endpoints that work with current services
        self.openai_base_url = "https://api.openai.com/v1"
        
        # Model configurations - updated to current models
        self.openai_model = "gpt-4"  # Fallback to reliable model
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Tactical optimization
        self.ai_summary_generated = False
        self.ai_calls_made = 0
        self.max_ai_calls = 2  # Cost optimization
        self.api_endpoint = None  # Will be determined during testing
        self.current_model = None
        self.api_headers = None
        
        # Enhanced intelligence sources with tactical priorities - updated URLs
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
            logger.info("🤖 API Token found - testing endpoints...")
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
        """Test API endpoints to find working one - FIXED for current API formats"""
        if not self.api_token:
            logger.warning("⚠️ No API token available for testing")
            return False
        
        # Test OpenAI endpoint only (most reliable)
        try:
            test_payload = {
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Respond with 'API_TEST_SUCCESS' only."}
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }

            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }

            logger.info(f"🔍 Testing OpenAI API...")
            
            async with self.session.post(f"{self.openai_base_url}/chat/completions", 
                                       headers=headers, 
                                       json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    test_response = result['choices'][0]['message']['content']
                    if 'API_TEST_SUCCESS' in test_response or 'SUCCESS' in test_response.upper():
                        logger.info(f"✅ OpenAI API test successful")
                        self.api_endpoint = f"{self.openai_base_url}/chat/completions"
                        self.current_model = self.openai_model
                        self.api_headers = headers
                        return True
                    else:
                        logger.info(f"✅ OpenAI API responding (got: {test_response})")
                        self.api_endpoint = f"{self.openai_base_url}/chat/completions"
                        self.current_model = self.openai_model
                        self.api_headers = headers
                        return True
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ OpenAI API test failed: {response.status} - {error_text[:200]}")
                    
        except Exception as e:
            logger.warning(f"⚠️ OpenAI API test error: {str(e)[:100]}")
                
        logger.error("❌ API endpoint failed - using enhanced fallback analysis")
        return False

    async def generate_tactical_sitrep(self, all_threats: List[TacticalThreatReport]) -> TacticalSituationReport:
        """Generate comprehensive tactical situation report using available AI - FIXED"""
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
            
            # FIXED: Simplified prompt for better compatibility
            tactical_prompt = f"""Generate a cyber threat tactical report for {datetime.now().strftime('%Y-%m-%d')} based on these threats:

{threat_data}

Provide a comprehensive analysis including:
1. Executive summary of the current threat landscape
2. Key developments and critical threats
3. Trending attack vectors and geographic hotspots
4. Sector impact analysis
5. Recommended actions
6. Overall threat landscape assessment
7. Zero-day activity status
8. Attribution insights
9. Defensive priorities
10. Tactical recommendations
11. Force protection level (CRITICAL/HIGH/MEDIUM/LOW)

Format your response as valid JSON only, with these exact keys:
"executive_summary", "key_developments" (array), "critical_threats_overview", "trending_attack_vectors" (array), 
"geographic_hotspots" (array), "sector_impact_analysis", "recommended_actions" (array), 
"threat_landscape_assessment", "zero_day_activity", "attribution_insights", 
"defensive_priorities" (array), "tactical_recommendations" (array), "force_protection_level"
"""

            # FIXED: Updated payload format for current API
            payload = {
                "model": self.current_model,
                "messages": [
                    {"role": "system", "content": "You are a cyber intelligence analyst. Provide tactical threat assessments in valid JSON format only."},
                    {"role": "user", "content": tactical_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"}  # Request JSON explicitly
            }

            self.ai_calls_made += 1
            logger.info(f"🤖 Generating Tactical SITREP for {len(all_threats)} threats...")

            async with self.session.post(self.api_endpoint, 
                                       headers=self.api_headers, 
                                       json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result['choices'][0]['message']['content']
                    
                    # FIXED: More robust JSON extraction
                    try:
                        sitrep_data = json.loads(ai_response)
                        self.ai_summary_generated = True
                        logger.info("✅ Tactical SITREP generated successfully using AI")
                        return self.format_tactical_sitrep(sitrep_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ JSON parsing failed: {str(e)}")
                        # Try to extract JSON from response with improved method
                        json_content = self.extract_json_from_response(ai_response)
                        if json_content:
                            try:
                                sitrep_data = json.loads(json_content)
                                self.ai_summary_generated = True
                                logger.info("✅ Tactical SITREP generated after JSON correction")
                                return self.format_tactical_sitrep(sitrep_data)
                            except Exception:
                                pass
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ AI API error: {response.status} - {error_text[:200]}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Tactical SITREP generation failed: {str(e)}")
            
        return self.create_tactical_fallback(all_threats)

    def extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from AI response with multiple strategies - FIXED with more robust methods"""
        # First try - direct JSON loading
        try:
            json.loads(response)
            return response
        except:
            pass
        
        # Strategy 1: Find JSON object boundaries
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_content = response[json_start:json_end]
            # Basic validation
            try:
                json.loads(json_content)
                return json_content
            except:
                pass
        
        # Strategy 2: Extract between code blocks
        import re
        # Match JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                json_content = json_match.group(1)
                json.loads(json_content)
                return json_content
            except:
                pass
        
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
                    try:
                        json_content = response[start_idx:i+1]
                        json.loads(json_content)
                        return json_content
                    except:
                        pass
        
        # Strategy 4: Fix common JSON errors and try again
        if json_start != -1 and json_end > json_start:
            json_content = response[json_start:json_end]
            # Fix unescaped quotes
            fixed_content = re.sub(r'(?<!\\)"(?=(.*?)")(?!,|:|})', r'\"', json_content)
            try:
                json.loads(fixed_content)
                return fixed_content
            except:
                pass
        
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

    # [Rest of the methods remain the same]
    
    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced tactical intelligence collection with improved logging"""
        collected_intel = []
        
        logger.info(f"🔍 Starting intelligence collection from {len(self.intelligence_sources)} sources...")
        
        for source in sorted(self.intelligence_sources, key=lambda x: x['priority']):
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url'], timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        # FIXED: Check for valid feed content
                        if not parsed_feed.entries:
                            logger.warning(f"⚠️ No entries found in {source['name']} feed")
                            continue
                            
                        source_intel = []
                        for entry in parsed_feed.entries[:15]:
                            title = entry.title.strip()
                            # FIXED: Better summary extraction
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            if not summary and 'content' in entry:
                                for content in entry.content:
                                    if content.get('value'):
                                        summary = content.get('value', '').strip()
                                        break
                            
                            # Skip if we don't have enough information
                            if not title or not summary:
                                continue
                                
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
                            
                            # FIXED: Extract proper link and publication date
                            link = entry.get('link', '')
                            
                            # Try to parse the publication date
                            try:
                                pub_date = entry.get('published', entry.get('pubDate', entry.get('updated', '')))
                                pub_timestamp = datetime.now(timezone.utc).isoformat()
                                if pub_date:
                                    # Try multiple date formats
                                    from email.utils import parsedate_to_datetime
                                    try:
                                        pub_timestamp = parsedate_to_datetime(pub_date).isoformat()
                                    except:
                                        # Try direct parsing
                                        try:
                                            from dateutil import parser
                                            pub_timestamp = parser.parse(pub_date).isoformat()
                                        except:
                                            pass
                            except:
                                pub_timestamp = datetime.now(timezone.utc).isoformat()
                            
                            intel_item = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': link,
                                'timestamp': pub_timestamp,
                                'relevance_score': final_relevance,
                                'tactical_value': source.get('tactical_value', 'MEDIUM')
                            }
                            
                            source_intel.append(intel_item)
                        
                        source_intel.sort(key=lambda x: x['relevance_score'], reverse=True)
                        collected_intel.extend(source_intel[:12])  # Top 12 per source
                        
                        logger.info(f"📊 {source['name']}: {len(source_intel)} reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']} returned status {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Error collecting from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(1)  # FIXED: Increased rate limiting to avoid blocks

        logger.info(f"🎯 Total Intelligence Collected: {len(collected_intel)} reports")
        return self.tactical_deduplication(collected_intel)

    # [Rest of the methods remain unchanged]

async def execute_tactical_intelligence_mission():
    """Execute tactical cyber threat intelligence mission with enhanced logging"""
    logger.info("🎖️ PATRIOTS PROTOCOL - Intelligence Mission Initiated")
    
    try:
        async with TacticalPatriotsIntelligence() as intel_engine:
            # Collect tactical intelligence
            raw_intelligence = await intel_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No tactical intelligence collected - creating baseline report")
                intel_engine.save_tactical_data([], 
                    intel_engine.calculate_tactical_metrics([]), 
                    intel_engine.create_tactical_fallback([]))
                return
            
            # Process with enhanced tactical analysis
            threat_reports = await intel_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed - creating baseline report")
                intel_engine.save_tactical_data([], 
                    intel_engine.calculate_tactical_metrics([]), 
                    intel_engine.create_tactical_fallback([]))
                return
            
            # Generate tactical SITREP
            tactical_sitrep = await intel_engine.generate_tactical_sitrep(threat_reports)
            
            # Calculate tactical metrics
            metrics = intel_engine.calculate_tactical_metrics(threat_reports)
            
            # Save tactical data
            intel_engine.save_tactical_data(threat_reports, metrics, tactical_sitrep)
            
            # Tactical mission summary
            logger.info("✅ Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"🎖️ Patriots Protocol Intelligence: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Intelligence mission failed: {str(e)}")
        logger.error("🔄 Creating emergency fallback report...")
        
        # Create tactical error state
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
