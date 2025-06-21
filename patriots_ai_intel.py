#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Enhanced AI-Powered Cyber Intelligence Engine v4.2
Daily AI Summary Approach with Smart Value Optimization

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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class ThreatReport:
    """Enhanced Cyber Threat Intelligence Report"""
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

@dataclass
class DailyIntelligenceSummary:
    """Daily AI-Generated Intelligence Summary"""
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

@dataclass
class IntelligenceMetrics:
    """Enhanced Intelligence Metrics"""
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

class SmartPatriotsIntelligence:
    """Enhanced AI-Powered Cyber Threat Intelligence with Daily Summary Approach"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4o-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Optimized AI usage - single daily summary call
        self.ai_summary_generated = False
        
        # Enhanced intelligence sources
        self.intelligence_sources = [
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.92,
                'priority': 1
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'priority': 1
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'priority': 1
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.88,
                'priority': 2
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.85,
                'priority': 2
            },
            {
                'name': 'SECURITY_WEEK',
                'url': 'https://www.securityweek.com/feed/',
                'reliability': 0.87,
                'priority': 2
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
        
        logger.info("🎖️ Smart Patriots Protocol Intelligence Engine v4.2 - Daily Summary Mode")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-Enhanced/4.2 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_daily_ai_summary(self, all_threats: List[ThreatReport]) -> DailyIntelligenceSummary:
        """Generate comprehensive daily AI summary from all collected threats"""
        if not self.api_token or not all_threats:
            return self.create_fallback_summary(all_threats)

        try:
            # Prepare comprehensive threat data for AI analysis
            threat_data = self.prepare_threat_summary_data(all_threats)
            
            summary_prompt = f"""As a senior cybersecurity analyst, provide a comprehensive DAILY CYBER THREAT INTELLIGENCE SUMMARY for {datetime.now().strftime('%Y-%m-%d')}.

THREAT DATA ANALYZED:
{threat_data}

Provide analysis in this EXACT JSON format:
{{
    "executive_summary": "2-3 sentence executive overview of today's threat landscape and key developments",
    "key_developments": [
        "Most significant security incident or development",
        "Important vulnerability disclosure or patch release",
        "Notable threat actor activity or campaign"
    ],
    "critical_threats_overview": "Detailed analysis of the most critical threats identified today, their impact and urgency",
    "trending_attack_vectors": ["vector1", "vector2", "vector3"],
    "geographic_hotspots": ["country/region where significant activity detected"],
    "sector_impact_analysis": "Analysis of which industry sectors are most affected and why",
    "recommended_actions": [
        "Immediate action organizations should take today",
        "Critical patches or updates to prioritize",
        "Enhanced monitoring or defensive measures to implement"
    ],
    "threat_landscape_assessment": "Overall assessment of how today's threats compare to recent patterns - escalating, stable, or improving",
    "zero_day_activity": "Specific analysis of any zero-day or critical vulnerability activity",
    "attribution_insights": "Any threat actor attribution or campaign intelligence identified",
    "defensive_priorities": [
        "Top defensive priority for security teams",
        "Second priority action",
        "Third priority action"
    ]
}}

Focus on ACTIONABLE intelligence that helps security teams make decisions TODAY. Be specific about what organizations should do."""

            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a senior cybersecurity analyst specializing in daily threat intelligence summaries. Provide specific, actionable insights."},
                    {"role": "user", "content": summary_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }

            logger.info(f"🤖 Generating Daily AI Summary for {len(all_threats)} threats...")

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
                        summary_data = json.loads(json_content)
                        self.ai_summary_generated = True
                        logger.info("✅ Daily AI Summary generated successfully")
                        return self.format_daily_summary(summary_data)
                else:
                    logger.warning(f"⚠️ AI API error: {response.status}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Daily AI summary failed: {str(e)[:100]}... - using enhanced fallback")
            
        return self.create_fallback_summary(all_threats)

    def prepare_threat_summary_data(self, threats: List[ThreatReport]) -> str:
        """Prepare concise threat data for AI analysis"""
        summary_data = []
        
        # Group threats by level for better analysis
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        
        summary_data.append(f"TOTAL THREATS: {len(threats)}")
        summary_data.append(f"CRITICAL: {len(critical_threats)}, HIGH: {len(high_threats)}")
        
        # Add critical threats details
        if critical_threats:
            summary_data.append("\nCRITICAL THREATS:")
            for threat in critical_threats[:3]:  # Top 3 critical
                summary_data.append(f"- {threat.title} ({threat.threat_family}) - Risk: {threat.risk_score}/10")
        
        # Add high threats details
        if high_threats:
            summary_data.append("\nHIGH THREATS:")
            for threat in high_threats[:3]:  # Top 3 high
                summary_data.append(f"- {threat.title} ({threat.threat_family}) - Risk: {threat.risk_score}/10")
        
        # Add threat families and geographic distribution
        families = {}
        geography = {}
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
            geography[threat.country_code] = geography.get(threat.country_code, 0) + 1
        
        top_families = sorted(families.items(), key=lambda x: x[1], reverse=True)[:3]
        top_regions = sorted(geography.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary_data.append(f"\nTOP THREAT FAMILIES: {', '.join([f'{k}({v})' for k, v in top_families])}")
        summary_data.append(f"GEOGRAPHIC HOTSPOTS: {', '.join([f'{k}({v})' for k, v in top_regions])}")
        
        # Add CVE and zero-day info
        cves = []
        zero_days = 0
        for threat in threats:
            cves.extend(threat.cve_references)
            if 'zero' in threat.threat_family.lower() or any('zero' in insight.lower() for insight in threat.key_insights):
                zero_days += 1
        
        if cves:
            summary_data.append(f"CVE REFERENCES: {', '.join(list(set(cves))[:5])}")
        if zero_days:
            summary_data.append(f"ZERO-DAY ACTIVITY: {zero_days} incidents")
        
        return '\n'.join(summary_data)

    def format_daily_summary(self, ai_data: Dict) -> DailyIntelligenceSummary:
        """Format AI response into daily summary structure"""
        return DailyIntelligenceSummary(
            date=datetime.now().strftime('%Y-%m-%d'),
            executive_summary=ai_data.get('executive_summary', 'Daily threat analysis complete'),
            key_developments=ai_data.get('key_developments', []),
            critical_threats_overview=ai_data.get('critical_threats_overview', 'No critical threats identified'),
            trending_attack_vectors=ai_data.get('trending_attack_vectors', []),
            geographic_hotspots=ai_data.get('geographic_hotspots', []),
            sector_impact_analysis=ai_data.get('sector_impact_analysis', 'Multi-sector monitoring active'),
            recommended_actions=ai_data.get('recommended_actions', []),
            threat_landscape_assessment=ai_data.get('threat_landscape_assessment', 'Threat landscape stable'),
            zero_day_activity=ai_data.get('zero_day_activity', 'No zero-day activity detected'),
            attribution_insights=ai_data.get('attribution_insights', 'Attribution analysis ongoing'),
            defensive_priorities=ai_data.get('defensive_priorities', [])
        )

    def create_fallback_summary(self, threats: List[ThreatReport]) -> DailyIntelligenceSummary:
        """Create enhanced fallback summary when AI is unavailable"""
        if not threats:
            return DailyIntelligenceSummary(
                date=datetime.now().strftime('%Y-%m-%d'),
                executive_summary="No significant threats detected in current monitoring cycle",
                key_developments=["Intelligence network operational", "Monitoring all sources"],
                critical_threats_overview="No critical threats identified in current analysis period",
                trending_attack_vectors=["monitoring"],
                geographic_hotspots=["Global"],
                sector_impact_analysis="No specific sector targeting identified",
                recommended_actions=["Continue standard monitoring", "Maintain security posture"],
                threat_landscape_assessment="Stable monitoring baseline",
                zero_day_activity="No zero-day activity detected",
                attribution_insights="No specific attribution intelligence",
                defensive_priorities=["Maintain vigilance", "Standard monitoring"]
            )

        # Enhanced fallback analysis
        critical_threats = [t for t in threats if t.threat_level == 'CRITICAL']
        high_threats = [t for t in threats if t.threat_level == 'HIGH']
        
        # Analyze threat families
        families = {}
        for threat in threats:
            families[threat.threat_family] = families.get(threat.threat_family, 0) + 1
        top_family = max(families.items(), key=lambda x: x[1])[0] if families else 'Mixed'
        
        # Analyze geography
        geography = {}
        for threat in threats:
            geography[threat.country_code] = geography.get(threat.country_code, 0) + 1
        top_regions = sorted(geography.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Check for zero-days
        zero_days = sum(1 for t in threats if 'zero' in t.threat_family.lower())
        
        # Generate summary
        exec_summary = f"Analysis of {len(threats)} threats shows "
        if critical_threats:
            exec_summary += f"{len(critical_threats)} critical incidents requiring immediate attention. "
        exec_summary += f"Primary threat category: {top_family}. "
        exec_summary += "Enhanced monitoring and standard security procedures recommended."
        
        key_developments = []
        if critical_threats:
            key_developments.append(f"Critical threat detected: {critical_threats[0].title[:80]}...")
        if high_threats:
            key_developments.append(f"High-priority incident: {high_threats[0].title[:80]}...")
        if zero_days:
            key_developments.append(f"Zero-day vulnerability activity detected ({zero_days} incidents)")
        
        if not key_developments:
            key_developments = ["Standard threat monitoring active", "No critical incidents detected"]
        
        return DailyIntelligenceSummary(
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
            ]
        )

    def enhanced_basic_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Enhanced basic analysis for individual threats"""
        full_text = (title + ' ' + content).lower()
        
        # Enhanced threat family detection
        threat_families = {
            'Ransomware': ['ransomware', 'encryption', 'ransom', 'lockbit', 'conti', 'revil', 'blackcat'],
            'Zero-Day Exploit': ['zero-day', 'zero day', '0-day', 'unknown vulnerability'],
            'Data Breach': ['data breach', 'breach', 'stolen data', 'exposed records'],
            'Vulnerability Disclosure': ['patch tuesday', 'security update', 'patches', 'vulnerability'],
            'APT Campaign': ['apt', 'nation-state', 'state-sponsored', 'advanced persistent'],
            'Supply Chain Attack': ['supply chain', 'software supply', 'third-party'],
            'Malware Campaign': ['malware', 'trojan', 'backdoor', 'spyware', 'botnet'],
            'Phishing Campaign': ['phishing', 'spear phishing', 'business email'],
            'Critical Infrastructure': ['infrastructure', 'utility', 'scada', 'industrial'],
            'Security Incident': ['systems restored', 'security incident', 'cyber attack']
        }
        
        detected_family = 'Security Incident'
        confidence = 0.6
        
        for family, keywords in threat_families.items():
            if any(keyword in full_text for keyword in keywords):
                detected_family = family
                confidence = 0.8
                break
        
        # Smart risk scoring
        risk_score = 3
        if any(term in full_text for term in ['zero-day', 'critical vulnerability', 'rce']):
            risk_score = 9
        elif any(term in full_text for term in ['ransomware', 'apt', 'nation-state']):
            risk_score = 7
        elif any(term in full_text for term in ['malware', 'phishing', 'vulnerability']):
            risk_score = 5
        
        # Extract key insights
        key_insights = []
        if 'zero-day' in full_text:
            key_insights.append("Zero-day vulnerability requires immediate attention")
        if 'ransomware' in full_text:
            key_insights.append("Ransomware threat - verify backup systems")
        if 'critical' in full_text and 'patch' in full_text:
            key_insights.append("Critical patches available for deployment")
        
        # Determine threat level
        if risk_score >= 8:
            threat_level = 'CRITICAL'
        elif risk_score >= 6:
            threat_level = 'HIGH'
        elif risk_score >= 4:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'
        
        return {
            'threat_family': detected_family,
            'risk_score': risk_score,
            'threat_level': threat_level,
            'confidence': confidence,
            'key_insights': key_insights,
            'attack_vectors': self.extract_attack_vectors(full_text),
            'affected_sectors': self.extract_sectors(full_text),
            'geographic_scope': self.extract_geography(full_text),
            'cve_references': re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()),
            'business_impact': self.assess_business_impact(full_text, risk_score)
        }

    def extract_attack_vectors(self, content: str) -> List[str]:
        """Extract attack vectors from content"""
        vectors = []
        vector_mapping = {
            'email': ['phishing', 'email', 'attachment'],
            'web': ['website', 'web', 'browser'],
            'network': ['network', 'remote', 'lateral'],
            'supply_chain': ['supply chain', 'third-party']
        }
        
        for vector, keywords in vector_mapping.items():
            if any(keyword in content for keyword in keywords):
                vectors.append(vector)
        
        return vectors[:3]

    def extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors"""
        sectors = []
        sector_keywords = {
            'healthcare': ['hospital', 'medical', 'health'],
            'financial': ['bank', 'finance', 'financial'],
            'government': ['government', 'federal', 'agency'],
            'technology': ['tech', 'software', 'microsoft'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        
        return sectors[:2]

    def extract_geography(self, content: str) -> List[str]:
        """Extract geographic scope"""
        geography = []
        for region, code in self.geographic_mapping.items():
            if region in content:
                geography.append(region.title())
                if len(geography) >= 2:
                    break
        
        return geography or ['Global']

    def assess_business_impact(self, content: str, risk_score: int) -> str:
        """Assess business impact"""
        if 'systems restored' in content:
            return "Operations restored - monitor for residual impact"
        elif risk_score >= 8:
            return "High business impact - immediate executive attention required"
        elif risk_score >= 6:
            return "Moderate business impact - security team escalation needed"
        else:
            return "Limited business impact - standard monitoring protocols apply"

    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced intelligence collection"""
        collected_intel = []
        
        for source in sorted(self.intelligence_sources, key=lambda x: x['priority']):
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:10]:
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Enhanced relevance filtering
                            full_content = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'zero-day',
                                'patch', 'cve', 'incident', 'compromise'
                            ]
                            
                            relevance_score = sum(1 for indicator in cyber_indicators if indicator in full_content)
                            if relevance_score < 1:
                                continue
                            
                            if len(summary) < 50 or len(title) < 15:
                                continue
                            
                            intel_item = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'relevance_score': relevance_score
                            }
                            
                            source_intel.append(intel_item)
                        
                        source_intel.sort(key=lambda x: x['relevance_score'], reverse=True)
                        collected_intel.extend(source_intel[:8])
                        
                        logger.info(f"📊 {source['name']}: {len(source_intel)} reports")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(0.3)

        logger.info(f"🎯 Total Intelligence: {len(collected_intel)} reports")
        return self.smart_deduplication(collected_intel)

    def smart_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Smart deduplication"""
        unique_intel = []
        seen_signatures = set()
        
        for intel in raw_intel:
            title_words = set(re.findall(r'\w+', intel['title'].lower()))
            content_signature = hashlib.sha256(''.join(sorted(title_words)).encode()).hexdigest()[:12]
            
            if content_signature not in seen_signatures:
                seen_signatures.add(content_signature)
                unique_intel.append(intel)
        
        unique_intel.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"🔄 Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel[:25]

    async def process_intelligence(self, raw_intel: List[Dict]) -> List[ThreatReport]:
        """Process intelligence with basic analysis (no individual AI calls)"""
        threat_reports = []
        
        for intel_item in raw_intel:
            try:
                # Use enhanced basic analysis for each item
                analysis = self.enhanced_basic_analysis(
                    intel_item['title'], 
                    intel_item['summary'],
                    intel_item['source']
                )
                
                # Smart geographic processing
                geographic_scope = 'Global'
                country_code = 'GLOBAL'
                
                if analysis.get('geographic_scope'):
                    geographic_scope = ', '.join(analysis['geographic_scope'][:2])
                    country_code = self.geographic_mapping.get(
                        analysis['geographic_scope'][0].lower(), 'GLOBAL'
                    )
                
                # Create threat report
                threat_report = ThreatReport(
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
                    threat_actors=[],
                    mitigation_priority='URGENT' if analysis['risk_score'] >= 7 else 'ROUTINE',
                    cve_references=analysis.get('cve_references', []),
                    threat_family=analysis['threat_family'],
                    attack_sophistication='HIGH' if analysis['risk_score'] >= 7 else 'MEDIUM',
                    risk_score=analysis['risk_score'],
                    correlation_id=hashlib.md5(intel_item['title'].encode()).hexdigest()[:8],
                    key_insights=analysis.get('key_insights', []),
                    business_impact=analysis['business_impact']
                )
                
                threat_reports.append(threat_report)
                logger.info(f"✅ Processed: {threat_report.title[:50]}... ({threat_report.threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_accurate_metrics(self, reports: List[ThreatReport]) -> IntelligenceMetrics:
        """Calculate accurate metrics with proper validation"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], threat_velocity="stable", fresh_intel_24h=0, 
                source_credibility=0.0, emerging_trends=[], threat_evolution="stable",
                daily_summary_confidence=85 if self.ai_summary_generated else 70,
                ai_insights_quality=95 if self.ai_summary_generated else 75
            )

        # Accurate threat level counting
        threat_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            level = report.threat_level.upper()
            if level in threat_counts:
                threat_counts[level] += 1

        # Smart global threat level assessment
        if threat_counts['CRITICAL'] >= 3:
            global_level = "CRITICAL"
        elif threat_counts['CRITICAL'] >= 1 or threat_counts['HIGH'] >= 5:
            global_level = "HIGH"
        elif threat_counts['HIGH'] >= 2:
            global_level = "ELEVATED"
        else:
            global_level = "MEDIUM"

        # Geographic distribution
        geo_dist = {}
        for report in reports:
            country = report.country_code
            geo_dist[country] = geo_dist.get(country, 0) + 1

        # Threat family analysis
        family_analysis = {}
        for report in reports:
            family = report.threat_family
            if family not in family_analysis:
                family_analysis[family] = {'count': 0, 'total_risk': 0}
            family_analysis[family]['count'] += 1
            family_analysis[family]['total_risk'] += report.risk_score

        top_families = []
        for family, data in sorted(family_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:6]:
            avg_risk = data['total_risk'] / data['count']
            risk_level = "CRITICAL" if avg_risk >= 7 else "HIGH" if avg_risk >= 5 else "MEDIUM"
            top_families.append({
                "name": family,
                "count": data['count'],
                "risk_level": risk_level,
                "avg_risk": round(avg_risk, 1)
            })

        # Zero-day and recent threat calculations
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower())

        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_24h = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff:
                    recent_24h += 1
            except:
                pass

        # Trending threats
        trending = sorted([r for r in reports if r.risk_score >= 6], 
                         key=lambda x: x.risk_score, reverse=True)[:5]

        trending_threats = [{
            'title': r.title,
            'risk_score': r.risk_score,
            'threat_family': r.threat_family,
            'threat_level': r.threat_level
        } for r in trending]

        # Enhanced metrics
        avg_confidence = sum(r.confidence_score for r in reports) / len(reports)
        intelligence_confidence = int(avg_confidence * 100)

        emerging_trends = []
        if threat_counts['CRITICAL'] > 0:
            emerging_trends.append("Critical Vulnerability Surge")
        if zero_day_count > 0:
            emerging_trends.append("Zero-Day Activity")
        if any('ransomware' in r.threat_family.lower() for r in reports):
            emerging_trends.append("Ransomware Campaign")

        return IntelligenceMetrics(
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
            threat_evolution="escalating" if threat_counts['CRITICAL'] > 0 else "stable",
            daily_summary_confidence=95 if self.ai_summary_generated else 75,
            ai_insights_quality=90 if self.ai_summary_generated else 70
        )

    def save_intelligence_data(self, reports: List[ThreatReport], metrics: IntelligenceMetrics, daily_summary: DailyIntelligenceSummary) -> None:
        """Save enhanced intelligence with daily summary"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "daily_summary": asdict(daily_summary),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "ai_usage": {
                "daily_summary_generated": self.ai_summary_generated,
                "approach": "Daily Summary Mode",
                "efficiency_score": 95 if self.ai_summary_generated else 75,
                "cost_optimization": "MAXIMUM_VALUE"
            },
            "intelligence_summary": {
                "mission_status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "intelligence_sources": len(self.intelligence_sources),
                "confidence_level": metrics.intelligence_confidence,
                "threat_landscape": metrics.global_threat_level,
                "daily_summary_quality": metrics.daily_summary_confidence,
                "next_update": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Enhanced Intelligence saved: {len(reports)} reports")
        logger.info(f"🎯 Global Threat Level: {metrics.global_threat_level}")
        logger.info(f"🤖 Daily AI Summary: {'Generated' if self.ai_summary_generated else 'Fallback Used'}")

async def execute_enhanced_intelligence_mission():
    """Execute enhanced cyber threat intelligence mission with daily summary"""
    logger.info("🎖️ PATRIOTS PROTOCOL v4.2 - Daily Summary Intelligence Mission")
    
    try:
        async with SmartPatriotsIntelligence() as intelligence_engine:
            # Collect intelligence
            raw_intelligence = await intelligence_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No intelligence collected")
                return
            
            # Process with basic analysis (no individual AI calls)
            threat_reports = await intelligence_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed")
                return
            
            # Generate single daily AI summary
            daily_summary = await intelligence_engine.generate_daily_ai_summary(threat_reports)
            
            # Calculate metrics
            metrics = intelligence_engine.calculate_accurate_metrics(threat_reports)
            
            # Save enhanced data
            intelligence_engine.save_intelligence_data(threat_reports, metrics, daily_summary)
            
            # Mission summary
            logger.info("✅ Enhanced Intelligence Mission Complete")
            logger.info(f"🎯 High-Value Threats: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"💥 Zero-Day Activity: {metrics.zero_day_count}")
            logger.info(f"🤖 AI Quality Score: {metrics.ai_insights_quality}%")
            logger.info(f"📈 Intelligence Confidence: {metrics.intelligence_confidence}%")
            logger.info(f"🎖️ Patriots Protocol Enhanced v4.2: MAXIMUM VALUE OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Enhanced intelligence mission failed: {str(e)}")
        
        # Create error state
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
                "daily_summary_confidence": 0, "ai_insights_quality": 0
            },
            "daily_summary": {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "executive_summary": "System temporarily offline - recovery in progress",
                "key_developments": ["System maintenance", "Recovery protocols active"],
                "critical_threats_overview": "System offline - no threat analysis available",
                "trending_attack_vectors": [],
                "geographic_hotspots": [],
                "sector_impact_analysis": "Analysis unavailable during maintenance",
                "recommended_actions": ["Monitor system status", "Await system recovery"],
                "threat_landscape_assessment": "System offline",
                "zero_day_activity": "Analysis unavailable",
                "attribution_insights": "Analysis unavailable", 
                "defensive_priorities": ["Await system recovery"]
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "intelligence_summary": {
                "mission_status": "ERROR",
                "threats_analyzed": 0,
                "intelligence_confidence": 0,
                "threat_landscape": "OFFLINE"
            }
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(execute_enhanced_intelligence_mission())
