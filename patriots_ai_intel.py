#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Cyber Threat Intelligence Engine
Real-time threat monitoring and analysis system
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

logging.basicConfig(
    level=logging.INFO,
    format='🎖️ %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class ThreatReport:
    """Cyber Threat Intelligence Report"""
    title: str
    summary: str
    source: str
    source_url: str
    timestamp: str
    threat_level: str
    ai_technical_analysis: str
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
    summary_preview: str
    full_summary: str
    key_insights: List[str]
    actionable_items: List[str]
    technical_indicators: List[str]
    business_impact: str
    timeline_urgency: str
    smart_analysis: Dict[str, Any]

@dataclass
class IntelligenceMetrics:
    """Intelligence Metrics"""
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
    ai_analysis_quality: int
    threat_velocity: str
    fresh_intel_24h: int
    source_credibility: float
    emerging_trends: List[str]
    threat_evolution: str

class SmartPatriotsIntelligence:
    """Cyber Threat Intelligence Collection and Analysis"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4o-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # AI usage tracking
        self.max_ai_calls_per_run = 12
        self.ai_calls_made = 0
        
        # Intelligence sources
        self.intelligence_sources = [
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.92,
                'priority': 1,
                'ai_analysis': True
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'priority': 1,
                'ai_analysis': True
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'priority': 1,
                'ai_analysis': True
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.88,
                'priority': 2,
                'ai_analysis': False
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.85,
                'priority': 2,
                'ai_analysis': False
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
        
        logger.info("🎖️ Patriots Protocol Intelligence Engine - Operational")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol/4.2 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def should_use_ai_analysis(self, content: str, source_config: Dict, basic_risk: int) -> bool:
        """Determine if AI analysis should be used"""
        if not self.api_token or self.ai_calls_made >= self.max_ai_calls_per_run:
            return False
        
        if not source_config.get('ai_analysis', False):
            return False
        
        content_lower = content.lower()
        
        # High-value indicators for AI analysis
        ai_indicators = [
            'zero-day', '0-day', 'critical vulnerability', 'remote code execution',
            'supply chain', 'ransomware', 'apt', 'nation-state', 'breach',
            'critical infrastructure', 'exclusive'
        ]
        
        has_high_value_content = any(indicator in content_lower for indicator in ai_indicators)
        is_high_risk = basic_risk >= 6
        is_priority_source = source_config.get('priority', 3) <= 1
        
        return has_high_value_content or (is_priority_source and is_high_risk)

    async def ai_analysis(self, title: str, content: str, source_config: Dict) -> Dict[str, Any]:
        """AI analysis for threat intelligence"""
        if not self.should_use_ai_analysis(title + ' ' + content, source_config, 5):
            return self.basic_analysis(title, content, source_config['name'])

        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            analysis_prompt = f"""Analyze this cybersecurity threat and extract actionable intelligence:

THREAT: {title}
CONTENT: {content[:1200]}
SOURCE: {source_config['name']} (reliability: {source_config['reliability']})

Provide analysis in JSON format:
{{
    "threat_assessment": {{
        "severity": "CRITICAL/HIGH/MEDIUM/LOW",
        "risk_score": 1-10,
        "urgency": "IMMEDIATE/URGENT/ROUTINE",
        "confidence": 0.1-1.0
    }},
    "classification": {{
        "family": "Specific threat type",
        "sophistication": "LOW/MEDIUM/HIGH",
        "attack_vectors": ["specific_attack_methods"],
        "indicators": ["technical_indicators"]
    }},
    "impact": {{
        "affected_sectors": ["sectors_mentioned"],
        "geographic_impact": ["countries_or_regions"],
        "business_impact": "Impact assessment"
    }},
    "intelligence": {{
        "key_insights": ["3-4 actionable insights"],
        "immediate_actions": ["specific_defensive_steps"],
        "cve_references": ["CVE_numbers_if_mentioned"],
        "technical_analysis": "Detailed technical analysis"
    }}
}}

Focus on actionable intelligence only."""

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a cybersecurity analyst. Provide factual threat intelligence."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }

            self.ai_calls_made += 1
            logger.info(f"🤖 AI Analysis ({self.ai_calls_made}/{self.max_ai_calls_per_run}): {title[:50]}...")

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
                        analysis_result = json.loads(json_content)
                        logger.info("✅ AI analysis completed successfully")
                        return self.format_ai_analysis(analysis_result)
                else:
                    logger.warning(f"⚠️ AI API error: {response.status}")
                    
        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {str(e)[:100]}... - using basic analysis")
            
        return self.basic_analysis(title, content, source_config['name'])

    def format_ai_analysis(self, ai_result: Dict) -> Dict[str, Any]:
        """Format AI analysis into standardized structure"""
        try:
            threat_assessment = ai_result.get('threat_assessment', {})
            classification = ai_result.get('classification', {})
            impact = ai_result.get('impact', {})
            intelligence = ai_result.get('intelligence', {})
            
            return {
                'technical_analysis': intelligence.get('technical_analysis', 'Threat analysis in progress'),
                'key_insights': intelligence.get('key_insights', []),
                'threat_family': classification.get('family', 'Security Incident'),
                'attack_sophistication': classification.get('sophistication', 'MEDIUM'),
                'attack_vectors': classification.get('attack_vectors', []),
                'technical_indicators': classification.get('indicators', []),
                'affected_sectors': impact.get('affected_sectors', []),
                'geographic_scope': impact.get('geographic_impact', []),
                'cve_references': intelligence.get('cve_references', []),
                'threat_actors': [],
                'risk_score': threat_assessment.get('risk_score', 5),
                'severity': threat_assessment.get('severity', 'MEDIUM'),
                'urgency': threat_assessment.get('urgency', 'ROUTINE'),
                'confidence': threat_assessment.get('confidence', 0.7),
                'immediate_actions': intelligence.get('immediate_actions', []),
                'business_impact': impact.get('business_impact', 'Standard monitoring required'),
                'analysis_type': 'AI Analysis'
            }
        except Exception as e:
            logger.warning(f"⚠️ Error formatting AI analysis: {e}")
            return self.basic_analysis("", "", "AI")

    def basic_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Basic threat analysis without AI"""
        full_text = (title + ' ' + content).lower()
        
        # Threat family detection
        threat_families = {
            'Ransomware': ['ransomware', 'encryption', 'ransom', 'lockbit', 'conti', 'revil'],
            'Zero-Day Exploit': ['zero-day', 'zero day', '0-day', 'unknown vulnerability'],
            'Data Breach': ['data breach', 'breach', 'stolen data', 'exposed records'],
            'Vulnerability Disclosure': ['patch tuesday', 'security update', 'patches', 'vulnerability'],
            'APT Campaign': ['apt', 'nation-state', 'state-sponsored', 'advanced persistent'],
            'Supply Chain Attack': ['supply chain', 'software supply', 'third-party'],
            'Malware Campaign': ['malware', 'trojan', 'backdoor', 'spyware', 'botnet'],
            'Phishing Campaign': ['phishing', 'spear phishing', 'business email'],
            'Critical Infrastructure': ['infrastructure', 'utility', 'scada', 'industrial']
        }
        
        detected_family = 'Security Incident'
        confidence = 0.6
        key_indicators = []
        
        for family, keywords in threat_families.items():
            matches = [keyword for keyword in keywords if keyword in full_text]
            if matches:
                detected_family = family
                confidence = min(0.9, 0.6 + len(matches) * 0.1)
                key_indicators = matches[:3]
                break
        
        # Risk scoring
        risk_score = 3
        if any(word in full_text for word in ['zero-day', 'critical vulnerability', 'rce']):
            risk_score = 8
        elif any(word in full_text for word in ['ransomware', 'apt', 'supply chain']):
            risk_score = 6
        elif any(word in full_text for word in ['malware', 'phishing', 'vulnerability']):
            risk_score = 4
        
        # Determine threat level
        if risk_score >= 8:
            threat_level, urgency = 'CRITICAL', 'IMMEDIATE'
        elif risk_score >= 6:
            threat_level, urgency = 'HIGH', 'URGENT'
        elif risk_score >= 4:
            threat_level, urgency = 'MEDIUM', 'ROUTINE'
        else:
            threat_level, urgency = 'LOW', 'ROUTINE'
        
        # Generate key insights
        key_insights = []
        if 'zero-day' in full_text:
            key_insights.append("Zero-day vulnerability requires immediate attention")
        if 'ransomware' in full_text:
            key_insights.append("Ransomware threat demands backup verification")
        if 'critical' in full_text and 'patch' in full_text:
            key_insights.append("Critical patches available - prioritize deployment")
        
        # Generate actionable items
        actionable_items = []
        if any(term in full_text for term in ['patch', 'update', 'fix']):
            actionable_items.append('Review and apply relevant security updates')
        if 'ransomware' in full_text:
            actionable_items.append('Verify backup systems integrity')
        if not actionable_items:
            actionable_items.append('Monitor threat developments')
        
        return {
            'technical_analysis': f"Security incident involving {detected_family.lower()}",
            'key_insights': key_insights,
            'threat_family': detected_family,
            'attack_sophistication': 'HIGH' if risk_score >= 7 else 'MEDIUM' if risk_score >= 5 else 'LOW',
            'attack_vectors': self.extract_attack_vectors(full_text),
            'technical_indicators': key_indicators,
            'affected_sectors': self.extract_sectors(full_text),
            'geographic_scope': self.extract_geography(full_text),
            'cve_references': re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()),
            'threat_actors': [],
            'risk_score': risk_score,
            'severity': threat_level,
            'urgency': urgency,
            'confidence': confidence,
            'immediate_actions': actionable_items,
            'business_impact': self.assess_business_impact(full_text, risk_score),
            'analysis_type': 'Basic Analysis'
        }

    def assess_business_impact(self, content: str, risk_score: int) -> str:
        """Assess business impact based on content and risk"""
        if risk_score >= 8:
            return "High business impact - immediate executive attention required"
        elif risk_score >= 6:
            return "Moderate business impact - security team response needed"
        elif risk_score >= 4:
            return "Limited business impact - standard monitoring applies"
        else:
            return "Minimal business impact - routine security review sufficient"

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
            'financial': ['bank', 'finance', 'payment'],
            'government': ['government', 'federal', 'agency'],
            'technology': ['tech', 'software', 'cloud'],
            'education': ['university', 'school', 'education']
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

    def create_summary(self, full_content: str) -> Tuple[str, str, List[str]]:
        """Create summary with key points"""
        clean_content = re.sub(r'<[^>]+>', '', full_content)
        clean_content = re.sub(r'&[^;]+;', ' ', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Simple preview (first 280 chars)
        preview = clean_content[:280]
        if len(clean_content) > 280:
            last_period = preview.rfind('.')
            if last_period > 200:
                preview = preview[:last_period + 1]
        
        # Full summary (up to 1200 chars)
        full_summary = clean_content[:1200]
        if len(clean_content) > 1200:
            last_period = full_summary.rfind('.')
            if last_period > 1000:
                full_summary = full_summary[:last_period + 1]
        
        # Key points (simple extraction)
        key_points = []
        sentences = clean_content.split('. ')
        for sentence in sentences[:3]:
            if len(sentence) > 30 and any(word in sentence.lower() for word in ['security', 'threat', 'attack', 'vulnerability']):
                key_points.append(sentence.strip())
        
        return preview or full_summary[:280], full_summary, key_points

    async def collect_intelligence(self) -> List[Dict]:
        """Collect threat intelligence from feeds"""
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
                            
                            # Filter for cybersecurity relevance
                            full_content = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'zero-day'
                            ]
                            
                            relevance_score = sum(1 for indicator in cyber_indicators if indicator in full_content)
                            if relevance_score < 1 or len(summary) < 100:
                                continue
                            
                            preview, full_summary, key_points = self.create_summary(summary)
                            
                            intel_item = {
                                'title': title,
                                'summary': preview,
                                'full_summary': full_summary,
                                'key_points': key_points,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'source_config': source,
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
        return self.deduplication(collected_intel)

    def deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Deduplicate intelligence reports"""
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
        """Process intelligence with AI analysis"""
        threat_reports = []
        
        for intel_item in raw_intel:
            try:
                # Get AI or basic analysis
                analysis = await self.ai_analysis(
                    intel_item['title'], 
                    intel_item['full_summary'],
                    intel_item['source_config']
                )
                
                # Geographic processing
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
                    threat_level=analysis['severity'],
                    ai_technical_analysis=analysis['technical_analysis'],
                    confidence_score=analysis['confidence'],
                    severity_rating=analysis['risk_score'],
                    attack_vectors=analysis.get('attack_vectors', []),
                    affected_sectors=analysis.get('affected_sectors', []),
                    geographic_scope=geographic_scope,
                    country_code=country_code,
                    threat_actors=analysis.get('threat_actors', []),
                    mitigation_priority=analysis['urgency'],
                    cve_references=analysis.get('cve_references', []),
                    threat_family=analysis['threat_family'],
                    attack_sophistication=analysis['attack_sophistication'],
                    risk_score=analysis['risk_score'],
                    correlation_id=hashlib.md5(intel_item['title'].encode()).hexdigest()[:8],
                    summary_preview=intel_item['summary'],
                    full_summary=intel_item['full_summary'],
                    key_insights=analysis.get('key_insights', intel_item.get('key_points', [])),
                    actionable_items=analysis.get('immediate_actions', []),
                    technical_indicators=analysis.get('technical_indicators', []),
                    business_impact=analysis['business_impact'],
                    timeline_urgency=analysis['urgency'],
                    smart_analysis={
                        'analysis_type': analysis.get('analysis_type', 'Basic'),
                        'confidence': analysis['confidence']
                    }
                )
                
                threat_reports.append(threat_report)
                logger.info(f"✅ Processed: {threat_report.title[:50]}... ({threat_report.threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_metrics(self, reports: List[ThreatReport]) -> IntelligenceMetrics:
        """Calculate accurate metrics"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], ai_analysis_quality=0, threat_velocity="stable",
                fresh_intel_24h=0, source_credibility=0.0, emerging_trends=[], threat_evolution="stable"
            )

        # Threat level counting
        threat_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            level = report.threat_level.upper()
            if level in threat_counts:
                threat_counts[level] += 1

        # Global threat level
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

        # Zero-day count
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower() or 
                            any('zero' in insight.lower() for insight in r.key_insights))

        # Recent threats
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_24h = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('
