#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Enhanced AI-Powered Cyber Intelligence Engine v4.1
Production-Ready Threat Intelligence with Optimized AI Usage

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
class EnhancedThreatReport:
    """Enhanced Cyber Threat Intelligence Report with Smart Data Management"""
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
    attack_timeline: str
    risk_score: int
    correlation_id: str
    # Enhanced fields for better data presentation
    summary_preview: str  # First 200 chars for cards
    full_summary: str     # Complete summary for details
    ai_insights: Dict[str, Any]  # Structured AI insights
    actionable_items: List[str]  # Key action items
    technical_details: Dict[str, Any]  # Technical indicators
    business_impact: str  # Business impact assessment

@dataclass
class IntelligenceMetrics:
    """Enhanced Intelligence Metrics with Smart Analytics"""
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
    sector_risk_matrix: Dict[str, int]
    ai_analysis_quality: int  # Quality score of AI analysis
    threat_velocity: str      # Rate of new threats
    impact_forecast: str      # Predicted impact trend

class SmartPatriotsIntelligence:
    """Enhanced AI-Powered Cyber Threat Intelligence System with Cost Optimization"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4o-mini"  # More cost-effective model
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cost optimization settings
        self.max_ai_calls_per_run = 15  # Limit API usage
        self.ai_calls_made = 0
        self.cache_duration = timedelta(hours=6)  # 6-hour cache
        
        # Enhanced intelligence sources with priorities
        self.intelligence_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'reliability': 0.98,
                'priority': 1,  # Highest priority
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
                'priority': 2,
                'ai_analysis': True
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.88,
                'priority': 2,
                'ai_analysis': False  # Use basic analysis to save costs
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.86,
                'priority': 3,
                'ai_analysis': False
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.84,
                'priority': 3,
                'ai_analysis': False
            }
        ]
        
        # Enhanced geographic mapping
        self.geographic_mapping = {
            'united states': 'US', 'usa': 'US', 'america': 'US', 'u.s.': 'US',
            'canada': 'CA', 'mexico': 'MX', 'china': 'CN', 'japan': 'JP',
            'south korea': 'KR', 'korea': 'KR', 'australia': 'AU', 'new zealand': 'NZ',
            'singapore': 'SG', 'india': 'IN', 'thailand': 'TH', 'vietnam': 'VN',
            'philippines': 'PH', 'malaysia': 'MY', 'indonesia': 'ID', 'taiwan': 'TW',
            'hong kong': 'HK', 'germany': 'DE', 'france': 'FR', 'united kingdom': 'GB',
            'uk': 'GB', 'italy': 'IT', 'spain': 'ES', 'netherlands': 'NL',
            'belgium': 'BE', 'sweden': 'SE', 'norway': 'NO', 'denmark': 'DK',
            'finland': 'FI', 'poland': 'PL', 'ukraine': 'UA', 'switzerland': 'CH',
            'austria': 'AT', 'israel': 'IL', 'saudi arabia': 'SA', 'uae': 'AE',
            'turkey': 'TR', 'iran': 'IR', 'egypt': 'EG', 'south africa': 'ZA',
            'nigeria': 'NG', 'russia': 'RU', 'north korea': 'KP', 'brazil': 'BR',
            'argentina': 'AR', 'europe': 'EU', 'european union': 'EU',
            'asia pacific': 'APAC', 'middle east': 'ME', 'africa': 'AF'
        }
        
        # Enhanced sector mapping
        self.sector_mapping = {
            'healthcare': ['hospital', 'medical', 'patient', 'clinic', 'health', 'pharmaceutical', 'medicare'],
            'financial': ['bank', 'finance', 'payment', 'credit', 'financial', 'fintech', 'cryptocurrency', 'bitcoin'],
            'government': ['government', 'federal', 'agency', 'military', 'defense', 'public sector', 'pentagon'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power', 'water', 'transportation', 'grid'],
            'education': ['university', 'school', 'education', 'academic', 'student', 'campus', 'college'],
            'manufacturing': ['manufacturing', 'industrial', 'factory', 'production', 'automotive', 'supply chain'],
            'technology': ['tech', 'software', 'cloud', 'saas', 'platform', 'microsoft', 'google', 'apple', 'amazon'],
            'telecommunications': ['telecom', 'communications', 'mobile', 'network', 'isp', 'cellular', 'wireless'],
            'retail': ['retail', 'shopping', 'commerce', 'store', 'consumer', 'ecommerce'],
            'media': ['media', 'news', 'journalism', 'broadcasting', 'entertainment', 'social media']
        }
        
        self.data_directory = Path('./data')
        self.data_directory.mkdir(exist_ok=True)
        
        logger.info("🎖️ Smart Patriots Protocol Intelligence Engine v4.1 - Operational")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-Enhanced/4.1 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def should_use_ai_analysis(self, source_config: Dict, content: str) -> bool:
        """Smart decision on whether to use AI analysis based on cost/value"""
        if not source_config.get('ai_analysis', False):
            return False
        
        if self.ai_calls_made >= self.max_ai_calls_per_run:
            logger.info(f"⚠️ AI call limit reached ({self.max_ai_calls_per_run}), using basic analysis")
            return False
        
        # Use AI for high-priority sources and critical content
        content_lower = content.lower()
        critical_indicators = [
            'zero-day', 'critical', 'remote code execution', 'rce', 'apt', 'ransomware',
            'supply chain', 'breach', 'compromise', 'exploit', 'vulnerability'
        ]
        
        has_critical_content = any(indicator in content_lower for indicator in critical_indicators)
        is_priority_source = source_config.get('priority', 3) <= 2
        
        return has_critical_content or is_priority_source

    async def smart_ai_analysis(self, title: str, content: str, source_config: Dict) -> Dict[str, Any]:
        """Enhanced AI analysis with cost optimization and better prompts"""
        if not self.api_token or not self.should_use_ai_analysis(source_config, title + ' ' + content):
            return self.enhanced_basic_analysis(title, content, source_config['name'])

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Enhanced, more efficient prompt
            analysis_prompt = f"""Analyze this cybersecurity threat and provide ACTIONABLE intelligence in JSON format:

THREAT: {title}
CONTENT: {content[:1800]}
SOURCE: {source_config['name']} (reliability: {source_config['reliability']})

Provide analysis as JSON:
{{
    "threat_assessment": {{
        "severity": "CRITICAL/HIGH/MEDIUM/LOW",
        "risk_score": 1-10,
        "urgency": "IMMEDIATE/URGENT/ROUTINE",
        "confidence": 0.1-1.0
    }},
    "technical_analysis": "2-3 sentence specific analysis of attack methods, vulnerabilities, or defensive measures",
    "threat_details": {{
        "family": "Specific threat type (e.g., 'Ransomware', 'APT Campaign', 'Zero-Day Exploit')",
        "sophistication": "LOW/MEDIUM/HIGH/ADVANCED",
        "attack_vectors": ["specific_methods_mentioned"],
        "affected_sectors": ["only_if_specifically_mentioned"],
        "geographic_scope": ["only_countries_specifically_mentioned"]
    }},
    "actionable_intelligence": {{
        "immediate_actions": ["specific_defensive_steps"],
        "indicators": ["any_IOCs_or_technical_indicators"],
        "business_impact": "Brief impact assessment for organizations"
    }},
    "references": {{
        "cves": ["only_actual_CVE_numbers"],
        "threat_actors": ["only_named_groups_mentioned"]
    }}
}}

CRITICAL: Be specific, avoid generic statements. If details aren't in content, use "Not specified"."""

            self.ai_calls_made += 1
            logger.info(f"🤖 AI Analysis ({self.ai_calls_made}/{self.max_ai_calls_per_run}): {title[:50]}...")

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior cybersecurity analyst. Provide specific, actionable threat intelligence. Avoid generic responses."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = ai_response[json_start:json_end]
                analysis_result = json.loads(json_content)
                logger.info("✅ AI analysis completed successfully")
                return self.format_ai_analysis(analysis_result)
                
        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {str(e)[:100]}... - using enhanced basic analysis")
            
        return self.enhanced_basic_analysis(title, content, source_config['name'])

    def format_ai_analysis(self, ai_result: Dict) -> Dict[str, Any]:
        """Format AI analysis result into standardized structure"""
        threat_assessment = ai_result.get('threat_assessment', {})
        threat_details = ai_result.get('threat_details', {})
        actionable = ai_result.get('actionable_intelligence', {})
        references = ai_result.get('references', {})
        
        return {
            'technical_analysis': ai_result.get('technical_analysis', 'AI analysis in progress'),
            'threat_family': threat_details.get('family', 'Unknown Threat'),
            'attack_sophistication': threat_details.get('sophistication', 'MEDIUM'),
            'attack_vectors': threat_details.get('attack_vectors', []),
            'affected_sectors': threat_details.get('affected_sectors', []),
            'geographic_scope': threat_details.get('geographic_scope', []),
            'cve_references': references.get('cves', []),
            'threat_actors': references.get('threat_actors', []),
            'risk_score': threat_assessment.get('risk_score', 5),
            'severity': threat_assessment.get('severity', 'MEDIUM'),
            'urgency': threat_assessment.get('urgency', 'ROUTINE'),
            'confidence': threat_assessment.get('confidence', 0.7),
            'actionable_items': actionable.get('immediate_actions', []),
            'business_impact': actionable.get('business_impact', 'Impact assessment pending'),
            'technical_indicators': actionable.get('indicators', [])
        }

    def enhanced_basic_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Enhanced basic analysis with better threat detection"""
        full_text = (title + ' ' + content).lower()
        
        # Enhanced threat family detection with scoring
        threat_families = {
            'zero-day exploit': ['zero-day', 'zero day', '0-day', 'unknown vulnerability'],
            'ransomware': ['ransomware', 'encryption', 'ransom', 'lockbit', 'conti', 'revil'],
            'nation-state apt': ['apt', 'nation-state', 'state-sponsored', 'lazarus', 'fancy bear'],
            'banking trojan': ['banking trojan', 'danabot', 'emotet', 'trickbot', 'qakbot'],
            'supply chain attack': ['supply chain', 'software supply', 'third-party'],
            'phishing campaign': ['phishing', 'spear phishing', 'business email compromise'],
            'vulnerability disclosure': ['patch tuesday', 'security update', 'vulnerability'],
            'ddos attack': ['ddos', 'denial of service', 'botnet'],
            'data breach': ['data breach', 'data leak', 'personal information'],
            'malware': ['malware', 'trojan', 'virus', 'backdoor', 'spyware']
        }
        
        detected_family = 'Unknown Threat'
        max_score = 0
        
        for family, keywords in threat_families.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > max_score:
                max_score = score
                detected_family = family.title()
        
        # Enhanced risk scoring
        risk_factors = {
            'zero-day': 9, 'critical': 8, 'remote code execution': 8, 'rce': 8,
            'ransomware': 7, 'apt': 7, 'supply chain': 7, 'breach': 6,
            'vulnerability': 5, 'phishing': 4, 'ddos': 4, 'malware': 5
        }
        
        risk_score = 3  # Base score
        for factor, score in risk_factors.items():
            if factor in full_text:
                risk_score = max(risk_score, score)
        
        # Determine threat level
        if risk_score >= 8:
            threat_level = 'CRITICAL'
            urgency = 'IMMEDIATE'
        elif risk_score >= 6:
            threat_level = 'HIGH'
            urgency = 'URGENT'
        elif risk_score >= 4:
            threat_level = 'MEDIUM'
            urgency = 'ROUTINE'
        else:
            threat_level = 'LOW'
            urgency = 'ROUTINE'
        
        # Extract actionable items
        actionable_items = []
        if 'patch' in full_text:
            actionable_items.append('Apply security patches immediately')
        if 'update' in full_text:
            actionable_items.append('Update affected systems')
        if 'monitor' in full_text:
            actionable_items.append('Enhance monitoring for indicators')
        
        return {
            'technical_analysis': f"{detected_family} detected with {threat_level.lower()} risk impact requiring {urgency.lower()} attention",
            'threat_family': detected_family,
            'attack_sophistication': 'HIGH' if risk_score >= 7 else 'MEDIUM',
            'attack_vectors': ['email_phishing'] if 'phishing' in full_text else ['network_intrusion'],
            'affected_sectors': self.extract_sectors(full_text),
            'geographic_scope': self.extract_geography(full_text),
            'cve_references': re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()),
            'threat_actors': [],
            'risk_score': risk_score,
            'severity': threat_level,
            'urgency': urgency,
            'confidence': 0.8,
            'actionable_items': actionable_items or ['Monitor for additional indicators'],
            'business_impact': f"{threat_level.title()} impact threat requiring organizational attention",
            'technical_indicators': []
        }

    def extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors from content"""
        sectors = []
        for sector, keywords in self.sector_mapping.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        return sectors[:3]  # Limit to top 3

    def extract_geography(self, content: str) -> List[str]:
        """Extract geographic indicators from content"""
        geography = []
        for region, code in self.geographic_mapping.items():
            if region in content:
                geography.append(region.title())
        return geography[:3]  # Limit to top 3

    def create_smart_summary(self, full_summary: str) -> Tuple[str, str]:
        """Create smart summary preview and full summary"""
        # Clean the summary
        clean_summary = re.sub(r'<[^>]+>', '', full_summary)
        clean_summary = re.sub(r'&[^;]+;', ' ', clean_summary)
        clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
        
        # Create preview (first sentence or 200 chars)
        sentences = clean_summary.split('. ')
        if len(sentences[0]) <= 200:
            preview = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
        else:
            preview = clean_summary[:200] + ('...' if len(clean_summary) > 200 else '')
        
        return preview, clean_summary[:1500]  # Limit full summary too

    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced intelligence collection with smart filtering"""
        collected_intel = []
        
        # Sort sources by priority
        sorted_sources = sorted(self.intelligence_sources, key=lambda x: x['priority'])
        
        for source in sorted_sources:
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:15]:  # Limit per source
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Enhanced cyber relevance check
                            content_check = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt',
                                'zero-day', 'backdoor', 'trojan', 'spyware', 'botnet', 'ddos',
                                'patch', 'cve', 'incident', 'compromise', 'espionage', 'scam'
                            ]
                            
                            if not any(indicator in content_check for indicator in cyber_indicators):
                                continue
                            
                            # Quality filters
                            if len(summary) < 80 or len(title) < 10:
                                continue
                            
                            preview, full_summary = self.create_smart_summary(summary)
                            
                            intel_item = {
                                'title': title,
                                'summary': preview,
                                'full_summary': full_summary,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'source_config': source
                            }
                            
                            source_intel.append(intel_item)
                        
                        collected_intel.extend(source_intel)
                        logger.info(f"📊 {source['name']}: {len(source_intel)} reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(0.5)  # Rate limiting

        logger.info(f"🎯 Total Intelligence Collected: {len(collected_intel)} reports")
        return self.smart_deduplication(collected_intel)

    def smart_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Smart deduplication with similarity detection"""
        seen_signatures = set()
        unique_intel = []
        
        for intel in raw_intel:
            # Create content signature
            title_clean = re.sub(r'[^\w\s]', '', intel['title'].lower())
            content_signature = hashlib.sha256(title_clean.encode()).hexdigest()[:16]
            
            if content_signature not in seen_signatures:
                seen_signatures.add(content_signature)
                unique_intel.append(intel)
        
        logger.info(f"🔄 Smart Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel

    async def process_intelligence(self, raw_intel: List[Dict]) -> List[EnhancedThreatReport]:
        """Process intelligence with smart AI usage"""
        threat_reports = []
        
        for intel_item in raw_intel:
            try:
                # Get AI or basic analysis
                analysis = await self.smart_ai_analysis(
                    intel_item['title'], 
                    intel_item['full_summary'],
                    intel_item['source_config']
                )
                
                # Determine geographic scope
                geographic_scope = 'Global'
                country_code = 'GLOBAL'
                
                if analysis['geographic_scope']:
                    geographic_scope = ', '.join(analysis['geographic_scope'][:2])
                    country_code = self.geographic_mapping.get(
                        analysis['geographic_scope'][0].lower(), 'GLOBAL'
                    )
                
                # Map severity to threat level
                severity_mapping = {
                    'CRITICAL': ('CRITICAL', 9),
                    'HIGH': ('HIGH', 7),
                    'MEDIUM': ('MEDIUM', 5),
                    'LOW': ('LOW', 3)
                }
                
                threat_level, severity_rating = severity_mapping.get(
                    analysis['severity'], ('MEDIUM', 5)
                )
                
                # Create enhanced threat report
                threat_report = EnhancedThreatReport(
                    title=intel_item['title'],
                    summary=intel_item['summary'],
                    source=intel_item['source'],
                    source_url=intel_item['source_url'],
                    timestamp=intel_item['timestamp'],
                    threat_level=threat_level,
                    ai_technical_analysis=analysis['technical_analysis'],
                    confidence_score=analysis['confidence'],
                    severity_rating=severity_rating,
                    attack_vectors=analysis['attack_vectors'],
                    affected_sectors=analysis['affected_sectors'],
                    geographic_scope=geographic_scope,
                    country_code=country_code,
                    threat_actors=analysis['threat_actors'],
                    mitigation_priority=analysis['urgency'],
                    cve_references=analysis['cve_references'],
                    threat_family=analysis['threat_family'],
                    attack_sophistication=analysis['attack_sophistication'],
                    attack_timeline=analysis['urgency'].lower(),
                    risk_score=analysis['risk_score'],
                    correlation_id=hashlib.md5(intel_item['title'].encode()).hexdigest()[:8],
                    # Enhanced fields
                    summary_preview=intel_item['summary'],
                    full_summary=intel_item['full_summary'],
                    ai_insights={
                        'confidence': analysis['confidence'],
                        'urgency': analysis['urgency'],
                        'analysis_type': 'AI' if self.ai_calls_made > 0 else 'Enhanced Basic'
                    },
                    actionable_items=analysis['actionable_items'],
                    technical_details={
                        'indicators': analysis['technical_indicators'],
                        'sophistication': analysis['attack_sophistication']
                    },
                    business_impact=analysis['business_impact']
                )
                
                threat_reports.append(threat_report)
                logger.info(f"✅ Processed: {threat_report.title[:50]}... (Level: {threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return threat_reports

    def calculate_metrics(self, reports: List[EnhancedThreatReport]) -> IntelligenceMetrics:
        """Calculate enhanced metrics with smart analytics"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], sector_risk_matrix={}, ai_analysis_quality=0,
                threat_velocity="stable", impact_forecast="low"
            )

        # Count by threat level
        level_counts = {
            'CRITICAL': sum(1 for r in reports if r.threat_level == 'CRITICAL'),
            'HIGH': sum(1 for r in reports if r.threat_level == 'HIGH'),
            'MEDIUM': sum(1 for r in reports if r.threat_level == 'MEDIUM'),
            'LOW': sum(1 for r in reports if r.threat_level == 'LOW')
        }
        
        # Global threat level assessment
        if level_counts['CRITICAL'] >= 2:
            global_threat_level = "CRITICAL"
        elif level_counts['CRITICAL'] >= 1 or level_counts['HIGH'] >= 4:
            global_threat_level = "HIGH"
        elif level_counts['HIGH'] >= 2:
            global_threat_level = "ELEVATED"
        else:
            global_threat_level = "MEDIUM"

        # Geographic distribution
        geo_distribution = {}
        for report in reports:
            country = report.country_code
            geo_distribution[country] = geo_distribution.get(country, 0) + 1

        # Threat families analysis
        family_counts = {}
        for report in reports:
            family = report.threat_family
            family_counts[family] = family_counts.get(family, 0) + 1
        
        top_families = [
            {"name": family, "count": count, "risk_level": "HIGH" if count >= 3 else "MEDIUM"}
            for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        ]

        # Zero-day count
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower() or 
                            'zero-day' in r.ai_technical_analysis.lower())

        # Recent threats (24h)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff:
                    recent_count += 1
            except:
                pass

        # Trending threats
        trending = sorted(
            [r for r in reports if r.risk_score >= 6],
            key=lambda x: x.risk_score,
            reverse=True
        )[:5]
        
        trending_threats = [
            {
                'title': r.title,
                'risk_score': r.risk_score,
                'threat_family': r.threat_family,
                'threat_level': r.threat_level
            }
            for r in trending
        ]

        # Sector risk matrix
        sector_risks = {}
        for report in reports:
            for sector in report.affected_sectors:
                if sector not in sector_risks:
                    sector_risks[sector] = []
                sector_risks[sector].append(report.risk_score)
        
        sector_risk_matrix = {
            sector: int(sum(scores) / len(scores)) if scores else 0 
            for sector, scores in sector_risks.items()
        }

        # AI analysis quality
        ai_analyzed = sum(1 for r in reports if 'AI' in r.ai_insights.get('analysis_type', ''))
        ai_quality = int((ai_analyzed / len(reports)) * 100) if reports else 0

        return IntelligenceMetrics(
            total_threats=len(reports),
            critical_threats=level_counts['CRITICAL'],
            high_threats=level_counts['HIGH'],
            medium_threats=level_counts['MEDIUM'],
            low_threats=level_counts['LOW'],
            global_threat_level=global_threat_level,
            intelligence_confidence=int(sum(r.confidence_score for r in reports) / len(reports) * 100),
            recent_threats_24h=recent_count,
            top_threat_families=top_families,
            geographic_distribution=geo_distribution,
            zero_day_count=zero_day_count,
            trending_threats=trending_threats,
            sector_risk_matrix=sector_risk_matrix,
            ai_analysis_quality=ai_quality,
            threat_velocity="accelerating" if recent_count > len(reports) * 0.3 else "stable",
            impact_forecast="elevated" if level_counts['CRITICAL'] > 0 else "moderate"
        )

    def save_intelligence_data(self, reports: List[EnhancedThreatReport], metrics: IntelligenceMetrics) -> None:
        """Save enhanced intelligence data"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.1",
            "ai_usage": {
                "api_calls_made": self.ai_calls_made,
                "api_calls_limit": self.max_ai_calls_per_run,
                "efficiency_score": min(100, int((metrics.ai_analysis_quality * 2 + 
                                               (100 - (self.ai_calls_made / self.max_ai_calls_per_run * 100))) / 3))
            },
            "intelligence_summary": {
                "mission_status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "intelligence_confidence": metrics.intelligence_confidence,
                "threat_landscape": metrics.global_threat_level,
                "cost_optimization": "ENABLED",
                "next_update": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Intelligence saved: {len(reports)} reports, {metrics.global_threat_level} threat level")
        logger.info(f"🤖 AI Usage: {self.ai_calls_made}/{self.max_ai_calls_per_run} calls, Quality: {metrics.ai_analysis_quality}%")

async def execute_enhanced_intelligence_mission():
    """Execute enhanced cyber threat intelligence mission"""
    logger.info("🎖️ PATRIOTS PROTOCOL - Enhanced Intelligence Mission Initiated")
    
    try:
        async with SmartPatriotsIntelligence() as intelligence_engine:
            # Collect intelligence
            raw_intelligence = await intelligence_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No intelligence collected")
                return
            
            # Process with smart AI usage
            threat_reports = await intelligence_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed")
                return
            
            # Calculate metrics
            metrics = intelligence_engine.calculate_metrics(threat_reports)
            
            # Save data
            intelligence_engine.save_intelligence_data(threat_reports, metrics)
            
            # Mission summary
            logger.info("✅ Enhanced Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"💥 Zero-Day Exploits: {metrics.zero_day_count}")
            logger.info(f"🤖 AI Quality Score: {metrics.ai_analysis_quality}%")
            logger.info(f"🎖️ Patriots Protocol Enhanced Intelligence: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Enhanced intelligence mission failed: {str(e)}")
        
        # Create minimal error state
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                "medium_threats": 0, "low_threats": 0, "global_threat_level": "OFFLINE",
                "intelligence_confidence": 0, "recent_threats_24h": 0,
                "top_threat_families": [], "geographic_distribution": {},
                "zero_day_count": 0, "trending_threats": [], "sector_risk_matrix": {},
                "ai_analysis_quality": 0, "threat_velocity": "unknown", 
                "impact_forecast": "unknown"
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.1",
            "intelligence_summary": {
                "mission_status": "ERROR",
                "threats_analyzed": 0,
                "intelligence_confidence": 0,
                "threat_landscape": "OFFLINE",
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(execute_enhanced_intelligence_mission())
