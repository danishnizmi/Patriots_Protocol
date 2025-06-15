#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Enhanced AI-Powered Cyber Intelligence Engine v4.2
Production-Ready Threat Intelligence with Smart AI Value Optimization

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
    """Enhanced Cyber Threat Intelligence Report with Smart Value-Driven Data"""
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
    # Enhanced value-driven fields
    summary_preview: str
    full_summary: str
    key_insights: List[str]  # AI-extracted key points
    actionable_items: List[str]
    technical_indicators: List[str]
    business_impact: str
    timeline_urgency: str
    smart_analysis: Dict[str, Any]  # Structured AI insights

@dataclass
class IntelligenceMetrics:
    """Enhanced Intelligence Metrics with Accurate Calculations"""
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
    """Enhanced AI-Powered Cyber Threat Intelligence with Maximum Value Extraction"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4o-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Smart AI optimization for maximum value
        self.max_ai_calls_per_run = 12  # Optimized limit
        self.ai_calls_made = 0
        self.high_value_threshold = 6  # Risk score threshold for AI analysis
        
        # Enhanced intelligence sources prioritized for value
        self.intelligence_sources = [
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.92,
                'priority': 1,
                'ai_analysis': True,
                'value_keywords': ['zero-day', 'ransomware', 'critical', 'exploit', 'breach']
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'priority': 1,
                'ai_analysis': True,
                'value_keywords': ['investigation', 'analysis', 'deep dive', 'exclusive']
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'priority': 1,
                'ai_analysis': True,
                'value_keywords': ['technical', 'analysis', 'malware', 'honeypot']
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.88,
                'priority': 2,
                'ai_analysis': False,
                'value_keywords': ['vulnerability', 'patch', 'security']
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.85,
                'priority': 2,
                'ai_analysis': False,
                'value_keywords': ['government', 'policy', 'regulation']
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
        
        logger.info("🎖️ Smart Patriots Protocol Intelligence Engine v4.2 - Enhanced Value Mode")

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

    def should_use_premium_ai_analysis(self, content: str, source_config: Dict, basic_risk: int) -> bool:
        """Smart decision for premium AI analysis based on value potential"""
        if not self.api_token or self.ai_calls_made >= self.max_ai_calls_per_run:
            return False
        
        if not source_config.get('ai_analysis', False):
            return False
        
        content_lower = content.lower()
        
        # High-value indicators for AI analysis
        premium_indicators = [
            'zero-day', '0-day', 'critical vulnerability', 'remote code execution',
            'supply chain', 'ransomware', 'apt', 'nation-state', 'breach',
            'millions affected', 'critical infrastructure', 'exclusive', 'analysis'
        ]
        
        value_keywords = source_config.get('value_keywords', [])
        
        # Prioritize AI for high-value content
        has_premium_content = any(indicator in content_lower for indicator in premium_indicators)
        has_value_keywords = any(keyword in content_lower for keyword in value_keywords)
        is_high_risk = basic_risk >= self.high_value_threshold
        is_priority_source = source_config.get('priority', 3) <= 1
        
        return (has_premium_content or (has_value_keywords and is_high_risk) or 
                (is_priority_source and is_high_risk))

    async def premium_ai_analysis(self, title: str, content: str, source_config: Dict) -> Dict[str, Any]:
        """Premium AI analysis focused on extracting maximum actionable value"""
        if not self.should_use_premium_ai_analysis(title + ' ' + content, source_config, 5):
            return self.enhanced_basic_analysis(title, content, source_config['name'])

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Value-focused analysis prompt
            analysis_prompt = f"""As a senior cybersecurity analyst, analyze this threat and extract MAXIMUM ACTIONABLE VALUE:

THREAT: {title}
CONTENT: {content[:1500]}
SOURCE: {source_config['name']} (reliability: {source_config['reliability']})

Provide comprehensive analysis in JSON format:
{{
    "executive_summary": {{
        "key_insights": ["3-4 specific, actionable insights that matter to security teams"],
        "critical_finding": "Most important takeaway in one sentence",
        "risk_assessment": {{
            "severity": "CRITICAL/HIGH/MEDIUM/LOW",
            "risk_score": 1-10,
            "urgency": "IMMEDIATE/URGENT/ROUTINE",
            "confidence": 0.1-1.0
        }}
    }},
    "technical_intelligence": {{
        "attack_analysis": "Detailed 2-3 sentence technical analysis of the specific attack method, vulnerability, or threat mechanism",
        "threat_classification": {{
            "family": "Specific threat type (e.g., 'Ransomware-as-a-Service', 'Supply Chain Attack', 'Zero-Day Exploit')",
            "sophistication": "LOW/MEDIUM/HIGH/ADVANCED",
            "attack_vectors": ["specific_attack_methods"],
            "indicators": ["any_technical_indicators_or_IOCs"]
        }},
        "impact_scope": {{
            "affected_sectors": ["only_sectors_specifically_mentioned"],
            "geographic_impact": ["countries_or_regions_mentioned"],
            "scale": "Scope and scale of impact"
        }}
    }},
    "actionable_response": {{
        "immediate_actions": ["3-4 specific defensive steps organizations should take"],
        "detection_guidance": ["specific_detection_methods_if_applicable"],
        "mitigation_steps": ["concrete_mitigation_strategies"],
        "business_impact": "Clear assessment of potential business impact and why this matters"
    }},
    "threat_context": {{
        "cve_references": ["only_actual_CVE_numbers_mentioned"],
        "threat_actors": ["only_named_groups_or_actors_mentioned"],
        "campaign_details": "Any campaign names, tools, or attribution mentioned"
    }}
}}

Focus on ACTIONABLE intelligence. Avoid generic advice. Be specific about what organizations should DO."""

            self.ai_calls_made += 1
            logger.info(f"🤖 Premium AI Analysis ({self.ai_calls_made}/{self.max_ai_calls_per_run}): {title[:50]}...")

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior cybersecurity analyst specializing in actionable threat intelligence. Provide specific, valuable insights that help security teams make decisions."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=1200
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = ai_response[json_start:json_end]
                analysis_result = json.loads(json_content)
                logger.info("✅ Premium AI analysis completed successfully")
                return self.format_premium_analysis(analysis_result)
                
        except Exception as e:
            logger.warning(f"⚠️ Premium AI analysis failed: {str(e)[:100]}... - using enhanced basic analysis")
            
        return self.enhanced_basic_analysis(title, content, source_config['name'])

    def format_premium_analysis(self, ai_result: Dict) -> Dict[str, Any]:
        """Format premium AI analysis into standardized high-value structure"""
        executive = ai_result.get('executive_summary', {})
        technical = ai_result.get('technical_intelligence', {})
        actionable = ai_result.get('actionable_response', {})
        context = ai_result.get('threat_context', {})
        
        risk_assessment = executive.get('risk_assessment', {})
        threat_classification = technical.get('threat_classification', {})
        impact_scope = technical.get('impact_scope', {})
        
        return {
            'technical_analysis': technical.get('attack_analysis', 'Advanced threat analysis in progress'),
            'key_insights': executive.get('key_insights', []),
            'critical_finding': executive.get('critical_finding', ''),
            'threat_family': threat_classification.get('family', 'Advanced Threat'),
            'attack_sophistication': threat_classification.get('sophistication', 'MEDIUM'),
            'attack_vectors': threat_classification.get('attack_vectors', []),
            'technical_indicators': threat_classification.get('indicators', []),
            'affected_sectors': impact_scope.get('affected_sectors', []),
            'geographic_scope': impact_scope.get('geographic_impact', []),
            'impact_scale': impact_scope.get('scale', ''),
            'cve_references': context.get('cve_references', []),
            'threat_actors': context.get('threat_actors', []),
            'campaign_details': context.get('campaign_details', ''),
            'risk_score': risk_assessment.get('risk_score', 5),
            'severity': risk_assessment.get('severity', 'MEDIUM'),
            'urgency': risk_assessment.get('urgency', 'ROUTINE'),
            'confidence': risk_assessment.get('confidence', 0.7),
            'immediate_actions': actionable.get('immediate_actions', []),
            'detection_guidance': actionable.get('detection_guidance', []),
            'mitigation_steps': actionable.get('mitigation_steps', []),
            'business_impact': actionable.get('business_impact', 'Impact assessment requires further analysis'),
            'analysis_type': 'Premium AI'
        }

    def enhanced_basic_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Enhanced basic analysis with improved value extraction"""
        full_text = (title + ' ' + content).lower()
        
        # Enhanced threat family detection with better keywords
        threat_families = {
            'Ransomware': ['ransomware', 'encryption', 'ransom', 'lockbit', 'conti', 'revil', 'blackcat', 'royal'],
            'Zero-Day Exploit': ['zero-day', 'zero day', '0-day', 'unknown vulnerability', 'cve-2024', 'cve-2025'],
            'Data Breach': ['data breach', 'breach', 'stolen data', 'exposed records', 'leaked database'],
            'Vulnerability Disclosure': ['patch tuesday', 'security update', 'patches', 'vulnerability', 'fixes'],
            'APT Campaign': ['apt', 'nation-state', 'state-sponsored', 'advanced persistent', 'sophisticated attack'],
            'Supply Chain Attack': ['supply chain', 'software supply', 'third-party', 'vendor compromise'],
            'Malware Campaign': ['malware', 'trojan', 'backdoor', 'spyware', 'botnet', 'stealer'],
            'Phishing Campaign': ['phishing', 'spear phishing', 'business email', 'email attack'],
            'Critical Infrastructure': ['infrastructure', 'utility', 'scada', 'industrial', 'power grid'],
            'Corporate Security Incident': ['systems restored', 'security incident', 'cyber attack', 'operations disrupted']
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
        
        # Smart risk scoring based on content analysis
        risk_score = 3
        severity_indicators = {
            'critical': (['zero-day', 'critical vulnerability', 'remote code execution', 'rce'], 9),
            'high': (['ransomware', 'apt', 'nation-state', 'supply chain', 'widespread'], 7),
            'medium': (['malware', 'phishing', 'vulnerability', 'patch'], 5),
            'low': (['spam', 'scam', 'minor', 'resolved'], 3)
        }
        
        for level, (indicators, score) in severity_indicators.items():
            if any(indicator in full_text for indicator in indicators):
                risk_score = score
                break
        
        # Enhanced key insights extraction
        key_insights = []
        if 'zero-day' in full_text:
            key_insights.append("Zero-day vulnerability requires immediate attention and patch management")
        if 'ransomware' in full_text:
            key_insights.append("Ransomware threat demands backup verification and incident response readiness")
        if 'supply chain' in full_text:
            key_insights.append("Supply chain attack affects downstream security dependencies")
        if 'critical' in full_text and 'patch' in full_text:
            key_insights.append("Critical patches available - prioritize deployment and testing")
        
        # Smart actionable items based on content
        actionable_items = []
        if any(term in full_text for term in ['patch', 'update', 'fix']):
            actionable_items.append('Review and apply relevant security updates immediately')
        if 'ransomware' in full_text:
            actionable_items.append('Verify backup systems integrity and test recovery procedures')
        if any(term in full_text for term in ['breach', 'leak', 'exposed']):
            actionable_items.append('Conduct data inventory and verify access controls')
        if 'phishing' in full_text:
            actionable_items.append('Enhance email security filters and user awareness training')
        if not actionable_items:
            actionable_items.append('Monitor threat developments and assess security posture')
        
        # Determine threat level and urgency
        if risk_score >= 8:
            threat_level, urgency = 'CRITICAL', 'IMMEDIATE'
        elif risk_score >= 6:
            threat_level, urgency = 'HIGH', 'URGENT'
        elif risk_score >= 4:
            threat_level, urgency = 'MEDIUM', 'ROUTINE'
        else:
            threat_level, urgency = 'LOW', 'ROUTINE'
        
        return {
            'technical_analysis': f"Security incident involving {detected_family.lower()} with {threat_level.lower()} impact potential",
            'key_insights': key_insights,
            'critical_finding': f"{detected_family} detected requiring {urgency.lower()} response",
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
            'analysis_type': 'Enhanced Basic'
        }

    def assess_business_impact(self, content: str, risk_score: int) -> str:
        """Assess business impact based on content and risk"""
        if 'systems restored' in content:
            return "Operations restored - monitor for residual impact and lessons learned"
        elif risk_score >= 8:
            return "High business impact - immediate executive attention and resource allocation required"
        elif risk_score >= 6:
            return "Moderate business impact - security team escalation and response planning needed"
        elif risk_score >= 4:
            return "Limited business impact - standard monitoring and assessment protocols apply"
        else:
            return "Minimal business impact - awareness and routine security review sufficient"

    def extract_attack_vectors(self, content: str) -> List[str]:
        """Extract specific attack vectors from content"""
        vectors = []
        vector_mapping = {
            'email': ['phishing', 'email', 'attachment', 'spam'],
            'web': ['website', 'web', 'browser', 'online'],
            'network': ['network', 'remote', 'lateral', 'privilege'],
            'supply_chain': ['supply chain', 'third-party', 'vendor'],
            'social_engineering': ['social', 'human', 'employee', 'insider']
        }
        
        for vector, keywords in vector_mapping.items():
            if any(keyword in content for keyword in keywords):
                vectors.append(vector)
        
        return vectors[:3]

    def extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors with better accuracy"""
        sectors = []
        sector_keywords = {
            'healthcare': ['hospital', 'medical', 'patient', 'health'],
            'financial': ['bank', 'finance', 'payment', 'financial'],
            'government': ['government', 'federal', 'agency', 'public'],
            'technology': ['tech', 'software', 'cloud', 'microsoft', 'google'],
            'education': ['university', 'school', 'education', 'academic'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        
        return sectors[:2]

    def extract_geography(self, content: str) -> List[str]:
        """Extract geographic scope with better mapping"""
        geography = []
        for region, code in self.geographic_mapping.items():
            if region in content:
                geography.append(region.title())
                if len(geography) >= 2:
                    break
        
        return geography or ['Global']

    def create_smart_summary(self, full_content: str) -> Tuple[str, str, List[str]]:
        """Create smart summary with key points extraction"""
        # Clean content
        clean_content = re.sub(r'<[^>]+>', '', full_content)
        clean_content = re.sub(r'&[^;]+;', ' ', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Extract key sentences based on importance indicators
        sentences = clean_content.split('. ')
        important_keywords = [
            'zero-day', 'critical', 'vulnerability', 'ransomware', 'breach', 'exploit',
            'patch', 'update', 'malware', 'attack', 'threat', 'security'
        ]
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword.lower() in sentence.lower())
            if len(sentence) > 30:  # Avoid very short sentences
                scored_sentences.append((sentence, score))
        
        # Sort by importance and length
        scored_sentences.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        # Create preview (best 1-2 sentences, max 300 chars)
        preview_sentences = []
        preview_length = 0
        for sentence, score in scored_sentences:
            if preview_length + len(sentence) <= 280 and len(preview_sentences) < 2:
                preview_sentences.append(sentence)
                preview_length += len(sentence)
            else:
                break
        
        preview = '. '.join(preview_sentences)
        if preview and not preview.endswith('.'):
            preview += '.'
        
        # Full summary (up to 1500 chars for meaningful content)
        full_summary = clean_content[:1500]
        if len(clean_content) > 1500:
            # Try to cut at sentence boundary
            last_period = full_summary.rfind('.')
            if last_period > 1200:
                full_summary = full_summary[:last_period + 1]
        
        # Extract key points
        key_points = []
        for sentence, score in scored_sentences[:3]:
            if score > 0 and len(sentence) > 20:
                # Simplify sentence for key point
                simplified = sentence.split(',')[0]  # Take first clause
                if len(simplified) <= 100:
                    key_points.append(simplified.strip())
        
        return preview or full_summary[:280], full_summary, key_points

    async def collect_intelligence(self) -> List[Dict]:
        """Enhanced intelligence collection with smart filtering"""
        collected_intel = []
        
        for source in sorted(self.intelligence_sources, key=lambda x: x['priority']):
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:12]:  # Optimized limit
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Enhanced relevance filtering
                            full_content = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'zero-day',
                                'patch', 'cve', 'incident', 'compromise', 'backdoor', 'trojan'
                            ]
                            
                            relevance_score = sum(1 for indicator in cyber_indicators if indicator in full_content)
                            if relevance_score < 1:
                                continue
                            
                            # Quality filters
                            if len(summary) < 100 or len(title) < 15:
                                continue
                            
                            preview, full_summary, key_points = self.create_smart_summary(summary)
                            
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
                        
                        # Sort by relevance and take best items
                        source_intel.sort(key=lambda x: x['relevance_score'], reverse=True)
                        collected_intel.extend(source_intel[:8])  # Best 8 per source
                        
                        logger.info(f"📊 {source['name']}: {len(source_intel)} high-value reports")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            await asyncio.sleep(0.3)  # Rate limiting

        logger.info(f"🎯 Total High-Value Intelligence: {len(collected_intel)} reports")
        return self.smart_deduplication(collected_intel)

    def smart_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Enhanced deduplication with similarity scoring"""
        unique_intel = []
        seen_signatures = set()
        
        for intel in raw_intel:
            # Create smarter signature
            title_words = set(re.findall(r'\w+', intel['title'].lower()))
            content_signature = hashlib.sha256(''.join(sorted(title_words)).encode()).hexdigest()[:12]
            
            if content_signature not in seen_signatures:
                seen_signatures.add(content_signature)
                unique_intel.append(intel)
        
        # Sort by relevance score
        unique_intel.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"🔄 Smart Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel[:25]  # Limit to top 25 most relevant

    async def process_intelligence(self, raw_intel: List[Dict]) -> List[EnhancedThreatReport]:
        """Process intelligence with smart AI optimization"""
        threat_reports = []
        
        for intel_item in raw_intel:
            try:
                # Get premium AI or enhanced basic analysis
                analysis = await self.premium_ai_analysis(
                    intel_item['title'], 
                    intel_item['full_summary'],
                    intel_item['source_config']
                )
                
                # Smart geographic processing
                geographic_scope = 'Global'
                country_code = 'GLOBAL'
                
                if analysis.get('geographic_scope'):
                    geographic_scope = ', '.join(analysis['geographic_scope'][:2])
                    country_code = self.geographic_mapping.get(
                        analysis['geographic_scope'][0].lower(), 'GLOBAL'
                    )
                
                # Create enhanced threat report with all value-driven fields
                threat_report = EnhancedThreatReport(
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
                    # Enhanced value fields
                    summary_preview=intel_item['summary'],
                    full_summary=intel_item['full_summary'],
                    key_insights=analysis.get('key_insights', intel_item.get('key_points', [])),
                    actionable_items=analysis.get('immediate_actions', []),
                    technical_indicators=analysis.get('technical_indicators', []),
                    business_impact=analysis['business_impact'],
                    timeline_urgency=analysis['urgency'],
                    smart_analysis={
                        'analysis_type': analysis.get('analysis_type', 'Basic'),
                        'confidence': analysis['confidence'],
                        'critical_finding': analysis.get('critical_finding', ''),
                        'detection_guidance': analysis.get('detection_guidance', []),
                        'mitigation_steps': analysis.get('mitigation_steps', [])
                    }
                )
                
                threat_reports.append(threat_report)
                logger.info(f"✅ Processed: {threat_report.title[:50]}... ({threat_report.threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_accurate_metrics(self, reports: List[EnhancedThreatReport]) -> IntelligenceMetrics:
        """Calculate accurate metrics with proper validation"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                top_threat_families=[], geographic_distribution={}, zero_day_count=0,
                trending_threats=[], ai_analysis_quality=0, threat_velocity="stable",
                fresh_intel_24h=0, source_credibility=0.0, emerging_trends=[], threat_evolution="stable"
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

        # Accurate geographic distribution
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
            risk_level = "CRITICAL" if avg_risk >= 7 else "HIGH" if avg_risk >= 5 else "MEDIUM"
            top_families.append({
                "name": family,
                "count": data['count'],
                "risk_level": risk_level,
                "avg_risk": round(avg_risk, 1)
            })

        # Zero-day and recent threat calculations
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower() or 
                            any('zero' in insight.lower() for insight in r.key_insights))

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
            'key_insight': r.key_insights[0] if r.key_insights else ''
        } for r in trending]

        # AI analysis quality calculation
        ai_analyzed = sum(1 for r in reports if 'AI' in r.smart_analysis.get('analysis_type', ''))
        ai_quality = int((ai_analyzed / len(reports)) * 100) if reports else 0

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
            ai_analysis_quality=ai_quality,
            threat_velocity="accelerating" if recent_24h > len(reports) * 0.4 else "stable",
            fresh_intel_24h=recent_24h,
            source_credibility=round(avg_confidence, 2),
            emerging_trends=emerging_trends or ["Intelligence Network Monitoring"],
            threat_evolution="escalating" if threat_counts['CRITICAL'] > 0 else "stable"
        )

    def save_intelligence_data(self, reports: List[EnhancedThreatReport], metrics: IntelligenceMetrics) -> None:
        """Save enhanced intelligence with complete data structure"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "ai_usage": {
                "api_calls_made": self.ai_calls_made,
                "api_calls_limit": self.max_ai_calls_per_run,
                "efficiency_score": min(100, int((metrics.ai_analysis_quality * 1.5 + 
                                               (100 - (self.ai_calls_made / self.max_ai_calls_per_run * 100))) / 2.5)),
                "cost_optimization": "MAXIMUM_VALUE"
            },
            "intelligence_summary": {
                "mission_status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "intelligence_sources": len(self.intelligence_sources),
                "confidence_level": metrics.intelligence_confidence,
                "threat_landscape": metrics.global_threat_level,
                "next_update": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Enhanced Intelligence saved: {len(reports)} reports")
        logger.info(f"🎯 Global Threat Level: {metrics.global_threat_level}")
        logger.info(f"🤖 AI Efficiency: {self.ai_calls_made}/{self.max_ai_calls_per_run} calls, Quality: {metrics.ai_analysis_quality}%")

async def execute_enhanced_intelligence_mission():
    """Execute enhanced cyber threat intelligence mission with maximum value"""
    logger.info("🎖️ PATRIOTS PROTOCOL v4.2 - Enhanced Value Intelligence Mission")
    
    try:
        async with SmartPatriotsIntelligence() as intelligence_engine:
            # Collect high-value intelligence
            raw_intelligence = await intelligence_engine.collect_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No intelligence collected")
                return
            
            # Process with smart AI optimization
            threat_reports = await intelligence_engine.process_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed")
                return
            
            # Calculate accurate metrics
            metrics = intelligence_engine.calculate_accurate_metrics(threat_reports)
            
            # Save enhanced data
            intelligence_engine.save_intelligence_data(threat_reports, metrics)
            
            # Enhanced mission summary
            logger.info("✅ Enhanced Intelligence Mission Complete")
            logger.info(f"🎯 High-Value Threats: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"💥 Zero-Day Activity: {metrics.zero_day_count}")
            logger.info(f"🤖 AI Quality Score: {metrics.ai_analysis_quality}%")
            logger.info(f"📈 Intelligence Confidence: {metrics.intelligence_confidence}%")
            logger.info(f"🎖️ Patriots Protocol Enhanced v4.2: MAXIMUM VALUE OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Enhanced intelligence mission failed: {str(e)}")
        
        # Create error state with proper structure
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                "medium_threats": 0, "low_threats": 0, "global_threat_level": "OFFLINE",
                "intelligence_confidence": 0, "recent_threats_24h": 0,
                "top_threat_families": [], "geographic_distribution": {},
                "zero_day_count": 0, "trending_threats": [], "ai_analysis_quality": 0,
                "threat_velocity": "unknown", "fresh_intel_24h": 0, "source_credibility": 0.0,
                "emerging_trends": ["System Recovery"], "threat_evolution": "offline"
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
