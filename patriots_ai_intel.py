#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Advanced AI-Powered Cyber Intelligence Engine v4.0
Professional Threat Intelligence with Advanced AI Analysis

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

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedThreatReport:
    """Advanced Cyber Threat Intelligence Report"""
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
    threat_keywords: List[str]
    geographic_scope: str
    country_code: str
    threat_actors: List[str]
    technical_indicators: Dict[str, List[str]]
    mitigation_priority: str
    cve_references: List[str]
    threat_family: str
    attack_sophistication: str
    iocs: Dict[str, List[str]]
    attack_timeline: str
    risk_score: int
    correlation_id: str

@dataclass
class AdvancedMetrics:
    """Advanced Intelligence Metrics"""
    total_threats: int
    critical_threats: int
    high_threats: int
    medium_threats: int
    low_threats: int
    active_threat_actors: int
    attack_techniques_detected: int
    sectors_under_threat: int
    global_threat_level: str
    intelligence_confidence: int
    recent_threats_24h: int
    source_reliability: float
    emerging_threat_vectors: List[str]
    threat_landscape_trend: str
    top_threat_families: List[Dict[str, Any]]
    geographic_distribution: Dict[str, int]
    critical_threat_names: List[str]
    zero_day_count: int
    attack_timeline_data: List[Dict[str, Any]]
    threat_evolution: Dict[str, Any]
    sector_risk_matrix: Dict[str, int]
    trending_threats: List[Dict[str, Any]]

class AdvancedPatriotsIntelligence:
    """Advanced AI-Powered Cyber Threat Intelligence System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Enhanced intelligence sources
        self.intelligence_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'reliability': 0.98,
                'geographic_focus': 'US',
                'specialization': 'government_advisories'
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'geographic_focus': 'Global',
                'specialization': 'investigative_journalism'
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'geographic_focus': 'Global',
                'specialization': 'incident_analysis'
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.88,
                'geographic_focus': 'Global',
                'specialization': 'malware_research'
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.86,
                'geographic_focus': 'Global',
                'specialization': 'threat_research'
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.84,
                'geographic_focus': 'US',
                'specialization': 'policy_analysis'
            }
        ]
        
        # Comprehensive country/region mapping
        self.geographic_mapping = {
            # North America
            'united states': 'US', 'usa': 'US', 'america': 'US', 'u.s.': 'US',
            'canada': 'CA', 'mexico': 'MX',
            
            # Asia Pacific
            'china': 'CN', 'japan': 'JP', 'south korea': 'KR', 'korea': 'KR',
            'australia': 'AU', 'new zealand': 'NZ', 'singapore': 'SG',
            'india': 'IN', 'thailand': 'TH', 'vietnam': 'VN', 'philippines': 'PH',
            'malaysia': 'MY', 'indonesia': 'ID', 'taiwan': 'TW', 'hong kong': 'HK',
            
            # Europe
            'germany': 'DE', 'france': 'FR', 'united kingdom': 'GB', 'uk': 'GB',
            'italy': 'IT', 'spain': 'ES', 'netherlands': 'NL', 'belgium': 'BE',
            'sweden': 'SE', 'norway': 'NO', 'denmark': 'DK', 'finland': 'FI',
            'poland': 'PL', 'ukraine': 'UA', 'switzerland': 'CH', 'austria': 'AT',
            
            # Middle East & Africa
            'israel': 'IL', 'saudi arabia': 'SA', 'uae': 'AE', 'turkey': 'TR',
            'iran': 'IR', 'egypt': 'EG', 'south africa': 'ZA', 'nigeria': 'NG',
            
            # Other Regions
            'russia': 'RU', 'north korea': 'KP', 'brazil': 'BR', 'argentina': 'AR',
            
            # Regional Groupings
            'europe': 'EU', 'european union': 'EU', 'asia pacific': 'APAC',
            'middle east': 'ME', 'africa': 'AF', 'latin america': 'LATAM'
        }
        
        # Enhanced sector mapping
        self.sector_mapping = {
            'healthcare': ['hospital', 'medical', 'patient', 'clinic', 'health', 'pharmaceutical'],
            'financial': ['bank', 'finance', 'payment', 'credit', 'financial', 'fintech', 'cryptocurrency'],
            'government': ['government', 'federal', 'agency', 'military', 'defense', 'public sector'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power', 'water', 'transportation'],
            'education': ['university', 'school', 'education', 'academic', 'student', 'campus'],
            'manufacturing': ['manufacturing', 'industrial', 'factory', 'production', 'automotive'],
            'technology': ['tech', 'software', 'cloud', 'saas', 'platform', 'microsoft', 'google', 'apple'],
            'telecommunications': ['telecom', 'communications', 'mobile', 'network', 'isp'],
            'retail': ['retail', 'shopping', 'commerce', 'store', 'consumer'],
            'media': ['media', 'news', 'journalism', 'broadcasting', 'entertainment']
        }
        
        self.data_directory = Path('./data')
        self.data_directory.mkdir(exist_ok=True)
        
        logger.info("🎖️ Advanced Patriots Protocol Intelligence Engine v4.0 - Operational")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-Advanced/4.0 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def advanced_ai_technical_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Advanced AI-powered technical analysis with specific prompting"""
        if not self.api_token:
            logger.warning("⚠️ No API token - using advanced basic analysis")
            return self.advanced_basic_analysis(title, content, source)

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Enhanced technical analysis prompt with specific instructions
            analysis_prompt = f"""You are a senior cybersecurity threat analyst with 15+ years experience. Analyze this threat intelligence and provide SPECIFIC, ACTIONABLE technical insights.

THREAT INTELLIGENCE:
Title: {title}
Content: {content[:2000]}
Source: {source}

Provide analysis in this EXACT JSON format:
{{
    "technical_analysis": "SPECIFIC technical analysis focusing on: attack methods, exploitation techniques, indicators, defensive countermeasures. Be SPECIFIC - mention actual CVEs, malware names, attack techniques, not generic statements. Max 2-3 sentences.",
    "threat_family": "SPECIFIC threat type (e.g., 'LockBit Ransomware', 'APT29 Campaign', 'Chrome Zero-Day Exploit', 'DanaBot Banking Trojan')",
    "attack_sophistication": "LOW/MEDIUM/HIGH/ADVANCED based on technical complexity and threat actor capabilities",
    "attack_vectors": ["specific_attack_methods_mentioned_in_content"],
    "cve_references": ["only_actual_CVE_numbers_mentioned"],
    "threat_actors": ["only_specific_group_names_mentioned"],
    "mitigation_priority": "IMMEDIATE/URGENT/STANDARD/INFORMATIONAL",
    "geographic_indicators": ["only_countries_specifically_mentioned"],
    "sector_targets": ["only_industries_specifically_mentioned"],
    "iocs": {{
        "domains": ["malicious_domains_if_mentioned"],
        "ips": ["malicious_ips_if_mentioned"],
        "file_hashes": ["file_hashes_if_mentioned"]
    }},
    "risk_score": 1-10 based on impact and likelihood,
    "attack_timeline": "immediate/hours/days/weeks based on urgency"
}}

CRITICAL INSTRUCTIONS:
- Be SPECIFIC, not generic. If no specific technical details are available, say "Limited technical details available"
- Only include CVEs, threat actors, IOCs that are ACTUALLY mentioned in the content
- For technical_analysis, focus on HOW the attack works, WHAT makes it dangerous, HOW to defend
- Avoid generic phrases like "requires defensive measures" - be specific about what kind
- If it's a patch Tuesday, mention specific vulnerabilities being patched
- If it's malware, mention specific capabilities and infection methods"""

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cybersecurity threat analyst. Provide specific technical analysis, avoid generic statements."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = ai_response[json_start:json_end]
                analysis_result = json.loads(json_content)
                logger.info("✅ Advanced AI technical analysis completed")
                return analysis_result
                
        except Exception as e:
            logger.warning(f"⚠️ Advanced AI analysis failed: {str(e)[:100]}... - using advanced basic analysis")
            
        return self.advanced_basic_analysis(title, content, source)

    def advanced_basic_analysis(self, title: str, content: str, source: str) -> Dict[str, Any]:
        """Advanced basic analysis with specific threat detection"""
        full_text = (title + ' ' + content).lower()
        
        # Enhanced threat family detection
        threat_family = "Unknown Threat"
        attack_sophistication = "MEDIUM"
        risk_score = 5
        
        # Specific threat family identification
        if any(term in full_text for term in ['lockbit', 'conti', 'revil', 'ryuk']):
            threat_family = "Ransomware-as-a-Service"
            attack_sophistication = "HIGH"
            risk_score = 8
        elif any(term in full_text for term in ['apt29', 'apt28', 'lazarus', 'fancy bear']):
            threat_family = "Nation-State APT"
            attack_sophistication = "ADVANCED"
            risk_score = 9
        elif 'zero-day' in full_text or 'zero day' in full_text:
            threat_family = "Zero-Day Exploit"
            attack_sophistication = "ADVANCED"
            risk_score = 9
        elif any(term in full_text for term in ['ransomware', 'encryption', 'ransom']):
            threat_family = "Ransomware"
            attack_sophistication = "HIGH"
            risk_score = 8
        elif any(term in full_text for term in ['danabot', 'emotet', 'trickbot', 'qakbot']):
            threat_family = "Banking Trojan"
            attack_sophistication = "HIGH"
            risk_score = 7
        elif any(term in full_text for term in ['phishing', 'spear phishing']):
            threat_family = "Phishing Campaign"
            attack_sophistication = "MEDIUM"
            risk_score = 6
        elif 'supply chain' in full_text:
            threat_family = "Supply Chain Attack"
            attack_sophistication = "HIGH"
            risk_score = 8
        elif any(term in full_text for term in ['patch tuesday', 'microsoft patch']):
            threat_family = "Vulnerability Disclosure"
            attack_sophistication = "MEDIUM"
            risk_score = 6
        elif any(term in full_text for term in ['ddos', 'denial of service']):
            threat_family = "DDoS Attack"
            attack_sophistication = "MEDIUM"
            risk_score = 5
        elif any(term in full_text for term in ['spyware', 'surveillance']):
            threat_family = "Spyware Campaign"
            attack_sophistication = "HIGH"
            risk_score = 7

        # Extract CVE references
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_refs = re.findall(cve_pattern, content.upper())
        
        # Extract geographic indicators
        geographic_indicators = []
        for region, code in self.geographic_mapping.items():
            if region in full_text:
                geographic_indicators.append(region.title())
        
        # Extract attack vectors based on content
        attack_vectors = []
        if any(term in full_text for term in ['phishing', 'email']):
            attack_vectors.append('email_phishing')
        if any(term in full_text for term in ['malware', 'trojan', 'virus']):
            attack_vectors.append('malware_delivery')
        if any(term in full_text for term in ['vulnerability', 'exploit', 'rce']):
            attack_vectors.append('vulnerability_exploitation')
        if any(term in full_text for term in ['network', 'lateral']):
            attack_vectors.append('network_intrusion')
        if any(term in full_text for term in ['social engineering', 'human']):
            attack_vectors.append('social_engineering')
        
        # Extract affected sectors
        affected_sectors = []
        for sector, keywords in self.sector_mapping.items():
            if any(keyword in full_text for keyword in keywords):
                affected_sectors.append(sector)
        
        # Generate specific technical analysis
        analysis_components = []
        
        if cve_refs:
            analysis_components.append(f"Addresses vulnerabilities {', '.join(cve_refs[:3])}")
        
        if 'zero-day' in full_text:
            analysis_components.append("exploits previously unknown vulnerabilities")
        
        if attack_vectors:
            analysis_components.append(f"utilizes {', '.join(attack_vectors[:2])} attack methods")
        
        if affected_sectors:
            analysis_components.append(f"targeting {', '.join(affected_sectors[:2])} sectors")
        
        if not analysis_components:
            if threat_family != "Unknown Threat":
                analysis_components.append(f"{threat_family.lower()} with {attack_sophistication.lower()} technical complexity")
            else:
                analysis_components.append("cybersecurity incident requiring assessment")
        
        technical_analysis = '. '.join(analysis_components).capitalize()
        
        # Determine timeline and priority
        if attack_sophistication == "ADVANCED" or 'zero-day' in full_text:
            timeline = "immediate"
            priority = "IMMEDIATE"
        elif attack_sophistication == "HIGH" or 'critical' in full_text:
            timeline = "hours"
            priority = "URGENT"
        else:
            timeline = "days"
            priority = "STANDARD"
        
        return {
            "technical_analysis": technical_analysis,
            "threat_family": threat_family,
            "attack_sophistication": attack_sophistication,
            "attack_vectors": attack_vectors,
            "cve_references": cve_refs,
            "threat_actors": [],
            "mitigation_priority": priority,
            "geographic_indicators": geographic_indicators,
            "sector_targets": affected_sectors,
            "iocs": {"domains": [], "ips": [], "file_hashes": []},
            "risk_score": risk_score,
            "attack_timeline": timeline
        }

    def extract_comprehensive_geographic_data(self, content: str) -> Tuple[str, str]:
        """Extract comprehensive geographic scope and country code"""
        content_lower = content.lower()
        
        # Check for specific countries/regions
        for region, code in self.geographic_mapping.items():
            if region in content_lower:
                return region.replace('_', ' ').title(), code
        
        # Fallback to global
        return 'Global', 'GLOBAL'

    def advanced_deduplication(self, raw_intel: List[Dict]) -> List[Dict]:
        """Advanced deduplication with similarity scoring"""
        seen_signatures = set()
        unique_intel = []
        
        for intel in raw_intel:
            # Create multiple signatures for comparison
            title_sig = re.sub(r'[^\w\s]', '', intel['title'].lower())[:100]
            content_sig = re.sub(r'[^\w\s]', '', intel['summary'].lower())[:200]
            
            # Combine signatures
            combined_sig = f"{title_sig}|{content_sig}"
            signature_hash = hashlib.sha256(combined_sig.encode()).hexdigest()
            
            # Check for exact duplicates
            if signature_hash in seen_signatures:
                continue
            
            # Check for near-duplicates
            is_duplicate = False
            for existing_sig in seen_signatures:
                # Simple similarity check - could be enhanced with edit distance
                if len(set(title_sig.split()).intersection(set(existing_sig.split()))) > len(title_sig.split()) * 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_signatures.add(signature_hash)
                unique_intel.append(intel)
        
        logger.info(f"🔄 Advanced Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel

    async def collect_advanced_intelligence(self) -> List[Dict]:
        """Enhanced intelligence collection with advanced filtering"""
        collected_intel = []
        
        for source in self.intelligence_sources:
            try:
                logger.info(f"🔍 Advanced Collection from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:20]:  # Increased limit
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Advanced content cleaning
                            summary = re.sub(r'<[^>]+>', '', summary)
                            summary = re.sub(r'&[^;]+;', ' ', summary)
                            summary = re.sub(r'\s+', ' ', summary).strip()
                            
                            # Enhanced cyber relevance check
                            content_check = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt',
                                'zero-day', 'backdoor', 'trojan', 'spyware', 'botnet', 'ddos',
                                'patch', 'cve', 'incident', 'compromise', 'espionage'
                            ]
                            
                            if not any(indicator in content_check for indicator in cyber_indicators):
                                continue
                            
                            # Quality filter
                            if len(summary) < 100 or len(title) < 10:
                                continue
                            
                            # Filter out non-English content (basic check)
                            if any(char in title for char in ['中', '日', '한', 'русский']):
                                continue
                            
                            intel_item = {
                                'title': title,
                                'summary': summary[:1500],
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'source_reliability': source['reliability'],
                                'source_geographic_focus': source['geographic_focus'],
                                'source_specialization': source['specialization']
                            }
                            
                            source_intel.append(intel_item)
                        
                        collected_intel.extend(source_intel)
                        logger.info(f"📊 {source['name']}: {len(source_intel)} advanced reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Advanced collection error from {source['name']}: {str(e)}")
                continue
                
            # Rate limiting
            await asyncio.sleep(1.0)

        logger.info(f"🎯 Total Advanced Intelligence: {len(collected_intel)} reports")
        return self.advanced_deduplication(collected_intel)

    async def process_advanced_intelligence(self, raw_intel: List[Dict]) -> List[AdvancedThreatReport]:
        """Process intelligence with advanced AI analysis"""
        threat_reports = []
        
        # Process in optimized batches
        batch_size = 2  # Smaller batches for better AI analysis
        
        for i in range(0, len(raw_intel), batch_size):
            batch = raw_intel[i:i + batch_size]
            
            for intel_item in batch:
                try:
                    # Get advanced AI analysis
                    ai_analysis = await self.advanced_ai_technical_analysis(
                        intel_item['title'], 
                        intel_item['summary'],
                        intel_item['source']
                    )
                    
                    # Extract geographic data
                    geographic_scope, country_code = self.extract_comprehensive_geographic_data(
                        intel_item['title'] + ' ' + intel_item['summary']
                    )
                    
                    # Advanced threat level determination
                    sophistication = ai_analysis.get('attack_sophistication', 'MEDIUM')
                    risk_score = ai_analysis.get('risk_score', 5)
                    mitigation_priority = ai_analysis.get('mitigation_priority', 'STANDARD')
                    
                    if sophistication == 'ADVANCED' or risk_score >= 9:
                        threat_level = 'CRITICAL'
                        severity_rating = 9
                    elif sophistication == 'HIGH' or risk_score >= 7:
                        threat_level = 'HIGH'
                        severity_rating = 7
                    elif risk_score >= 5:
                        threat_level = 'MEDIUM'
                        severity_rating = 5
                    else:
                        threat_level = 'LOW'
                        severity_rating = 3
                    
                    # Generate correlation ID
                    correlation_content = f"{intel_item['title']}{ai_analysis.get('threat_family', '')}"
                    correlation_id = hashlib.md5(correlation_content.encode()).hexdigest()[:8]
                    
                    # Create advanced threat report
                    threat_report = AdvancedThreatReport(
                        title=intel_item['title'],
                        summary=intel_item['summary'],
                        source=intel_item['source'],
                        source_url=intel_item['source_url'],
                        timestamp=intel_item['timestamp'],
                        threat_level=threat_level,
                        ai_technical_analysis=ai_analysis.get('technical_analysis', 'Technical analysis in progress'),
                        confidence_score=intel_item['source_reliability'],
                        severity_rating=severity_rating,
                        attack_vectors=ai_analysis.get('attack_vectors', []),
                        affected_sectors=ai_analysis.get('sector_targets', []),
                        threat_keywords=[],  # Dynamically extracted
                        geographic_scope=geographic_scope,
                        country_code=country_code,
                        threat_actors=ai_analysis.get('threat_actors', []),
                        technical_indicators={'cve_references': ai_analysis.get('cve_references', [])},
                        mitigation_priority=mitigation_priority,
                        cve_references=ai_analysis.get('cve_references', []),
                        threat_family=ai_analysis.get('threat_family', 'Unknown Threat'),
                        attack_sophistication=sophistication,
                        iocs=ai_analysis.get('iocs', {}),
                        attack_timeline=ai_analysis.get('attack_timeline', 'unknown'),
                        risk_score=risk_score,
                        correlation_id=correlation_id
                    )
                    
                    threat_reports.append(threat_report)
                    logger.info(f"✅ Advanced Processing: {threat_report.title[:50]}... (Level: {threat_level})")
                    
                except Exception as e:
                    logger.error(f"❌ Advanced processing error: {str(e)}")
                    continue
            
            # Rate limiting for API calls
            if i + batch_size < len(raw_intel):
                await asyncio.sleep(3.0)

        return threat_reports

    def calculate_advanced_metrics(self, reports: List[AdvancedThreatReport]) -> AdvancedMetrics:
        """Calculate comprehensive advanced metrics"""
        if not reports:
            return AdvancedMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                active_threat_actors=0, attack_techniques_detected=0, sectors_under_threat=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                source_reliability=0.0, emerging_threat_vectors=[], threat_landscape_trend="unknown",
                top_threat_families=[], geographic_distribution={}, critical_threat_names=[],
                zero_day_count=0, attack_timeline_data=[], threat_evolution={},
                sector_risk_matrix={}, trending_threats=[]
            )

        # Count by threat level
        critical_count = sum(1 for r in reports if r.threat_level == 'CRITICAL')
        high_count = sum(1 for r in reports if r.threat_level == 'HIGH')
        medium_count = sum(1 for r in reports if r.threat_level == 'MEDIUM')
        low_count = sum(1 for r in reports if r.threat_level == 'LOW')
        
        # Advanced global threat assessment
        if critical_count >= 3:
            global_threat_level = "CRITICAL"
        elif critical_count >= 1 or high_count >= 5:
            global_threat_level = "HIGH"
        elif high_count >= 2 or medium_count >= 8:
            global_threat_level = "ELEVATED"
        else:
            global_threat_level = "MEDIUM"

        # Geographic distribution
        geo_distribution = {}
        for report in reports:
            country = report.country_code
            geo_distribution[country] = geo_distribution.get(country, 0) + 1

        # Advanced threat families analysis
        family_counts = {}
        family_risk_scores = {}
        for report in reports:
            family = report.threat_family
            family_counts[family] = family_counts.get(family, 0) + 1
            family_risk_scores[family] = max(family_risk_scores.get(family, 0), report.risk_score)
        
        top_families = [
            {
                "name": family, 
                "count": count, 
                "risk_level": "CRITICAL" if family_risk_scores[family] >= 8 else "HIGH" if family_risk_scores[family] >= 6 else "MEDIUM",
                "avg_risk": family_risk_scores[family]
            }
            for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        ]

        # Critical threat names
        critical_names = [r.title[:80] + "..." if len(r.title) > 80 else r.title 
                         for r in reports if r.threat_level == 'CRITICAL'][:10]

        # Zero-day analysis
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower() or 
                            any('zero' in cve.lower() for cve in r.cve_references) or
                            'zero-day' in r.ai_technical_analysis.lower())

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

        # Timeline analysis
        timeline_data = []
        for report in reports:
            try:
                timestamp = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                timeline_data.append({
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'threat_level': report.threat_level,
                    'risk_score': report.risk_score,
                    'family': report.threat_family
                })
            except:
                pass

        # Trending threats (recent high-impact)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        trending = []
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff and report.risk_score >= 7:
                    trending.append({
                        'title': report.title,
                        'risk_score': report.risk_score,
                        'threat_family': report.threat_family,
                        'timestamp': report.timestamp
                    })
            except:
                pass
        
        trending_threats = sorted(trending, key=lambda x: x['risk_score'], reverse=True)[:5]

        # Recent threats count
        recent_cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff_24h:
                    recent_count += 1
            except:
                pass

        return AdvancedMetrics(
            total_threats=len(reports),
            critical_threats=critical_count,
            high_threats=high_count,
            medium_threats=medium_count,
            low_threats=low_count,
            active_threat_actors=len(set(actor for r in reports for actor in r.threat_actors)),
            attack_techniques_detected=len(set(vector for r in reports for vector in r.attack_vectors)),
            sectors_under_threat=len(set(sector for r in reports for sector in r.affected_sectors)),
            global_threat_level=global_threat_level,
            intelligence_confidence=int(sum(r.confidence_score for r in reports) / len(reports) * 100),
            recent_threats_24h=recent_count,
            source_reliability=sum(r.confidence_score for r in reports) / len(reports),
            emerging_threat_vectors=list(set(vector for r in reports for vector in r.attack_vectors))[:8],
            threat_landscape_trend="escalating" if critical_count > 0 else "stable",
            top_threat_families=top_families,
            geographic_distribution=geo_distribution,
            critical_threat_names=critical_names,
            zero_day_count=zero_day_count,
            attack_timeline_data=timeline_data,
            threat_evolution={"trend": "escalating" if critical_count > 0 else "stable"},
            sector_risk_matrix=sector_risk_matrix,
            trending_threats=trending_threats
        )

    def save_advanced_data(self, reports: List[AdvancedThreatReport], metrics: AdvancedMetrics) -> None:
        """Save advanced intelligence data with enhanced structure"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0",
            "intelligence_summary": {
                "mission_status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "intelligence_sources": len(self.intelligence_sources),
                "confidence_level": metrics.intelligence_confidence,
                "threat_landscape": metrics.global_threat_level,
                "advanced_features": {
                    "ai_analysis": True,
                    "geographic_mapping": True,
                    "threat_correlation": True,
                    "risk_scoring": True,
                    "timeline_analysis": True
                },
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Advanced intelligence saved: {len(reports)} reports, {metrics.global_threat_level} threat level")

async def execute_advanced_intelligence_mission():
    """Execute advanced cyber threat intelligence mission"""
    logger.info("🎖️ PATRIOTS PROTOCOL - Advanced Intelligence Mission Initiated")
    
    try:
        async with AdvancedPatriotsIntelligence() as intelligence_engine:
            # Collect advanced intelligence
            raw_intelligence = await intelligence_engine.collect_advanced_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No advanced intelligence collected")
                return
            
            # Process with advanced AI
            threat_reports = await intelligence_engine.process_advanced_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No advanced threats processed")
                return
            
            # Calculate advanced metrics
            metrics = intelligence_engine.calculate_advanced_metrics(threat_reports)
            
            # Save advanced data
            intelligence_engine.save_advanced_data(threat_reports, metrics)
            
            # Advanced mission summary
            logger.info("✅ Advanced Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"💥 Zero-Day Exploits: {metrics.zero_day_count}")
            logger.info(f"🌍 Geographic Coverage: {len(metrics.geographic_distribution)} regions")
            logger.info(f"🎖️ Patriots Protocol Advanced Intelligence: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Advanced intelligence mission failed: {str(e)}")
        
        # Minimal error state
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                "medium_threats": 0, "low_threats": 0, "active_threat_actors": 0,
                "attack_techniques_detected": 0, "sectors_under_threat": 0,
                "global_threat_level": "OFFLINE", "intelligence_confidence": 0,
                "recent_threats_24h": 0, "source_reliability": 0.0,
                "emerging_threat_vectors": [], "threat_landscape_trend": "unknown",
                "top_threat_families": [], "geographic_distribution": {},
                "critical_threat_names": [], "zero_day_count": 0,
                "attack_timeline_data": [], "threat_evolution": {},
                "sector_risk_matrix": {}, "trending_threats": []
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0",
            "intelligence_summary": {
                "mission_status": "ERROR",
                "threats_analyzed": 0,
                "intelligence_sources": 0,
                "confidence_level": 0,
                "threat_landscape": "OFFLINE",
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(execute_advanced_intelligence_mission())
