#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Enhanced AI-Powered Cyber Intelligence Engine v4.0
Dynamic Threat Intelligence with AI-Driven Technical Analysis

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

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedThreatReport:
    """Enhanced Cyber Threat Intelligence Report with AI Analysis"""
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

@dataclass
class IntelligenceMetrics:
    """Enhanced Intelligence Metrics"""
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

class EnhancedPatriotsIntelligence:
    """Enhanced AI-Powered Cyber Threat Intelligence System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Premium intelligence sources with geographic focus
        self.intelligence_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'reliability': 0.98,
                'geographic_focus': 'US'
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'geographic_focus': 'Global'
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'geographic_focus': 'Global'
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.88,
                'geographic_focus': 'Global'
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.86,
                'geographic_focus': 'Global'
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.84,
                'geographic_focus': 'US'
            }
        ]
        
        # Country code mapping for geographic intelligence
        self.country_mapping = {
            'united states': 'US', 'usa': 'US', 'america': 'US',
            'china': 'CN', 'russia': 'RU', 'iran': 'IR', 'north korea': 'KP',
            'germany': 'DE', 'france': 'FR', 'united kingdom': 'GB', 'uk': 'GB',
            'japan': 'JP', 'south korea': 'KR', 'australia': 'AU',
            'canada': 'CA', 'mexico': 'MX', 'brazil': 'BR',
            'india': 'IN', 'israel': 'IL', 'ukraine': 'UA'
        }
        
        self.data_directory = Path('./data')
        self.data_directory.mkdir(exist_ok=True)
        
        logger.info("🎖️ Enhanced Patriots Protocol Intelligence Engine v4.0 - Operational")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=45),
            headers={
                'User-Agent': 'Patriots-Protocol-Enhanced/4.0 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def ai_powered_technical_analysis(self, title: str, content: str) -> Dict[str, Any]:
        """AI-powered technical analysis for real cybersecurity insights"""
        if not self.api_token:
            logger.warning("⚠️ No API token - skipping AI technical analysis")
            return self.basic_technical_analysis(title, content)

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Focused technical analysis prompt
            analysis_prompt = f"""You are a senior cybersecurity analyst. Analyze this threat intelligence and provide SPECIFIC technical insights in JSON format:

Title: {title}
Content: {content[:1500]}

Provide analysis in this exact JSON structure:
{{
    "technical_analysis": "Detailed technical analysis with specific attack vectors, exploitation methods, and defensive countermeasures (2-3 sentences max)",
    "threat_family": "Specific malware family or threat type (e.g., 'Ransomware-as-a-Service', 'Nation-State APT', 'Banking Trojan')",
    "attack_sophistication": "LOW/MEDIUM/HIGH/ADVANCED based on technical complexity",
    "attack_vectors": ["specific_attack_methods"],
    "cve_references": ["CVE-YYYY-XXXX if mentioned"],
    "threat_actors": ["specific_group_names_if_mentioned"],
    "mitigation_priority": "IMMEDIATE/URGENT/STANDARD/INFORMATIONAL",
    "geographic_indicators": ["countries_or_regions_mentioned"],
    "sector_targets": ["specific_industries_targeted"]
}}

Focus on TECHNICAL DETAILS, not generic security advice. Be specific about exploitation methods, attack techniques, and technical countermeasures."""

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cybersecurity threat analyst. Provide specific technical analysis, not generic advice."},
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
                logger.info("✅ AI technical analysis completed")
                return analysis_result
                
        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {str(e)[:100]}... - using basic analysis")
            
        return self.basic_technical_analysis(title, content)

    def basic_technical_analysis(self, title: str, content: str) -> Dict[str, Any]:
        """Dynamic technical analysis without hardcoded patterns"""
        full_text = (title + ' ' + content).lower()
        
        # Dynamic threat classification
        threat_indicators = {
            'zero_day': ['zero-day', 'zero day', '0-day', 'unknown vulnerability'],
            'ransomware': ['ransomware', 'encryption', 'ransom', 'crypto-locker'],
            'apt': ['apt', 'advanced persistent threat', 'nation-state', 'state-sponsored'],
            'supply_chain': ['supply chain', 'software supply', 'third-party'],
            'critical_infra': ['critical infrastructure', 'power grid', 'utility', 'scada']
        }
        
        # Dynamic severity assessment
        severity_keywords = {
            'CRITICAL': ['critical', 'emergency', 'actively exploited', 'widespread'],
            'HIGH': ['high', 'severe', 'dangerous', 'significant'],
            'MEDIUM': ['medium', 'moderate', 'notable'],
            'LOW': ['low', 'minor', 'limited']
        }
        
        # Extract CVE references dynamically
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_refs = re.findall(cve_pattern, content.upper())
        
        # Determine threat family
        if any(keyword in full_text for keyword in threat_indicators['ransomware']):
            threat_family = "Ransomware"
            sophistication = "HIGH"
        elif any(keyword in full_text for keyword in threat_indicators['apt']):
            threat_family = "Advanced Persistent Threat"
            sophistication = "ADVANCED"
        elif any(keyword in full_text for keyword in threat_indicators['zero_day']):
            threat_family = "Zero-Day Exploit"
            sophistication = "ADVANCED"
        elif any(keyword in full_text for keyword in threat_indicators['supply_chain']):
            threat_family = "Supply Chain Attack"
            sophistication = "HIGH"
        else:
            threat_family = "General Cyber Threat"
            sophistication = "MEDIUM"
        
        # Dynamic geographic extraction
        geographic_indicators = []
        for country, code in self.country_mapping.items():
            if country in full_text:
                geographic_indicators.append(country.title())
        
        # Dynamic attack vector identification
        attack_vectors = []
        vector_keywords = {
            'phishing': ['phishing', 'email attack', 'malicious email'],
            'malware_delivery': ['malware', 'trojan', 'backdoor'],
            'vulnerability_exploitation': ['exploit', 'vulnerability', 'rce'],
            'social_engineering': ['social engineering', 'human factor'],
            'network_intrusion': ['network', 'lateral movement', 'privilege escalation']
        }
        
        for vector, keywords in vector_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                attack_vectors.append(vector)
        
        # Generate specific technical analysis
        analysis_parts = []
        
        if cve_refs:
            analysis_parts.append(f"Exploits vulnerabilities {', '.join(cve_refs[:3])}")
        
        if attack_vectors:
            analysis_parts.append(f"Uses {', '.join(attack_vectors[:2])} attack vectors")
        
        if geographic_indicators:
            analysis_parts.append(f"Geographic focus: {', '.join(geographic_indicators[:2])}")
        
        technical_analysis = '. '.join(analysis_parts) if analysis_parts else f"{threat_family} requiring {sophistication.lower()} defensive measures"
        
        return {
            "technical_analysis": technical_analysis,
            "threat_family": threat_family,
            "attack_sophistication": sophistication,
            "attack_vectors": attack_vectors,
            "cve_references": cve_refs,
            "threat_actors": [],
            "mitigation_priority": "URGENT" if sophistication in ['ADVANCED', 'HIGH'] else "STANDARD",
            "geographic_indicators": geographic_indicators,
            "sector_targets": []
        }

    def extract_geographic_data(self, content: str) -> Tuple[str, str]:
        """Extract geographic scope and country code"""
        content_lower = content.lower()
        
        # Check for specific countries
        for country, code in self.country_mapping.items():
            if country in content_lower:
                return country.replace('_', ' ').title(), code
        
        # Regional indicators
        if any(indicator in content_lower for indicator in ['global', 'worldwide', 'international']):
            return 'Global', 'GLOBAL'
        elif any(indicator in content_lower for indicator in ['europe', 'european']):
            return 'Europe', 'EU'
        elif any(indicator in content_lower for indicator in ['asia', 'asian']):
            return 'Asia Pacific', 'APAC'
        
        return 'Global', 'GLOBAL'

    def deduplicate_intelligence(self, raw_intel: List[Dict]) -> List[Dict]:
        """Advanced deduplication with content similarity"""
        seen_hashes = set()
        unique_intel = []
        
        for intel in raw_intel:
            # Create content fingerprint
            content_sig = f"{intel['title'][:100]}{intel['summary'][:200]}".lower()
            content_hash = hashlib.sha256(content_sig.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_intel.append(intel)
        
        logger.info(f"🔄 Deduplication: {len(raw_intel)} → {len(unique_intel)} unique reports")
        return unique_intel

    async def collect_cyber_intelligence(self) -> List[Dict]:
        """Enhanced intelligence collection with geographic tracking"""
        collected_intel = []
        
        for source in self.intelligence_sources:
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intel = []
                        for entry in parsed_feed.entries[:15]:
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Clean content
                            summary = re.sub(r'<[^>]+>', '', summary)
                            summary = re.sub(r'&[^;]+;', ' ', summary)
                            summary = re.sub(r'\s+', ' ', summary).strip()
                            
                            # Enhanced cyber relevance check
                            content_check = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt',
                                'zero-day', 'backdoor', 'trojan', 'spyware', 'botnet'
                            ]
                            
                            if not any(indicator in content_check for indicator in cyber_indicators):
                                continue
                            
                            if len(summary) < 150:
                                continue
                            
                            intel_item = {
                                'title': title,
                                'summary': summary[:1200],
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'source_reliability': source['reliability'],
                                'source_geographic_focus': source['geographic_focus']
                            }
                            
                            source_intel.append(intel_item)
                        
                        collected_intel.extend(source_intel)
                        logger.info(f"📊 {source['name']}: {len(source_intel)} reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue

        logger.info(f"🎯 Total Intelligence: {len(collected_intel)} reports")
        return self.deduplicate_intelligence(collected_intel)

    async def process_enhanced_intelligence(self, raw_intel: List[Dict]) -> List[EnhancedThreatReport]:
        """Process intelligence with AI-powered analysis"""
        threat_reports = []
        
        # Process in small batches to manage API usage
        batch_size = 3
        
        for i in range(0, len(raw_intel), batch_size):
            batch = raw_intel[i:i + batch_size]
            
            for intel_item in batch:
                try:
                    # Get AI-powered technical analysis
                    ai_analysis = await self.ai_powered_technical_analysis(
                        intel_item['title'], intel_item['summary']
                    )
                    
                    # Extract geographic data
                    geographic_scope, country_code = self.extract_geographic_data(
                        intel_item['title'] + ' ' + intel_item['summary']
                    )
                    
                    # Determine threat level based on AI analysis
                    sophistication = ai_analysis.get('attack_sophistication', 'MEDIUM')
                    mitigation_priority = ai_analysis.get('mitigation_priority', 'STANDARD')
                    
                    if sophistication == 'ADVANCED' or mitigation_priority == 'IMMEDIATE':
                        threat_level = 'CRITICAL'
                        severity_rating = 9
                    elif sophistication == 'HIGH' or mitigation_priority == 'URGENT':
                        threat_level = 'HIGH'
                        severity_rating = 7
                    elif sophistication == 'MEDIUM':
                        threat_level = 'MEDIUM'
                        severity_rating = 5
                    else:
                        threat_level = 'LOW'
                        severity_rating = 3
                    
                    # Create enhanced threat report
                    threat_report = EnhancedThreatReport(
                        title=intel_item['title'],
                        summary=intel_item['summary'],
                        source=intel_item['source'],
                        source_url=intel_item['source_url'],
                        timestamp=intel_item['timestamp'],
                        threat_level=threat_level,
                        ai_technical_analysis=ai_analysis.get('technical_analysis', 'Technical analysis pending'),
                        confidence_score=intel_item['source_reliability'],
                        severity_rating=severity_rating,
                        attack_vectors=ai_analysis.get('attack_vectors', []),
                        affected_sectors=ai_analysis.get('sector_targets', []),
                        threat_keywords=[],  # We're not using hardcoded keywords
                        geographic_scope=geographic_scope,
                        country_code=country_code,
                        threat_actors=ai_analysis.get('threat_actors', []),
                        technical_indicators={'cve_references': ai_analysis.get('cve_references', [])},
                        mitigation_priority=mitigation_priority,
                        cve_references=ai_analysis.get('cve_references', []),
                        threat_family=ai_analysis.get('threat_family', 'Unknown'),
                        attack_sophistication=sophistication
                    )
                    
                    threat_reports.append(threat_report)
                    logger.info(f"✅ Processed: {threat_report.title[:50]}... (Level: {threat_level})")
                    
                except Exception as e:
                    logger.error(f"❌ Processing error: {str(e)}")
                    continue
            
            # Rate limiting for API calls
            if i + batch_size < len(raw_intel):
                await asyncio.sleep(2.0)

        return threat_reports

    def calculate_enhanced_metrics(self, reports: List[EnhancedThreatReport]) -> IntelligenceMetrics:
        """Calculate enhanced metrics with filtering and geographic data"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                active_threat_actors=0, attack_techniques_detected=0, sectors_under_threat=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                source_reliability=0.0, emerging_threat_vectors=[], threat_landscape_trend="unknown",
                top_threat_families=[], geographic_distribution={}, critical_threat_names=[],
                zero_day_count=0
            )

        # Count by threat level
        critical_count = sum(1 for r in reports if r.threat_level == 'CRITICAL')
        high_count = sum(1 for r in reports if r.threat_level == 'HIGH')
        medium_count = sum(1 for r in reports if r.threat_level == 'MEDIUM')
        low_count = sum(1 for r in reports if r.threat_level == 'LOW')
        
        # Global threat assessment
        if critical_count >= 2:
            global_threat_level = "CRITICAL"
        elif critical_count >= 1 or high_count >= 3:
            global_threat_level = "HIGH"
        elif high_count >= 1 or medium_count >= 4:
            global_threat_level = "ELEVATED"
        else:
            global_threat_level = "MEDIUM"

        # Geographic distribution
        geo_distribution = {}
        for report in reports:
            country = report.country_code
            geo_distribution[country] = geo_distribution.get(country, 0) + 1

        # Top threat families
        family_counts = {}
        for report in reports:
            family = report.threat_family
            family_counts[family] = family_counts.get(family, 0) + 1
        
        top_families = [
            {"name": family, "count": count, "severity": "HIGH" if count >= 3 else "MEDIUM"}
            for family, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Critical threat names
        critical_names = [r.title[:60] + "..." if len(r.title) > 60 else r.title 
                         for r in reports if r.threat_level == 'CRITICAL'][:5]

        # Zero-day count
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower() or 
                            any('zero' in cve.lower() for cve in r.cve_references))

        # Recent threats
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = 0
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff:
                    recent_count += 1
            except:
                pass

        return IntelligenceMetrics(
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
            emerging_threat_vectors=list(set(vector for r in reports for vector in r.attack_vectors))[:5],
            threat_landscape_trend="escalating" if critical_count > 0 else "stable",
            top_threat_families=top_families,
            geographic_distribution=geo_distribution,
            critical_threat_names=critical_names,
            zero_day_count=zero_day_count
        )

    def save_enhanced_data(self, reports: List[EnhancedThreatReport], metrics: IntelligenceMetrics) -> None:
        """Save enhanced intelligence data"""
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
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }

        output_file = self.data_directory / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Enhanced intelligence saved: {len(reports)} reports, {metrics.global_threat_level} threat level")

async def execute_enhanced_intelligence_mission():
    """Execute enhanced cyber threat intelligence mission"""
    logger.info("🎖️ PATRIOTS PROTOCOL - Enhanced Intelligence Mission Initiated")
    
    try:
        async with EnhancedPatriotsIntelligence() as intelligence_engine:
            # Collect intelligence
            raw_intelligence = await intelligence_engine.collect_cyber_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No intelligence collected")
                return
            
            # Process with AI enhancement
            threat_reports = await intelligence_engine.process_enhanced_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed")
                return
            
            # Calculate enhanced metrics
            metrics = intelligence_engine.calculate_enhanced_metrics(threat_reports)
            
            # Save enhanced data
            intelligence_engine.save_enhanced_data(threat_reports, metrics)
            
            # Mission summary
            logger.info("✅ Enhanced Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"🌍 Zero-Day Exploits: {metrics.zero_day_count}")
            logger.info(f"🎖️ Patriots Protocol Enhanced Intelligence: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Enhanced intelligence mission failed: {str(e)}")
        
        # Minimal error state data
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
                "critical_threat_names": [], "zero_day_count": 0
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
    asyncio.run(execute_enhanced_intelligence_mission())
