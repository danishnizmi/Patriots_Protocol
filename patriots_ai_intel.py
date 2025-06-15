#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Cyber Intelligence Engine v4.0
Professional Cyber Threat Intelligence System

Repository: https://github.com/danishnizmi/Patriots_Protocol
"""

import os
import json
import asyncio
import aiohttp
import time
import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import feedparser

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class ThreatIntelReport:
    """Professional Cyber Threat Intelligence Report"""
    title: str
    summary: str
    source: str
    source_url: str
    timestamp: str
    threat_level: str
    technical_analysis: str
    confidence_score: float
    severity_rating: int
    attack_vectors: List[str]
    affected_sectors: List[str]
    threat_keywords: List[str]
    geographic_scope: str
    threat_actors: List[str]
    technical_indicators: Dict[str, List[str]]
    mitigation_priority: str

@dataclass
class ThreatMetrics:
    """Comprehensive Threat Intelligence Metrics"""
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

class PatriotsIntelligenceEngine:
    """Professional Cyber Threat Intelligence Collection and Analysis System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Premium cyber intelligence sources
        self.intelligence_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'reliability': 0.98,
                'specialization': 'government_threats'
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95,
                'specialization': 'investigative_cyber'
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.93,
                'specialization': 'incident_analysis'
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.88,
                'specialization': 'malware_research'
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.86,
                'specialization': 'threat_research'
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.84,
                'specialization': 'policy_threats'
            }
        ]
        
        # Technical threat classification
        self.threat_classification = {
            'CRITICAL': {
                'indicators': ['zero-day', 'nation-state', 'critical infrastructure', 'emergency patch', 'actively exploited', 'widespread campaign'],
                'base_score': 9
            },
            'HIGH': {
                'indicators': ['ransomware', 'apt', 'advanced persistent threat', 'supply chain', 'data breach', 'remote code execution'],
                'base_score': 7
            },
            'MEDIUM': {
                'indicators': ['vulnerability disclosure', 'malware campaign', 'phishing operation', 'security bypass'],
                'base_score': 5
            },
            'LOW': {
                'indicators': ['security advisory', 'patch available', 'awareness campaign'],
                'base_score': 3
            }
        }
        
        # MITRE ATT&CK technique mapping
        self.attack_techniques = {
            'initial_access': ['spear phishing', 'exploit public application', 'supply chain compromise'],
            'execution': ['command line interface', 'powershell', 'windows management'],
            'persistence': ['registry modification', 'scheduled task', 'service installation'],
            'privilege_escalation': ['process injection', 'dll hijacking', 'bypass uac'],
            'defense_evasion': ['obfuscated files', 'process hollowing', 'rootkit'],
            'credential_access': ['credential dumping', 'brute force', 'keylogging'],
            'discovery': ['network scanning', 'system information', 'process discovery'],
            'lateral_movement': ['remote services', 'smb admin shares', 'pass the hash'],
            'collection': ['data from local system', 'clipboard data', 'screen capture'],
            'command_control': ['standard protocols', 'custom protocols', 'dns tunneling'],
            'exfiltration': ['data transfer', 'encrypted channel', 'physical medium'],
            'impact': ['data destruction', 'service stop', 'system shutdown']
        }
        
        self.data_directory = Path('./data')
        self.data_directory.mkdir(exist_ok=True)
        
        logger.info("🎖️ Patriots Protocol Intelligence Engine v4.0 - Operational")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=45),
            headers={
                'User-Agent': 'Patriots-Protocol-Intelligence/4.0 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def extract_technical_indicators(self, content: str) -> Dict[str, List[str]]:
        """Extract comprehensive technical indicators from threat intelligence"""
        content_lower = content.lower()
        indicators = {
            'malware_families': [],
            'attack_techniques': [],
            'vulnerabilities': [],
            'threat_actors': [],
            'infrastructure': []
        }
        
        # Malware family detection
        malware_signatures = {
            'emotet': ['emotet'],
            'trickbot': ['trickbot'],
            'qakbot': ['qakbot', 'qbot'],
            'cobalt_strike': ['cobalt strike', 'cobaltstrike'],
            'lockbit': ['lockbit'],
            'conti': ['conti'],
            'revil': ['revil', 'sodinokibi'],
            'ryuk': ['ryuk'],
            'danabot': ['danabot'],
            'anubis': ['anubis'],
            'quasar': ['quasar rat'],
            'scanbox': ['scanbox'],
            'predator': ['predator spyware'],
            'paragon': ['paragon spyware'],
            'darkgate': ['darkgate'],
            'icedid': ['icedid']
        }
        
        for family, signatures in malware_signatures.items():
            if any(sig in content_lower for sig in signatures):
                indicators['malware_families'].append(family)
        
        # Attack technique identification
        for category, techniques in self.attack_techniques.items():
            for technique in techniques:
                if technique in content_lower:
                    indicators['attack_techniques'].append(f"{category}:{technique}")
        
        # CVE and vulnerability extraction
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cves = re.findall(cve_pattern, content.upper())
        indicators['vulnerabilities'].extend(cves)
        
        # Vulnerability types
        vuln_types = {
            'rce': ['remote code execution', 'rce'],
            'sqli': ['sql injection'],
            'xss': ['cross-site scripting', 'xss'],
            'lfi': ['local file inclusion'],
            'rfi': ['remote file inclusion'],
            'privilege_escalation': ['privilege escalation', 'elevation of privilege'],
            'authentication_bypass': ['authentication bypass', 'auth bypass']
        }
        
        for vuln_type, patterns in vuln_types.items():
            if any(pattern in content_lower for pattern in patterns):
                indicators['vulnerabilities'].append(vuln_type)
        
        # Threat actor identification
        threat_actors = [
            'lazarus', 'apt29', 'apt28', 'apt1', 'fancy bear', 'cozy bear',
            'carbanak', 'fin7', 'ta505', 'conti group', 'lockbit group',
            'cl0p', 'alphv', 'black basta', 'royal ransomware'
        ]
        
        for actor in threat_actors:
            if actor in content_lower:
                indicators['threat_actors'].append(actor.replace(' ', '_'))
        
        # Infrastructure indicators
        if 'command and control' in content_lower or 'c2' in content_lower:
            indicators['infrastructure'].append('command_control_servers')
        if 'botnet' in content_lower:
            indicators['infrastructure'].append('botnet_infrastructure')
        if 'tor' in content_lower or 'dark web' in content_lower:
            indicators['infrastructure'].append('anonymization_network')
        
        return indicators

    def classify_threat_level(self, title: str, summary: str) -> Tuple[str, int, str]:
        """Professional threat classification with technical analysis"""
        content = (title + ' ' + summary).lower()
        
        for level, config in self.threat_classification.items():
            if any(indicator in content for indicator in config['indicators']):
                # Calculate priority based on threat level and content analysis
                if level == 'CRITICAL':
                    priority = 'immediate_response'
                elif level == 'HIGH':
                    priority = 'urgent_assessment'
                elif level == 'MEDIUM':
                    priority = 'standard_monitoring'
                else:
                    priority = 'informational'
                
                return level, config['base_score'], priority
        
        return 'LOW', 3, 'informational'

    def generate_technical_analysis(self, content: str, threat_level: str, indicators: Dict[str, List[str]]) -> str:
        """Generate professional technical analysis"""
        content_lower = content.lower()
        analysis_components = []
        
        # Threat vector analysis
        if indicators['attack_techniques']:
            techniques = [t.split(':')[1] for t in indicators['attack_techniques']]
            analysis_components.append(f"Attack vectors include {', '.join(techniques[:3])}")
        
        # Malware analysis
        if indicators['malware_families']:
            analysis_components.append(f"Malware families detected: {', '.join(indicators['malware_families'][:3])}")
        
        # Vulnerability analysis
        if indicators['vulnerabilities']:
            cves = [v for v in indicators['vulnerabilities'] if v.startswith('CVE')]
            if cves:
                analysis_components.append(f"Exploiting vulnerabilities: {', '.join(cves[:2])}")
        
        # Infrastructure analysis
        if indicators['infrastructure']:
            analysis_components.append(f"Infrastructure elements: {', '.join(indicators['infrastructure'])}")
        
        # Specific threat analysis
        threat_specifics = []
        if 'zero-day' in content_lower:
            threat_specifics.append("zero-day exploitation confirmed")
        if 'supply chain' in content_lower:
            threat_specifics.append("supply chain compromise detected")
        if 'nation-state' in content_lower:
            threat_specifics.append("nation-state level sophistication")
        if 'critical infrastructure' in content_lower:
            threat_specifics.append("critical infrastructure targeting")
        
        # Combine analysis
        base_analysis = '. '.join(analysis_components) if analysis_components else "Technical analysis in progress"
        
        if threat_specifics:
            base_analysis += f". Notable characteristics: {', '.join(threat_specifics)}"
        
        # Add threat level specific guidance
        if threat_level == 'CRITICAL':
            base_analysis += ". Immediate containment and incident response required."
        elif threat_level == 'HIGH':
            base_analysis += ". Enhanced security posture and monitoring recommended."
        elif threat_level == 'MEDIUM':
            base_analysis += ". Standard security protocols and awareness required."
        
        return base_analysis

    def extract_threat_scope(self, content: str) -> Tuple[List[str], str]:
        """Extract affected sectors and geographic scope"""
        content_lower = content.lower()
        
        # Sector identification
        sector_keywords = {
            'healthcare': ['hospital', 'healthcare', 'medical', 'patient', 'clinic'],
            'financial': ['bank', 'financial', 'finance', 'payment', 'credit card'],
            'government': ['government', 'federal', 'agency', 'military', 'defense'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power grid', 'water'],
            'education': ['university', 'school', 'education', 'academic', 'student'],
            'manufacturing': ['manufacturing', 'industrial', 'factory', 'production'],
            'technology': ['tech company', 'software', 'cloud provider', 'saas'],
            'transportation': ['airline', 'transportation', 'logistics', 'shipping'],
            'telecommunications': ['telecom', 'telecommunications', 'mobile carrier']
        }
        
        affected_sectors = []
        for sector, keywords in sector_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                affected_sectors.append(sector)
        
        # Geographic scope determination
        geographic_indicators = {
            'global': ['global', 'worldwide', 'international', 'multiple countries'],
            'north_america': ['united states', 'usa', 'canada', 'north america'],
            'europe': ['europe', 'european union', 'uk', 'germany', 'france'],
            'asia_pacific': ['china', 'japan', 'korea', 'asia pacific', 'australia'],
            'middle_east': ['middle east', 'israel', 'saudi arabia', 'uae'],
            'latin_america': ['brazil', 'mexico', 'latin america', 'south america']
        }
        
        geographic_scope = 'global'  # Default
        for region, indicators in geographic_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                geographic_scope = region
                break
        
        return affected_sectors[:4], geographic_scope

    def extract_cyber_keywords(self, content: str) -> List[str]:
        """Extract relevant cybersecurity keywords"""
        content_lower = content.lower()
        
        cyber_terminology = [
            'ransomware', 'malware', 'phishing', 'apt', 'zero-day', 'vulnerability',
            'data breach', 'exploit', 'trojan', 'botnet', 'backdoor', 'spyware',
            'nation-state', 'critical infrastructure', 'supply chain', 'social engineering',
            'privilege escalation', 'lateral movement', 'command and control', 'exfiltration',
            'threat actor', 'incident response', 'threat hunting', 'cyber espionage'
        ]
        
        detected_keywords = []
        for keyword in cyber_terminology:
            if keyword in content_lower:
                detected_keywords.append(keyword.upper().replace(' ', '_'))
        
        return detected_keywords[:8]

    def deduplicate_intelligence(self, raw_intelligence: List[Dict]) -> List[Dict]:
        """Professional deduplication of intelligence reports"""
        seen_hashes = set()
        seen_titles = set()
        unique_intelligence = []
        
        for intel in raw_intelligence:
            # Create content signature
            content_signature = f"{intel['title'][:100]}{intel['summary'][:200]}".lower()
            content_hash = hashlib.sha256(content_signature.encode()).hexdigest()
            
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', intel['title'].lower()).strip()
            
            # Check for duplicates
            if content_hash in seen_hashes:
                continue
            
            # Check for title similarity
            is_duplicate = False
            for seen_title in seen_titles:
                title_words = set(normalized_title.split())
                seen_words = set(seen_title.split())
                
                if title_words and seen_words:
                    intersection = title_words.intersection(seen_words)
                    similarity = len(intersection) / max(len(title_words), len(seen_words))
                    
                    if similarity > 0.75:  # 75% similarity threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_hashes.add(content_hash)
                seen_titles.add(normalized_title)
                unique_intelligence.append(intel)
        
        logger.info(f"🔄 Intelligence Deduplication: {len(raw_intelligence)} → {len(unique_intelligence)} unique reports")
        return unique_intelligence

    async def collect_cyber_intelligence(self) -> List[Dict]:
        """Collect real-time cyber threat intelligence from premium sources"""
        collected_intelligence = []
        
        for source in self.intelligence_sources:
            try:
                logger.info(f"🔍 Collecting from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        source_intelligence = []
                        for entry in parsed_feed.entries[:12]:  # Top 12 per source
                            title = entry.title.strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Clean HTML tags
                            summary = re.sub(r'<[^>]+>', '', summary)
                            summary = re.sub(r'\s+', ' ', summary).strip()
                            
                            # Filter for cyber relevance
                            content_check = (title + ' ' + summary).lower()
                            cyber_indicators = [
                                'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
                                'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt'
                            ]
                            
                            if not any(indicator in content_check for indicator in cyber_indicators):
                                continue
                            
                            if len(summary) < 100:  # Minimum content threshold
                                continue
                            
                            intelligence_item = {
                                'title': title,
                                'summary': summary[:1000],  # Limit summary length
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'source_reliability': source['reliability'],
                                'specialization': source['specialization']
                            }
                            
                            source_intelligence.append(intelligence_item)
                        
                        collected_intelligence.extend(source_intelligence)
                        logger.info(f"📊 {source['name']}: {len(source_intelligence)} reports collected")
                        
                    else:
                        logger.warning(f"⚠️ {source['name']} returned status {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Collection error from {source['name']}: {str(e)}")
                continue
                
            # Rate limiting between sources
            await asyncio.sleep(1.0)

        logger.info(f"🎯 Total Intelligence Collected: {len(collected_intelligence)} reports")
        
        # Deduplicate collected intelligence
        unique_intelligence = self.deduplicate_intelligence(collected_intelligence)
        
        return unique_intelligence

    async def process_threat_intelligence(self, raw_intelligence: List[Dict]) -> List[ThreatIntelReport]:
        """Process raw intelligence into professional threat reports"""
        threat_reports = []
        
        for intel_item in raw_intelligence:
            try:
                # Extract technical indicators
                technical_indicators = self.extract_technical_indicators(
                    intel_item['title'] + ' ' + intel_item['summary']
                )
                
                # Classify threat
                threat_level, severity_rating, mitigation_priority = self.classify_threat_level(
                    intel_item['title'], intel_item['summary']
                )
                
                # Generate technical analysis
                technical_analysis = self.generate_technical_analysis(
                    intel_item['title'] + ' ' + intel_item['summary'],
                    threat_level,
                    technical_indicators
                )
                
                # Extract scope
                affected_sectors, geographic_scope = self.extract_threat_scope(
                    intel_item['title'] + ' ' + intel_item['summary']
                )
                
                # Extract attack vectors
                attack_vectors = []
                for technique in technical_indicators['attack_techniques']:
                    category = technique.split(':')[0]
                    if category not in attack_vectors:
                        attack_vectors.append(category)
                
                # Extract keywords
                threat_keywords = self.extract_cyber_keywords(
                    intel_item['title'] + ' ' + intel_item['summary']
                )
                
                # Create comprehensive threat report
                threat_report = ThreatIntelReport(
                    title=intel_item['title'],
                    summary=intel_item['summary'],
                    source=intel_item['source'],
                    source_url=intel_item['source_url'],
                    timestamp=intel_item['timestamp'],
                    threat_level=threat_level,
                    technical_analysis=technical_analysis,
                    confidence_score=intel_item['source_reliability'],
                    severity_rating=severity_rating,
                    attack_vectors=attack_vectors,
                    affected_sectors=affected_sectors,
                    threat_keywords=threat_keywords,
                    geographic_scope=geographic_scope,
                    threat_actors=technical_indicators['threat_actors'],
                    technical_indicators=technical_indicators,
                    mitigation_priority=mitigation_priority
                )
                
                threat_reports.append(threat_report)
                logger.info(f"✅ Processed: {threat_report.title[:50]}... (Level: {threat_level})")
                
            except Exception as e:
                logger.error(f"❌ Processing error: {str(e)}")
                continue

        return threat_reports

    def calculate_threat_metrics(self, reports: List[ThreatIntelReport]) -> ThreatMetrics:
        """Calculate comprehensive threat intelligence metrics"""
        if not reports:
            return ThreatMetrics(
                total_threats=0, critical_threats=0, high_threats=0, medium_threats=0, low_threats=0,
                active_threat_actors=0, attack_techniques_detected=0, sectors_under_threat=0,
                global_threat_level="MONITORING", intelligence_confidence=0, recent_threats_24h=0,
                source_reliability=0.0, emerging_threat_vectors=[], threat_landscape_trend="unknown"
            )

        # Count threats by level
        critical_count = sum(1 for r in reports if r.threat_level == 'CRITICAL')
        high_count = sum(1 for r in reports if r.threat_level == 'HIGH')
        medium_count = sum(1 for r in reports if r.threat_level == 'MEDIUM')
        low_count = sum(1 for r in reports if r.threat_level == 'LOW')
        
        # Global threat level assessment
        if critical_count >= 2:
            global_threat_level = "CRITICAL"
        elif critical_count >= 1 or high_count >= 4:
            global_threat_level = "HIGH"
        elif high_count >= 2 or medium_count >= 6:
            global_threat_level = "ELEVATED"
        elif high_count >= 1 or medium_count >= 3:
            global_threat_level = "MEDIUM"
        else:
            global_threat_level = "LOW"

        # Calculate unique threat actors
        all_actors = []
        for report in reports:
            all_actors.extend(report.threat_actors)
        unique_actors = len(set(all_actors))

        # Calculate unique attack techniques
        all_techniques = []
        for report in reports:
            all_techniques.extend(report.attack_vectors)
        unique_techniques = len(set(all_techniques))

        # Calculate sectors under threat
        all_sectors = []
        for report in reports:
            all_sectors.extend(report.affected_sectors)
        unique_sectors = len(set(all_sectors))

        # Recent threats (24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = 0
        
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > recent_cutoff:
                    recent_count += 1
            except:
                pass

        # Emerging threat vectors
        vector_frequency = {}
        for report in reports:
            for vector in report.attack_vectors:
                vector_frequency[vector] = vector_frequency.get(vector, 0) + 1
        
        emerging_vectors = [vector for vector, count in 
                          sorted(vector_frequency.items(), key=lambda x: x[1], reverse=True)[:5]]

        # Calculate average confidence
        avg_confidence = int(sum(r.confidence_score for r in reports) / len(reports) * 100)
        
        # Calculate average source reliability
        avg_reliability = sum(r.confidence_score for r in reports) / len(reports)

        return ThreatMetrics(
            total_threats=len(reports),
            critical_threats=critical_count,
            high_threats=high_count,
            medium_threats=medium_count,
            low_threats=low_count,
            active_threat_actors=unique_actors,
            attack_techniques_detected=unique_techniques,
            sectors_under_threat=unique_sectors,
            global_threat_level=global_threat_level,
            intelligence_confidence=avg_confidence,
            recent_threats_24h=recent_count,
            source_reliability=avg_reliability,
            emerging_threat_vectors=emerging_vectors,
            threat_landscape_trend="escalating" if critical_count > 0 else "stable"
        )

    def save_intelligence_data(self, reports: List[ThreatIntelReport], metrics: ThreatMetrics) -> None:
        """Save processed intelligence data"""
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
        
        logger.info(f"💾 Intelligence data saved: {len(reports)} reports, {metrics.global_threat_level} threat level")

async def execute_intelligence_mission():
    """Execute comprehensive cyber threat intelligence mission"""
    logger.info("🎖️ PATRIOTS PROTOCOL - Intelligence Mission Initiated")
    
    try:
        async with PatriotsIntelligenceEngine() as intelligence_engine:
            # Collect real-time intelligence
            raw_intelligence = await intelligence_engine.collect_cyber_intelligence()
            
            if not raw_intelligence:
                logger.warning("⚠️ No intelligence collected - check source availability")
                return
            
            # Process into threat reports
            threat_reports = await intelligence_engine.process_threat_intelligence(raw_intelligence)
            
            if not threat_reports:
                logger.warning("⚠️ No threats processed from collected intelligence")
                return
            
            # Calculate metrics
            metrics = intelligence_engine.calculate_threat_metrics(threat_reports)
            
            # Save data
            intelligence_engine.save_intelligence_data(threat_reports, metrics)
            
            # Mission summary
            logger.info("✅ Intelligence Mission Complete")
            logger.info(f"🎯 Threats Analyzed: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️ Critical Threats: {metrics.critical_threats}")
            logger.info(f"🎖️ Patriots Protocol Intelligence: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Intelligence mission failed: {str(e)}")
        
        # Create minimal operational data for error state
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0, "critical_threats": 0, "high_threats": 0, 
                "medium_threats": 0, "low_threats": 0, "active_threat_actors": 0,
                "attack_techniques_detected": 0, "sectors_under_threat": 0,
                "global_threat_level": "OFFLINE", "intelligence_confidence": 0,
                "recent_threats_24h": 0, "source_reliability": 0.0,
                "emerging_threat_vectors": [], "threat_landscape_trend": "unknown"
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
    asyncio.run(execute_intelligence_mission())
