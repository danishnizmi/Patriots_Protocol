#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Cyber Intelligence Engine v4.0
AI-Driven Threat Intelligence Network

Focus: Real-time cyber threat intelligence from premium sources
Source: PATRIOTS PROTOCOL - https://github.com/danishnizmi/Patriots_Protocol
"""

import os
import json
import asyncio
import aiohttp
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import feedparser
import hashlib
import logging
from pathlib import Path

# Configure logging for intelligence only
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - INTEL - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class ThreatIntelligence:
    """Cyber Threat Intelligence Report"""
    title: str
    summary: str
    source: str
    source_url: str
    timestamp: str
    threat_level: str
    ai_analysis: str
    confidence: float
    keywords: List[str]
    attack_vectors: List[str]
    affected_sectors: List[str]
    severity_score: int
    geolocation: str
    threat_actors: List[str]
    technical_details: Dict[str, Any]

@dataclass
class IntelligenceMetrics:
    """Intelligence-Focused Metrics"""
    total_threats: int
    critical_threats: int
    high_threats: int
    medium_threats: int
    low_threats: int
    threat_actors_identified: int
    attack_techniques_observed: int
    sectors_targeted: int
    global_threat_level: str
    intelligence_confidence: int
    fresh_intel_24h: int
    source_credibility: float
    emerging_trends: List[str]
    threat_evolution: str

class PatriotsCyberIntel:
    """Patriots Protocol Cyber Intelligence System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session = None
        
        # Premium cyber intelligence sources
        self.intel_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'credibility': 0.98,
                'focus': 'government_advisories'
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'credibility': 0.94,
                'focus': 'investigative_cyber'
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'credibility': 0.93,
                'focus': 'incident_response'
            },
            {
                'name': 'DARK_READING',
                'url': 'https://www.darkreading.com/rss_simple.asp',
                'credibility': 0.89,
                'focus': 'enterprise_security'
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'credibility': 0.87,
                'focus': 'malware_analysis'
            },
            {
                'name': 'SECURITY_WEEK',
                'url': 'https://www.securityweek.com/feed',
                'credibility': 0.86,
                'focus': 'industry_news'
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'credibility': 0.85,
                'focus': 'threat_research'
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'credibility': 0.83,
                'focus': 'policy_threats'
            }
        ]
        
        self.data_dir = Path('./data')
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info("🎖️  Patriots Protocol Intelligence Engine v4.0 - Intelligence Network Active")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-Intel/v4.0 (+https://github.com/danishnizmi/Patriots_Protocol)',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def analyze_threat_intelligence(self, articles_batch: List[Dict]) -> List[Dict]:
        """AI-powered threat intelligence analysis"""
        if not self.api_token:
            logger.info("⚠️  Using fallback intelligence analysis")
            return self._fallback_intelligence_analysis(articles_batch)

        try:
            # Dynamic import to handle missing library
            try:
                from openai import AsyncOpenAI
            except ImportError:
                logger.warning("⚠️  OpenAI library not available - using fallback")
                return self._fallback_intelligence_analysis(articles_batch)
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Prepare intelligence content for analysis
            intel_content = ""
            for i, article in enumerate(articles_batch):
                intel_content += f"\n--- THREAT INTELLIGENCE {i+1} ---\n"
                intel_content += f"Title: {article['title']}\n"
                intel_content += f"Content: {article['summary'][:800]}\n"
                intel_content += f"Source: {article['source']}\n"

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an elite cyber threat intelligence analyst for PATRIOTS PROTOCOL. Analyze cyber security threats and provide detailed intelligence assessments in JSON format:

{
    "intelligence_analyses": [
        {
            "article_index": 0,
            "threat_level": "CRITICAL/HIGH/MEDIUM/LOW",
            "severity_score": 8,
            "confidence": 0.92,
            "ai_analysis": "Comprehensive threat intelligence analysis with actionable insights, technical details, and strategic implications (200-300 words)",
            "attack_vectors": ["advanced_persistent_threat", "ransomware", "supply_chain"],
            "affected_sectors": ["healthcare", "finance", "critical_infrastructure"],
            "keywords": ["apt", "zero_day", "nation_state"],
            "geolocation": "Global/North_America/Europe/Asia_Pacific",
            "threat_actors": ["lazarus_group", "apt29", "unknown_actor"],
            "technical_details": {
                "malware_families": ["emotet", "trickbot"],
                "attack_techniques": ["spear_phishing", "lateral_movement"],
                "indicators": ["file_hashes", "domains", "ips"],
                "vulnerabilities": ["CVE-2024-XXXX"]
            },
            "emerging_trends": ["ai_powered_attacks", "cloud_targeting"],
            "threat_evolution": "escalating/stable/declining"
        }
    ],
    "global_assessment": {
        "overall_threat_level": "HIGH",
        "key_trends": ["trend1", "trend2"],
        "intelligence_confidence": 89
    }
}

Focus on actionable threat intelligence, technical analysis, and strategic cyber security insights."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these cyber threat intelligence reports and provide comprehensive assessment:\n{intel_content}"
                    }
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI intelligence response
            try:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = ai_response[json_start:json_end]
                    results = json.loads(json_content)
                    logger.info(f"✅ AI Intelligence Analysis Complete - {len(results.get('intelligence_analyses', []))} threats analyzed")
                    return results.get('intelligence_analyses', [])
            
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  AI response parsing failed: {str(e)} - using fallback")
                return self._fallback_intelligence_analysis(articles_batch)
            
        except Exception as e:
            logger.error(f"❌ AI intelligence analysis failed: {str(e)}")
            return self._fallback_intelligence_analysis(articles_batch)

    def _fallback_intelligence_analysis(self, articles: List[Dict]) -> List[Dict]:
        """Fallback threat intelligence analysis"""
        analyses = []
        
        # Advanced threat detection patterns
        threat_patterns = {
            'CRITICAL': {
                'keywords': ['zero-day', 'nation-state', 'critical infrastructure', 'emergency patch', 'active exploitation'],
                'score': 9
            },
            'HIGH': {
                'keywords': ['ransomware', 'apt', 'data breach', 'vulnerability', 'exploit', 'malware campaign'],
                'score': 7
            },
            'MEDIUM': {
                'keywords': ['phishing', 'trojan', 'botnet', 'scam', 'security update'],
                'score': 5
            },
            'LOW': {
                'keywords': ['awareness', 'training', 'policy', 'recommendation'],
                'score': 3
            }
        }
        
        for i, article in enumerate(articles):
            content = (article['title'] + ' ' + article['summary']).lower()
            
            # Determine threat level using pattern matching
            threat_level = "LOW"
            severity_score = 3
            
            for level, pattern in threat_patterns.items():
                if any(keyword in content for keyword in pattern['keywords']):
                    threat_level = level
                    severity_score = pattern['score']
                    break

            # Extract technical intelligence
            attack_vectors = self._extract_attack_vectors(content)
            sectors = self._extract_sectors(content)
            actors = self._extract_threat_actors(content)
            technical_details = self._extract_technical_details(content)

            analyses.append({
                'article_index': i,
                'threat_level': threat_level,
                'severity_score': severity_score,
                'confidence': 0.75,
                'ai_analysis': f"Intelligence Assessment: {threat_level} threat identified. {self._generate_analysis_summary(content, threat_level)}",
                'attack_vectors': attack_vectors,
                'affected_sectors': sectors,
                'keywords': self._extract_cyber_keywords(content),
                'geolocation': self._determine_geolocation(content),
                'threat_actors': actors,
                'technical_details': technical_details,
                'emerging_trends': self._identify_trends(content),
                'threat_evolution': 'stable'
            })
        
        return analyses

    def _extract_attack_vectors(self, content: str) -> List[str]:
        """Extract attack vectors from content"""
        vectors = []
        vector_keywords = {
            'ransomware': ['ransomware', 'crypto-locker', 'file encryption'],
            'phishing': ['phishing', 'spear phishing', 'email attack'],
            'malware': ['malware', 'trojan', 'virus', 'backdoor'],
            'apt': ['apt', 'advanced persistent threat', 'nation-state'],
            'supply_chain': ['supply chain', 'third-party', 'vendor compromise'],
            'zero_day': ['zero-day', 'zero day', 'unknown vulnerability'],
            'ddos': ['ddos', 'denial of service', 'amplification'],
            'insider_threat': ['insider threat', 'employee', 'credential abuse']
        }
        
        for vector, keywords in vector_keywords.items():
            if any(keyword in content for keyword in keywords):
                vectors.append(vector)
        
        return vectors[:5]  # Limit to top 5

    def _extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors"""
        sectors = []
        sector_keywords = {
            'healthcare': ['hospital', 'healthcare', 'medical', 'patient', 'clinic'],
            'finance': ['bank', 'financial', 'finance', 'payment', 'credit'],
            'government': ['government', 'federal', 'agency', 'military', 'defense'],
            'critical_infrastructure': ['infrastructure', 'utility', 'energy', 'power grid'],
            'education': ['university', 'school', 'education', 'academic'],
            'manufacturing': ['manufacturing', 'industrial', 'factory', 'production'],
            'technology': ['tech', 'software', 'cloud', 'saas', 'platform'],
            'transportation': ['airline', 'transportation', 'logistics', 'shipping']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        
        return sectors[:3]  # Limit to top 3

    def _extract_threat_actors(self, content: str) -> List[str]:
        """Extract threat actors"""
        actors = []
        actor_patterns = [
            'lazarus', 'apt29', 'apt28', 'apt1', 'fancy bear', 'cozy bear',
            'carbanak', 'fin7', 'ta505', 'conti', 'revil', 'lockbit'
        ]
        
        for actor in actor_patterns:
            if actor in content:
                actors.append(actor.replace(' ', '_'))
        
        return actors if actors else ['unknown_actor']

    def _extract_technical_details(self, content: str) -> Dict[str, Any]:
        """Extract technical details"""
        details = {
            'malware_families': [],
            'attack_techniques': [],
            'vulnerabilities': [],
            'indicators': []
        }
        
        # Malware families
        malware_families = ['emotet', 'trickbot', 'qakbot', 'cobalt strike', 'metasploit']
        details['malware_families'] = [m for m in malware_families if m in content]
        
        # Attack techniques (MITRE ATT&CK)
        techniques = ['spear_phishing', 'lateral_movement', 'privilege_escalation', 'data_exfiltration']
        details['attack_techniques'] = [t for t in techniques if t.replace('_', ' ') in content]
        
        # Look for CVE patterns
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cves = re.findall(cve_pattern, content.upper())
        details['vulnerabilities'] = cves[:3]
        
        return details

    def _extract_cyber_keywords(self, content: str) -> List[str]:
        """Extract cyber security keywords"""
        cyber_keywords = [
            'ransomware', 'malware', 'phishing', 'apt', 'zero-day', 'vulnerability',
            'breach', 'exploit', 'trojan', 'botnet', 'backdoor', 'spyware',
            'nation-state', 'critical infrastructure', 'data theft', 'cyber espionage'
        ]
        
        found_keywords = []
        for keyword in cyber_keywords:
            if keyword in content.lower():
                found_keywords.append(keyword.upper().replace('-', '_'))
        
        return found_keywords[:8]

    def _determine_geolocation(self, content: str) -> str:
        """Determine geographic focus"""
        regions = {
            'North_America': ['us', 'usa', 'america', 'canada', 'mexico'],
            'Europe': ['eu', 'europe', 'uk', 'germany', 'france'],
            'Asia_Pacific': ['china', 'japan', 'korea', 'asia', 'australia'],
            'Global': ['global', 'worldwide', 'international']
        }
        
        for region, keywords in regions.items():
            if any(keyword in content for keyword in keywords):
                return region
        
        return 'Global'

    def _identify_trends(self, content: str) -> List[str]:
        """Identify emerging trends"""
        trends = []
        trend_keywords = {
            'ai_powered_attacks': ['ai', 'artificial intelligence', 'machine learning'],
            'cloud_targeting': ['cloud', 'aws', 'azure', 'saas'],
            'mobile_threats': ['mobile', 'android', 'ios', 'smartphone'],
            'iot_exploitation': ['iot', 'internet of things', 'smart device'],
            'supply_chain_attacks': ['supply chain', 'third party', 'vendor']
        }
        
        for trend, keywords in trend_keywords.items():
            if any(keyword in content for keyword in keywords):
                trends.append(trend)
        
        return trends[:3]

    def _generate_analysis_summary(self, content: str, threat_level: str) -> str:
        """Generate intelligent analysis summary"""
        if threat_level == 'CRITICAL':
            return "Immediate response required. High-confidence threat with potential for significant impact."
        elif threat_level == 'HIGH':
            return "Priority monitoring recommended. Elevated threat requiring tactical assessment."
        elif threat_level == 'MEDIUM':
            return "Standard monitoring protocols. Moderate threat with limited immediate impact."
        else:
            return "Informational intelligence. Low-priority threat for awareness purposes."

    async def fetch_cyber_intelligence(self) -> List[Dict]:
        """Fetch real-time cyber threat intelligence"""
        all_intel = []
        
        for source in self.intel_sources:
            try:
                logger.info(f"🔍 Gathering intelligence from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        source_intel = []
                        for entry in feed.entries[:10]:  # Top 10 per source
                            title = entry.title
                            summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                            
                            if len(summary) < 50:
                                continue
                            
                            # Filter for cyber security intelligence
                            if not self._is_cyber_intelligence(title + ' ' + summary):
                                continue
                            
                            intel = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'credibility': source['credibility'],
                                'focus_area': source['focus']
                            }
                            
                            source_intel.append(intel)
                        
                        all_intel.extend(source_intel)
                        logger.info(f"📊 {source['name']}: {len(source_intel)} intelligence reports collected")
                        
                    else:
                        logger.warning(f"⚠️  {source['name']} returned {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Intelligence gathering error from {source['name']}: {str(e)}")
                continue

        logger.info(f"🎯 Total Intelligence Collected: {len(all_intel)} reports from {len(self.intel_sources)} sources")
        return all_intel

    def _is_cyber_intelligence(self, content: str) -> bool:
        """Filter for relevant cyber security intelligence"""
        cyber_indicators = [
            'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
            'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt',
            'zero-day', 'incident', 'compromise', 'botnet', 'trojan', 'backdoor',
            'espionage', 'nation-state', 'critical infrastructure'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in cyber_indicators)

    def _clean_text(self, text: str) -> str:
        """Clean and format text content"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text

    async def process_intelligence(self, intel_data: List[Dict]) -> List[ThreatIntelligence]:
        """Process raw intelligence into structured threat reports"""
        threat_reports = []
        
        # Process in batches for AI analysis
        batch_size = 4
        
        for i in range(0, len(intel_data), batch_size):
            batch = intel_data[i:i + batch_size]
            
            # Get AI analysis
            ai_analyses = await self.analyze_threat_intelligence(batch)
            
            # Create threat intelligence reports
            for j, intel in enumerate(batch):
                try:
                    # Find corresponding AI analysis
                    ai_analysis = None
                    for analysis in ai_analyses:
                        if analysis.get('article_index') == j:
                            ai_analysis = analysis
                            break
                    
                    if not ai_analysis:
                        continue
                    
                    # Create comprehensive threat intelligence report
                    report = ThreatIntelligence(
                        title=intel['title'],
                        summary=intel['summary'][:800],
                        source=intel['source'],
                        source_url=intel.get('source_url', ''),
                        timestamp=intel['timestamp'],
                        threat_level=ai_analysis.get('threat_level', 'MEDIUM'),
                        ai_analysis=ai_analysis.get('ai_analysis', 'Intelligence analysis in progress'),
                        confidence=ai_analysis.get('confidence', 0.8),
                        keywords=ai_analysis.get('keywords', []),
                        attack_vectors=ai_analysis.get('attack_vectors', []),
                        affected_sectors=ai_analysis.get('affected_sectors', []),
                        severity_score=ai_analysis.get('severity_score', 5),
                        geolocation=ai_analysis.get('geolocation', 'Global'),
                        threat_actors=ai_analysis.get('threat_actors', ['unknown_actor']),
                        technical_details=ai_analysis.get('technical_details', {})
                    )
                    
                    threat_reports.append(report)
                    logger.info(f"✅ Intelligence Processed: {report.title[:60]}... (Level: {report.threat_level})")
                    
                except Exception as e:
                    logger.error(f"❌ Intelligence processing error: {str(e)}")
                    continue
            
            # Rate limiting between batches
            await asyncio.sleep(2.0)

        return threat_reports

    def calculate_intelligence_metrics(self, reports: List[ThreatIntelligence]) -> IntelligenceMetrics:
        """Calculate comprehensive intelligence metrics"""
        if not reports:
            return IntelligenceMetrics(
                total_threats=0,
                critical_threats=0,
                high_threats=0,
                medium_threats=0,
                low_threats=0,
                threat_actors_identified=0,
                attack_techniques_observed=0,
                sectors_targeted=0,
                global_threat_level="LOW",
                intelligence_confidence=85,
                fresh_intel_24h=0,
                source_credibility=0.9,
                emerging_trends=[],
                threat_evolution="stable"
            )

        # Count threats by level
        critical_count = len([r for r in reports if r.threat_level == 'CRITICAL'])
        high_count = len([r for r in reports if r.threat_level == 'HIGH'])
        medium_count = len([r for r in reports if r.threat_level == 'MEDIUM'])
        low_count = len([r for r in reports if r.threat_level == 'LOW'])
        
        # Determine global threat level
        if critical_count > 0:
            global_threat = "CRITICAL"
        elif high_count >= 3:
            global_threat = "HIGH"
        elif high_count > 0 or medium_count >= 5:
            global_threat = "MEDIUM"
        else:
            global_threat = "LOW"

        # Analyze threat actors
        all_actors = []
        for report in reports:
            all_actors.extend(report.threat_actors)
        unique_actors = len(set(actor for actor in all_actors if actor != 'unknown_actor'))

        # Analyze attack techniques
        all_vectors = []
        for report in reports:
            all_vectors.extend(report.attack_vectors)
        unique_techniques = len(set(all_vectors))

        # Analyze targeted sectors
        all_sectors = []
        for report in reports:
            all_sectors.extend(report.affected_sectors)
        unique_sectors = len(set(all_sectors))

        # Fresh intelligence check (last 24 hours)
        fresh_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > cutoff:
                    fresh_count += 1
            except:
                pass

        # Extract emerging trends
        all_trends = []
        for report in reports:
            if hasattr(report, 'technical_details') and 'emerging_trends' in report.technical_details:
                all_trends.extend(report.technical_details['emerging_trends'])
        
        trend_freq = {}
        for trend in all_trends:
            trend_freq[trend] = trend_freq.get(trend, 0) + 1
        
        top_trends = [trend for trend, _ in sorted(trend_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

        return IntelligenceMetrics(
            total_threats=len(reports),
            critical_threats=critical_count,
            high_threats=high_count,
            medium_threats=medium_count,
            low_threats=low_count,
            threat_actors_identified=unique_actors,
            attack_techniques_observed=unique_techniques,
            sectors_targeted=unique_sectors,
            global_threat_level=global_threat,
            intelligence_confidence=int(sum(r.confidence for r in reports) / len(reports) * 100),
            fresh_intel_24h=fresh_count,
            source_credibility=0.9,  # Average credibility of sources
            emerging_trends=top_trends,
            threat_evolution="escalating" if critical_count > 0 else "stable"
        )

async def main():
    """Execute Patriots Protocol Intelligence Mission"""
    logger.info("🎖️  PATRIOTS PROTOCOL - Intelligence Gathering Mission Initiated")
    
    try:
        async with PatriotsCyberIntel() as intel_system:
            # Gather real-time cyber intelligence
            intel_data = await intel_system.fetch_cyber_intelligence()
            
            # Process into structured threat intelligence
            threat_reports = await intel_system.process_intelligence(intel_data)
            
            # Calculate intelligence metrics
            metrics = intel_system.calculate_intelligence_metrics(threat_reports)
            
            # Prepare intelligence output
            output_data = {
                "articles": [asdict(report) for report in threat_reports],
                "metrics": asdict(metrics),
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "version": "4.0",
                "intelligence_summary": {
                    "mission_status": "OPERATIONAL",
                    "threats_analyzed": len(threat_reports),
                    "intelligence_sources": len(intel_system.intel_sources),
                    "confidence_level": metrics.intelligence_confidence,
                    "threat_landscape": metrics.global_threat_level,
                    "repository": "https://github.com/danishnizmi/Patriots_Protocol"
                }
            }

            # Save intelligence data
            with open('./data/news-analysis.json', 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info("✅ Intelligence Mission Complete")
            logger.info(f"🎯 Threat Intelligence Reports: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️  Critical Threats: {metrics.critical_threats}")
            logger.info(f"🎖️  Patriots Protocol Intelligence Network: OPERATIONAL")
            
    except Exception as e:
        logger.error(f"❌ Intelligence mission error: {str(e)}")
        
        # Generate minimal intelligence data
        fallback_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0,
                "critical_threats": 0,
                "global_threat_level": "LOW",
                "intelligence_confidence": 85,
                "fresh_intel_24h": 0
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0",
            "intelligence_summary": {
                "mission_status": "STANDBY",
                "repository": "https://github.com/danishnizmi/Patriots_Protocol"
            }
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(fallback_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
