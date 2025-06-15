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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS PROTOCOL - %(levelname)s - %(message)s',
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
    iocs: List[str]  # Indicators of Compromise

@dataclass
class ThreatMetrics:
    """Threat Intelligence Metrics"""
    total_threats: int
    critical_threats: int
    high_priority_threats: int
    active_campaigns: int
    threat_actors: List[str]
    attack_techniques: List[str]
    targeted_sectors: List[str]
    global_threat_level: str
    confidence_level: int
    fresh_intel_count: int

class PatriotsCyberIntel:
    """Patriots Protocol Cyber Intelligence System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session = None
        
        # Intelligence sources - premium cyber security feeds
        self.intel_sources = [
            {
                'name': 'CISA_ADVISORIES',
                'url': 'https://www.cisa.gov/cybersecurity-advisories/rss.xml',
                'credibility': 0.98,
                'priority': 1
            },
            {
                'name': 'MITRE_ATTACK',
                'url': 'https://attack.mitre.org/resources/rss.xml',
                'credibility': 0.96,
                'priority': 1
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'credibility': 0.94,
                'priority': 1
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'credibility': 0.93,
                'priority': 1
            },
            {
                'name': 'CERT_EU',
                'url': 'https://cert.europa.eu/en/publications/security-advisories',
                'credibility': 0.92,
                'priority': 2
            },
            {
                'name': 'DARK_READING',
                'url': 'https://www.darkreading.com/rss_simple.asp',
                'credibility': 0.89,
                'priority': 2
            },
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'credibility': 0.87,
                'priority': 2
            },
            {
                'name': 'SECURITY_WEEK',
                'url': 'https://www.securityweek.com/feed',
                'credibility': 0.86,
                'priority': 3
            }
        ]
        
        self.data_dir = Path('./data')
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info("🎖️  Patriots Protocol Cyber Intelligence Engine v4.0 Initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-Intel/v4.0',
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
            logger.warning("⚠️  No API token - using fallback analysis")
            return self._fallback_analysis(articles_batch)

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Prepare batch content for analysis
            batch_content = ""
            for i, article in enumerate(articles_batch):
                batch_content += f"\n--- THREAT INTEL {i+1} ---\n"
                batch_content += f"Title: {article['title']}\n"
                batch_content += f"Content: {article['summary'][:500]}\n"

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an elite cyber threat intelligence analyst. Analyze these cyber security reports and return detailed JSON with this structure:
{
    "threat_analyses": [
        {
            "article_index": 0,
            "threat_level": "CRITICAL/HIGH/MEDIUM/LOW",
            "severity_score": 8,
            "confidence": 0.92,
            "ai_analysis": "Detailed threat analysis with actionable insights (150-200 words)",
            "attack_vectors": ["malware", "phishing", "social_engineering"],
            "affected_sectors": ["healthcare", "finance", "government"],
            "keywords": ["apt", "ransomware", "zero-day"],
            "geolocation": "Global/US/EU/APAC",
            "iocs": ["hash", "domain", "ip"],
            "threat_actors": ["lazarus", "apt29", "unknown"]
        }
    ]
}

Focus on actionable threat intelligence. Identify attack techniques, threat actors, IOCs, and provide strategic recommendations."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these cyber threat intelligence reports:\n{batch_content}"
                    }
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response
            try:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = ai_response[json_start:json_end]
                    results = json.loads(json_content)
                    return results.get('threat_analyses', [])
            
            except json.JSONDecodeError:
                logger.warning("⚠️  AI response parsing failed - using fallback")
                return self._fallback_analysis(articles_batch)
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {str(e)}")
            return self._fallback_analysis(articles_batch)

    def _fallback_analysis(self, articles: List[Dict]) -> List[Dict]:
        """Fallback threat analysis when AI is unavailable"""
        analyses = []
        
        for i, article in enumerate(articles):
            # Determine threat level based on keywords
            content = (article['title'] + ' ' + article['summary']).lower()
            
            threat_level = "LOW"
            severity_score = 3
            
            critical_indicators = ['zero-day', 'critical', 'emergency', 'nation-state', 'apt']
            high_indicators = ['exploit', 'vulnerability', 'breach', 'ransomware', 'malware']
            medium_indicators = ['phishing', 'scam', 'leak', 'exposed']
            
            if any(indicator in content for indicator in critical_indicators):
                threat_level = "CRITICAL"
                severity_score = 9
            elif any(indicator in content for indicator in high_indicators):
                threat_level = "HIGH"
                severity_score = 7
            elif any(indicator in content for indicator in medium_indicators):
                threat_level = "MEDIUM"
                severity_score = 5

            # Extract attack vectors
            attack_vectors = []
            if 'malware' in content: attack_vectors.append('malware')
            if 'phishing' in content: attack_vectors.append('phishing')
            if 'ransomware' in content: attack_vectors.append('ransomware')
            if 'ddos' in content: attack_vectors.append('ddos')
            
            # Extract affected sectors
            sectors = []
            if any(word in content for word in ['hospital', 'healthcare', 'medical']): sectors.append('healthcare')
            if any(word in content for word in ['bank', 'financial', 'finance']): sectors.append('finance')
            if any(word in content for word in ['government', 'federal', 'agency']): sectors.append('government')
            if any(word in content for word in ['critical infrastructure', 'utility', 'energy']): sectors.append('critical_infrastructure')

            analyses.append({
                'article_index': i,
                'threat_level': threat_level,
                'severity_score': severity_score,
                'confidence': 0.75,
                'ai_analysis': f"Automated threat assessment indicates {threat_level.lower()} priority cyber security incident. Monitoring required for potential impact assessment and threat evolution.",
                'attack_vectors': attack_vectors or ['unknown'],
                'affected_sectors': sectors or ['multiple'],
                'keywords': self._extract_keywords(content),
                'geolocation': 'Global',
                'iocs': [],
                'threat_actors': ['unknown']
            })
        
        return analyses

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract cyber security keywords"""
        cyber_keywords = [
            'apt', 'malware', 'ransomware', 'phishing', 'exploit', 'vulnerability',
            'breach', 'leak', 'hack', 'attack', 'threat', 'cyber', 'security',
            'zero-day', 'trojan', 'botnet', 'ddos', 'spear-phishing'
        ]
        
        found_keywords = []
        for keyword in cyber_keywords:
            if keyword in content.lower():
                found_keywords.append(keyword.upper())
        
        return found_keywords[:8]  # Limit to top 8 keywords

    async def fetch_cyber_intelligence(self) -> List[Dict]:
        """Fetch real-time cyber threat intelligence"""
        all_intel = []
        
        for source in self.intel_sources:
            try:
                logger.info(f"🔍 Fetching from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:8]:  # Top 8 per source
                            title = entry.title
                            summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                            
                            if len(summary) < 50:
                                continue
                            
                            # Filter for cyber security content
                            if not self._is_cyber_content(title + ' ' + summary):
                                continue
                            
                            intel = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'credibility': source['credibility']
                            }
                            
                            all_intel.append(intel)
                            
                    else:
                        logger.warning(f"⚠️  {source['name']} returned {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Error fetching {source['name']}: {str(e)}")
                continue

        logger.info(f"📊 Collected {len(all_intel)} threat intelligence reports")
        return all_intel

    def _is_cyber_content(self, content: str) -> bool:
        """Filter for relevant cyber security content"""
        cyber_indicators = [
            'security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability',
            'attack', 'threat', 'ransomware', 'phishing', 'exploit', 'apt',
            'zero-day', 'incident', 'compromise', 'botnet', 'trojan'
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
                    
                    # Create threat intelligence report
                    report = ThreatIntelligence(
                        title=intel['title'],
                        summary=intel['summary'][:600],  # Limit summary length
                        source=intel['source'],
                        source_url=intel.get('source_url', ''),
                        timestamp=intel['timestamp'],
                        threat_level=ai_analysis.get('threat_level', 'MEDIUM'),
                        ai_analysis=ai_analysis.get('ai_analysis', 'Threat analysis pending'),
                        confidence=ai_analysis.get('confidence', 0.8),
                        keywords=ai_analysis.get('keywords', []),
                        attack_vectors=ai_analysis.get('attack_vectors', []),
                        affected_sectors=ai_analysis.get('affected_sectors', []),
                        severity_score=ai_analysis.get('severity_score', 5),
                        geolocation=ai_analysis.get('geolocation', 'Global'),
                        iocs=ai_analysis.get('iocs', [])
                    )
                    
                    threat_reports.append(report)
                    logger.info(f"✅ Processed: {report.title[:50]}... (Threat: {report.threat_level})")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing intel: {str(e)}")
                    continue
            
            # Rate limiting
            await asyncio.sleep(2.0)

        return threat_reports

    def calculate_threat_metrics(self, reports: List[ThreatIntelligence]) -> ThreatMetrics:
        """Calculate comprehensive threat landscape metrics"""
        if not reports:
            return ThreatMetrics(
                total_threats=0,
                critical_threats=0,
                high_priority_threats=0,
                active_campaigns=0,
                threat_actors=[],
                attack_techniques=[],
                targeted_sectors=[],
                global_threat_level="LOW",
                confidence_level=85,
                fresh_intel_count=0
            )

        # Count threats by level
        critical_count = len([r for r in reports if r.threat_level == 'CRITICAL'])
        high_count = len([r for r in reports if r.threat_level == 'HIGH'])
        
        # Determine global threat level
        if critical_count > 0:
            global_threat = "CRITICAL"
        elif high_count > 2:
            global_threat = "HIGH"
        elif high_count > 0:
            global_threat = "MEDIUM"
        else:
            global_threat = "LOW"

        # Extract attack techniques
        all_vectors = []
        for report in reports:
            all_vectors.extend(report.attack_vectors)
        
        technique_freq = {}
        for vector in all_vectors:
            technique_freq[vector] = technique_freq.get(vector, 0) + 1
        
        top_techniques = [t for t, _ in sorted(technique_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

        # Extract targeted sectors
        all_sectors = []
        for report in reports:
            all_sectors.extend(report.affected_sectors)
        
        sector_freq = {}
        for sector in all_sectors:
            sector_freq[sector] = sector_freq.get(sector, 0) + 1
        
        top_sectors = [s for s, _ in sorted(sector_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

        # Check for fresh intelligence (last 24 hours)
        fresh_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report.timestamp.replace('Z', '+00:00'))
                if report_time > cutoff:
                    fresh_count += 1
            except:
                pass

        return ThreatMetrics(
            total_threats=len(reports),
            critical_threats=critical_count,
            high_priority_threats=high_count,
            active_campaigns=len(set(all_vectors)),  # Unique attack vectors as campaigns
            threat_actors=[],  # Will be populated from reports
            attack_techniques=top_techniques,
            targeted_sectors=top_sectors,
            global_threat_level=global_threat,
            confidence_level=int(sum(r.confidence for r in reports) / len(reports) * 100),
            fresh_intel_count=fresh_count
        )

async def main():
    """Execute Patriots Protocol Cyber Intelligence Mission"""
    logger.info("🎖️  PATRIOTS PROTOCOL - Cyber Intelligence Mission Starting...")
    
    try:
        async with PatriotsCyberIntel() as intel_system:
            # Fetch real-time intelligence
            intel_data = await intel_system.fetch_cyber_intelligence()
            
            # Process into threat intelligence
            threat_reports = await intel_system.process_intelligence(intel_data)
            
            # Calculate threat metrics
            metrics = intel_system.calculate_threat_metrics(threat_reports)
            
            # Prepare output
            output_data = {
                "articles": [asdict(report) for report in threat_reports],
                "metrics": asdict(metrics),
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "version": "4.0",
                "patriots_protocol_info": {
                    "system_name": "PATRIOTS PROTOCOL",
                    "description": "CYBER INTELLIGENCE NETWORK",
                    "repository": "https://github.com/danishnizmi/Patriots_Protocol",
                    "status": "OPERATIONAL"
                }
            }

            # Save intelligence data
            with open('./data/news-analysis.json', 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info("✅ Patriots Protocol Intelligence Mission Complete")
            logger.info(f"📊 Threat Reports Generated: {len(threat_reports)}")
            logger.info(f"🔥 Global Threat Level: {metrics.global_threat_level}")
            logger.info(f"⚠️  Critical Threats: {metrics.critical_threats}")
            
    except Exception as e:
        logger.error(f"❌ Mission error: {str(e)}")
        
        # Generate minimal fallback data
        fallback_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0,
                "critical_threats": 0,
                "global_threat_level": "LOW",
                "confidence_level": 85
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0"
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(fallback_data, f)

if __name__ == "__main__":
    asyncio.run(main())
