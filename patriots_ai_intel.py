#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Cyber Threat Intelligence Engine
Real-time threat monitoring and analysis system
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
    format='%(asctime)s - PATRIOTS - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    confidence_score: float
    risk_score: int
    threat_family: str
    geographic_scope: str
    country_code: str
    cve_references: List[str]
    attack_vectors: List[str]
    affected_sectors: List[str]
    mitigation_priority: str

@dataclass
class ThreatMetrics:
    """Threat Intelligence Metrics"""
    total_threats: int
    critical_threats: int
    high_threats: int
    medium_threats: int
    low_threats: int
    global_threat_level: str
    confidence_level: int
    zero_day_count: int
    geographic_distribution: Dict[str, int]
    top_threat_families: List[Dict[str, Any]]

class ThreatIntelligence:
    """Cyber Threat Intelligence Collection and Analysis"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4o-mini"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Intelligence sources
        self.sources = [
            {
                'name': 'BLEEPING_COMPUTER',
                'url': 'https://www.bleepingcomputer.com/feed/',
                'reliability': 0.9
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'reliability': 0.95
            },
            {
                'name': 'SANS_ISC',
                'url': 'https://isc.sans.edu/rssfeed.xml',
                'reliability': 0.9
            },
            {
                'name': 'THREAT_POST',
                'url': 'https://threatpost.com/feed/',
                'reliability': 0.85
            },
            {
                'name': 'CYBER_SCOOP',
                'url': 'https://www.cyberscoop.com/feed/',
                'reliability': 0.8
            }
        ]
        
        self.country_codes = {
            'united states': 'US', 'usa': 'US', 'america': 'US',
            'canada': 'CA', 'china': 'CN', 'japan': 'JP',
            'australia': 'AU', 'germany': 'DE', 'france': 'FR',
            'united kingdom': 'GB', 'uk': 'GB', 'russia': 'RU',
            'global': 'GLOBAL', 'worldwide': 'GLOBAL'
        }
        
        self.data_dir = Path('./data')
        self.data_dir.mkdir(exist_ok=True)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Patriots-Protocol/1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def analyze_with_ai(self, title: str, content: str) -> Dict[str, Any]:
        """AI analysis for threat intelligence"""
        if not self.api_token:
            return self.basic_analysis(title, content)

        try:
            prompt = f"""Analyze this cybersecurity threat and provide structured intelligence:

TITLE: {title}
CONTENT: {content[:1000]}

Return JSON with:
{{
    "threat_assessment": {{
        "severity": "CRITICAL/HIGH/MEDIUM/LOW",
        "risk_score": 1-10,
        "confidence": 0.1-1.0
    }},
    "classification": {{
        "family": "Specific threat type",
        "attack_vectors": ["list of attack methods"],
        "affected_sectors": ["industries mentioned"],
        "geographic_impact": ["countries/regions"]
    }},
    "intelligence": {{
        "cve_references": ["CVE numbers if mentioned"],
        "key_indicators": ["important technical details"],
        "business_impact": "Brief impact assessment"
    }}
}}

Focus on factual analysis only."""

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a cybersecurity analyst. Provide factual threat intelligence without speculation."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }

            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }

            async with self.session.post(f"{self.base_url}/chat/completions", 
                                       headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    ai_response = result['choices'][0]['message']['content']
                    
                    # Extract JSON
                    json_start = ai_response.find('{')
                    json_end = ai_response.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = ai_response[json_start:json_end]
                        return json.loads(json_content)
                        
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            
        return self.basic_analysis(title, content)

    def basic_analysis(self, title: str, content: str) -> Dict[str, Any]:
        """Basic threat analysis without AI"""
        full_text = (title + ' ' + content).lower()
        
        # Threat family detection
        threat_families = {
            'Ransomware': ['ransomware', 'encryption', 'ransom', 'lockbit'],
            'Zero-Day Exploit': ['zero-day', '0-day', 'unknown vulnerability'],
            'Data Breach': ['data breach', 'breach', 'stolen data', 'leaked'],
            'Vulnerability': ['vulnerability', 'patch', 'security update'],
            'Malware': ['malware', 'trojan', 'backdoor', 'virus'],
            'Phishing': ['phishing', 'email attack', 'social engineering'],
            'APT': ['apt', 'nation-state', 'targeted attack']
        }
        
        detected_family = 'Security Incident'
        for family, keywords in threat_families.items():
            if any(keyword in full_text for keyword in keywords):
                detected_family = family
                break
        
        # Risk scoring
        risk_score = 3
        if any(word in full_text for word in ['critical', 'zero-day', 'rce']):
            risk_score = 8
        elif any(word in full_text for word in ['high', 'ransomware', 'breach']):
            risk_score = 6
        elif any(word in full_text for word in ['medium', 'vulnerability']):
            risk_score = 4
        
        # Severity mapping
        if risk_score >= 7:
            severity = 'CRITICAL'
        elif risk_score >= 5:
            severity = 'HIGH'
        elif risk_score >= 3:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        return {
            'threat_assessment': {
                'severity': severity,
                'risk_score': risk_score,
                'confidence': 0.7
            },
            'classification': {
                'family': detected_family,
                'attack_vectors': self.extract_attack_vectors(full_text),
                'affected_sectors': self.extract_sectors(full_text),
                'geographic_impact': self.extract_geography(full_text)
            },
            'intelligence': {
                'cve_references': re.findall(r'CVE-\d{4}-\d{4,7}', content.upper()),
                'key_indicators': [],
                'business_impact': 'Standard monitoring required'
            }
        }

    def extract_attack_vectors(self, content: str) -> List[str]:
        """Extract attack vectors from content"""
        vectors = []
        vector_map = {
            'email': ['email', 'phishing', 'attachment'],
            'web': ['web', 'browser', 'website'],
            'network': ['network', 'remote', 'lateral'],
            'supply_chain': ['supply chain', 'third-party']
        }
        
        for vector, keywords in vector_map.items():
            if any(keyword in content for keyword in keywords):
                vectors.append(vector)
        
        return vectors[:3]

    def extract_sectors(self, content: str) -> List[str]:
        """Extract affected sectors"""
        sectors = []
        sector_map = {
            'healthcare': ['hospital', 'medical', 'health'],
            'financial': ['bank', 'finance', 'payment'],
            'government': ['government', 'federal', 'agency'],
            'technology': ['tech', 'software', 'cloud'],
            'education': ['university', 'school', 'education']
        }
        
        for sector, keywords in sector_map.items():
            if any(keyword in content for keyword in keywords):
                sectors.append(sector)
        
        return sectors[:2]

    def extract_geography(self, content: str) -> List[str]:
        """Extract geographic scope"""
        geography = []
        for location, code in self.country_codes.items():
            if location in content:
                geography.append(location.title())
                if len(geography) >= 2:
                    break
        
        return geography or ['Global']

    async def collect_feeds(self) -> List[Dict]:
        """Collect threat intelligence from feeds"""
        all_items = []
        
        for source in self.sources:
            try:
                logger.info(f"Collecting from {source['name']}")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        feed_content = await response.text()
                        parsed_feed = feedparser.parse(feed_content)
                        
                        for entry in parsed_feed.entries[:10]:
                            title = entry.title.strip()
                            summary = entry.get('summary', '').strip()
                            
                            # Filter for cybersecurity relevance
                            content = (title + ' ' + summary).lower()
                            cyber_keywords = [
                                'security', 'cyber', 'hack', 'breach', 'malware',
                                'vulnerability', 'attack', 'threat', 'exploit'
                            ]
                            
                            if any(keyword in content for keyword in cyber_keywords):
                                all_items.append({
                                    'title': title,
                                    'summary': summary,
                                    'source': source['name'],
                                    'source_url': entry.get('link', ''),
                                    'timestamp': entry.get('published', ''),
                                    'reliability': source['reliability']
                                })
                        
                        logger.info(f"Collected {len([i for i in all_items if i['source'] == source['name']])} items from {source['name']}")
                        
            except Exception as e:
                logger.error(f"Error collecting from {source['name']}: {e}")
                
            await asyncio.sleep(0.2)  # Rate limiting
        
        return self.deduplicate(all_items)

    def deduplicate(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items"""
        unique_items = []
        seen_titles = set()
        
        for item in items:
            title_hash = hashlib.md5(item['title'].lower().encode()).hexdigest()[:10]
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_items.append(item)
        
        logger.info(f"Deduplicated: {len(items)} -> {len(unique_items)} unique items")
        return unique_items[:20]  # Limit to 20 items

    async def process_threats(self, raw_items: List[Dict]) -> List[ThreatReport]:
        """Process raw items into threat reports"""
        threat_reports = []
        
        for item in raw_items:
            try:
                # Get AI analysis
                analysis = await self.analyze_with_ai(item['title'], item['summary'])
                
                # Extract data from analysis
                threat_assessment = analysis.get('threat_assessment', {})
                classification = analysis.get('classification', {})
                intelligence = analysis.get('intelligence', {})
                
                # Geographic processing
                geo_impact = classification.get('geographic_impact', ['Global'])
                country_code = 'GLOBAL'
                if geo_impact:
                    country_code = self.country_codes.get(geo_impact[0].lower(), 'GLOBAL')
                
                # Create threat report
                report = ThreatReport(
                    title=item['title'],
                    summary=item['summary'][:500],  # Limit summary length
                    source=item['source'],
                    source_url=item['source_url'],
                    timestamp=item['timestamp'] or datetime.now(timezone.utc).isoformat(),
                    threat_level=threat_assessment.get('severity', 'MEDIUM'),
                    confidence_score=threat_assessment.get('confidence', 0.7),
                    risk_score=threat_assessment.get('risk_score', 5),
                    threat_family=classification.get('family', 'Unknown'),
                    geographic_scope=', '.join(geo_impact[:2]),
                    country_code=country_code,
                    cve_references=intelligence.get('cve_references', []),
                    attack_vectors=classification.get('attack_vectors', []),
                    affected_sectors=classification.get('affected_sectors', []),
                    mitigation_priority=threat_assessment.get('severity', 'MEDIUM')
                )
                
                threat_reports.append(report)
                logger.info(f"Processed: {report.title[:50]}... ({report.threat_level})")
                
            except Exception as e:
                logger.error(f"Error processing threat: {e}")
                continue
        
        return sorted(threat_reports, key=lambda x: x.risk_score, reverse=True)

    def calculate_metrics(self, reports: List[ThreatReport]) -> ThreatMetrics:
        """Calculate threat intelligence metrics"""
        if not reports:
            return ThreatMetrics(
                total_threats=0, critical_threats=0, high_threats=0,
                medium_threats=0, low_threats=0, global_threat_level="MONITORING",
                confidence_level=0, zero_day_count=0, geographic_distribution={},
                top_threat_families=[]
            )

        # Count by threat level
        level_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            level = report.threat_level.upper()
            if level in level_counts:
                level_counts[level] += 1

        # Global threat level
        if level_counts['CRITICAL'] >= 2:
            global_level = "CRITICAL"
        elif level_counts['CRITICAL'] >= 1 or level_counts['HIGH'] >= 3:
            global_level = "HIGH"
        else:
            global_level = "MEDIUM"

        # Geographic distribution
        geo_dist = {}
        for report in reports:
            country = report.country_code
            geo_dist[country] = geo_dist.get(country, 0) + 1

        # Threat families
        family_counts = {}
        for report in reports:
            family = report.threat_family
            if family not in family_counts:
                family_counts[family] = {'count': 0, 'avg_risk': 0, 'total_risk': 0}
            family_counts[family]['count'] += 1
            family_counts[family]['total_risk'] += report.risk_score

        top_families = []
        for family, data in sorted(family_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
            avg_risk = data['total_risk'] / data['count']
            top_families.append({
                "name": family,
                "count": data['count'],
                "avg_risk": round(avg_risk, 1)
            })

        # Zero-day count
        zero_day_count = sum(1 for r in reports if 'zero' in r.threat_family.lower())

        # Average confidence
        avg_confidence = sum(r.confidence_score for r in reports) / len(reports)

        return ThreatMetrics(
            total_threats=len(reports),
            critical_threats=level_counts['CRITICAL'],
            high_threats=level_counts['HIGH'],
            medium_threats=level_counts['MEDIUM'],
            low_threats=level_counts['LOW'],
            global_threat_level=global_level,
            confidence_level=int(avg_confidence * 100),
            zero_day_count=zero_day_count,
            geographic_distribution=geo_dist,
            top_threat_families=top_families
        )

    def save_data(self, reports: List[ThreatReport], metrics: ThreatMetrics):
        """Save threat intelligence data"""
        output_data = {
            "articles": [asdict(report) for report in reports],
            "metrics": asdict(metrics),
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "summary": {
                "status": "OPERATIONAL",
                "threats_analyzed": len(reports),
                "confidence_level": metrics.confidence_level,
                "threat_level": metrics.global_threat_level
            }
        }

        output_file = self.data_dir / 'news-analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(reports)} threat reports")
        logger.info(f"Global threat level: {metrics.global_threat_level}")

async def run_intelligence_collection():
    """Main intelligence collection function"""
    logger.info("Starting Patriots Protocol Threat Intelligence")
    
    try:
        async with ThreatIntelligence() as intel:
            # Collect raw feeds
            raw_items = await intel.collect_feeds()
            
            if not raw_items:
                logger.warning("No threat data collected")
                return
            
            # Process into threat reports
            threat_reports = await intel.process_threats(raw_items)
            
            if not threat_reports:
                logger.warning("No threats processed")
                return
            
            # Calculate metrics
            metrics = intel.calculate_metrics(threat_reports)
            
            # Save data
            intel.save_data(threat_reports, metrics)
            
            # Summary
            logger.info("Intelligence collection complete")
            logger.info(f"Threats: {len(threat_reports)}")
            logger.info(f"Critical: {metrics.critical_threats}")
            logger.info(f"High: {metrics.high_threats}")
            logger.info(f"Zero-day: {metrics.zero_day_count}")
            
    except Exception as e:
        logger.error(f"Intelligence collection failed: {e}")
        
        # Create minimal error data
        error_data = {
            "articles": [],
            "metrics": {
                "total_threats": 0,
                "critical_threats": 0,
                "high_threats": 0,
                "medium_threats": 0,
                "low_threats": 0,
                "global_threat_level": "OFFLINE",
                "confidence_level": 0,
                "zero_day_count": 0,
                "geographic_distribution": {},
                "top_threat_families": []
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "summary": {"status": "ERROR", "threats_analyzed": 0}
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_intelligence_collection())
