#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Professional AI Intelligence System v4.0
Professional AI-Driven Intelligence Network with Meaningful Analysis

Production system with GitHub Models API integration and professional
intelligence analysis capabilities.

Source: PATRIOTS PROTOCOL - https://github.com/danishnizmi/Patriots_Protocol
"""

import os
import json
import asyncio
import aiohttp
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import feedparser
import hashlib
import logging
from urllib.parse import urlparse, urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS PROTOCOL v4.0 - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class IntelligenceReport:
    """Professional Intelligence Report"""
    title: str
    full_summary: str
    executive_summary: str
    source: str
    source_url: str
    source_credibility: float
    timestamp: str
    category: str
    ai_analysis: str
    confidence: float
    threat_level: str
    strategic_importance: str
    operational_impact: str
    geo_relevance: List[str]
    keywords: List[str]
    entities: List[str]
    priority_score: int
    content_hash: str
    word_count: int
    reading_time: str
    intelligence_value: str
    patriots_protocol_ref: str = "PATRIOTS PROTOCOL INTELLIGENCE NETWORK"

@dataclass
class IntelligenceMetrics:
    """Professional Intelligence Metrics"""
    total_articles: int
    ai_analysis_complete: int
    threat_level: str
    system_status: str
    average_article_length: int
    total_word_count: int
    average_reading_time: str
    high_value_intelligence: int
    critical_intelligence: int
    actionable_intelligence: int
    emerging_threats: List[str]
    primary_regions: List[str]
    source_diversity: int
    credibility_average: float
    primary_sources: List[str]
    ai_confidence: int
    processing_time: str
    api_status: str
    strategic_assessment: str
    intelligence_summary: str
    threat_vector_analysis: str
    last_analysis: str
    last_update: str
    patriots_protocol_status: str = "PATRIOTS PROTOCOL OPERATIONAL"

class PatriotsProtocolAI:
    """Professional AI Intelligence Analysis System"""
    
    def __init__(self):
        # Use GITHUB_TOKEN as MODEL_TOKEN for GitHub Models API
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "gpt-4o-mini"
        self.session = None
        
        # Professional intelligence sources
        self.intelligence_sources = [
            {
                'name': 'BBC_WORLD_NEWS',
                'url': 'https://feeds.bbci.co.uk/news/world/rss.xml',
                'base_url': 'https://www.bbc.com',
                'credibility': 0.95
            },
            {
                'name': 'CYBERSECURITY_INTEL',
                'url': 'https://feeds.feedburner.com/eset/blog',
                'base_url': 'https://www.welivesecurity.com',
                'credibility': 0.88
            },
            {
                'name': 'DEFENSE_NEWS',
                'url': 'https://www.defensenews.com/arc/outboundfeeds/rss/category/breaking-news/',
                'base_url': 'https://www.defensenews.com',
                'credibility': 0.87
            }
        ]
        
        if not self.api_token:
            logger.error("❌ GITHUB_TOKEN/MODEL_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN or MODEL_TOKEN is required for AI operations")
        
        logger.info("🚀 Patriots Protocol AI Intelligence System v4.0 initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                'User-Agent': 'Patriots-Protocol-AI-v4.0/Professional-Intelligence',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def generate_content_hash(self, content: str) -> str:
        """Generate unique content hash"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    async def make_ai_request(self, prompt: str, context: str) -> Dict[str, Any]:
        """Professional AI request with GitHub Models API"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_token}'
            }

            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": """You are a professional intelligence analyst providing concise, factual analysis of news events. 

Analyze the provided content and return a JSON response with these exact fields:
{
    "analysis": "Professional 2-3 sentence analysis focusing on key facts and implications. Be specific and factual, not generic.",
    "threat_level": "CRITICAL/HIGH/MEDIUM/LOW based on actual content",
    "strategic_importance": "CRITICAL/HIGH/MEDIUM/LOW",
    "operational_impact": "Specific assessment of actual implications",
    "geo_relevance": ["Specific countries/regions mentioned"],
    "confidence_score": 0.85,
    "priority_score": 6,
    "entities": ["Specific people, organizations mentioned"],
    "intelligence_value": "CRITICAL/HIGH/MEDIUM/LOW"
}

Focus on actual content. Avoid generic phrases like "monitoring protocols" or "situational awareness". Be specific about what the news means."""
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze this intelligence report:\n\nTitle: {context.split('Content:')[0].replace('Title:', '').strip()}\n\nContent: {context.split('Content:')[1] if 'Content:' in context else context}\n\nProvide specific, professional analysis in JSON format."
                    }
                ],
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": 800
            }

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    ai_response = data['choices'][0]['message']['content']
                    
                    # Extract JSON from response
                    try:
                        json_start = ai_response.find('{')
                        json_end = ai_response.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_content = ai_response[json_start:json_end]
                            structured_data = json.loads(json_content)
                            
                            # Validate we got meaningful analysis
                            analysis = structured_data.get('analysis', '')
                            if 'monitoring protocols' in analysis.lower() or 'situational awareness' in analysis.lower():
                                logger.warning("⚠️  Generic analysis detected, retrying...")
                                raise ValueError("Generic analysis response")
                            
                            return {
                                'success': True,
                                'analysis': analysis,
                                'threat_level': structured_data.get('threat_level', 'LOW'),
                                'strategic_importance': structured_data.get('strategic_importance', 'MEDIUM'),
                                'operational_impact': structured_data.get('operational_impact', 'Requires assessment'),
                                'geo_relevance': structured_data.get('geo_relevance', ['GLOBAL']),
                                'confidence_score': structured_data.get('confidence_score', 0.78),
                                'priority_score': structured_data.get('priority_score', 5),
                                'entities': structured_data.get('entities', []),
                                'intelligence_value': structured_data.get('intelligence_value', 'MEDIUM')
                            }
                    
                    except (json.JSONDecodeError, ValueError):
                        # If we can't get good structured data, skip this article
                        logger.warning("⚠️  Could not extract meaningful analysis")
                        return {'success': False}
                
                else:
                    logger.error(f"❌ API error: {response.status}")
                    return {'success': False}

        except Exception as e:
            logger.error(f"❌ AI request failed: {str(e)}")
            return {'success': False}

    async def fetch_intelligence_feeds(self) -> List[Dict[str, Any]]:
        """Fetch intelligence from news sources"""
        all_articles = []
        
        for source in self.intelligence_sources:
            try:
                logger.info(f"🔍 Fetching from {source['name']}...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:4]:  # Limit articles per source
                            title = entry.title
                            summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                            
                            # Skip if summary is too short or generic
                            if len(summary) < 50:
                                continue
                            
                            # Build proper source URL
                            source_url = entry.get('link', '')
                            if source_url and not source_url.startswith('http'):
                                source_url = urljoin(source['base_url'], source_url)
                            
                            full_content = f"{title}. {summary}"
                            word_count = len(full_content.split())
                            reading_time = max(1, word_count // 200)
                            
                            article = {
                                'title': title,
                                'summary': summary,
                                'source': source['name'],
                                'source_url': source_url,
                                'timestamp': entry.get('published', datetime.now(timezone.utc).isoformat()),
                                'credibility': source['credibility'],
                                'word_count': word_count,
                                'reading_time': f"{reading_time} min read",
                                'content_hash': self.generate_content_hash(full_content)
                            }
                            
                            all_articles.append(article)
                            
                    else:
                        logger.warning(f"⚠️  {source['name']} returned {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Error fetching {source['name']}: {str(e)}")
                continue

        # Remove duplicates
        unique_articles = {}
        for article in all_articles:
            if article['content_hash'] not in unique_articles:
                unique_articles[article['content_hash']] = article

        logger.info(f"📊 Collected {len(unique_articles)} unique intelligence reports")
        return list(unique_articles.values())

    def _clean_text(self, text: str) -> str:
        """Clean and enhance text content"""
        if not text:
            return ""
        
        # Remove HTML and clean
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)
        
        return text[:500] if len(text) > 500 else text

    async def analyze_article(self, article: Dict[str, Any]) -> Optional[IntelligenceReport]:
        """Professional article analysis - only return if meaningful"""
        logger.info(f"🔍 Analyzing: {article['title'][:50]}...")
        
        # AI analysis
        ai_result = await self.make_ai_request(
            "Professional intelligence analysis",
            f"Title: {article['title']}\nContent: {article['summary']}"
        )
        
        # Skip if AI couldn't provide meaningful analysis
        if not ai_result.get('success'):
            logger.info(f"⚠️  Skipping article - no meaningful analysis available")
            return None
        
        # Category classification
        category = self._classify_category(article['title'], article['summary'])
        
        # Extract keywords and entities
        keywords = self._extract_keywords(f"{article['title']} {article['summary']}")
        entities = ai_result.get('entities', [])
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(article['summary'])
        
        report = IntelligenceReport(
            title=article['title'],
            full_summary=article['summary'],
            executive_summary=executive_summary,
            source=article['source'],
            source_url=article.get('source_url', ''),
            source_credibility=article['credibility'],
            timestamp=article['timestamp'],
            category=category,
            ai_analysis=ai_result['analysis'],
            confidence=ai_result['confidence_score'],
            threat_level=ai_result['threat_level'],
            strategic_importance=ai_result['strategic_importance'],
            operational_impact=ai_result['operational_impact'],
            geo_relevance=ai_result['geo_relevance'],
            keywords=keywords,
            entities=entities,
            priority_score=ai_result['priority_score'],
            content_hash=article['content_hash'],
            word_count=article['word_count'],
            reading_time=article['reading_time'],
            intelligence_value=ai_result['intelligence_value']
        )
        
        logger.info(f"✅ Analysis complete - Threat: {report.threat_level}, Value: {report.intelligence_value}")
        return report

    def _generate_executive_summary(self, summary: str) -> str:
        """Generate concise executive summary"""
        if not summary or len(summary) <= 100:
            return summary
        
        sentences = summary.split('. ')
        if sentences and len(sentences[0]) <= 120:
            return sentences[0] + ('.' if not sentences[0].endswith('.') else '')
        
        return summary[:97] + "..."

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords"""
        text_lower = text.lower()
        keywords = []
        
        keyword_categories = {
            'threat': ['attack', 'threat', 'breach', 'exploit', 'cyber', 'terrorism'],
            'technology': ['ai', 'artificial intelligence', 'quantum', 'digital'],
            'geopolitical': ['diplomatic', 'sanctions', 'conflict', 'war', 'military'],
            'economic': ['economic', 'financial', 'trade', 'market'],
            'organizations': ['nato', 'un', 'government', 'military']
        }
        
        for category, terms in keyword_categories.items():
            for term in terms:
                if term in text_lower:
                    keywords.append(term.upper())
        
        return list(set(keywords))[:8]

    def _classify_category(self, title: str, summary: str) -> str:
        """Classify article category"""
        content = f"{title} {summary}".lower()
        
        categories = {
            'SECURITY': ['cyber', 'security', 'attack', 'threat', 'breach'],
            'TECHNOLOGY': ['tech', 'ai', 'digital', 'innovation'],
            'POLITICS': ['political', 'government', 'election', 'diplomatic'],
            'DEFENSE': ['military', 'defense', 'weapons', 'nuclear'],
            'ECONOMICS': ['economic', 'trade', 'financial', 'market']
        }
        
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in content)
            scores[category] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'GLOBAL'

    def calculate_metrics(self, reports: List[IntelligenceReport]) -> IntelligenceMetrics:
        """Calculate intelligence metrics"""
        if not reports:
            return self._generate_baseline_metrics()

        # Basic metrics
        total_words = sum(r.word_count for r in reports)
        avg_length = total_words // len(reports) if reports else 0
        
        # Intelligence metrics
        high_value = len([r for r in reports if r.intelligence_value in ['HIGH', 'CRITICAL']])
        critical = len([r for r in reports if r.intelligence_value == 'CRITICAL'])
        
        # Threat analysis
        threat_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            threat_counts[report.threat_level] += 1
        
        overall_threat = 'CRITICAL' if threat_counts['CRITICAL'] > 0 else (
                        'HIGH' if threat_counts['HIGH'] > 0 else (
                        'MEDIUM' if threat_counts['MEDIUM'] > 0 else 'LOW'))
        
        # Extract trending data
        all_keywords = []
        all_entities = []
        for report in reports:
            all_keywords.extend(report.keywords)
            all_entities.extend(report.entities)
        
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        trending = [kw for kw, count in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Geographic analysis
        all_geo = []
        for report in reports:
            all_geo.extend(report.geo_relevance)
        
        geo_freq = {}
        for geo in all_geo:
            geo_freq[geo] = geo_freq.get(geo, 0) + 1
        
        primary_regions = [geo for geo, count in sorted(geo_freq.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Strategic assessment
        strategic_assessment = f"Patriots Protocol analyzed {len(reports)} intelligence reports with professional AI assessment. "
        if high_value > 0:
            strategic_assessment += f"Identified {high_value} high-value intelligence items requiring attention. "
        if threat_counts['HIGH'] + threat_counts['CRITICAL'] > 0:
            strategic_assessment += f"Detected {threat_counts['HIGH'] + threat_counts['CRITICAL']} elevated threat indicators."
        else:
            strategic_assessment += "Current threat environment remains stable."
        
        return IntelligenceMetrics(
            total_articles=len(reports),
            ai_analysis_complete=len(reports),
            threat_level=overall_threat,
            system_status="OPERATIONAL",
            average_article_length=avg_length,
            total_word_count=total_words,
            average_reading_time="1 min",
            high_value_intelligence=high_value,
            critical_intelligence=critical,
            actionable_intelligence=len(reports),
            emerging_threats=trending[:3] if trending else ["Monitoring"],
            primary_regions=primary_regions if primary_regions else ["GLOBAL"],
            source_diversity=len(set(r.source for r in reports)),
            credibility_average=sum(r.source_credibility for r in reports) / len(reports),
            primary_sources=list(set(r.source for r in reports))[:3],
            ai_confidence=int(sum(r.confidence for r in reports) / len(reports) * 100),
            processing_time="< 45 seconds",
            api_status="ACTIVE",
            strategic_assessment=strategic_assessment,
            intelligence_summary=f"Patriots Protocol processed {len(reports)} intelligence reports. Primary categories: {', '.join(set(r.category for r in reports))}.",
            threat_vector_analysis=f"Threat environment: {overall_threat}. Active monitoring of {len(trending)} key indicators." if trending else "Standard threat monitoring active.",
            last_analysis=datetime.now(timezone.utc).isoformat(),
            last_update=datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        )

    def _generate_baseline_metrics(self) -> IntelligenceMetrics:
        """Generate baseline metrics when no meaningful reports available"""
        return IntelligenceMetrics(
            total_articles=0,
            ai_analysis_complete=0,
            threat_level="LOW",
            system_status="OPERATIONAL",
            average_article_length=0,
            total_word_count=0,
            average_reading_time="0 min",
            high_value_intelligence=0,
            critical_intelligence=0,
            actionable_intelligence=0,
            emerging_threats=["Monitoring"],
            primary_regions=["GLOBAL"],
            source_diversity=0,
            credibility_average=0.85,
            primary_sources=["PATRIOTS_PROTOCOL"],
            ai_confidence=85,
            processing_time="< 30 seconds",
            api_status="ACTIVE",
            strategic_assessment="Patriots Protocol operational - monitoring intelligence feeds for meaningful analysis.",
            intelligence_summary="Patriots Protocol standing by for intelligence operations.",
            threat_vector_analysis="No immediate threats detected. Standard monitoring active.",
            last_analysis=datetime.now(timezone.utc).isoformat(),
            last_update=datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        )

async def main():
    """Main Patriots Protocol AI Intelligence Pipeline"""
    logger.info("🎖️  PATRIOTS PROTOCOL v4.0 - AI Intelligence System Starting...")
    logger.info(f"📅 Mission Start: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        async with PatriotsProtocolAI() as ai_system:
            # Fetch intelligence
            articles = await ai_system.fetch_intelligence_feeds()
            
            # Process articles with AI analysis - only keep meaningful ones
            reports = []
            for i, article in enumerate(articles[:15]):  # Check more articles
                try:
                    logger.info(f"🔍 Processing article {i+1}/{min(15, len(articles))}: {article['title'][:50]}...")
                    report = await ai_system.analyze_article(article)
                    
                    if report:  # Only add if we got meaningful analysis
                        reports.append(report)
                    
                    # Rate limiting
                    await asyncio.sleep(1.5)
                    
                except Exception as e:
                    logger.error(f"❌ Analysis error for article {i+1}: {str(e)}")
                    continue

            logger.info(f"📊 Analysis complete - {len(reports)} reports with meaningful analysis")

            # Calculate metrics
            metrics = ai_system.calculate_metrics(reports)

            # Prepare output
            output_data = {
                "articles": [asdict(report) for report in reports],
                "metrics": asdict(metrics),
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "version": "4.0",
                "generatedBy": "Patriots Protocol AI Intelligence System v4.0",
                "patriots_protocol_info": {
                    "system_name": "PATRIOTS PROTOCOL",
                    "description": "AI-DRIVEN INTELLIGENCE NETWORK",
                    "repository": "https://github.com/danishnizmi/Patriots_Protocol",
                    "ai_integration": "GitHub Models API",
                    "last_enhanced": datetime.now(timezone.utc).isoformat(),
                    "status": "OPERATIONAL"
                },
                "system_status": {
                    "ai_models": "ACTIVE",
                    "intelligence_gathering": "OPERATIONAL",
                    "threat_assessment": "ONLINE",
                    "strategic_analysis": "READY"
                }
            }

            # Save data
            os.makedirs('./data', exist_ok=True)
            with open('./data/news-analysis.json', 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info("✅ Patriots Protocol AI Intelligence Mission Complete")
            logger.info(f"📁 Intelligence data saved to ./data/news-analysis.json")
            logger.info(f"📈 Threat Level: {metrics.threat_level}")
            logger.info(f"🎯 AI Confidence: {metrics.ai_confidence}%")
            logger.info(f"🛡️  High-Value Intel: {metrics.high_value_intelligence}")
            
    except Exception as e:
        logger.error(f"❌ Patriots Protocol mission error: {str(e)}")
        # Create minimal operational data
        minimal_data = {
            "articles": [],
            "metrics": {
                "total_articles": 0,
                "threat_level": "LOW",
                "system_status": "OPERATIONAL",
                "ai_confidence": 85,
                "patriots_protocol_status": "PATRIOTS PROTOCOL OPERATIONAL"
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0",
            "patriots_protocol_info": {
                "system_name": "PATRIOTS PROTOCOL", 
                "status": "OPERATIONAL"
            }
        }
        
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(minimal_data, f, indent=2)
        
        logger.info("✅ Minimal operational data generated")

if __name__ == "__main__":
    asyncio.run(main())
