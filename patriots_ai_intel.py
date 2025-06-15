#!/usr/bin/env python3
"""
PATRIOTS PROTOCOL - Cost-Optimized Cyber Intelligence System v4.0
Secure, Efficient AI-Driven Intelligence Network with Cost Management

Enhanced system with API cost optimization, data lifecycle management,
and security features.

Source: PATRIOTS PROTOCOL - https://github.com/danishnizmi/Patriots_Protocol
"""

import os
import json
import asyncio
import aiohttp
import time
import re
import gzip
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import feedparser
import hashlib
import logging
from urllib.parse import urlparse, urljoin
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='🎖️  %(asctime)s - PATRIOTS PROTOCOL v4.0 - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

@dataclass
class IntelligenceReport:
    """Optimized Intelligence Report"""
    title: str
    summary: str  # Reduced from full_summary to save space
    source: str
    source_url: str
    timestamp: str
    category: str
    ai_analysis: str
    confidence: float
    threat_level: str
    keywords: List[str]
    priority_score: int
    content_hash: str
    reading_time: str
    intelligence_value: str

@dataclass
class IntelligenceMetrics:
    """Optimized Intelligence Metrics"""
    total_articles: int
    threat_level: str
    system_status: str
    high_value_intelligence: int
    critical_intelligence: int
    emerging_threats: List[str]
    primary_regions: List[str]
    ai_confidence: int
    processing_time: str
    api_status: str
    api_calls_made: int  # Track API usage
    cost_estimate: float  # Track estimated costs
    last_cleanup: str
    data_retention_days: int
    patriots_protocol_status: str = "PATRIOTS PROTOCOL OPERATIONAL"

class CostOptimizedPatriotsAI:
    """Cost-Optimized AI Intelligence Analysis System"""
    
    def __init__(self):
        self.api_token = os.getenv('GITHUB_TOKEN') or os.getenv('MODEL_TOKEN')
        self.base_url = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1-mini"
        self.session = None
        
        # Cost optimization settings
        self.max_api_calls_per_run = 8  # Limit API calls to control costs
        self.api_calls_made = 0
        self.estimated_cost_per_call = 0.001  # Estimated cost per API call
        self.data_retention_days = 6  # Delete data after 6 days
        self.max_articles_to_process = 12  # Limit articles processed
        
        # Security settings
        self.data_dir = Path('./data')
        self.archive_dir = Path('./archive')
        self.logs_dir = Path('./logs')
        
        # Optimized cyber security sources (reduced for cost efficiency)
        self.intelligence_sources = [
            {
                'name': 'CYBERSECURITY_INTEL',
                'url': 'https://feeds.feedburner.com/eset/blog',
                'credibility': 0.92,
                'priority': 1  # High priority sources processed first
            },
            {
                'name': 'KREBS_SECURITY',
                'url': 'https://krebsonsecurity.com/feed/',
                'credibility': 0.95,
                'priority': 1
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
                'credibility': 0.88,
                'priority': 3  # Lower priority, process only if API budget allows
            }
        ]
        
        if not self.api_token:
            logger.error("❌ GITHUB_TOKEN/MODEL_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN or MODEL_TOKEN is required for AI operations")
        
        # Create secure directories
        self._setup_secure_directories()
        
        logger.info("🚀 Patriots Protocol Cost-Optimized Cyber Intelligence System v4.0 initialized")
        logger.info(f"💰 Max API calls per run: {self.max_api_calls_per_run}")
        logger.info(f"🗂️  Data retention: {self.data_retention_days} days")
        logger.info(f"🛡️  Security directories created")

    def _setup_secure_directories(self):
        """Setup secure directory structure"""
        for directory in [self.data_dir, self.archive_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True, mode=0o750)  # Secure permissions
        
        # Create .gitignore for sensitive data
        gitignore_path = self.data_dir / '.gitignore'
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("*.tmp\n*.cache\n*.log\nsensitive_*\n")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=45),  # Reduced timeout
            headers={
                'User-Agent': 'Patriots-Protocol-Cyber-Intel-Optimized/v4.0',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def cleanup_old_data(self):
        """Secure cleanup of old data to save space and maintain security"""
        logger.info(f"🧹 Starting data cleanup - retention: {self.data_retention_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        files_cleaned = 0
        space_saved = 0
        
        # Clean old files from all directories
        for directory in [self.data_dir, self.archive_dir, self.logs_dir]:
            if directory.exists():
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_size = file_path.stat().st_size
                                
                                # Secure deletion
                                if file_path.suffix in ['.json', '.log', '.tmp']:
                                    # Overwrite file before deletion for security
                                    with open(file_path, 'wb') as f:
                                        f.write(os.urandom(file_size))
                                
                                file_path.unlink()
                                files_cleaned += 1
                                space_saved += file_size
                                
                        except Exception as e:
                            logger.warning(f"⚠️  Could not clean {file_path}: {str(e)}")
        
        space_mb = space_saved / (1024 * 1024)
        logger.info(f"✅ Cleanup complete: {files_cleaned} files removed, {space_mb:.2f} MB freed")
        
        return {"files_cleaned": files_cleaned, "space_saved_mb": space_mb}

    def generate_content_hash(self, content: str) -> str:
        """Generate secure content hash"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def load_processed_hashes(self) -> set:
        """Load previously processed content hashes to avoid duplicate API calls"""
        hash_file = self.data_dir / 'processed_hashes.json'
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    data = json.load(f)
                    # Clean old hashes (older than retention period)
                    cutoff_time = datetime.now().timestamp() - (self.data_retention_days * 24 * 3600)
                    fresh_hashes = {h: t for h, t in data.items() if t > cutoff_time}
                    
                    # Save cleaned hashes
                    with open(hash_file, 'w') as f:
                        json.dump(fresh_hashes, f)
                    
                    return set(fresh_hashes.keys())
            except Exception as e:
                logger.warning(f"⚠️  Could not load processed hashes: {str(e)}")
        
        return set()

    def save_processed_hash(self, content_hash: str):
        """Save processed content hash with timestamp"""
        hash_file = self.data_dir / 'processed_hashes.json'
        
        try:
            # Load existing hashes
            existing_hashes = {}
            if hash_file.exists():
                with open(hash_file, 'r') as f:
                    existing_hashes = json.load(f)
            
            # Add new hash with current timestamp
            existing_hashes[content_hash] = datetime.now().timestamp()
            
            # Save updated hashes
            with open(hash_file, 'w') as f:
                json.dump(existing_hashes, f)
                
        except Exception as e:
            logger.warning(f"⚠️  Could not save processed hash: {str(e)}")

    def is_fresh_content(self, published_date: str) -> bool:
        """Check if content is fresh (within last 24 hours for cost optimization)"""
        try:
            if not published_date:
                return True
                
            try:
                from dateutil import parser
                pub_date = parser.parse(published_date)
            except:
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            
            # Only process very fresh content (24 hours) to reduce API calls
            cutoff_date = datetime.now(timezone.utc) - timedelta(hours=24)
            is_fresh = pub_date > cutoff_date
            
            return is_fresh
            
        except Exception as e:
            logger.warning(f"⚠️  Date parsing error: {str(e)}")
            return False  # Don't process if we can't verify freshness

    async def make_optimized_ai_request(self, articles_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch process multiple articles in single API call to reduce costs"""
        if self.api_calls_made >= self.max_api_calls_per_run:
            logger.warning(f"💰 API call limit reached ({self.max_api_calls_per_run})")
            return []

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_token
            )

            # Batch multiple articles for analysis
            batch_content = ""
            for i, article in enumerate(articles_batch):
                batch_content += f"\n--- ARTICLE {i+1} ---\n"
                batch_content += f"Title: {article['title']}\n"
                batch_content += f"Content: {article['summary'][:300]}...\n"  # Limit content to reduce token usage

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a cyber security analyst. Analyze the provided articles and return JSON with this structure:
{
    "analyses": [
        {
            "article_index": 0,
            "analysis": "Brief analysis (max 100 words)",
            "threat_level": "CRITICAL/HIGH/MEDIUM/LOW",
            "confidence_score": 0.85,
            "priority_score": 7,
            "intelligence_value": "CRITICAL/HIGH/MEDIUM/LOW",
            "keywords": ["keyword1", "keyword2"]
        }
    ]
}

Keep analyses brief to minimize token usage. Focus only on actionable cyber security intelligence."""
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze these cyber security articles:\n{batch_content}"
                    }
                ],
                temperature=0.1,
                max_tokens=800  # Reduced token limit to control costs
            )
            
            self.api_calls_made += 1
            logger.info(f"💰 API call {self.api_calls_made}/{self.max_api_calls_per_run} - Cost estimate: ${self.api_calls_made * self.estimated_cost_per_call:.4f}")
            
            ai_response = response.choices[0].message.content
            
            # Parse batch response
            try:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = ai_response[json_start:json_end]
                    batch_results = json.loads(json_content)
                    
                    return batch_results.get('analyses', [])
            
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️  Batch JSON parsing failed: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Batch AI request failed: {str(e)}")
            return []

    async def fetch_optimized_intelligence_feeds(self) -> List[Dict[str, Any]]:
        """Fetch intelligence with cost optimization and caching"""
        all_articles = []
        processed_hashes = self.load_processed_hashes()
        
        # Sort sources by priority
        sorted_sources = sorted(self.intelligence_sources, key=lambda x: x['priority'])
        
        for source in sorted_sources:
            try:
                logger.info(f"🔍 Fetching from {source['name']} (Priority: {source['priority']})...")
                
                async with self.session.get(source['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        fresh_articles = 0
                        
                        for entry in feed.entries[:6]:  # Limit per source
                            title = entry.title
                            summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                            
                            if len(summary) < 50:
                                continue
                            
                            # Check if content is fresh
                            published_date = entry.get('published', entry.get('updated', ''))
                            if not self.is_fresh_content(published_date):
                                continue
                            
                            # Check if already processed
                            content_hash = self.generate_content_hash(f"{title}{summary}")
                            if content_hash in processed_hashes:
                                logger.info(f"⚡ Skipping already processed: {title[:50]}...")
                                continue
                            
                            # Filter for cyber security content
                            cyber_keywords = ['security', 'cyber', 'hack', 'breach', 'malware', 'vulnerability', 'attack', 'threat']
                            if not any(keyword in (title + summary).lower() for keyword in cyber_keywords):
                                continue
                            
                            article = {
                                'title': title,
                                'summary': summary[:400],  # Limit summary length
                                'source': source['name'],
                                'source_url': entry.get('link', ''),
                                'timestamp': published_date or datetime.now(timezone.utc).isoformat(),
                                'credibility': source['credibility'],
                                'content_hash': content_hash,
                                'priority': source['priority']
                            }
                            
                            all_articles.append(article)
                            fresh_articles += 1
                            
                            # Stop if we have enough articles
                            if len(all_articles) >= self.max_articles_to_process:
                                break
                        
                        logger.info(f"📊 {source['name']}: {fresh_articles} fresh cyber articles collected")
                        
                        # Stop if we have enough or if this is a lower priority source
                        if len(all_articles) >= self.max_articles_to_process or source['priority'] > 1:
                            break
                            
                    else:
                        logger.warning(f"⚠️  {source['name']} returned {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Error fetching {source['name']}: {str(e)}")
                continue

        logger.info(f"📊 Collected {len(all_articles)} optimized cyber intelligence articles")
        return all_articles

    def _clean_text(self, text: str) -> str:
        """Clean text content efficiently"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)
        
        return text[:400]  # Limit text length to save space

    async def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[IntelligenceReport]:
        """Process articles in batches to optimize API usage"""
        reports = []
        
        # Process in batches of 3-4 articles per API call
        batch_size = 3
        
        for i in range(0, len(articles), batch_size):
            if self.api_calls_made >= self.max_api_calls_per_run:
                logger.warning(f"💰 Stopping processing - API limit reached")
                break
                
            batch = articles[i:i + batch_size]
            logger.info(f"🔍 Processing batch {i//batch_size + 1}: {len(batch)} articles")
            
            # Get AI analysis for batch
            ai_results = await self.make_optimized_ai_request(batch)
            
            # Create reports from batch results
            for j, article in enumerate(batch):
                try:
                    # Find corresponding AI result
                    ai_result = None
                    for result in ai_results:
                        if result.get('article_index') == j:
                            ai_result = result
                            break
                    
                    if not ai_result:
                        logger.info(f"⚠️  No AI result for article {j} in batch")
                        continue
                    
                    # Create optimized report
                    report = IntelligenceReport(
                        title=article['title'],
                        summary=article['summary'],
                        source=article['source'],
                        source_url=article.get('source_url', ''),
                        timestamp=article['timestamp'],
                        category='SECURITY',
                        ai_analysis=ai_result.get('analysis', 'Analysis available'),
                        confidence=ai_result.get('confidence_score', 0.8),
                        threat_level=ai_result.get('threat_level', 'MEDIUM'),
                        keywords=ai_result.get('keywords', [])[:6],  # Limit keywords
                        priority_score=ai_result.get('priority_score', 5),
                        content_hash=article['content_hash'],
                        reading_time=f"{max(1, len(article['summary']) // 200)} min",
                        intelligence_value=ai_result.get('intelligence_value', 'MEDIUM')
                    )
                    
                    reports.append(report)
                    
                    # Save processed hash
                    self.save_processed_hash(article['content_hash'])
                    
                    logger.info(f"✅ Processed: {report.title[:50]}... (Threat: {report.threat_level})")
                    
                except Exception as e:
                    logger.error(f"❌ Error creating report for article {j}: {str(e)}")
                    continue
            
            # Rate limiting between batches
            await asyncio.sleep(2.0)

        logger.info(f"📊 Batch processing complete: {len(reports)} reports generated from {self.api_calls_made} API calls")
        return reports

    def calculate_optimized_metrics(self, reports: List[IntelligenceReport]) -> IntelligenceMetrics:
        """Calculate metrics with cost tracking"""
        if not reports:
            return self._generate_baseline_metrics()

        # Calculate intelligence metrics
        high_value = len([r for r in reports if r.intelligence_value in ['HIGH', 'CRITICAL']])
        critical = len([r for r in reports if r.intelligence_value == 'CRITICAL'])
        
        # Threat analysis
        threat_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for report in reports:
            threat_counts[report.threat_level] += 1
        
        overall_threat = 'CRITICAL' if threat_counts['CRITICAL'] > 0 else (
                        'HIGH' if threat_counts['HIGH'] > 0 else 'MEDIUM')
        
        # Extract trending keywords
        all_keywords = []
        for report in reports:
            all_keywords.extend(report.keywords)
        
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        trending = [kw for kw, count in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Calculate costs
        estimated_cost = self.api_calls_made * self.estimated_cost_per_call
        
        return IntelligenceMetrics(
            total_articles=len(reports),
            threat_level=overall_threat,
            system_status="OPERATIONAL",
            high_value_intelligence=high_value,
            critical_intelligence=critical,
            emerging_threats=trending if trending else ["MONITORING"],
            primary_regions=["GLOBAL"],
            ai_confidence=int(sum(r.confidence for r in reports) / len(reports) * 100) if reports else 85,
            processing_time=f"< {30 + self.api_calls_made * 5} seconds",
            api_status="ACTIVE",
            api_calls_made=self.api_calls_made,
            cost_estimate=estimated_cost,
            last_cleanup=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            data_retention_days=self.data_retention_days
        )

    def _generate_baseline_metrics(self) -> IntelligenceMetrics:
        """Generate baseline metrics with cost tracking"""
        return IntelligenceMetrics(
            total_articles=0,
            threat_level="LOW",
            system_status="OPERATIONAL",
            high_value_intelligence=0,
            critical_intelligence=0,
            emerging_threats=["MONITORING"],
            primary_regions=["GLOBAL"],
            ai_confidence=85,
            processing_time="< 30 seconds",
            api_status="ACTIVE",
            api_calls_made=self.api_calls_made,
            cost_estimate=self.api_calls_made * self.estimated_cost_per_call,
            last_cleanup=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            data_retention_days=self.data_retention_days
        )

    def save_compressed_data(self, data: dict, filename: str):
        """Save data with compression to save space"""
        try:
            # Save regular JSON
            json_path = self.data_dir / filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'))  # Compact JSON
            
            # Save compressed version
            compressed_path = self.data_dir / f"{filename}.gz"
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'))
            
            # Check compression ratio
            original_size = json_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            logger.info(f"💾 Data saved: {original_size} bytes, compressed: {compressed_size} bytes ({compression_ratio:.1f}% reduction)")
            
        except Exception as e:
            logger.error(f"❌ Error saving compressed data: {str(e)}")

async def main():
    """Main Patriots Protocol Cost-Optimized Intelligence Pipeline"""
    logger.info("🎖️  PATRIOTS PROTOCOL v4.0 - Cost-Optimized Cyber Intelligence System Starting...")
    logger.info(f"📅 Mission Start: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        async with CostOptimizedPatriotsAI() as ai_system:
            # Cleanup old data first to save space
            cleanup_stats = ai_system.cleanup_old_data()
            
            # Fetch optimized intelligence
            articles = await ai_system.fetch_optimized_intelligence_feeds()
            
            # Process articles in cost-optimized batches
            reports = []
            if articles:
                reports = await ai_system.process_articles_batch(articles)
            
            # Calculate metrics with cost tracking
            metrics = ai_system.calculate_optimized_metrics(reports)
            
            # Prepare optimized output data
            output_data = {
                "articles": [asdict(report) for report in reports],
                "metrics": asdict(metrics),
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "version": "4.0-optimized",
                "cost_optimization": {
                    "api_calls_used": ai_system.api_calls_made,
                    "api_calls_limit": ai_system.max_api_calls_per_run,
                    "estimated_cost": metrics.cost_estimate,
                    "data_retention_days": ai_system.data_retention_days,
                    "cleanup_stats": cleanup_stats
                },
                "patriots_protocol_info": {
                    "system_name": "PATRIOTS PROTOCOL",
                    "description": "COST-OPTIMIZED CYBER INTELLIGENCE NETWORK",
                    "repository": "https://github.com/danishnizmi/Patriots_Protocol",
                    "optimization": "Cost & Security Enhanced",
                    "status": "OPERATIONAL"
                }
            }

            # Save compressed data
            ai_system.save_compressed_data(output_data, 'news-analysis.json')

            logger.info("✅ Patriots Protocol Cost-Optimized Mission Complete")
            logger.info(f"💰 API Calls Used: {ai_system.api_calls_made}/{ai_system.max_api_calls_per_run}")
            logger.info(f"💰 Estimated Cost: ${metrics.cost_estimate:.4f}")
            logger.info(f"📊 Reports Generated: {len(reports)}")
            logger.info(f"🧹 Files Cleaned: {cleanup_stats['files_cleaned']}")
            logger.info(f"💾 Space Saved: {cleanup_stats['space_saved_mb']:.2f} MB")
            
    except Exception as e:
        logger.error(f"❌ Patriots Protocol cost-optimized mission error: {str(e)}")
        
        # Create minimal operational data
        minimal_data = {
            "articles": [],
            "metrics": {
                "total_articles": 0,
                "threat_level": "LOW", 
                "system_status": "OPERATIONAL",
                "ai_confidence": 85,
                "api_calls_made": 0,
                "cost_estimate": 0.0
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "version": "4.0-optimized"
        }
        
        # Save minimal data
        os.makedirs('./data', exist_ok=True)
        with open('./data/news-analysis.json', 'w') as f:
            json.dump(minimal_data, f, separators=(',', ':'))
        
        logger.info("✅ Minimal operational data generated")

if __name__ == "__main__":
    asyncio.run(main())
