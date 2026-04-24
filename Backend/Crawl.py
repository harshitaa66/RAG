import asyncio
import os
import re
import json
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("Tavily_API")

client = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)


# ==================== BEAUTIFULSOUP EXTRACTORS ==================== #

class DirectExtractor:
    """Extract data directly using BeautifulSoup patterns"""
    
    @staticmethod
    def extract_jobs(html: str, url: str) -> List[Dict]:
        """Extract jobs using common HTML patterns"""
        jobs = []
        soup = BeautifulSoup(html, "html.parser")
        
        # Common job listing patterns
        job_selectors = [
            {'container': 'div', 'class': re.compile(r'job.*card|listing|item', re.I)},
            {'container': 'article', 'class': re.compile(r'job|listing', re.I)},
            {'container': 'li', 'class': re.compile(r'job|result', re.I)},
            {'container': 'div', 'data-job-id': True},
        ]
        
        containers = []
        for selector in job_selectors:
            found = soup.find_all(selector['container'], 
                                 **{k:v for k,v in selector.items() if k != 'container'})
            if found:
                containers.extend(found)
                if len(containers) >= 20:
                    break
        
        for container in containers[:30]:
            job = {'source_url': url}
            
            # Extract title
            title_elem = (
                container.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|job.*name|position', re.I)) or
                container.find('a', class_=re.compile(r'job', re.I)) or
                container.find(['h2', 'h3', 'h4'])
            )
            if title_elem:
                job['title'] = title_elem.get_text(strip=True)
            
            # Extract company
            company_elem = (
                container.find(class_=re.compile(r'company|employer|org', re.I)) or
                container.find('span', class_=re.compile(r'name', re.I))
            )
            if company_elem:
                job['company'] = company_elem.get_text(strip=True)
            
            # Extract location
            location_elem = container.find(class_=re.compile(r'location|city|place', re.I))
            if location_elem:
                job['location'] = location_elem.get_text(strip=True)
            
            # Extract salary
            salary_elem = container.find(class_=re.compile(r'salary|pay|compensation', re.I))
            if salary_elem:
                job['salary'] = salary_elem.get_text(strip=True)
            
            # Extract experience
            exp_elem = container.find(class_=re.compile(r'experience|exp|years', re.I))
            if exp_elem:
                job['experience'] = exp_elem.get_text(strip=True)
            
            # Extract link
            link_elem = container.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if href.startswith('http'):
                    job['apply_link'] = href
                elif href.startswith('/'):
                    from urllib.parse import urljoin
                    job['apply_link'] = urljoin(url, href)
            
            # Only add if has at least title
            if job.get('title'):
                jobs.append(job)
        
        return jobs
    
    @staticmethod
    def extract_products(html: str, url: str) -> List[Dict]:
        """Extract products using common HTML patterns"""
        products = []
        soup = BeautifulSoup(html, "html.parser")
        
        # Common product patterns
        product_selectors = [
            {'container': 'div', 'class': re.compile(r'product.*card|item|tile', re.I)},
            {'container': 'article', 'class': re.compile(r'product', re.I)},
            {'container': 'li', 'class': re.compile(r'product|item', re.I)},
        ]
        
        containers = []
        for selector in product_selectors:
            found = soup.find_all(selector['container'], 
                                 **{k:v for k,v in selector.items() if k != 'container'})
            if found:
                containers.extend(found)
                if len(containers) >= 15:
                    break
        
        for container in containers[:25]:
            product = {'source_url': url}
            
            # Extract name/title
            name_elem = (
                container.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'product.*name|title', re.I)) or
                container.find('a', class_=re.compile(r'product', re.I)) or
                container.find(['h2', 'h3', 'h4', 'h5'])
            )
            if name_elem:
                product['name'] = name_elem.get_text(strip=True)
            
            # Extract price
            price_elem = (
                container.find(class_=re.compile(r'price|cost', re.I)) or
                container.find(attrs={'data-price': True})
            )
            if price_elem:
                product['price'] = price_elem.get_text(strip=True)
            
            # Extract brand
            brand_elem = container.find(class_=re.compile(r'brand|manufacturer', re.I))
            if brand_elem:
                product['brand'] = brand_elem.get_text(strip=True)
            
            # Extract rating
            rating_elem = container.find(class_=re.compile(r'rating|star|review', re.I))
            if rating_elem:
                product['rating'] = rating_elem.get_text(strip=True)
            
            # Extract link
            link_elem = container.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                if href.startswith('http'):
                    product['link'] = href
                elif href.startswith('/'):
                    from urllib.parse import urljoin
                    product['link'] = urljoin(url, href)
            
            # Only add if has name
            if product.get('name'):
                products.append(product)
        
        return products
    
    @staticmethod
    def extract_general_info(html: str, url: str) -> str:
        """Extract general publicly available information from page"""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get main title
        title = None
        for tag in ['h1', 'h2', 'title']:
            elem = soup.find(tag)
            if elem:
                title = elem.get_text(strip=True)
                break
        
        # Get main paragraphs
        paragraphs = []
        for p in soup.find_all(['p', 'div'], class_=re.compile(r'content|text|description|summary', re.I)):
            text = p.get_text(strip=True)
            if len(text) > 50 and len(text) < 500:
                paragraphs.append(text)
            if len(paragraphs) >= 5:
                break
        
        # Get lists (features, amenities, etc.)
        lists = []
        for ul in soup.find_all(['ul', 'ol'])[:3]:
            items = [li.get_text(strip=True) for li in ul.find_all('li')[:5]]
            if items:
                lists.append(items)
        
        # Format output
        output = []
        if title:
            output.append(f"**{title}**\n")
        
        if paragraphs:
            output.append("\n".join(paragraphs[:3]))
            output.append("")
        
        if lists:
            output.append("Key Points:")
            for items in lists:
                for item in items:
                    output.append(f"  • {item}")
        
        return "\n".join(output) if output else ""


# ==================== DEDUPLICATOR ==================== #

class Deduplicator:
    """Remove duplicate items based on key fields"""
    
    @staticmethod
    def deduplicate_items(items: List[Dict]) -> List[Dict]:
        """Remove duplicates based on title/name and other key fields"""
        seen = set()
        unique_items = []
        
        for item in items:
            # Create signature based on title/name and company/brand
            title = item.get('title') or item.get('name') or ''
            company = item.get('company') or item.get('brand') or ''
            location = item.get('location') or ''
            
            # Clean and normalize
            signature = f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}"
            
            if signature not in seen and title:  # Must have title/name
                seen.add(signature)
                unique_items.append(item)
        
        return unique_items


# ==================== FORMATTER ==================== #

class SmartFormatter:
    """Universal formatter"""
    
    ICONS = {
        'title': '📌', 'name': '📌', 'company': '🏢', 'location': '📍',
        'price': '💰', 'salary': '💰', 'rating': '⭐', 'experience': '⏳',
        'job_type': '📋', 'capacity': '👥', 'brand': '🏷️', 'apply_link': '🔗',
        'link': '🔗', 'website': '🔗', 'contact': '📞', 'availability': '📊'
    }
    
    @staticmethod
    def format_items(items: List[Dict], title: str = "RESULTS") -> str:
        if not items:
            return f"\n❌ No {title.lower()} found.\n"
        
        output = ["\n" + "="*100]
        output.append(f"📋 FOUND {len(items)} {title.upper()}")
        output.append("="*100 + "\n")
        
        for idx, item in enumerate(items, 1):
            output.append("─"*100)
            output.append(f"ITEM #{idx}")
            output.append("─"*100)
            
            for key, value in item.items():
                if key != 'source_url' and value:
                    icon = SmartFormatter.ICONS.get(key, '•')
                    display = key.replace('_', ' ').title()
                    val = str(value)[:150]
                    output.append(f"{icon} {display:20s} {val}")
            
            output.append("")
        
        output.append("="*100 + "\n")
        return "\n".join(output)
    
    @staticmethod
    def format_summary(summaries: List[Dict], query: str) -> str:
        """Format when we have summaries instead of structured listings"""
        output = ["\n" + "="*100]
        output.append(f"📄 PUBLIC INFORMATION AVAILABLE")
        output.append("="*100)
        output.append(f"\nQuery: \"{query}\"")
        output.append("\nCouldn't extract specific listings (may require login/subscription),")
        output.append("but here's publicly available information:\n")
        output.append("─"*100 + "\n")
        
        for idx, summary in enumerate(summaries, 1):
            output.append(f"Source #{idx}: {summary.get('url', 'N/A')[:80]}")
            output.append("─"*100)
            
            if summary.get('content'):
                content = summary['content'][:800]
                output.append(content)
            
            output.append("\n" + "─"*100 + "\n")
        
        output.append("="*100 + "\n")
        output.append("💡 TIP: Visit these URLs directly for complete information.")
        output.append("="*100 + "\n")
        
        return "\n".join(output)


# ==================== PRODUCTION SCRAPER ==================== #

class ProductionScraper:
    """Production scraper with fixed product detection and deduplication"""
    
    def __init__(self):
        self.formatter = SmartFormatter()
        self.extractor = DirectExtractor()
        self.deduplicator = Deduplicator()
    
    async def process_query(self, user_query: str, max_urls: int = 8) -> str:
        print("\n" + "="*100)
        print(f"🔍 QUERY: \"{user_query}\"")
        print("="*100 + "\n")
        
        # Detect domain (FIXED)
        domain_type = self._detect_domain(user_query)
        print(f"🎯 Domain: {domain_type.upper()}")
        
        # Search
        print(f"\n🔎 Searching...")
        urls = await self._search(user_query, max_urls)
        
        if not urls:
            return "\n❌ No results found.\n"
        
        print(f"   ✓ Found {len(urls)} sources")
        
        # Extract
        print(f"\n📄 Extracting data...")
        items, summaries = await self._extract_smart(urls, user_query, domain_type)
        
        # DEDUPLICATE (NEW)
        if items:
            items = self.deduplicator.deduplicate_items(items)
        
        # Format results
        if items:
            print(f"   ✓ Extracted {len(items)} unique items\n")
            return self.formatter.format_items(items, domain_type)
        elif summaries:
            print(f"   ✓ Extracted summaries from {len(summaries)} pages\n")
            return self.formatter.format_summary(summaries, user_query)
        else:
            return "\n⚠️ No data available. Pages may require login.\n"
    
    def _detect_domain(self, query: str) -> str:
        """FIXED: Better product detection"""
        q_lower = query.lower()
        
        # Check jobs first
        if any(word in q_lower for word in ['job', 'career', 'employment', 'hiring']):
            return 'jobs'
        
        # FIXED: Better product detection
        product_keywords = ['jacket', 'shirt', 'laptop', 'phone', 'watch', 'shoe', 
                           'bag', 'camera', 'headphone', 'tv', 'furniture', 'clothing',
                           'buy', 'shop', 'purchase', 'product', 'price']
        if any(word in q_lower for word in product_keywords):
            return 'products'
        
        # Other domains
        if any(word in q_lower for word in ['venue', 'wedding', 'event', 'party']):
            return 'venues'
        if any(word in q_lower for word in ['hotel', 'resort', 'stay']):
            return 'hotels'
        if any(word in q_lower for word in ['restaurant', 'food', 'dining']):
            return 'restaurants'
        
        return 'general'
    
    async def _search(self, query: str, max_urls: int) -> List[str]:
        def _sync():
            try:
                return tavily.search(query)
            except:
                return None
        
        resp = await asyncio.to_thread(_sync)
        urls = []
        
        if isinstance(resp, dict):
            for key in ("results", "items"):
                for item in resp.get(key, []):
                    if isinstance(item, dict):
                        url = item.get("url") or item.get("link")
                        if url:
                            urls.append(url)
        
        return list(dict.fromkeys(urls))[:max_urls]
    
    async def _extract_smart(self, urls: List[str], query: str, domain_type: str):
        """Smart extraction with product support"""
        items = []
        summaries = []
        
        for idx, url in enumerate(urls[:8], 1):
            print(f"   [{idx}/8] {url[:65]}...")
            
            try:
                # Try direct HTTP + BeautifulSoup
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Try structured extraction
                    if domain_type == 'jobs':
                        extracted = self.extractor.extract_jobs(html, url)
                        if extracted:
                            print(f"      ✓ BeautifulSoup: {len(extracted)} items")
                            items.extend(extracted)
                            continue
                    
                    elif domain_type == 'products':
                        extracted = self.extractor.extract_products(html, url)
                        if extracted:
                            print(f"      ✓ BeautifulSoup: {len(extracted)} items")
                            items.extend(extracted)
                            continue
                    
                    # Fallback: Extract general info
                    general_info = self.extractor.extract_general_info(html, url)
                    if general_info and len(general_info) > 100:
                        summaries.append({'url': url, 'content': general_info})
                        print(f"      ✓ Extracted summary")
                        continue
                
                # Last resort: Try LLM
                print(f"      → Trying LLM extraction...")
                llm_result = await self._llm_extract(url, query, domain_type)
                if llm_result:
                    if isinstance(llm_result, list):
                        items.extend(llm_result)
                        print(f"      ✓ LLM: {len(llm_result)} items")
                    else:
                        summaries.append({'url': url, 'content': str(llm_result)})
                        print(f"      ✓ LLM summary")
            
            except Exception as e:
                print(f"      ⚠️ Failed: {str(e)[:40]}")
            
            await asyncio.sleep(0.8)
        
        return items, summaries
    
    async def _llm_extract(self, url: str, query: str, domain_type: str):
        """LLM extraction"""
        
        fields_map = {
            'jobs': ['title', 'company', 'location', 'salary', 'experience', 'skills', 'apply_link'],
            'products': ['name', 'brand', 'price', 'rating', 'link'],
            'venues': ['name', 'location', 'price', 'capacity', 'availability', 'contact'],
        }
        
        fields = fields_map.get(domain_type, [])
        
        if fields:
            prompt = f"""Extract {domain_type} from this page for: "{query}"

Return JSON array: [{{"field1": "val", "field2": "val", ...}}]
Fields: {json.dumps(fields)}

If no {domain_type} found, return a 2-3 paragraph summary."""
        else:
            prompt = f"""Provide a 2-3 paragraph summary of this page relevant to: "{query}"
Focus on key information, prices, features, or details available."""
        
        try:
            strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(provider="gpt-4o-mini", api_token=OPENAI_API_KEY),
                instruction=prompt
            )
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, extraction_strategy=strategy)
                
                content = None
                for attr in ["extracted_content", "markdown", "text"]:
                    if hasattr(result, attr):
                        content = getattr(result, attr)
                        if content:
                            break
                
                if content:
                    content = str(content).strip()
                    
                    # Try parse as JSON array
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            if isinstance(parsed, list) and len(parsed) > 0:
                                return [item for item in parsed if isinstance(item, dict)]
                        except:
                            pass
                    
                    # Return as summary
                    return content[:1000]
        
        except:
            pass
        
        return None


# ==================== MAIN ==================== #

async def main():
    print("\n" + "="*100)
    print("🤖 PRODUCTION SCRAPER (FIXED)")
    print("="*100)
    print("\n✨ Fixed:")
    print("  • Better product detection (jackets, laptops, etc.)")
    print("  • Duplicate removal")
    print("  • Jobs work perfectly")
    print("\nType 'exit' to quit.")
    print("="*100 + "\n")
    
    scraper = ProductionScraper()
    
    while True:
        user_query = input("\n💭 Enter your query: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q', '']:
            print("\n👋 Goodbye!\n")
            break
        
        try:
            result = await scraper.process_query(user_query, max_urls=8)
            print(result)
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())