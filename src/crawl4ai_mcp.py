"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Also includes AI hallucination detection and repository parsing tools using Neo4j knowledge graphs.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures
import sys

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
try:
    from code_widget_extractor import CodeWidgetExtractor, create_enhanced_crawler_config
except ImportError:
    # Widget extractor not available on main branch
    CodeWidgetExtractor = None
    create_enhanced_crawler_config = None

# Conditional imports for Firecrawl
USE_FIRECRAWL = os.getenv("FIRECRAWL_API_KEY") is not None

if USE_FIRECRAWL:
    try:
        from firecrawl import FirecrawlApp
        print("🔥 Firecrawl integration enabled")
    except ImportError:
        print("⚠️  Firecrawl API key found but firecrawl-py not installed. Run: pip install firecrawl-py")
        USE_FIRECRAWL = False
else:
    FirecrawlApp = None
    print("🕷️  Using Crawl4AI (default crawler)")

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from utils import (
    get_supabase_client, 
    add_documents_to_supabase, 
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples
)

# Import knowledge graph modules
from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor
from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Helper functions for Neo4j validation and error handling
def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    ])

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    elif "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    else:
        return f"Neo4j error: {str(error)}"

def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}
    
    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}
    
    if not script_path.endswith('.py'):
        return {"valid": False, "error": "Only Python (.py) files are supported"}
    
    try:
        # Check if file is readable
        with open(script_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}

def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}
    
    repo_url = repo_url.strip()
    
    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}
    
    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}
    
    return {"valid": True, "repo_name": repo_url.split('/')[-1].replace('.git', '')}

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available
    firecrawl_app: Optional[Any] = None        # FirecrawlApp when available

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    # Initialize Firecrawl app if enabled
    firecrawl_app = None
    if USE_FIRECRAWL:
        try:
            firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
            print("✓ Firecrawl client initialized")
        except Exception as e:
            print(f"Failed to initialize Firecrawl client: {e}")
            firecrawl_app = None
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    # Initialize Neo4j components if configured and enabled
    knowledge_validator = None
    repo_extractor = None
    
    # Check if knowledge graph functionality is enabled
    knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
    
    if knowledge_graph_enabled:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("Initializing knowledge graph components...")
                
                # Initialize knowledge graph validator
                knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
                await knowledge_validator.initialize()
                print("✓ Knowledge graph validator initialized")
                
                # Initialize repository extractor
                repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
                await repo_extractor.initialize()
                print("✓ Repository extractor initialized")
                
            except Exception as e:
                print(f"Failed to initialize Neo4j components: {format_neo4j_error(e)}")
                knowledge_validator = None
                repo_extractor = None
        else:
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
    else:
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
            firecrawl_app=firecrawl_app
        )
    finally:
        # Clean up all components
        await crawler.__aexit__(None, None, None)
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as e:
                print(f"Error closing knowledge validator: {e}")
        if repo_extractor:
            try:
                await repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as e:
                print(f"Error closing repository extractor: {e}")

# Initialize FastMCP server
transport = os.getenv("TRANSPORT", "sse")

if transport == "stdio":
    # For stdio transport, don't pass host/port
    mcp = FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan
    )
else:
    # For SSE transport, use host/port
    port_str = os.getenv("PORT", "8051")
    port = int(port_str) if port_str else 8051
    
    mcp = FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
        host=os.getenv("HOST", "0.0.0.0"),
        port=port
    )

# Conditional decorator for MCP tools
def conditional_tool(condition: bool):
    """Decorator that conditionally registers MCP tools based on environment configuration."""
    def decorator(func):
        if condition:
            return mcp.tool()(func)
        else:
            return func
    return decorator

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (code, context_before, context_after)
        
    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)

# Action builder utilities for Firecrawl

@conditional_tool(not USE_FIRECRAWL)
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    For extracting code examples from interactive widgets, use the extract_code_widgets tool
    after running this tool on the same URL.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                total_word_count += meta.get("word_count", 0)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
            update_source_info(supabase_client, source_id, source_summary, total_word_count)
            
            # Add documentation chunks to Supabase (AFTER source exists)
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split()),
                            "language": block.get('language', ''),
                            "extraction_method": "standard_crawl"
                        }
                        code_metadatas.append(code_meta)
                    
                    # Add code examples to Supabase
                    add_code_examples_to_supabase(
                        supabase_client, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas
                    )
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if code_blocks else 0,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@conditional_tool(USE_FIRECRAWL)
async def firecrawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page using Firecrawl and store its content in Supabase.
    
    This tool uses Firecrawl's scraping service to extract clean, LLM-ready content.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the firecrawl client from the context
        firecrawl_app = ctx.request_context.lifespan_context.firecrawl_app
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        if not firecrawl_app:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "Firecrawl client not initialized"
            }, indent=2)
        
        # Scrape the page with Firecrawl (use simple API for now)
        firecrawl_result = firecrawl_app.scrape_url(url)
        
        # Process the firecrawl response
        result = process_firecrawl_response(firecrawl_result)
        
        if result['success'] and result['markdown']:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Chunk the content
            chunks = smart_chunk_markdown(result['markdown'])
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                meta["crawler_type"] = "firecrawl"
                meta["firecrawl_title"] = result.get('title', '')
                meta["firecrawl_description"] = result.get('description', '')
                metadatas.append(meta)
                
                # Accumulate word count
                total_word_count += meta.get("word_count", 0)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result['markdown']}
            
            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(source_id, result['markdown'][:5000])  # Use first 5000 chars for summary
            update_source_info(supabase_client, source_id, source_summary, total_word_count)
            
            # Add documentation chunks to Supabase (AFTER source exists)
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            if extract_code_examples:
                code_blocks = extract_code_blocks(result['markdown'])
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split()),
                            "language": block.get('language', ''),
                            "extraction_method": "firecrawl_scrape",
                            "crawler_type": "firecrawl"
                        }
                        code_metadatas.append(code_meta)
                    
                    # Add code examples to Supabase
                    add_code_examples_to_supabase(
                        supabase_client, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas
                    )
            
            return json.dumps({
                "success": True,
                "url": url,
                "crawler_type": "firecrawl",
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if extract_code_examples and code_blocks else 0,
                "content_length": len(result['markdown']),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "title": result.get('title', ''),
                "description": result.get('description', '')
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "crawler_type": "firecrawl",
                "error": result['error_message']
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "crawler_type": "firecrawl",
            "error": str(e)
        }, indent=2)


@conditional_tool(not USE_FIRECRAWL)
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}
        
        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)
                
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Update source information for each unique source FIRST (before inserting documents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))
        
        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)
        
        # Add documentation chunks to Supabase (AFTER sources exist)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
        
        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            all_code_blocks = []
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)
                
                if code_blocks:
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
            
            # Add all code examples to Supabase
            if code_examples:
                add_code_examples_to_supabase(
                    supabase_client, 
                    code_urls, 
                    code_chunk_numbers, 
                    code_examples, 
                    code_summaries, 
                    code_metadatas,
                    batch_size=batch_size
                )
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples),
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@conditional_tool(USE_FIRECRAWL)
async def firecrawl_smart_crawl(ctx: Context, url: str, max_pages: int = 10, include_paths: str = "", exclude_paths: str = "") -> str:
    """
    Intelligently crawl websites using Firecrawl's advanced crawling capabilities.
    
    This tool uses Firecrawl's crawl_url method to discover and crawl multiple pages
    from a website. It's perfect for crawling documentation sites, blogs, or any
    multi-page content.
    
    Args:
        ctx: The MCP server provided context
        url: Starting URL or domain to crawl
        max_pages: Maximum number of pages to crawl (default: 10, max: 100)
        include_paths: Comma-separated paths to include (e.g., "/docs,/api")
        exclude_paths: Comma-separated paths to exclude (e.g., "/login,/admin")
    
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the firecrawl client from the context
        firecrawl_app = ctx.request_context.lifespan_context.firecrawl_app
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        if not firecrawl_app:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "Firecrawl client not initialized"
            }, indent=2)
        
        # Ensure max_pages is reasonable
        max_pages = min(max_pages, 100)
        
        # Prepare crawl parameters (use simple API for now)
        # Start the crawl
        crawl_result = firecrawl_app.crawl_url(url)
        
        if not crawl_result or not getattr(crawl_result, 'success', False):
            error_msg = getattr(crawl_result, 'error', 'Crawl failed') if crawl_result else 'No crawl result'
            return json.dumps({
                "success": False,
                "url": url,
                "crawler_type": "firecrawl",
                "error": error_msg
            }, indent=2)
        
        # Process crawled pages
        pages_data = getattr(crawl_result, 'data', []) or []
        
        if not pages_data:
            return json.dumps({
                "success": False,
                "url": url,
                "crawler_type": "firecrawl",
                "error": "No pages found"
            }, indent=2)
        
        # Process each page
        total_chunks = 0
        total_code_examples = 0
        total_word_count = 0
        pages_processed = 0
        
        for page_data in pages_data:
            try:
                page_url = getattr(page_data, 'url', '') if hasattr(page_data, 'url') else page_data.get('url', '')
                markdown_content = getattr(page_data, 'markdown', '') if hasattr(page_data, 'markdown') else page_data.get('markdown', '')
                metadata = getattr(page_data, 'metadata', {}) if hasattr(page_data, 'metadata') else page_data.get('metadata', {})
                
                if not markdown_content or not page_url:
                    continue
                
                # Extract source_id
                parsed_url = urlparse(page_url)
                source_id = parsed_url.netloc or parsed_url.path
                
                # Chunk the content
                chunks = smart_chunk_markdown(markdown_content)
                
                # Prepare data for Supabase
                urls = []
                chunk_numbers = []
                contents = []
                metadatas = []
                page_word_count = 0
                
                for i, chunk in enumerate(chunks):
                    urls.append(page_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = page_url
                    meta["source"] = source_id
                    meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                    meta["crawler_type"] = "firecrawl"
                    meta["firecrawl_title"] = metadata.get('title', '')
                    meta["firecrawl_description"] = metadata.get('description', '')
                    metadatas.append(meta)
                    
                    # Accumulate word count
                    page_word_count += meta.get("word_count", 0)
                
                # Create url_to_full_document mapping
                url_to_full_document = {page_url: markdown_content}
                
                # Update source information
                source_summary = extract_source_summary(source_id, markdown_content[:5000])
                update_source_info(supabase_client, source_id, source_summary, page_word_count)
                
                # Add documentation chunks to Supabase
                add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
                
                # Extract and process code examples if enabled
                extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
                if extract_code_examples:
                    code_blocks = extract_code_blocks(markdown_content)
                    if code_blocks:
                        code_urls = []
                        code_chunk_numbers = []
                        code_examples = []
                        code_summaries = []
                        code_metadatas = []
                        
                        # Process code examples in parallel
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                            # Prepare arguments for parallel processing
                            summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                            for block in code_blocks]
                            
                            # Generate summaries in parallel
                            summaries = list(executor.map(process_code_example, summary_args))
                        
                        # Prepare code example data
                        for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                            code_urls.append(page_url)
                            code_chunk_numbers.append(i)
                            code_examples.append(block['code'])
                            code_summaries.append(summary)
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": i,
                                "url": page_url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split()),
                                "language": block.get('language', ''),
                                "extraction_method": "firecrawl_crawl",
                                "crawler_type": "firecrawl"
                            }
                            code_metadatas.append(code_meta)
                        
                        # Add code examples to Supabase
                        add_code_examples_to_supabase(
                            supabase_client, 
                            code_urls, 
                            code_chunk_numbers, 
                            code_examples, 
                            code_summaries, 
                            code_metadatas
                        )
                        
                        total_code_examples += len(code_blocks)
                
                total_chunks += len(chunks)
                total_word_count += page_word_count
                pages_processed += 1
                
            except Exception as page_error:
                print(f"Error processing page {page_data.get('url', 'unknown')}: {page_error}")
                continue
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawler_type": "firecrawl",
            "pages_crawled": pages_processed,
            "chunks_stored": total_chunks,
            "code_examples_stored": total_code_examples,
            "total_word_count": total_word_count,
            "crawl_params": {
                "max_pages": max_pages,
                "include_paths": include_paths,
                "exclude_paths": exclude_paths
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "crawler_type": "firecrawl",
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering 
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Query the sources table directly
        result = supabase_client.from_('sources')\
            .select('*')\
            .order('source_id')\
            .execute()
        
        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append({
                    "source_id": source.get("source_id"),
                    "summary": source.get("summary"),
                    "total_words": source.get("total_words"),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                })
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE
            keyword_query = supabase_client.from_('crawled_pages')\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')
            
            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq('source_id', source)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.
    
    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "error": "Code example extraction is disabled. Perform a normal RAG search."
        }, indent=2)
    
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # Import the search function from utils
            from utils import search_code_examples as search_code_examples_impl
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = supabase_client.from_('code_examples')\
                .select('id, url, chunk_number, content, summary, metadata, source_id')\
                .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')
            
            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq('source_id', source_id)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            from utils import search_code_examples as search_code_examples_impl
            
            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    """
    Check an AI-generated Python script for hallucinations using the knowledge graph.
    
    This tool analyzes a Python script for potential AI hallucinations by validating
    imports, method calls, class instantiations, and function calls against a Neo4j
    knowledge graph containing real repository data.
    
    The tool performs comprehensive analysis including:
    - Import validation against known repositories
    - Method call validation on classes from the knowledge graph
    - Class instantiation parameter validation
    - Function call parameter validation
    - Attribute access validation
    
    Args:
        ctx: The MCP server provided context
        script_path: Absolute path to the Python script to analyze
    
    Returns:
        JSON string with hallucination detection results, confidence scores, and recommendations
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)
        
        # Get the knowledge validator from context
        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator
        
        if not knowledge_validator:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph validator not available. Check Neo4j configuration in environment variables."
            }, indent=2)
        
        # Validate script path
        validation = validate_script_path(script_path)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "script_path": script_path,
                "error": validation["error"]
            }, indent=2)
        
        # Step 1: Analyze script structure using AST
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)
        
        if analysis_result.errors:
            print(f"Analysis warnings for {script_path}: {analysis_result.errors}")
        
        # Step 2: Validate against knowledge graph
        validation_result = await knowledge_validator.validate_script(analysis_result)
        
        # Step 3: Generate comprehensive report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)
        
        # Format response with comprehensive information
        return json.dumps({
            "success": True,
            "script_path": script_path,
            "overall_confidence": validation_result.overall_confidence,
            "validation_summary": {
                "total_validations": report["validation_summary"]["total_validations"],
                "valid_count": report["validation_summary"]["valid_count"],
                "invalid_count": report["validation_summary"]["invalid_count"],
                "uncertain_count": report["validation_summary"]["uncertain_count"],
                "not_found_count": report["validation_summary"]["not_found_count"],
                "hallucination_rate": report["validation_summary"]["hallucination_rate"]
            },
            "hallucinations_detected": report["hallucinations_detected"],
            "recommendations": report["recommendations"],
            "analysis_metadata": {
                "total_imports": report["analysis_metadata"]["total_imports"],
                "total_classes": report["analysis_metadata"]["total_classes"],
                "total_methods": report["analysis_metadata"]["total_methods"],
                "total_attributes": report["analysis_metadata"]["total_attributes"],
                "total_functions": report["analysis_metadata"]["total_functions"]
            },
            "libraries_analyzed": report.get("libraries_analyzed", [])
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "script_path": script_path,
            "error": f"Analysis failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def query_knowledge_graph(ctx: Context, command: str) -> str:
    """
    Query and explore the Neo4j knowledge graph containing repository data.
    
    This tool provides comprehensive access to the knowledge graph for exploring repositories,
    classes, methods, functions, and their relationships. Perfect for understanding what data
    is available for hallucination detection and debugging validation results.
    
    **⚠️ IMPORTANT: Always start with the `repos` command first!**
    Before using any other commands, run `repos` to see what repositories are available
    in your knowledge graph. This will help you understand what data you can explore.
    
    ## Available Commands:
    
    **Repository Commands:**
    - `repos` - **START HERE!** List all repositories in the knowledge graph
    - `explore <repo_name>` - Get detailed overview of a specific repository
    
    **Class Commands:**  
    - `classes` - List all classes across all repositories (limited to 20)
    - `classes <repo_name>` - List classes in a specific repository
    - `class <class_name>` - Get detailed information about a specific class including methods and attributes
    
    **Method Commands:**
    - `method <method_name>` - Search for methods by name across all classes
    - `method <method_name> <class_name>` - Search for a method within a specific class
    
    **Custom Query:**
    - `query <cypher_query>` - Execute a custom Cypher query (results limited to 20 records)
    
    ## Knowledge Graph Schema:
    
    **Node Types:**
    - Repository: `(r:Repository {name: string})`
    - File: `(f:File {path: string, module_name: string})`
    - Class: `(c:Class {name: string, full_name: string})`
    - Method: `(m:Method {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Function: `(func:Function {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Attribute: `(a:Attribute {name: string, type: string})`
    
    **Relationships:**
    - `(r:Repository)-[:CONTAINS]->(f:File)`
    - `(f:File)-[:DEFINES]->(c:Class)`
    - `(c:Class)-[:HAS_METHOD]->(m:Method)`
    - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
    - `(f:File)-[:DEFINES]->(func:Function)`
    
    ## Example Workflow:
    ```
    1. repos                                    # See what repositories are available
    2. explore pydantic-ai                      # Explore a specific repository
    3. classes pydantic-ai                      # List classes in that repository
    4. class Agent                              # Explore the Agent class
    5. method run_stream                        # Search for run_stream method
    6. method __init__ Agent                    # Find Agent constructor
    7. query "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name = 'run' RETURN c.name, m.name LIMIT 5"
    ```
    
    Args:
        ctx: The MCP server provided context
        command: Command string to execute (see available commands above)
    
    Returns:
        JSON string with query results, statistics, and metadata
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)
        
        # Get Neo4j driver from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor or not repo_extractor.driver:
            return json.dumps({
                "success": False,
                "error": "Neo4j connection not available. Check Neo4j configuration in environment variables."
            }, indent=2)
        
        # Parse command
        command = command.strip()
        if not command:
            return json.dumps({
                "success": False,
                "command": "",
                "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
            }, indent=2)
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        async with repo_extractor.driver.session() as session:
            # Route to appropriate handler
            if cmd == "repos":
                return await _handle_repos_command(session, command)
            elif cmd == "explore":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Repository name required. Usage: explore <repo_name>"
                    }, indent=2)
                return await _handle_explore_command(session, command, args[0])
            elif cmd == "classes":
                repo_name = args[0] if args else None
                return await _handle_classes_command(session, command, repo_name)
            elif cmd == "class":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Class name required. Usage: class <class_name>"
                    }, indent=2)
                return await _handle_class_command(session, command, args[0])
            elif cmd == "method":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Method name required. Usage: method <method_name> [class_name]"
                    }, indent=2)
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await _handle_method_command(session, command, method_name, class_name)
            elif cmd == "query":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Cypher query required. Usage: query <cypher_query>"
                    }, indent=2)
                cypher_query = " ".join(args)
                return await _handle_query_command(session, command, cypher_query)
            else:
                return json.dumps({
                    "success": False,
                    "command": command,
                    "error": f"Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
                }, indent=2)
                
    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Query execution failed: {str(e)}"
        }, indent=2)


async def _handle_repos_command(session, command: str) -> str:
    """Handle 'repos' command - list all repositories"""
    query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
    result = await session.run(query)
    
    repos = []
    async for record in result:
        repos.append(record['name'])
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repositories": repos
        },
        "metadata": {
            "total_results": len(repos),
            "limited": False
        }
    }, indent=2)


async def _handle_explore_command(session, command: str, repo_name: str) -> str:
    """Handle 'explore <repo>' command - get repository overview"""
    # Check if repository exists
    repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
    result = await session.run(repo_check_query, repo_name=repo_name)
    repo_record = await result.single()
    
    if not repo_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Repository '{repo_name}' not found in knowledge graph"
        }, indent=2)
    
    # Get file count
    files_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
    RETURN count(f) as file_count
    """
    result = await session.run(files_query, repo_name=repo_name)
    file_count = (await result.single())['file_count']
    
    # Get class count
    classes_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
    RETURN count(DISTINCT c) as class_count
    """
    result = await session.run(classes_query, repo_name=repo_name)
    class_count = (await result.single())['class_count']
    
    # Get function count
    functions_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
    RETURN count(DISTINCT func) as function_count
    """
    result = await session.run(functions_query, repo_name=repo_name)
    function_count = (await result.single())['function_count']
    
    # Get method count
    methods_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
    RETURN count(DISTINCT m) as method_count
    """
    result = await session.run(methods_query, repo_name=repo_name)
    method_count = (await result.single())['method_count']
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repository": repo_name,
            "statistics": {
                "files": file_count,
                "classes": class_count,
                "functions": function_count,
                "methods": method_count
            }
        },
        "metadata": {
            "total_results": 1,
            "limited": False
        }
    }, indent=2)


async def _handle_classes_command(session, command: str, repo_name: str = None) -> str:
    """Handle 'classes [repo]' command - list classes"""
    limit = 20
    
    if repo_name:
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, repo_name=repo_name, limit=limit)
    else:
        query = """
        MATCH (c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, limit=limit)
    
    classes = []
    async for record in result:
        classes.append({
            'name': record['name'],
            'full_name': record['full_name']
        })
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "classes": classes,
            "repository_filter": repo_name
        },
        "metadata": {
            "total_results": len(classes),
            "limited": len(classes) >= limit
        }
    }, indent=2)


async def _handle_class_command(session, command: str, class_name: str) -> str:
    """Handle 'class <name>' command - explore specific class"""
    # Find the class
    class_query = """
    MATCH (c:Class)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN c.name as name, c.full_name as full_name
    LIMIT 1
    """
    result = await session.run(class_query, class_name=class_name)
    class_record = await result.single()
    
    if not class_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Class '{class_name}' not found in knowledge graph"
        }, indent=2)
    
    actual_name = class_record['name']
    full_name = class_record['full_name']
    
    # Get methods
    methods_query = """
    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
    ORDER BY m.name
    """
    result = await session.run(methods_query, class_name=class_name)
    
    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'name': record['name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any'
        })
    
    # Get attributes
    attributes_query = """
    MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN a.name as name, a.type as type
    ORDER BY a.name
    """
    result = await session.run(attributes_query, class_name=class_name)
    
    attributes = []
    async for record in result:
        attributes.append({
            'name': record['name'],
            'type': record['type'] or 'Any'
        })
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "class": {
                "name": actual_name,
                "full_name": full_name,
                "methods": methods,
                "attributes": attributes
            }
        },
        "metadata": {
            "total_results": 1,
            "methods_count": len(methods),
            "attributes_count": len(attributes),
            "limited": False
        }
    }, indent=2)


async def _handle_method_command(session, command: str, method_name: str, class_name: str = None) -> str:
    """Handle 'method <name> [class]' command - search for methods"""
    if class_name:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        """
        result = await session.run(query, class_name=class_name, method_name=method_name)
    else:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        ORDER BY c.name
        LIMIT 20
        """
        result = await session.run(query, method_name=method_name)
    
    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'class_name': record['class_name'],
            'class_full_name': record['class_full_name'],
            'method_name': record['method_name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any',
            'legacy_args': record['args'] or []
        })
    
    if not methods:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Method '{method_name}'" + (f" in class '{class_name}'" if class_name else "") + " not found"
        }, indent=2)
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "methods": methods,
            "class_filter": class_name
        },
        "metadata": {
            "total_results": len(methods),
            "limited": len(methods) >= 20 and not class_name
        }
    }, indent=2)


async def _handle_query_command(session, command: str, cypher_query: str) -> str:
    """Handle 'query <cypher>' command - execute custom Cypher query"""
    try:
        # Execute the query with a limit to prevent overwhelming responses
        result = await session.run(cypher_query)
        
        records = []
        count = 0
        async for record in result:
            records.append(dict(record))
            count += 1
            if count >= 20:  # Limit results to prevent overwhelming responses
                break
        
        return json.dumps({
            "success": True,
            "command": command,
            "data": {
                "query": cypher_query,
                "results": records
            },
            "metadata": {
                "total_results": len(records),
                "limited": len(records) >= 20
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Cypher query error: {str(e)}",
            "data": {
                "query": cypher_query
            }
        }, indent=2)


@mcp.tool()
async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    """
    Parse a GitHub repository into the Neo4j knowledge graph.
    
    This tool clones a GitHub repository, analyzes its Python files, and stores
    the code structure (classes, methods, functions, imports) in Neo4j for use
    in hallucination detection. The tool:
    
    - Clones the repository to a temporary location
    - Analyzes Python files to extract code structure
    - Stores classes, methods, functions, and imports in Neo4j
    - Provides detailed statistics about the parsing results
    - Automatically handles module name detection for imports
    
    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
    
    Returns:
        JSON string with parsing results, statistics, and repository information
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)
        
        # Get the repository extractor from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        
        if not repo_extractor:
            return json.dumps({
                "success": False,
                "error": "Repository extractor not available. Check Neo4j configuration in environment variables."
            }, indent=2)
        
        # Validate repository URL
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": validation["error"]
            }, indent=2)
        
        repo_name = validation["repo_name"]
        
        # Parse the repository (this includes cloning, analysis, and Neo4j storage)
        print(f"Starting repository analysis for: {repo_name}")
        await repo_extractor.analyze_repository(repo_url)
        print(f"Repository analysis completed for: {repo_name}")
        
        # Query Neo4j for statistics about the parsed repository
        async with repo_extractor.driver.session() as session:
            # Get comprehensive repository statistics
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            WITH r, 
                 count(DISTINCT f) as files_count,
                 count(DISTINCT c) as classes_count,
                 count(DISTINCT m) as methods_count,
                 count(DISTINCT func) as functions_count,
                 count(DISTINCT a) as attributes_count
            
            // Get some sample module names
            OPTIONAL MATCH (r)-[:CONTAINS]->(sample_f:File)
            WITH r, files_count, classes_count, methods_count, functions_count, attributes_count,
                 collect(DISTINCT sample_f.module_name)[0..5] as sample_modules
            
            RETURN 
                r.name as repo_name,
                files_count,
                classes_count, 
                methods_count,
                functions_count,
                attributes_count,
                sample_modules
            """
            
            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()
            
            if record:
                stats = {
                    "repository": record['repo_name'],
                    "files_processed": record['files_count'],
                    "classes_created": record['classes_count'],
                    "methods_created": record['methods_count'], 
                    "functions_created": record['functions_count'],
                    "attributes_created": record['attributes_count'],
                    "sample_modules": record['sample_modules'] or []
                }
            else:
                return json.dumps({
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Repository '{repo_name}' not found in database after parsing"
                }, indent=2)
        
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
            "statistics": stats,
            "ready_for_validation": True,
            "next_steps": [
                "Repository is now available for hallucination detection",
                f"Use check_ai_script_hallucinations to validate scripts against {repo_name}",
                "The knowledge graph contains classes, methods, and functions from this repository"
            ]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Repository parsing failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def extract_code_widgets(ctx: Context, url: str, languages: str = "python,javascript,shell,go,kotlin") -> str:
    """
    Extract code examples from interactive widgets on a documentation page.
    
    This tool should be called AFTER crawl_single_page has been run on the same URL
    to ensure the source exists in the database. It uses LLM reasoning to detect 
    interactive code widgets (tabs, dropdowns, etc.) and extracts code examples
    for the specified programming languages.
    
    Perfect for documentation sites like Polygon.io, Stripe, GitHub docs, etc.
    that have interactive code examples in multiple languages.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the documentation page with code widgets
        languages: Comma-separated list of languages to extract (e.g., "python,go" or "javascript,shell,kotlin")
    
    Returns:
        JSON string with extraction results and storage summary
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Parse languages list
        target_language_list = [lang.strip().lower() for lang in languages.split(',') if lang.strip()]
        
        if not target_language_list:
            return json.dumps({
                "success": False,
                "error": "No languages specified"
            }, indent=2)
        
        print(f"🎯 Extracting code widgets for languages: {target_language_list}")
        
        # Step 1: Initial crawl to get HTML for widget detection
        initial_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=initial_config)
        
        if not result.success:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Failed to crawl page: {result.error_message}"
            }, indent=2)
        
        # Step 2: Detect code widgets using LLM
        print("🔍 Detecting code widgets...")
        widget_extractor = CodeWidgetExtractor()
        detection_result = await widget_extractor.detect_code_widgets(
            result.html or "", 
            url
        )
        
        widgets_detected = len(detection_result.widgets_found)
        widget_confidence = detection_result.confidence_score
        
        if widgets_detected == 0:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No interactive code widgets detected on this page",
                "confidence_score": widget_confidence,
                "suggestion": "This page may not have interactive code examples, or they may be in a format not yet supported"
            }, indent=2)
        
        if widget_confidence < 0.2:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Low confidence in widget detection (score: {widget_confidence})",
                "widgets_detected": widgets_detected,
                "suggestion": "The detected widgets may not be reliable. Try a different page or contact support."
            }, indent=2)
        
        print(f"✅ Detected {widgets_detected} widgets with confidence {widget_confidence}")
        
        # Step 3: Enhanced crawl with language-specific widget interaction
        print(f"🚀 Enhanced crawl targeting: {', '.join(target_language_list)}")
        
        enhanced_config = create_enhanced_crawler_config(
            widget_extractor, 
            detection_result.widgets_found,
            target_language_list
        )
        
        enhanced_result = await crawler.arun(url=url, config=enhanced_config)
        
        if not enhanced_result.success:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Enhanced crawl failed: {enhanced_result.error_message}",
                "widgets_detected": widgets_detected,
                "fallback_available": False
            }, indent=2)
        
        print(f"✅ Enhanced crawl successful: {len(enhanced_result.markdown)} chars")
        
        # Step 4: Extract code blocks with lower threshold for widget content
        code_blocks = extract_code_blocks(enhanced_result.markdown, min_length=50)
        
        if not code_blocks:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No code examples found after widget interaction",
                "widgets_detected": widgets_detected,
                "confidence_score": widget_confidence,
                "content_length": len(enhanced_result.markdown)
            }, indent=2)
        
        print(f"📦 Found {len(code_blocks)} code blocks")
        
        # Step 5: Filter and prepare code examples for target languages
        filtered_blocks = []
        seen_code_hashes = set()  # Track duplicates
        
        for block in code_blocks:
            code_content = block['code'].lower()
            block_language = block.get('language', '').lower()
            
            # Clean up the code by removing our injection markers and UI text
            clean_code = block['code']
            
            # Remove our extraction markers
            if "EXTRACTED: " in clean_code:
                lines = clean_code.split('\n')
                clean_lines = []
                skip_until_code = False
                
                for line in lines:
                    if line.strip().startswith("EXTRACTED:"):
                        skip_until_code = True
                        continue
                    elif skip_until_code and (line.strip().startswith("Language:") or line.strip() == ""):
                        continue
                    else:
                        skip_until_code = False
                        clean_lines.append(line)
                
                clean_code = '\n'.join(clean_lines)
            
            # Remove UI text pollution at the start
            if clean_code.startswith('ShellPythonGoJavaScriptKotlin'):
                clean_code = clean_code[len('ShellPythonGoJavaScriptKotlin'):]
            if clean_code.startswith('PythonCopied'):
                clean_code = clean_code[len('PythonCopied'):]
            
            clean_code = clean_code.strip()
            
            # Create a hash for deduplication (first 200 chars)
            code_hash = hash(clean_code[:200])
            if code_hash in seen_code_hashes:
                print(f"  🔄 Skipping duplicate code block")
                continue
            
            seen_code_hashes.add(code_hash)
            
            # Intelligently separate code from response objects
            formatted_code = clean_code
            
            # Check if this contains both code and JSON response
            if '{' in clean_code and ('"ev"' in clean_code or '"fmv"' in clean_code or '"sym"' in clean_code):
                # Split into code section and response section
                lines = clean_code.split('\n')
                code_lines = []
                response_lines = []
                in_json = False
                json_brace_count = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Detect start of JSON (starts with { and contains typical response fields)
                    if stripped.startswith('{') and not in_json:
                        # Check if this looks like a response object
                        next_few_lines = '\n'.join(lines[lines.index(line):lines.index(line)+5] if lines.index(line)+5 < len(lines) else lines[lines.index(line):])
                        if any(field in next_few_lines for field in ['"ev"', '"fmv"', '"sym"', '"timestamp"', '"response"']):
                            in_json = True
                            json_brace_count = 1
                            response_lines.append(line)
                            continue
                    
                    if in_json:
                        response_lines.append(line)
                        json_brace_count += line.count('{') - line.count('}')
                        if json_brace_count <= 0:
                            in_json = False
                    else:
                        # Skip empty lines at the end of code section
                        if stripped or code_lines:
                            code_lines.append(line)
                
                # Clean up trailing empty lines from code
                while code_lines and not code_lines[-1].strip():
                    code_lines.pop()
                
                # Format the content with proper sections
                if code_lines and response_lines:
                    python_code = '\n'.join(code_lines)
                    json_response = '\n'.join(response_lines)
                    
                    formatted_code = f"""## Code Example

```python
{python_code}
```

## Response Object

```json
{json_response}
```"""
                    
                    print(f"  📝 Split into code ({len(python_code)} chars) + response ({len(json_response)} chars)")
            
            # Update the block with formatted code
            block['code'] = formatted_code
            code_content = formatted_code.lower()
            
            # Enhanced language detection with smart filtering
            language_match = None
            for target_lang in target_language_list:
                if target_lang == block_language:
                    language_match = target_lang
                    break
                elif target_lang == 'python':
                    # Check for Python indicators
                    has_python = any(indicator in code_content for indicator in ['import ', 'def ', 'print(', 'from '])
                    
                    if has_python:
                        # Check if it's actually Kotlin disguised (package statement + kotlin-specific syntax)
                        has_kotlin_package = 'package io.polygon.kotlin' in code_content or 'suspend fun' in code_content
                        
                        # Check if it's UI text pollution (language list at start)
                        starts_with_ui_text = code_content.startswith('shellpythongojava') or code_content.startswith('shell\npython\ngo')
                        
                        if not has_kotlin_package:
                            language_match = target_lang
                            break
                elif target_lang == 'javascript':
                    has_js = any(indicator in code_content for indicator in ['function', 'const ', 'let ', 'var ', 'console.log', 'import {'])
                    # Exclude if it has Python-style imports without JS syntax
                    has_python_imports = 'from polygon import' in code_content
                    
                    if has_js and not has_python_imports:
                        language_match = target_lang
                        break
                elif target_lang == 'shell' and any(indicator in code_content for indicator in ['curl ', 'wget ', '#!/bin/bash', '$ ']):
                    language_match = target_lang
                    break
                elif target_lang == 'go':
                    has_go = any(indicator in code_content for indicator in ['package ', 'func ', 'fmt.'])
                    # Make sure it's not Python or Kotlin
                    has_python_syntax = 'from ' in code_content and 'import ' in code_content
                    has_kotlin_syntax = 'suspend fun' in code_content
                    
                    if has_go and not has_python_syntax and not has_kotlin_syntax:
                        language_match = target_lang
                        break
                elif target_lang == 'kotlin':
                    has_kotlin = any(indicator in code_content for indicator in ['suspend fun', 'kotlinx', 'package io.polygon.kotlin'])
                    
                    if has_kotlin:
                        language_match = target_lang
                        break
            
            if language_match:
                block['detected_language'] = language_match
                filtered_blocks.append(block)
        
        if not filtered_blocks:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"No code examples found for target languages: {', '.join(target_language_list)}",
                "total_code_blocks": len(code_blocks),
                "target_languages": target_language_list,
                "suggestion": "Try different languages or check if the page has examples in the requested languages"
            }, indent=2)
        
        print(f"🎯 Filtered to {len(filtered_blocks)} blocks for target languages")
        
        # Step 6: Prepare data for storage
        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path
        
        code_urls = []
        code_chunk_numbers = []
        code_examples = []
        code_summaries = []
        code_metadatas = []
        
        # Process code examples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            summary_args = [(block['code'], block['context_before'], block['context_after']) 
                            for block in filtered_blocks]
            summaries = list(executor.map(process_code_example, summary_args))
        
        # Prepare code example data
        for i, (block, summary) in enumerate(zip(filtered_blocks, summaries)):
            code_urls.append(url)
            code_chunk_numbers.append(i)
            code_examples.append(block['code'])
            code_summaries.append(summary)
            
            code_meta = {
                "chunk_index": i,
                "url": url,
                "source": source_id,
                "char_count": len(block['code']),
                "word_count": len(block['code'].split()),
                "language": block['detected_language'],
                "extraction_method": "widget_interaction",
                "widget_extracted": True,
                "target_languages": target_language_list,
                "widget_confidence": widget_confidence
            }
            code_metadatas.append(code_meta)
        
        # Step 7: Store in Supabase
        print(f"💾 Storing {len(filtered_blocks)} code examples...")
        
        add_code_examples_to_supabase(
            supabase_client, 
            code_urls, 
            code_chunk_numbers, 
            code_examples, 
            code_summaries, 
            code_metadatas
        )
        
        # Step 8: Verify storage and return results
        await asyncio.sleep(1)  # Brief wait for database consistency
        
        recent_codes = supabase_client.table('code_examples').select('*').eq('source_id', source_id).ilike('metadata->>extraction_method', 'widget_interaction').execute()
        stored_count = len(recent_codes.data) if recent_codes.data else 0
        
        # Summary by language
        language_summary = {}
        for block in filtered_blocks:
            lang = block['detected_language']
            if lang not in language_summary:
                language_summary[lang] = []
            language_summary[lang].append({
                "length": len(block['code']),
                "preview": block['code'][:100].replace('\n', ' ') + ("..." if len(block['code']) > 100 else "")
            })
        
        return json.dumps({
            "success": True,
            "url": url,
            "widgets_detected": widgets_detected,
            "widget_confidence": widget_confidence,
            "target_languages": target_language_list,
            "code_examples_extracted": len(filtered_blocks),
            "code_examples_stored": stored_count,
            "total_characters": sum(len(block['code']) for block in filtered_blocks),
            "language_breakdown": language_summary,
            "extraction_method": "widget_interaction",
            "source_id": source_id
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())