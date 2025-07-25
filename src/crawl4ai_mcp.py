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

# Conditional import for Firecrawl
FIRECRAWL_AVAILABLE = os.getenv("FIRECRAWL_API_KEY") is not None
if FIRECRAWL_AVAILABLE:
    from firecrawl import FirecrawlApp
    
# OpenAI import for LLM cleaning
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from utils import (
    get_supabase_client, 
    add_documents_to_supabase, 
    search_documents,
    extract_code_blocks_with_llm,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples,
    chunk_document_single,
    chunk_document_llm,
    smart_chunk_markdown
)

# Import knowledge graph modules
from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor
from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter

def clean_markdown_for_rag(markdown_content: str, url: str) -> str:
    """
    Clean and optimize markdown content for RAG consumption using an LLM.
    
    Args:
        markdown_content: Raw markdown content from web scraping
        url: Source URL for context
        
    Returns:
        Cleaned and structured markdown optimized for RAG
    """
    if not OPENAI_AVAILABLE:
        return markdown_content
    
    try:
        # Get OpenAI configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
        
        if not api_key:
            return markdown_content
            
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = """You are an expert content curator specializing in optimizing technical documentation for RAG (Retrieval-Augmented Generation) systems.

Your task is to clean and restructure markdown content to be maximally useful for LLM consumption while preserving all technical information.

REMOVE these elements:
- Navigation breadcrumbs and UI elements
- Duplicate or redundant headings
- Marketing CTAs and signup prompts
- Copy buttons, "Copied" indicators, and UI artifacts
- Social sharing buttons and feedback prompts
- Irrelevant footer content and support links
- Broken or incomplete sentences from HTML parsing

ENHANCE these elements:
- Create clear hierarchical structure with logical H1/H2/H3 headings
- Consolidate fragmented information into coherent sections
- Add context to code examples and technical specifications
- Ensure all API parameters, responses, and examples are properly documented
- Structure content for easy chunking and semantic search
- Preserve all technical details, URLs, and reference links

STRUCTURE the output with these sections (when applicable):
1. Overview/Description
2. Use Cases/Applications  
3. Technical Specifications/Parameters
4. API Endpoints/Methods
5. Code Examples/Implementation
6. Response Formats/Data Structures
7. Related Resources/References

Maintain technical accuracy while making the content more consumable for RAG systems."""

        user_prompt = f"""Please clean and optimize this markdown content from {url} for RAG consumption:

{markdown_content}

Return only the cleaned markdown content with no additional commentary or metadata headers. The content should be ready for direct indexing into a RAG knowledge base."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        cleaned_content = response.choices[0].message.content.strip()
        return cleaned_content
        
    except Exception as e:
        # If LLM cleaning fails, return original content
        print(f"Warning: LLM cleaning failed: {e}")
        return markdown_content

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
            repo_extractor=repo_extractor
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


# Only register the original crawl_single_page if Firecrawl is not available
if not FIRECRAWL_AVAILABLE:
    @mcp.tool()
    async def crawl_single_page(ctx: Context, url: str) -> str:
        """
        Crawl a single web page and store its content in Supabase.
        
        This tool is ideal for quickly retrieving content from a specific URL without following links.
        The content is stored in Supabase for later retrieval and querying.
        
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
                    code_blocks = extract_code_blocks_with_llm(result.markdown, url)
                    if code_blocks:
                        code_urls = []
                        code_chunk_numbers = []
                        code_examples = []
                        code_summaries = []
                        code_metadatas = []
                        
                        # Prepare code example data - LLM already provides summaries
                        for i, block in enumerate(code_blocks):
                            code_urls.append(url)
                            code_chunk_numbers.append(i)
                            code_examples.append(block['code'])
                            code_summaries.append(block.get('summary', f"{block.get('language', 'code')} example"))
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": i,
                                "url": url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split()),
                                "language": block.get('language', ''),
                                "type": block.get('type', 'code_block'),
                                "extraction_method": "llm"
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

@mcp.tool()
async def process_local_markdown(
    ctx: Context,
    file_path: str,
    # Processing Options
    chunking_strategy: str = "character",
    chunk_size: int = 5000,
    chunking_prompt: Optional[str] = None,
    title: Optional[str] = None,
    source_name: Optional[str] = None
) -> str:
    """
    Process local markdown files for RAG using the same smart agentic methods as scrape_single_page.
    
    This tool reads local markdown files and processes them using LLM-based code block detection,
    content optimization, and flexible chunking strategies. Content is cleaned and optimized
    for RAG when USE_AGENTIC_RAG is enabled.
    
    Args:
        ctx: The MCP server provided context
        file_path: Path to the local markdown file to process
        
        # CHUNKING OPTIONS:
        chunking_strategy: Strategy for chunking content - "character" (default), "document", or "llm"
            - "character": Smart character-based chunking with configurable size (respects code blocks/paragraphs)
            - "document": Store entire document as single chunk (useful for known document types)
            - "llm": Use LLM to intelligently determine chunk boundaries based on content structure
        chunk_size: For character strategy - maximum characters per chunk (default: 5000)
        chunking_prompt: For LLM strategy - custom prompt to guide chunking decisions (optional)
        title: Optional title for the document (defaults to filename if not provided)
        source_name: Optional source name for organization (defaults to file_path if not provided)
    
    Returns:
        JSON string with processing summary including: success status, file processed, 
        content chunks stored, code blocks found, and processing parameters used
    """
    try:
        # Get Supabase client from MCP context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Import and call the standalone function - single source of truth!
        from markdown_processor import process_local_markdown_standalone
        
        result = await process_local_markdown_standalone(
            file_path=file_path,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunking_prompt=chunking_prompt,
            title=title,
            source_name=source_name,
            supabase_client=supabase_client
        )
        
        return result
        
    except Exception as e:
        import json
        return json.dumps({
            "success": False,
            "file_path": file_path,
            "error": f"MCP tool error: {str(e)}"
        }, indent=2)

# Firecrawl version of crawl_single_page
if FIRECRAWL_AVAILABLE:
    @mcp.tool()
    async def scrape_single_page(
        ctx: Context,
        url: str,
        format: str = "markdown",
        agent_model: str = "FIRE-1",
        agent_prompt: Optional[str] = None,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        wait_for: int = 1000,
        timeout: int = 15000,
        # Chunking Options
        chunking_strategy: str = "character",
        chunk_size: int = 5000,
        chunking_prompt: Optional[str] = None
    ) -> str:
        """
        Crawl a single web page using Firecrawl and store its content in Supabase.
        
        This tool uses Firecrawl for web crawling with optional AI agent capabilities.
        When agent_prompt is provided, it uses the agent mode to interact with the page.
        The scraped markdown content is automatically cleaned and optimized for RAG using an LLM 
        when USE_AGENTIC_RAG environment variable is set to "true".
        
        Args:
            ctx: The MCP server provided context
            url: URL of the web page to crawl
            format: Format for the scraped content (default: "markdown")
            agent_model: AI model to use for agent mode (default: "FIRE-1")
            agent_prompt: Optional prompt for agent mode. If provided, enables agent mode.
            only_main_content: Extract only main content, removing ads/navigation (default: True)
            include_tags: List of HTML tags/selectors to include (e.g., ["h1", "p", ".main-content"])
            exclude_tags: List of HTML tags/selectors to exclude (e.g., ["#ad", "#footer"])
            wait_for: Time to wait after page load in milliseconds (default: 1000)
            timeout: Maximum time to wait for page load in milliseconds (default: 15000)
            
            # Chunking Options:
            chunking_strategy: Strategy for chunking content - "character" (default), "document", or "llm"
                - "character": Smart character-based chunking with configurable size (respects code blocks/paragraphs)
                - "document": Store entire document as single chunk (useful for known document types)
                - "llm": Use LLM to intelligently determine chunk boundaries based on content structure
            chunk_size: For character strategy - maximum characters per chunk (default: 5000)
            chunking_prompt: For LLM strategy - custom prompt to guide chunking decisions (optional)
        
        Returns:
            Summary of the crawling operation and storage in Supabase
        """
        try:
            # Initialize Firecrawl
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "FIRECRAWL_API_KEY not found in environment variables"
                }, indent=2)
                
            app = FirecrawlApp(api_key=api_key)
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            
            # Configure scrape options
            scrape_options = {
                'formats': [format] if format != "markdown" else ['markdown'],
                'onlyMainContent': only_main_content,
                'waitFor': wait_for,
                'timeout': timeout
            }
            
            # Add include/exclude tags if provided
            if include_tags:
                scrape_options['includeTags'] = include_tags
            if exclude_tags:
                scrape_options['excludeTags'] = exclude_tags
            
            # Add agent configuration if prompt is provided
            if agent_prompt:
                scrape_options['agent'] = {
                    'model': agent_model,
                    'prompt': agent_prompt
                }
            
            # Scrape the page
            result = app.scrape_url(url, **scrape_options)
            
            if result and hasattr(result, 'markdown') and result.markdown:
                raw_markdown_content = result.markdown
                
                # Clean markdown content for RAG using LLM if agentic RAG is enabled
                use_agentic_rag = os.getenv("USE_AGENTIC_RAG", "false") == "true"
                if use_agentic_rag:
                    markdown_content = clean_markdown_for_rag(raw_markdown_content, url)
                else:
                    markdown_content = raw_markdown_content
                
                # Extract source_id
                parsed_url = urlparse(url)
                source_id = parsed_url.netloc or parsed_url.path
                
                # Chunk the content using the specified strategy
                if chunking_strategy == "document":
                    chunks = chunk_document_single(markdown_content)
                    print(f"Using document chunking strategy: {len(chunks)} chunk(s)")
                elif chunking_strategy == "llm":
                    chunks = chunk_document_llm(markdown_content, chunking_prompt, url)
                    print(f"Using LLM chunking strategy: {len(chunks)} chunk(s)")
                else:  # Default to "character" strategy
                    chunks = smart_chunk_markdown(markdown_content, chunk_size)
                    print(f"Using character chunking strategy (size: {chunk_size}): {len(chunks)} chunk(s)")
                
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
                    meta["crawl_time"] = "firecrawl_scrape"
                    meta["crawl_method"] = "firecrawl"
                    if agent_prompt:
                        meta["agent_used"] = True
                        meta["agent_model"] = agent_model
                    metadatas.append(meta)
                    
                    # Accumulate word count
                    total_word_count += meta.get("word_count", 0)
                
                # Create url_to_full_document mapping
                url_to_full_document = {url: markdown_content}
                
                # Update source information FIRST (before inserting documents)
                source_summary = extract_source_summary(source_id, markdown_content[:5000])
                update_source_info(supabase_client, source_id, source_summary, total_word_count)
                
                # Add documentation chunks to Supabase
                add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
                
                # Extract and process code examples only if enabled
                extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
                code_blocks = []
                if extract_code_examples:
                    code_blocks = extract_code_blocks_with_llm(markdown_content, url)
                    if code_blocks:
                        code_urls = []
                        code_chunk_numbers = []
                        code_examples = []
                        code_summaries = []
                        code_metadatas = []
                        
                        # Prepare code example data - LLM already provides summaries
                        for i, block in enumerate(code_blocks):
                            code_urls.append(url)
                            code_chunk_numbers.append(i)
                            code_examples.append(block['code'])
                            code_summaries.append(block.get('summary', f"{block.get('language', 'code')} example"))
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": i,
                                "url": url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split()),
                                "crawl_method": "firecrawl",
                                "language": block.get('language', ''),
                                "type": block.get('type', 'code_block'),
                                "extraction_method": "llm"
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
                    "content_length": len(markdown_content),
                    "raw_content_length": len(raw_markdown_content),
                    "total_word_count": total_word_count,
                    "source_id": source_id,
                    "crawl_method": "firecrawl",
                    "agent_used": agent_prompt is not None,
                    "agent_model": agent_model if agent_prompt else None,
                    "llm_cleaned": use_agentic_rag and OPENAI_AVAILABLE,
                    "format": format,
                    "chunking_strategy": chunking_strategy,
                    "chunk_size": chunk_size if chunking_strategy == "character" else None,
                    "chunking_prompt_used": chunking_prompt is not None if chunking_strategy == "llm" else None
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No markdown content returned from Firecrawl"
                }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)

# Conditional tools based on Firecrawl availability
if FIRECRAWL_AVAILABLE:
    @mcp.tool()
    async def crawl_web_sync(
        ctx: Context,
        url: str,
        # Crawler Options
        limit: int = 10,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_depth: int = 5,
        allow_backward_links: bool = False,
        allow_external_links: bool = False,
        allow_subdomains: bool = False,
        delay: Optional[int] = None,
        # Scrape Options (same as scrape_single_page)
        format: str = "markdown",
        agent_model: str = "FIRE-1",
        agent_prompt: Optional[str] = None,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        wait_for: int = 2000,
        timeout: int = 15000,
        # Storage Options
        chunk_size: int = 5000,
        # Chunking Options
        chunking_strategy: str = "character",
        chunking_prompt: Optional[str] = None
    ) -> str:
        """
        Synchronously crawl multiple web pages using Firecrawl and store content in Supabase.
        
        This tool uses Firecrawl to crawl a website starting from the given URL and following
        links within the specified constraints. All scraped content is automatically cleaned
        and optimized for RAG when USE_AGENTIC_RAG is enabled.
        
        Args:
            ctx: The MCP server provided context
            url: Starting URL to crawl from
            
            # CRAWLER OPTIONS (control which pages to visit):
            limit: Maximum number of pages to crawl. Use lower values for focused crawls, higher for comprehensive site coverage (default: 10)
            include_paths: List of regex patterns for URLs to include. Only URLs matching these patterns will be crawled. Examples: ["^/docs/.*$", "^/api/.*$"] for docs and API pages only
            exclude_paths: List of regex patterns for URLs to exclude. URLs matching these patterns will be skipped. Examples: ["^/admin/.*$", "^/private/.*$"] to avoid admin/private areas
            max_depth: Maximum crawl depth from starting URL. Depth 1 = direct links, depth 2 = links from those pages, etc. Higher values crawl deeper into site structure (default: 5)
            allow_backward_links: Whether to follow internal links to sibling or parent URLs, not just child paths. False = only crawl deeper URLs, True = crawl any internal links including navigation (default: False)
            allow_external_links: Whether to follow links that point to external domains. Be careful as this can cause unlimited crawling (default: False)
            allow_subdomains: Whether to follow links to subdomains of the main domain. For example, if crawling example.com, this allows blog.example.com (default: False)
            delay: Delay in seconds (integer) between page scrapes to respect rate limits and avoid overwhelming the target website. If not provided, respects robots.txt crawl delay (default: None)
            
            # SCRAPE OPTIONS (control how each page is processed):
            format: Format for scraped content. Options: "markdown", "html", "text". Markdown is best for RAG applications (default: "markdown")
            agent_model: AI model to use for agent mode when agent_prompt is provided. Options: "FIRE-1", "gpt-4o", etc. (default: "FIRE-1")
            agent_prompt: Optional prompt for AI agent mode. If provided, the AI agent will interact with the page using this prompt before scraping. Enables dynamic content interaction (default: None)
            only_main_content: Whether to extract only main content, removing navigation, ads, footers, etc. True = cleaner content for RAG, False = full page content (default: True)
            include_tags: List of HTML tags/CSS selectors to specifically include in scraping. Examples: ["h1", "h2", "p", ".main-content", "#article-body"] (default: None = include all)
            exclude_tags: List of HTML tags/CSS selectors to exclude from scraping. Examples: ["#footer", ".ad", ".navigation", "#sidebar"] (default: None = exclude nothing)
            wait_for: Time in milliseconds to wait after page load before scraping. Useful for pages with dynamic content, JavaScript loading, etc. (default: 2000)
            timeout: Maximum time in milliseconds to wait for page load. Increase for slow-loading pages (default: 15000)
            
            # STORAGE OPTIONS:
            chunk_size: Maximum size in characters for each content chunk stored in database. Smaller chunks = more granular search, larger chunks = more context per result (default: 5000)
            
            # CHUNKING OPTIONS:
            chunking_strategy: Strategy for chunking content - "character" (default), "document", or "llm"
                - "character": Smart character-based chunking with configurable size (respects code blocks/paragraphs)
                - "document": Store entire document as single chunk (useful for known document types)
                - "llm": Use LLM to intelligently determine chunk boundaries based on content structure
            chunking_prompt: For LLM strategy - custom prompt to guide chunking decisions (optional)
        
        Returns:
            JSON string with detailed crawl summary including: success status, pages crawled, failed pages, total content chunks stored, code blocks found, list of crawled URLs, and all parameters used
        """
        try:
            # Initialize Firecrawl
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "FIRECRAWL_API_KEY not found in environment variables"
                }, indent=2)
                
            app = FirecrawlApp(api_key=api_key)
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            
            # Configure scrape options using ScrapeOptions class
            from firecrawl.firecrawl import ScrapeOptions
            scrape_options_dict = {
                'formats': [format] if format != "markdown" else ['markdown'],
                'onlyMainContent': only_main_content,
                'waitFor': wait_for,
                'timeout': timeout
            }
            
            # Add include/exclude tags if provided
            if include_tags:
                scrape_options_dict['includeTags'] = include_tags
            if exclude_tags:
                scrape_options_dict['excludeTags'] = exclude_tags
            
            # Add agent configuration if prompt is provided
            if agent_prompt:
                scrape_options_dict['agent'] = {
                    'model': agent_model,
                    'prompt': agent_prompt
                }
            
            scrape_options = ScrapeOptions(**scrape_options_dict)
            
            # Configure crawl parameters with correct parameter names
            crawl_params = {
                'limit': limit,
                'max_depth': max_depth,
                'allow_backward_links': allow_backward_links,
                'allow_external_links': allow_external_links,
                'allow_subdomains': allow_subdomains,
                'scrape_options': scrape_options
            }
            
            # Add path restrictions if provided
            if include_paths:
                crawl_params['include_paths'] = include_paths
            if exclude_paths:
                crawl_params['exclude_paths'] = exclude_paths
            
            # Add delay if provided
            if delay is not None:
                crawl_params['delay'] = delay
            
            # Perform the crawl
            result = app.crawl_url(url, **crawl_params)
            
            if result and hasattr(result, 'data') and result.data:
                # Create source record first (required for foreign key constraint)
                from urllib.parse import urlparse
                from utils import update_source_info
                
                parsed_url = urlparse(url)
                source_id = parsed_url.netloc or parsed_url.path
                try:
                    # Create/update source record to satisfy foreign key constraint
                    update_source_info(supabase_client, source_id, f"Documentation from {source_id}", 0)
                except Exception as source_error:
                    print(f"Warning: Failed to create source record for {source_id}: {source_error}")
                
                # Process each crawled page
                success_count = 0
                error_count = 0
                total_chunks = 0
                code_blocks_found = 0
                crawled_urls = []
                
                for page in result.data:
                    try:
                        # Extract URL from metadata
                        page_url = url  # fallback
                        if hasattr(page, 'metadata') and page.metadata and 'url' in page.metadata:
                            page_url = page.metadata['url']
                        
                        crawled_urls.append(page_url)
                        
                        if hasattr(page, 'markdown') and page.markdown:
                            raw_markdown_content = page.markdown
                            
                            # Clean markdown content for RAG using LLM if agentic RAG is enabled
                            use_agentic_rag = os.getenv("USE_AGENTIC_RAG", "false") == "true"
                            if use_agentic_rag:
                                markdown_content = clean_markdown_for_rag(raw_markdown_content, page_url)
                            else:
                                markdown_content = raw_markdown_content
                            
                            # Extract code blocks using LLM
                            code_blocks = extract_code_blocks_with_llm(markdown_content, page_url)
                            
                            # Store code blocks in database
                            if code_blocks:
                                try:
                                    # Prepare code blocks data for database storage
                                    urls = []
                                    chunk_numbers = []
                                    code_examples = []
                                    summaries = []
                                    metadatas = []
                                    
                                    for i, block in enumerate(code_blocks):
                                        urls.append(page_url)
                                        chunk_numbers.append(i + 1)
                                        code_examples.append(block.get('code', ''))
                                        summaries.append(block.get('summary', ''))
                                        metadatas.append({
                                            'language': block.get('language', ''),
                                            'context': block.get('context', ''),
                                            'type': block.get('type', 'code_block'),
                                            'extraction_method': 'llm'
                                        })
                                    
                                    add_code_examples_to_supabase(
                                        supabase_client, 
                                        urls, 
                                        chunk_numbers, 
                                        code_examples, 
                                        summaries, 
                                        metadatas
                                    )
                                    code_blocks_found += len(code_blocks)  # Only count if successfully stored
                                except Exception as code_error:
                                    print(f"Warning: Failed to store code blocks for {page_url}: {code_error}")
                                    # Continue processing - don't fail the whole page for code block storage issues
                            
                            # Store the main content
                            try:
                                # Extract source_id
                                parsed_url = urlparse(page_url)
                                source_id = parsed_url.netloc or parsed_url.path
                                
                                # Chunk the content using the specified strategy
                                if chunking_strategy == "document":
                                    chunks = chunk_document_single(markdown_content)
                                elif chunking_strategy == "llm":
                                    chunks = chunk_document_llm(markdown_content, chunking_prompt, page_url)
                                else:  # Default to "character" strategy
                                    chunks = smart_chunk_markdown(markdown_content, chunk_size=chunk_size)
                                
                                # Prepare data for Supabase
                                urls = []
                                chunk_numbers = []
                                contents = []
                                metadatas = []
                                url_to_full_document = {}
                                
                                for i, chunk in enumerate(chunks):
                                    urls.append(page_url)
                                    chunk_numbers.append(i + 1)
                                    contents.append(chunk)
                                    metadatas.append({
                                        "extraction_method": "firecrawl_llm"
                                    })
                                    url_to_full_document[page_url] = markdown_content
                                
                                # Add documents to Supabase
                                add_documents_to_supabase(
                                    supabase_client, 
                                    urls, 
                                    chunk_numbers, 
                                    contents, 
                                    metadatas, 
                                    url_to_full_document
                                )
                                total_chunks += len(chunks)
                                success_count += 1  # Only count as success if main content was stored
                            except Exception as content_error:
                                error_count += 1
                                print(f"Error: Failed to store main content for {page_url}: {content_error}")
                                print(f"  Content length: {len(markdown_content)} chars")
                                print(f"  Source ID: {parsed_url.netloc}")
                        else:
                            error_count += 1
                            print(f"Warning: No markdown content found for {page_url}")
                            if hasattr(page, 'markdown'):
                                print(f"  Markdown attribute exists but is empty/falsy: {repr(page.markdown)}")
                            else:
                                print(f"  No markdown attribute found")
                            
                    except Exception as page_error:
                        error_count += 1
                        print(f"Error processing page {page_url}: {page_error}")
                        import traceback
                        traceback.print_exc()
                
                return json.dumps({
                    "success": True,
                    "crawl_type": "firecrawl_sync",
                    "url": url,
                    "pages_crawled": success_count,
                    "pages_failed": error_count,
                    "total_chunks": total_chunks,
                    "code_blocks_found": code_blocks_found,
                    "crawled_urls": crawled_urls[:10],  # Show first 10 URLs
                    "parameters_used": {
                        "crawler_options": {
                            "limit": limit,
                            "max_depth": max_depth,
                            "include_paths": include_paths,
                            "exclude_paths": exclude_paths,
                            "allow_backward_links": allow_backward_links,
                            "allow_external_links": allow_external_links,
                            "allow_subdomains": allow_subdomains,
                            "delay": delay
                        },
                        "scrape_options": {
                            "format": format,
                            "agent_model": agent_model,
                            "agent_prompt": agent_prompt,
                            "only_main_content": only_main_content,
                            "include_tags": include_tags,
                            "exclude_tags": exclude_tags,
                            "wait_for": wait_for,
                            "timeout": timeout
                        }
                    }
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No data returned from Firecrawl crawl"
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)
    
    @mcp.tool()
    async def crawl_web_async(
        ctx: Context,
        url: str,
        # Crawler Options
        limit: int = 10,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_depth: int = 5,
        allow_backward_links: bool = False,
        allow_external_links: bool = False,
        allow_subdomains: bool = False,
        delay: Optional[int] = None,
        # Scrape Options (same as scrape_single_page)
        format: str = "markdown",
        agent_model: str = "FIRE-1",
        agent_prompt: Optional[str] = None,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        wait_for: int = 2000,
        timeout: int = 15000,
        # Storage Options
        chunk_size: int = 5000,
        # Chunking Options
        chunking_strategy: str = "character",
        chunking_prompt: Optional[str] = None
    ) -> str:
        """
        Asynchronously crawl multiple web pages using Firecrawl and return job information.
        
        This tool starts an asynchronous crawl job using Firecrawl. The job will crawl
        the website in the background. Use this for large websites where you don't want
        to wait for completion. You'll get a job ID that you can use to check status.
        
        Args:
            ctx: The MCP server provided context
            url: Starting URL to crawl from
            
            # CRAWLER OPTIONS (control which pages to visit):
            limit: Maximum number of pages to crawl. Use lower values for focused crawls, higher for comprehensive site coverage (default: 10)
            include_paths: List of regex patterns for URLs to include. Only URLs matching these patterns will be crawled. Examples: ["^/docs/.*$", "^/api/.*$"] for docs and API pages only
            exclude_paths: List of regex patterns for URLs to exclude. URLs matching these patterns will be skipped. Examples: ["^/admin/.*$", "^/private/.*$"] to avoid admin/private areas
            max_depth: Maximum crawl depth from starting URL. Depth 1 = direct links, depth 2 = links from those pages, etc. Higher values crawl deeper into site structure (default: 5)
            allow_backward_links: Whether to follow internal links to sibling or parent URLs, not just child paths. False = only crawl deeper URLs, True = crawl any internal links including navigation (default: False)
            allow_external_links: Whether to follow links that point to external domains. Be careful as this can cause unlimited crawling (default: False)
            allow_subdomains: Whether to follow links to subdomains of the main domain. For example, if crawling example.com, this allows blog.example.com (default: False)
            delay: Delay in seconds (integer) between page scrapes to respect rate limits and avoid overwhelming the target website. If not provided, respects robots.txt crawl delay (default: None)
            
            # SCRAPE OPTIONS (control how each page is processed):
            format: Format for scraped content. Options: "markdown", "html", "text". Markdown is best for RAG applications (default: "markdown")
            agent_model: AI model to use for agent mode when agent_prompt is provided. Options: "FIRE-1", "gpt-4o", etc. (default: "FIRE-1")
            agent_prompt: Optional prompt for AI agent mode. If provided, the AI agent will interact with the page using this prompt before scraping. Enables dynamic content interaction (default: None)
            only_main_content: Whether to extract only main content, removing navigation, ads, footers, etc. True = cleaner content for RAG, False = full page content (default: True)
            include_tags: List of HTML tags/CSS selectors to specifically include in scraping. Examples: ["h1", "h2", "p", ".main-content", "#article-body"] (default: None = include all)
            exclude_tags: List of HTML tags/CSS selectors to exclude from scraping. Examples: ["#footer", ".ad", ".navigation", "#sidebar"] (default: None = exclude nothing)
            wait_for: Time in milliseconds to wait after page load before scraping. Useful for pages with dynamic content, JavaScript loading, etc. (default: 2000)
            timeout: Maximum time in milliseconds to wait for page load. Increase for slow-loading pages (default: 15000)
            
            # STORAGE OPTIONS:
            chunk_size: Maximum size in characters for each content chunk stored in database. Smaller chunks = more granular search, larger chunks = more context per result (default: 5000)
            
            # CHUNKING OPTIONS:
            chunking_strategy: Strategy for chunking content - "character" (default), "document", or "llm"
                - "character": Smart character-based chunking with configurable size (respects code blocks/paragraphs)
                - "document": Store entire document as single chunk (useful for known document types)
                - "llm": Use LLM to intelligently determine chunk boundaries based on content structure
            chunking_prompt: For LLM strategy - custom prompt to guide chunking decisions (optional)
        
        Returns:
            JSON string with async job information including: job ID for status checking, crawl parameters used, and note about retrieving results when completed
        """
        try:
            # Initialize Firecrawl
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "FIRECRAWL_API_KEY not found in environment variables"
                }, indent=2)
                
            app = FirecrawlApp(api_key=api_key)
            
            # Configure scrape options using ScrapeOptions class
            from firecrawl.firecrawl import ScrapeOptions
            scrape_options_dict = {
                'formats': [format] if format != "markdown" else ['markdown'],
                'onlyMainContent': only_main_content,
                'waitFor': wait_for,
                'timeout': timeout
            }
            
            # Add include/exclude tags if provided
            if include_tags:
                scrape_options_dict['includeTags'] = include_tags
            if exclude_tags:
                scrape_options_dict['excludeTags'] = exclude_tags
            
            # Add agent configuration if prompt is provided
            if agent_prompt:
                scrape_options_dict['agent'] = {
                    'model': agent_model,
                    'prompt': agent_prompt
                }
            
            scrape_options = ScrapeOptions(**scrape_options_dict)
            
            # Configure crawl parameters with correct parameter names
            crawl_params = {
                'limit': limit,
                'max_depth': max_depth,
                'allow_backward_links': allow_backward_links,
                'allow_external_links': allow_external_links,
                'allow_subdomains': allow_subdomains,
                'scrape_options': scrape_options
            }
            
            # Add path restrictions if provided
            if include_paths:
                crawl_params['include_paths'] = include_paths
            if exclude_paths:
                crawl_params['exclude_paths'] = exclude_paths
            
            # Add delay if provided
            if delay is not None:
                crawl_params['delay'] = delay
            
            # Start async crawl
            result = app.async_crawl_url(url, **crawl_params)
            
            if result and hasattr(result, 'id'):
                job_id = result.id
                return json.dumps({
                    "success": True,
                    "crawl_type": "firecrawl_async",
                    "url": url,
                    "job_id": job_id,
                    "status": "started",
                    "parameters_used": {
                        "crawler_options": {
                            "limit": limit,
                            "max_depth": max_depth,
                            "include_paths": include_paths,
                            "exclude_paths": exclude_paths,
                            "allow_backward_links": allow_backward_links,
                            "allow_external_links": allow_external_links,
                            "allow_subdomains": allow_subdomains,
                            "delay": delay
                        },
                        "scrape_options": {
                            "format": format,
                            "agent_model": agent_model,
                            "agent_prompt": agent_prompt,
                            "only_main_content": only_main_content,
                            "include_tags": include_tags,
                            "exclude_tags": exclude_tags,
                            "wait_for": wait_for,
                            "timeout": timeout
                        },
                        "chunking_options": {
                            "chunking_strategy": chunking_strategy,
                            "chunk_size": chunk_size,
                            "chunking_prompt": chunking_prompt
                        }
                    },
                    "note": "Use the job_id to check crawl status and retrieve results when completed"
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "Failed to start async crawl - no job ID returned"
                }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)

# Original Crawl4AI tool - only available when Firecrawl is NOT available
if not FIRECRAWL_AVAILABLE:
    @mcp.tool()
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
                # Chunk the content using the specified strategy
                if chunking_strategy == "document":
                    chunks = chunk_document_single(md)
                elif chunking_strategy == "llm":
                    chunks = chunk_document_llm(md, chunking_prompt, source_url)
                else:  # Default to "character" strategy
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
                    code_blocks = extract_code_blocks_with_llm(md, source_url)
                    
                    if code_blocks:
                        # Prepare code example data - LLM already provides summaries
                        parsed_url = urlparse(source_url)
                        source_id = parsed_url.netloc or parsed_url.path
                        
                        for i, block in enumerate(code_blocks):
                            code_urls.append(source_url)
                            code_chunk_numbers.append(len(code_examples))  # Use global code example index
                            code_examples.append(block['code'])
                            code_summaries.append(block.get('summary', f"{block.get('language', 'code')} example"))
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": len(code_examples) - 1,
                                "url": source_url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split()),
                                "language": block.get('language', ''),
                                "type": block.get('type', 'code_block'),
                                "extraction_method": "llm"
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
                "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else []),
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size if chunking_strategy == "character" else None,
                "chunking_prompt_used": chunking_prompt is not None if chunking_strategy == "llm" else None
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
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